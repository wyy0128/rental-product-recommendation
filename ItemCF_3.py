import json
import csv
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# =================== 路径（按你的文件夹） ===================
DATA_DIR = Path(r"C:\pycharm\DL_Project_1\rental-product-recommendation-system")

TRAIN_VISITS = DATA_DIR / "metrika_visits.csv"
TRAIN_HITS   = DATA_DIR / "metrika_hits.csv"
TEST_VISITS  = DATA_DIR / "metrika_visits_test.csv"
TEST_HITS    = DATA_DIR / "metrika_hits_test.csv"

OLD_PROD = DATA_DIR / "old_site_products.csv"
NEW_PROD = DATA_DIR / "new_site_products.csv"

OUT_SUB  = DATA_DIR / "submission_itemcf_full.csv"


# =================== 超参 ===================
K_RECS = 6
TOP_SIM_PER_ITEM = 200   # 每个 item 最多保留 topN 相似邻居（省内存）
USE_IUF = True           # 惩罚超热门 item 的影响，通常更好


# =================== 工具函数（ID 一律用字符串） ===================
def read_csv_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")
    return df

def safe_id(x):
    """只做字符串清洗，绝不 float/int，避免19位整数精度丢失"""
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    # 有些 CSV 会把整数写成 "123.0"
    if s.endswith(".0"):
        s = s[:-2]
    # 去掉首尾引号
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1]
    return s.strip()

def parse_watch_ids(x):
    s = safe_id(x)
    if not s:
        return []
    try:
        arr = json.loads(s)
        return [safe_id(v) for v in arr if safe_id(v)]
    except Exception:
        # 兜底解析
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [p.strip().strip('"').strip("'") for p in s.split(",") if p.strip()]
        return [safe_id(p) for p in parts if safe_id(p)]

def dedup_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# =================== 0) 读文件 + 检查列 ===================
print("Loading files...")
visits_train = read_csv_auto(TRAIN_VISITS)
hits_train   = read_csv_auto(TRAIN_HITS)
visits_test  = read_csv_auto(TEST_VISITS)
hits_test    = read_csv_auto(TEST_HITS)
old_prod     = read_csv_auto(OLD_PROD)
new_prod     = read_csv_auto(NEW_PROD)

required = {
    "visits_train": ["project_id", "visit_id", "watch_ids"],
    "hits_train":   ["project_id", "watch_id", "page_type", "is_page_view", "slug"],
    "visits_test":  ["project_id", "visit_id", "watch_ids"],
    "hits_test":    ["project_id", "watch_id", "page_type", "is_page_view", "slug"],
    "old_prod":     ["id", "slug"],
    "new_prod":     ["id", "slug"],
}
for name, cols in required.items():
    df = locals()[name]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing columns: {miss}")

# 清洗关键列（ID 用字符串）
for df, col in [(visits_train,"visit_id"), (visits_test,"visit_id")]:
    df[col] = df[col].map(safe_id)
for df, col in [(hits_train,"watch_id"), (hits_test,"watch_id")]:
    df[col] = df[col].map(safe_id)

for df in [visits_train, visits_test, hits_train, hits_test]:
    df["project_id"] = df["project_id"].map(safe_id)

old_prod["id"] = old_prod["id"].map(safe_id)
new_prod["id"] = new_prod["id"].map(safe_id)
old_prod["slug"] = old_prod["slug"].astype(str)
new_prod["slug"] = new_prod["slug"].astype(str)


# =================== 1) slug -> product_id（按 project_id） ===================
print("Building slug->product_id...")
old_map = old_prod[["id","slug"]].rename(columns={"id":"product_id"}).copy()
old_map["project_id"] = "1"
new_map = new_prod[["id","slug"]].rename(columns={"id":"product_id"}).copy()
new_map["project_id"] = "0"
slug2pid = pd.concat([old_map, new_map], ignore_index=True)
slug2pid["slug"] = slug2pid["slug"].astype(str)
slug2pid["product_id"] = slug2pid["product_id"].map(safe_id)


# =================== 2) TRAIN: visits_train 的 watch_ids 展开 -> join hits_train -> 得到 (visit_id, product_id) ===================
print("Building train interactions...")
vw = visits_train[["project_id","visit_id","watch_ids"]].copy()
vw["watch_list"] = vw["watch_ids"].apply(parse_watch_ids)
vw = vw.explode("watch_list").rename(columns={"watch_list":"watch_id"}).dropna(subset=["watch_id"])
vw["watch_id"] = vw["watch_id"].map(safe_id)

hits_j = hits_train.merge(
    vw[["project_id","visit_id","watch_id"]],
    on=["project_id","watch_id"],
    how="inner"
)

pv = hits_j[(hits_j["page_type"] == "PRODUCT") & (hits_j["is_page_view"] == "1")][["project_id","visit_id","slug"]].copy()
pv["slug"] = pv["slug"].astype(str)

pv = pv.merge(slug2pid, on=["project_id","slug"], how="left").dropna(subset=["product_id"])
pv["product_id"] = pv["product_id"].map(safe_id)
pv = pv.drop_duplicates(subset=["project_id","visit_id","product_id"])

print("Train product-view interactions:", len(pv))
if len(pv) == 0:
    raise RuntimeError("No train interactions found. Check slug mapping / filters.")

# session -> items
train_session_items = (
    pv.groupby(["project_id","visit_id"])["product_id"]
      .apply(lambda x: dedup_keep_order([safe_id(v) for v in x.tolist() if safe_id(v)]))
      .to_dict()
)

# 站点热门兜底
popular_by_proj = (
    pv.groupby("project_id")["product_id"]
      .value_counts()
      .groupby(level=0)
      .apply(lambda s: s.index.get_level_values(1).tolist())
      .to_dict()
)

print("Popular[0] top6:", popular_by_proj.get("0", [])[:6])
print("Popular[1] top6:", popular_by_proj.get("1", [])[:6])


# =================== 3) 训练 ItemCF（按 project_id 分开） ===================
print("Training ItemCF similarity...")
item_sim_by_proj = {}
item_cnt_by_proj = {}

for proj in ["0","1"]:
    sessions = [items for (p, _vid), items in train_session_items.items() if p == proj]
    if not sessions:
        item_sim_by_proj[proj] = {}
        item_cnt_by_proj[proj] = Counter()
        print(f"project {proj}: no sessions")
        continue

    item_cnt = Counter()
    co = defaultdict(Counter)

    for items in sessions:
        uniq = dedup_keep_order(items)
        for i in uniq:
            item_cnt[i] += 1
        L = len(uniq)
        for a in range(L):
            i = uniq[a]
            for b in range(L):
                if a == b:
                    continue
                j = uniq[b]
                co[i][j] += 1

    item_sim = {}
    for i, nbrs in co.items():
        ci = item_cnt[i]
        sims = []
        for j, cij in nbrs.items():
            cj = item_cnt[j]
            s = cij / (np.sqrt(ci * cj) + 1e-12)
            sims.append((j, s))
        sims.sort(key=lambda x: x[1], reverse=True)
        item_sim[i] = sims[:TOP_SIM_PER_ITEM]

    item_sim_by_proj[proj] = item_sim
    item_cnt_by_proj[proj] = item_cnt
    print(f"project {proj}: distinct items={len(item_cnt)}, items with sim={len(item_sim)}")


# =================== 4) TEST: visits_test 的 watch_ids 展开 -> join hits_test -> 得到 test_session_items ===================
print("Building test session items (must work,否则全走热门)...")
test_keys_df = visits_test[["project_id","visit_id"]].copy()
test_keys_df["project_id"] = test_keys_df["project_id"].map(safe_id)
test_keys_df["visit_id"] = test_keys_df["visit_id"].map(safe_id)

test_visit_ids = test_keys_df["visit_id"].tolist()
test_set = set(test_visit_ids)

vt = visits_test[["project_id","visit_id","watch_ids"]].copy()
vt["watch_list"] = vt["watch_ids"].apply(parse_watch_ids)
vt = vt.explode("watch_list").rename(columns={"watch_list":"watch_id"}).dropna(subset=["watch_id"])
vt["watch_id"] = vt["watch_id"].map(safe_id)

hits_test_j = hits_test.merge(
    vt[["project_id","visit_id","watch_id"]],
    on=["project_id","watch_id"],
    how="inner"
)

pv_test = hits_test_j[(hits_test_j["page_type"] == "PRODUCT") & (hits_test_j["is_page_view"] == "1")][["project_id","visit_id","slug"]].copy()
pv_test["slug"] = pv_test["slug"].astype(str)

pv_test = pv_test.merge(slug2pid, on=["project_id","slug"], how="left").dropna(subset=["product_id"])
pv_test["product_id"] = pv_test["product_id"].map(safe_id)
pv_test = pv_test.drop_duplicates(subset=["project_id","visit_id","product_id"])

test_session_items = (
    pv_test.groupby(["project_id","visit_id"])["product_id"]
          .apply(lambda x: dedup_keep_order([safe_id(v) for v in x.tolist() if safe_id(v)]))
          .to_dict()
)

with_views = sum((p, v) in test_session_items for p, v in zip(test_keys_df["project_id"], test_keys_df["visit_id"]))
print("Test visits:", len(test_visit_ids), "with product views:", with_views)

# =================== 5) 推荐 + 去重补齐 ===================
def popular_fallback(proj: str, seen_set: set, k=6):
    pop = popular_by_proj.get(proj, [])
    recs = []
    for pid in pop:
        if pid not in seen_set and pid not in recs:
            recs.append(pid)
        if len(recs) == k:
            break
    # 极端补齐（跨站点）
    if len(recs) < k:
        pop2 = (popular_by_proj.get("0", []) + popular_by_proj.get("1", []))
        for pid in pop2:
            if pid not in seen_set and pid not in recs:
                recs.append(pid)
            if len(recs) == k:
                break
    return recs[:k]

def recommend_itemcf(proj: str, vid: str, k=6):
    seen = test_session_items.get((proj, vid), [])
    seen_set = set(seen)

    # 没有浏览记录 -> 兜底
    if not seen:
        return popular_fallback(proj, seen_set, k=k), False

    scores = Counter()
    item_sim = item_sim_by_proj.get(proj, {})
    item_cnt = item_cnt_by_proj.get(proj, Counter())

    for i in seen:
        w = 1.0
        if USE_IUF:
            w = 1.0 / np.log1p(item_cnt.get(i, 1))
        for j, s in item_sim.get(i, []):
            if j in seen_set:
                continue
            scores[j] += s * w

    if not scores:
        return popular_fallback(proj, seen_set, k=k), False

    recs = [pid for pid, _ in scores.most_common(k * 3)]  # 先取多一点再去重/补齐
    recs = dedup_keep_order(recs)
    recs = [pid for pid in recs if pid not in seen_set]
    recs = recs[:k]

    if len(recs) < k:
        recs = recs + [x for x in popular_fallback(proj, seen_set, k=k) if x not in recs]
        recs = recs[:k]

    return recs, True


# =================== 6) 生成 submission（严格 key 对齐 + 诊断不全一样） ===================
print("Generating submission...")
rows = []
used_itemcf = 0
for proj, vid in zip(test_keys_df["project_id"], test_keys_df["visit_id"]):
    recs, used = recommend_itemcf(proj, vid, k=K_RECS)
    used_itemcf += int(used)
    rows.append((vid, " ".join(map(str, recs))))

sub = pd.DataFrame(rows, columns=["visit_id","product_ids"])

# ---- 严格验证：visit_id 必须完全等于 test 的 key 集合 ----
sub_set = set(sub["visit_id"].tolist())
missing = test_set - sub_set
extra = sub_set - test_set
print("Key validation: missing =", len(missing), "extra =", len(extra))
assert len(missing) == 0 and len(extra) == 0, "visit_id mismatch! Do NOT submit this file."

# ---- 诊断：是否还会“全一样” ----
unique_lists = sub["product_ids"].nunique()
print("Diagnostics:")
print("  used ItemCF:", used_itemcf, f"({used_itemcf/len(sub):.1%})")
print("  unique recommendation lists:", unique_lists)
print("  example head:")
print(sub.head(5))

# 写出（可加引号保险）
sub.to_csv(OUT_SUB, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
print("Saved:", OUT_SUB)

# raw preview
with open(OUT_SUB, "r", encoding="utf-8") as f:
    print("---- raw preview ----")
    for _ in range(3):
        print(f.readline().rstrip())
