import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# =================== 路径：按你文件夹写死 ===================
DATA_DIR = Path(r"C:\pycharm\DL_Project_1\rental-product-recommendation-system")

TRAIN_VISITS = DATA_DIR / "metrika_visits.csv"
TRAIN_HITS   = DATA_DIR / "metrika_hits.csv"
TEST_VISITS  = DATA_DIR / "metrika_visits_test.csv"
TEST_HITS    = DATA_DIR / "metrika_hits_test.csv"

OLD_PROD = DATA_DIR / "old_site_products.csv"
NEW_PROD = DATA_DIR / "new_site_products.csv"

OUT_SUB  = DATA_DIR / "submission_itemcf_final.csv"


# =================== 超参（先用稳的） ===================
K_RECS = 6
TOP_SIM_PER_ITEM = 200   # 每个商品保留前 N 个相似商品（省内存）
USE_IUF = True           # IUF：惩罚过热商品，通常更好


# =================== 工具函数 ===================
def safe_id(x):
    s = "" if x is None else str(x).strip()
    if s.endswith(".0"):  # 有些CSV会把整数写成 123.0
        s = s[:-2]
    return s


def read_csv_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="utf-8")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")
    return df

def to_int_str(x):
    """把科学计数法/float/str 全部统一成整数字符串（visit_id/watch_id 等非常大）"""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return str(int(float(s)))
    except Exception:
        return s

def parse_watch_ids(x):
    if x is None:
        return []
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return []
    try:
        return json.loads(s)
    except Exception:
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [p.strip().strip('"').strip("'") for p in s.split(",") if p.strip()]
        return parts


# =================== 0) 读文件 ===================
print("Loading files...")
visits_train = read_csv_auto(TRAIN_VISITS)
hits_train   = read_csv_auto(TRAIN_HITS)
visits_test  = read_csv_auto(TEST_VISITS)
hits_test    = read_csv_auto(TEST_HITS)

old_prod = read_csv_auto(OLD_PROD)
new_prod = read_csv_auto(NEW_PROD)

# 基础字段检查
for name, df, cols in [
    ("metrika_visits.csv", visits_train, ["project_id", "visit_id", "watch_ids"]),
    ("metrika_hits.csv", hits_train, ["project_id", "watch_id", "page_type", "is_page_view", "slug"]),
    ("metrika_visits_test.csv", visits_test, ["project_id", "visit_id", "watch_ids"]),
    ("metrika_hits_test.csv", hits_test, ["project_id", "watch_id", "page_type", "is_page_view", "slug"]),
]:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing columns: {miss}")

# =================== 1) slug -> product_id 映射 ===================
print("Building slug -> product_id mapping...")
# 旧站 project_id=1，新站 project_id=0（按你数据描述）
old_map = old_prod[["id", "slug"]].rename(columns={"id": "product_id"}).copy()
old_map["project_id"] = "1"

new_map = new_prod[["id", "slug"]].rename(columns={"id": "product_id"}).copy()
new_map["project_id"] = "0"

slug2pid = pd.concat([old_map, new_map], ignore_index=True)
slug2pid["project_id"] = slug2pid["project_id"].astype(str)
slug2pid["slug"] = slug2pid["slug"].astype(str)
slug2pid["product_id"] = slug2pid["product_id"].apply(to_int_str)

# =================== 2) 训练集：visit_id -> watch_id -> product_id ===================
print("Building train interactions...")
visits_train["project_id"] = visits_train["project_id"].astype(str)
visits_train["visit_id"] = visits_train["visit_id"].apply(to_int_str)

vw = visits_train[["project_id", "visit_id", "watch_ids"]].copy()
vw["watch_list"] = vw["watch_ids"].apply(parse_watch_ids)
vw = vw.explode("watch_list").rename(columns={"watch_list": "watch_id"}).dropna(subset=["watch_id"])
vw["watch_id"] = vw["watch_id"].apply(to_int_str)

hits_train["project_id"] = hits_train["project_id"].astype(str)
hits_train["watch_id"] = hits_train["watch_id"].apply(to_int_str)

hits_j = hits_train.merge(
    vw[["project_id", "visit_id", "watch_id"]],
    on=["project_id", "watch_id"],
    how="inner"
)

pv = hits_j[(hits_j["page_type"] == "PRODUCT") & (hits_j["is_page_view"] == "1")][["project_id","visit_id","slug"]].copy()
pv["slug"] = pv["slug"].astype(str)

pv = pv.merge(slug2pid, on=["project_id","slug"], how="left").dropna(subset=["product_id"])
pv["product_id"] = pv["product_id"].apply(to_int_str)
pv = pv.drop_duplicates(subset=["project_id","visit_id","product_id"])

# session -> items（按站点分开存）
train_session_items = (
    pv.groupby(["project_id","visit_id"])["product_id"]
      .apply(lambda x: list(dict.fromkeys(x.tolist())))
      .to_dict()
)

# 分站点热门兜底
popular_by_proj = (
    pv.groupby("project_id")["product_id"]
      .value_counts()
      .groupby(level=0)
      .apply(lambda s: s.index.get_level_values(1).tolist())
      .to_dict()
)

print("Train sessions:", len(train_session_items))
print("Popular[0] top6:", popular_by_proj.get("0", [])[:6])
print("Popular[1] top6:", popular_by_proj.get("1", [])[:6])

# =================== 3) 训练 ItemCF 相似度（按 project_id 分开） ===================
print("Training ItemCF similarity per project_id...")
item_sim_by_proj = {}
item_cnt_by_proj = {}

for proj in ["0", "1"]:
    sessions = [items for (p, _vid), items in train_session_items.items() if p == proj]
    if not sessions:
        item_sim_by_proj[proj] = {}
        item_cnt_by_proj[proj] = Counter()
        print(f"project {proj}: no sessions")
        continue

    item_cnt = Counter()
    co = defaultdict(Counter)

    for items in sessions:
        uniq = list(dict.fromkeys(items))
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
    print(f"project {proj}: item_sim items={len(item_sim)}, distinct items={len(item_cnt)}")

# =================== 4) Test：用 visits_test 的 watch_ids 把 hits_test 归到 visit_id ===================
print("Building test session items via watch_id -> visit_id mapping...")
visits_test["project_id"] = visits_test["project_id"].astype(str)
visits_test["visit_id"] = visits_test["visit_id"].map(safe_id)

# 这就是 Kaggle 官方要求的 key（必须完全匹配）
test_keys_df = visits_test[["project_id", "visit_id"]].copy()
test_visit_ids = test_keys_df["visit_id"].tolist()

vt = visits_test[["project_id","visit_id","watch_ids"]].copy()
vt["watch_list"] = vt["watch_ids"].apply(parse_watch_ids)
vt = vt.explode("watch_list").rename(columns={"watch_list": "watch_id"}).dropna(subset=["watch_id"])
vt["watch_id"] = vt["watch_id"].apply(to_int_str)

hits_test["project_id"] = hits_test["project_id"].astype(str)
hits_test["watch_id"] = hits_test["watch_id"].map(safe_id)

hits_test_j = hits_test.merge(
    vt[["project_id","visit_id","watch_id"]],
    on=["project_id","watch_id"],
    how="inner"
)

pv_test = hits_test_j[(hits_test_j["page_type"] == "PRODUCT") & (hits_test_j["is_page_view"] == "1")][["project_id","visit_id","slug"]].copy()
pv_test["slug"] = pv_test["slug"].astype(str)

pv_test = pv_test.merge(slug2pid, on=["project_id","slug"], how="left").dropna(subset=["product_id"])
pv_test["product_id"] = pv_test["product_id"].apply(to_int_str)
pv_test = pv_test.drop_duplicates(subset=["project_id","visit_id","product_id"])

test_session_items = (
    pv_test.groupby(["project_id","visit_id"])["product_id"]
          .apply(lambda x: list(dict.fromkeys(x.tolist())))
          .to_dict()
)

with_views = sum((p, v) in test_session_items for p, v in zip(test_keys_df["project_id"], test_keys_df["visit_id"]))
print("Test visits:", len(test_visit_ids))
print("Test visits with some product views:", with_views)

# =================== 5) 推荐函数（ItemCF + 分站点热门兜底） ===================
def popular_fallback(proj: str, seen_set: set, k=6):
    pop = popular_by_proj.get(proj, [])
    recs = []
    for pid in pop:
        if pid not in seen_set:
            recs.append(pid)
        if len(recs) == k:
            break
    # 极端补齐
    if len(recs) < k:
        pop2 = (popular_by_proj.get("0", []) + popular_by_proj.get("1", []))
        for pid in pop2:
            if pid not in recs and pid not in seen_set:
                recs.append(pid)
            if len(recs) == k:
                break
    return recs

def recommend_itemcf(proj: str, vid: str, k=6):
    seen = test_session_items.get((proj, vid), [])
    seen_set = set(seen)

    if len(seen) == 0:
        return popular_fallback(proj, seen_set, k=k)

    scores = Counter()
    item_sim = item_sim_by_proj.get(proj, {})
    item_cnt = item_cnt_by_proj.get(proj, Counter())

    for i in seen:
        w = 1.0
        if USE_IUF:
            ci = item_cnt.get(i, 1)
            w = 1.0 / np.log1p(ci)
        for j, s in item_sim.get(i, []):
            if j in seen_set:
                continue
            scores[j] += s * w

    if not scores:
        return popular_fallback(proj, seen_set, k=k)

    recs = [pid for pid, _ in scores.most_common(k)]
    if len(recs) < k:
        extra = popular_fallback(proj, seen_set, k=k)
        for pid in extra:
            if pid not in recs:
                recs.append(pid)
            if len(recs) == k:
                break

    # 保证无重复且长度=k
    recs = list(dict.fromkeys(recs))[:k]
    if len(recs) < k:
        extra = popular_fallback(proj, seen_set, k=k)
        for pid in extra:
            if pid not in recs:
                recs.append(pid)
            if len(recs) == k:
                break
    return recs

# =================== 6) 生成 submission（关键：visit_id 只来自 visits_test） ===================
print("Generating submission with strict key alignment...")
rows = []
for proj, vid in zip(test_keys_df["project_id"], test_keys_df["visit_id"]):
    recs = recommend_itemcf(proj, vid, k=K_RECS)
    rows.append((vid, " ".join(map(str, recs))))

sub = pd.DataFrame(rows, columns=["visit_id", "product_ids"])

# =================== 7) 你要求的验证：所有 visit_id 必须在 metrika_visits_test.csv 中 ===================
test_set = set(test_visit_ids)
sub_set = set(sub["visit_id"].tolist())

missing = test_set - sub_set
extra = sub_set - test_set

print("Validation:")
print("  test unique:", len(test_set))
print("  sub  unique:", len(sub_set))
print("  missing in submission:", len(missing))
print("  extra in submission:", len(extra))

# 强制保证 0 missing / 0 extra，否则直接失败，不让你提交错误文件
assert len(missing) == 0 and len(extra) == 0, (
    f"visit_id mismatch! missing={len(missing)} extra={len(extra)}"
)

sub.to_csv(OUT_SUB, index=False)
print("Saved:", OUT_SUB)

# 打印原始前3行，确认没有科学计数法
with open(OUT_SUB, "r", encoding="utf-8") as f:
    print("---- raw preview ----")
    for _ in range(3):
        print(f.readline().rstrip())
