import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# ========= 配置 =========
DATA_DIR = Path("C:/pycharm/DL_Project_1/rental-product-recommendation-system")

TRAIN_VISITS = DATA_DIR / "metrika_visits.csv"
TRAIN_HITS   = DATA_DIR / "metrika_hits.csv"
TEST_VISITS  = DATA_DIR / "metrika_visits_test.csv"
TEST_HITS    = DATA_DIR / "metrika_hits_test.csv"

OLD_PROD = DATA_DIR / "old_site_products.csv"
NEW_PROD = DATA_DIR / "new_site_products.csv"

OUT_SUB  = DATA_DIR / "submission_itemcf.csv"

K_RECS = 6
TOP_SIM_PER_ITEM = 200   # 每个商品只保留最相似的前N个，省内存/加速
CAND_LIMIT = 5000        # 候选池上限（防止极端慢）


# ========= 工具函数 =========
def read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")
    return df

def to_int_str(x):
    if pd.isna(x):
        return None
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()

def parse_watch_ids(x):
    if pd.isna(x) or x == "":
        return []
    try:
        return json.loads(x)
    except Exception:
        s = str(x).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [p.strip().strip('"').strip("'") for p in s.split(",") if p.strip()]
        return parts


# ========= 1) 读数据 =========
print("Loading data...")
visits_train = read_csv_auto(TRAIN_VISITS)
hits_train   = read_csv_auto(TRAIN_HITS)
visits_test  = read_csv_auto(TEST_VISITS)
hits_test    = read_csv_auto(TEST_HITS)

old_prod = read_csv_auto(OLD_PROD)
new_prod = read_csv_auto(NEW_PROD)

# slug->product_id 映射（按 project_id）
old_map = old_prod[["id", "slug"]].rename(columns={"id": "product_id"}).copy()
old_map["project_id"] = 1
new_map = new_prod[["id", "slug"]].rename(columns={"id": "product_id"}).copy()
new_map["project_id"] = 0
slug2pid = pd.concat([old_map, new_map], ignore_index=True)

# ========= 2) 训练集：visit_id -> watch_id -> hits -> (visit_id, product_id) =========
print("Building train interactions...")
vw = visits_train[["project_id", "visit_id", "watch_ids"]].copy()
vw["visit_id"] = vw["visit_id"].apply(to_int_str)
vw["watch_list"] = vw["watch_ids"].apply(parse_watch_ids)
vw = vw.explode("watch_list").rename(columns={"watch_list": "watch_id"}).dropna(subset=["watch_id"])
vw["watch_id"] = vw["watch_id"].apply(to_int_str)

hits_train["watch_id"] = hits_train["watch_id"].apply(to_int_str)

hits_j = hits_train.merge(
    vw[["project_id", "visit_id", "watch_id"]],
    on=["project_id", "watch_id"],
    how="inner"
)

# 只用商品页浏览作为交互（起步最稳）
pv = hits_j[(hits_j["page_type"] == "PRODUCT") & (hits_j["is_page_view"] == 1)][["project_id","visit_id","slug"]].copy()
pv = pv.merge(slug2pid, on=["project_id","slug"], how="left").dropna(subset=["product_id"])
pv["product_id"] = pv["product_id"].astype("int64")

pv = pv.drop_duplicates(subset=["visit_id","product_id"])

# session -> items
train_session_items = pv.groupby("visit_id")["product_id"].apply(list).to_dict()

# 全局热门（兜底）
popular_products = pv["product_id"].value_counts().index.tolist()

print("Train sessions:", len(train_session_items))
print("Popular top6:", popular_products[:6])

# ========= 3) 训练 ItemCF 相似度（共现 + 归一化） =========
print("Training ItemCF similarity (co-occurrence)...")

# item出现次数
item_cnt = Counter()
# 共现计数：sim[i][j] += 1
co = defaultdict(Counter)

for items in train_session_items.values():
    # 去重（同一session重复浏览不重复算）
    uniq = list(dict.fromkeys(items))
    for i in uniq:
        item_cnt[i] += 1
    # 共现：两两组合（O(n^2)，但每个session通常不长）
    L = len(uniq)
    for a in range(L):
        i = uniq[a]
        for b in range(L):
            if a == b:
                continue
            j = uniq[b]
            co[i][j] += 1

# 归一化（类似 cosine）：c_ij / sqrt(cnt_i * cnt_j)
# 并为每个 item 只保留 topN 相似项
item_sim = {}
for i, nbrs in co.items():
    sims = []
    ci = item_cnt[i]
    for j, cij in nbrs.items():
        cj = item_cnt[j]
        s = cij / (np.sqrt(ci * cj) + 1e-12)
        sims.append((j, s))
    sims.sort(key=lambda x: x[1], reverse=True)
    item_sim[i] = sims[:TOP_SIM_PER_ITEM]

print("Built item_sim for items:", len(item_sim))

# ========= 4) Test：用 visits_test 的 watch_ids 把 hits_test 归属到 visit_id =========
print("Preparing test session items from hits_test (via watch_id -> visit_id)...")

# 4.1 visits_test: visit_id -> watch_id
vt = visits_test[["project_id", "visit_id", "watch_ids"]].copy()
vt["visit_id"] = vt["visit_id"].apply(to_int_str)
vt["watch_list"] = vt["watch_ids"].apply(parse_watch_ids)
vt = vt.explode("watch_list").rename(columns={"watch_list": "watch_id"}).dropna(subset=["watch_id"])
vt["watch_id"] = vt["watch_id"].apply(to_int_str)

# 4.2 hits_test: watch_id 统一格式
hits_test["watch_id"] = hits_test["watch_id"].apply(to_int_str)

# 4.3 join：把 hits_test 归到 visit_id
hits_test_j = hits_test.merge(
    vt[["project_id", "visit_id", "watch_id"]],
    on=["project_id", "watch_id"],
    how="inner"
)

# 4.4 抽取 test 内的商品页浏览
pv_test = hits_test_j[
    (hits_test_j["page_type"] == "PRODUCT") &
    (hits_test_j["is_page_view"] == 1)
][["project_id", "visit_id", "slug"]].copy()

pv_test = pv_test.merge(slug2pid, on=["project_id", "slug"], how="left").dropna(subset=["product_id"])
pv_test["product_id"] = pv_test["product_id"].astype("int64")
pv_test = pv_test.drop_duplicates(subset=["visit_id", "product_id"])

test_session_items = pv_test.groupby("visit_id")["product_id"].apply(list).to_dict()

# test visit 列表（必须来自 visits_test）
visits_test["visit_id"] = visits_test["visit_id"].apply(to_int_str).astype(str)
test_visits = visits_test["visit_id"].tolist()


print("Test visits:", len(test_visits))
print("Test visits with some product views:", sum(v in test_session_items for v in test_visits))


# ========= 5) 推荐函数：对一个 visit 用 ItemCF 聚合相似商品 =========
def popular_fallback(seen_set, k=6):
    recs = []
    for pid in popular_products:
        if pid not in seen_set:
            recs.append(pid)
        if len(recs) == k:
            break
    if len(recs) < k:
        for pid in popular_products:
            if pid not in recs:
                recs.append(pid)
            if len(recs) == k:
                break
    return recs

def recommend_itemcf(vid, k=6):
    seen = test_session_items.get(vid, [])
    seen_set = set(seen)

    # 没有浏览记录：只能热门兜底
    if len(seen) == 0:
        return popular_fallback(seen_set, k=k)

    scores = Counter()
    # 对已看商品，累加其相似商品分数
    for i in seen:
        for j, s in item_sim.get(i, []):
            if j in seen_set:
                continue
            scores[j] += s

            # 防止极端情况候选过多
            if len(scores) > CAND_LIMIT:
                break

    # 如果没有候选（比如 seen 是训练里没出现过的商品），兜底热门
    if not scores:
        return popular_fallback(seen_set, k=k)

    recs = [pid for pid, _ in scores.most_common(k)]
    if len(recs) < k:
        extra = popular_fallback(seen_set, k=k)
        for pid in extra:
            if pid not in recs:
                recs.append(pid)
            if len(recs) == k:
                break

    # 确保无重复且长度=k
    recs = list(dict.fromkeys(recs))[:k]
    if len(recs) < k:
        for pid in popular_products:
            if pid not in recs and pid not in seen_set:
                recs.append(pid)
            if len(recs) == k:
                break
    return recs


# ========= 6) 生成 submission =========
print("Generating submission...")
out = []
for vid in test_visits:
    recs = recommend_itemcf(vid, k=K_RECS)
    out.append((vid, " ".join(map(str, recs))))
# 1) 永远从 visits_test 取官方要求的 key
visits_test = pd.read_csv(TEST_VISITS, sep=None, engine="python", dtype=str)
visits_test.columns = visits_test.columns.str.strip().str.replace("\ufeff", "")
test_visits = visits_test["visit_id"].str.strip().tolist()

# 2) 你前面推荐阶段产生一个字典：rec_by_visit[visit_id] = "id1 id2 ... id6"
# 这里我假设你已经构建好了 out 列表：out = [(vid, "a b c d e f"), ...]
rec_by_visit = dict(out)  # out 来自你的推荐循环

# 3) 用 test_visits 按顺序写出，缺的用热门兜底
def fallback_str(k=6):
    return " ".join(map(str, popular_products[:k]))

sub = pd.DataFrame({
    "visit_id": test_visits,
    "product_ids": [rec_by_visit.get(vid, fallback_str(6)) for vid in test_visits]
})

sub.to_csv(OUT_SUB, index=False)
print(sub.head())
