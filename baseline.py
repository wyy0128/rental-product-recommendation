import pandas as pd
import json
from pathlib import Path

DATA_DIR = Path("C:/pycharm/DL_Project_1/rental-product-recommendation-system")  # 改成你的数据目录

# ========= 1) 读入核心表 =========
visits_path = DATA_DIR / "metrika_visits.csv"
hits_path   = DATA_DIR / "metrika_hits.csv"

visits = pd.read_csv(visits_path, sep=None, engine="python")
hits   = pd.read_csv(hits_path,   sep=None, engine="python")


# ========= 2) visit_id -> watch_id 映射（关键！）=========
# visits.watch_ids 是形如 ["6973...","6973..."] 的 JSON 字符串
def parse_watch_ids(x):
    if pd.isna(x) or x == "":
        return []
    try:
        return json.loads(x)
    except Exception:
        # 有时会有奇怪格式，兜底
        x = x.strip()
        if x.startswith("[") and x.endswith("]"):
            x = x[1:-1]
        return [s.strip().strip('"') for s in x.split(",") if s.strip()]

visit_watch = visits[["project_id", "visit_id", "watch_ids", "date_time"]].copy()
visit_watch["watch_id_list"] = visit_watch["watch_ids"].apply(parse_watch_ids)
visit_watch = visit_watch.explode("watch_id_list").rename(columns={"watch_id_list": "watch_id"})
visit_watch = visit_watch.dropna(subset=["watch_id"])

# 注意：你文件里 watch_id / visit_id 经常被读成科学计数法浮点
# 我们统一转成“整数字符串”，避免 merge 对不上
def to_int_str(s):
    # 处理 6.97E+18 这种
    if pd.isna(s):
        return None
    try:
        return str(int(float(s)))
    except Exception:
        return str(s)

visit_watch["watch_id"] = visit_watch["watch_id"].apply(to_int_str)
visit_watch["visit_id"] = visit_watch["visit_id"].apply(to_int_str)

# hits 里的 watch_id 在你的 header 叫 watch_id（第一列后面）
hits = hits.rename(columns={"watch_id": "watch_id"})  # 保持一致
hits["watch_id"] = hits["watch_id"].apply(to_int_str)

# ========= 3) 把 hit 事件归到 visit_id（watch_id join）=========
hits_joined = hits.merge(
    visit_watch[["project_id", "visit_id", "watch_id"]],
    on=["project_id", "watch_id"],
    how="inner"
)

# ========= 4) 从 hits 中抽“商品浏览”事件 =========
# 最稳的起步定义：page_type=='PRODUCT' 且 is_page_view==1
prod_views = hits_joined[
    (hits_joined["page_type"] == "PRODUCT") &
    (hits_joined["is_page_view"] == 1)
].copy()

# ========= 5) slug -> product_id 映射（用产品表）=========
old_prod_path = DATA_DIR / "old_site_products.csv"
new_prod_path = DATA_DIR / "new_site_products.csv"

old_prod = pd.read_csv(old_prod_path)
new_prod = pd.read_csv(new_prod_path)

# 统一列名：id, slug
old_map = old_prod[["id", "slug"]].rename(columns={"id": "product_id"})
old_map["project_id"] = 1  # old site
new_map = new_prod[["id", "slug"]].rename(columns={"id": "product_id"})
new_map["project_id"] = 0  # new site

slug2pid = pd.concat([old_map, new_map], ignore_index=True)

# join：用 (project_id, slug) 把 slug 映射成 product_id
prod_views = prod_views.merge(
    slug2pid,
    on=["project_id", "slug"],
    how="left"
)

# 丢掉无法映射商品ID的记录
prod_views = prod_views.dropna(subset=["product_id"])

# 统一 product_id 类型为 int
prod_views["product_id"] = prod_views["product_id"].astype("int64")

# ========= 6) 得到 session(visit_id) 的交互表 =========
interactions = prod_views[["visit_id", "product_id", "date_time", "project_id"]].copy()
interactions["date_time"] = pd.to_datetime(interactions["date_time"], errors="coerce")

# 去重：同一 session 反复看同一商品，只算一次（baseline）
interactions = interactions.drop_duplicates(subset=["visit_id", "product_id"])

print("Interactions sample:")
print(interactions.head())

# ========= 7) Popular baseline：全局最热门商品TopN =========
popular = interactions["product_id"].value_counts()
top_products = popular.index.tolist()

# ========= 8) 读取 test set 的 visit_id =========
# 你需要把下面这个文件名改成你数据里实际的 test 文件
# 这个文件应该至少有一列：visit_id
test_path = DATA_DIR / "metrika_visits_test.csv"  # <-- 改这里（可能是 test_visits.csv 等）
test_df = pd.read_csv(test_path)

# 确保列名是 visit_id
if "visit_id" not in test_df.columns:
    raise ValueError(f"test file must contain 'visit_id' column, but got: {test_df.columns.tolist()}")

test_df["visit_id"] = test_df["visit_id"].apply(to_int_str)

# ========= 9) 为每个 test visit_id 生成 6 个推荐 =========
# 可选：排除该 session 已经看过的商品（更合理）
seen_by_visit = interactions.groupby("visit_id")["product_id"].apply(set).to_dict()

def recommend_for_visit(vid, k=6):
    seen = seen_by_visit.get(vid, set())
    recs = []
    for pid in top_products:
        if pid in seen:
            continue
        recs.append(pid)
        if len(recs) == k:
            break
    # 如果热门里不够（极少），就补齐（不排除）
    if len(recs) < k:
        for pid in top_products:
            if pid not in recs:
                recs.append(pid)
            if len(recs) == k:
                break
    return recs

sub = pd.DataFrame({
    "visit_id": test_df["visit_id"],
    "product_ids": test_df["visit_id"].apply(lambda vid: " ".join(map(str, recommend_for_visit(vid, k=6))))
})



out_path = DATA_DIR / "submission.csv"
sub.to_csv(out_path, index=False)
print(f"Saved submission to: {out_path}")
print(sub.head())
