import json
import csv
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import lightgbm as lgb


# =================== 路径（按你的目录） ===================
DATA_DIR = Path(r"C:\pycharm\DL_Project_1\rental-product-recommendation-system")

TRAIN_VISITS = DATA_DIR / "metrika_visits.csv"
TRAIN_HITS   = DATA_DIR / "metrika_hits.csv"
TEST_VISITS  = DATA_DIR / "metrika_visits_test.csv"
TEST_HITS    = DATA_DIR / "metrika_hits_test.csv"

OLD_PROD = DATA_DIR / "old_site_products.csv"
NEW_PROD = DATA_DIR / "new_site_products.csv"
OLD2NEW  = DATA_DIR / "old_site_new_site_products.csv"

OLD_ORDERS = DATA_DIR / "old_site_orders.csv"
NEW_ORDERS = DATA_DIR / "new_site_orders.csv"

OUT_SUB = DATA_DIR / "submission_cf_lgbm.csv"

# =================== 超参 ===================
K_RECS = 6
TOP_SIM_PER_ITEM = 300          # ItemCF邻居数
CAND_PER_SESSION = 120          # 召回候选数（给 LGBM 重排）
NEG_PER_POS = 6                 # 训练时每个正样本配多少负样本
SEED = 42
USE_IUF = True
USE_POP_PENALTY = True

np.random.seed(SEED)


# =================== 工具函数（ID 一律字符串） ===================
def read_csv_auto(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "")
    return df

def safe_id(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    if s.endswith(".0"):
        s = s[:-2]
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


# =================== 1) slug -> product_id (按 project_id) ===================
def build_slug2pid(old_prod, new_prod):
    old_map = old_prod[["id","slug"]].rename(columns={"id":"product_id"}).copy()
    old_map["project_id"] = "1"
    new_map = new_prod[["id","slug"]].rename(columns={"id":"product_id"}).copy()
    new_map["project_id"] = "0"
    m = pd.concat([old_map, new_map], ignore_index=True)
    m["project_id"] = m["project_id"].map(safe_id)
    m["slug"] = m["slug"].astype(str)
    m["product_id"] = m["product_id"].map(safe_id)
    return m

def build_pv(visits_df, hits_df, slug2pid):
    vw = visits_df[["project_id","visit_id","watch_ids"]].copy()
    vw["project_id"] = vw["project_id"].map(safe_id)
    vw["visit_id"] = vw["visit_id"].map(safe_id)
    vw["watch_list"] = vw["watch_ids"].apply(parse_watch_ids)
    vw = vw.explode("watch_list").rename(columns={"watch_list":"watch_id"}).dropna(subset=["watch_id"])
    vw["watch_id"] = vw["watch_id"].map(safe_id)

    h = hits_df.copy()
    h["project_id"] = h["project_id"].map(safe_id)
    h["watch_id"] = h["watch_id"].map(safe_id)

    hj = h.merge(vw[["project_id","visit_id","watch_id"]], on=["project_id","watch_id"], how="inner")

    pv = hj[(hj["page_type"] == "PRODUCT") & (hj["is_page_view"] == "1")][["project_id","visit_id","slug"]].copy()
    pv["slug"] = pv["slug"].astype(str)

    pv = pv.merge(slug2pid, on=["project_id","slug"], how="left").dropna(subset=["product_id"])
    pv["product_id"] = pv["product_id"].map(safe_id)

    pv = pv.drop_duplicates(subset=["project_id","visit_id","product_id"])
    return pv

def build_session_items(pv: pd.DataFrame):
    return (
        pv.groupby(["project_id","visit_id"])["product_id"]
          .apply(lambda x: dedup_keep_order([safe_id(v) for v in x.tolist() if safe_id(v)]))
          .to_dict()
    )


# =================== 2) old->new 商品映射（跨站泛化） ===================
def build_old2new_map(old2new_df):
    if old2new_df is None or len(old2new_df) == 0:
        return {}
    # columns: old_site_id, new_site_id
    cols = old2new_df.columns
    if "old_site_id" not in cols or "new_site_id" not in cols:
        return {}
    mp = {}
    for _, r in old2new_df.iterrows():
        o = safe_id(r["old_site_id"])
        n = safe_id(r["new_site_id"])
        if o and n:
            mp[o] = n
    return mp

def project0ize_item(proj, pid, old2new):
    """把 old_site 商品映射成 new_site 商品（仅用于特征/召回增强，不改变原 label）"""
    if proj == "1" and pid in old2new:
        return old2new[pid]
    return pid


# =================== 3) 订单强信号热度（完成/下单） ===================
def build_order_popularity(old_orders, new_orders, old2new):
    """
    输出一个“在 new 空间”的 popularity:
      - 对 new_site：直接用 product_id
      - 对 old_site：先 old->new 映射后计数（提升跨站）
    """
    pop = Counter()

    def add_orders(df, proj):
        if df is None or len(df) == 0:
            return
        # product_id 列存在
        if "product_id" not in df.columns:
            return
        # status_code 可能包含 ORDERFINISH / CANCEL / ERROR 等
        status = df["status_code"].fillna("").astype(str) if "status_code" in df.columns else pd.Series([""]*len(df))
        pid = df["product_id"].map(safe_id)

        # 只把“更强的状态”算更大权重（可调）
        for p, st in zip(pid.tolist(), status.tolist()):
            if not p:
                continue
            w = 1.0
            st = st.upper()
            if "ORDERFINISH" in st:
                w = 5.0
            elif "ORDER" in st:  # 其它含 ORDER 的状态
                w = 3.0
            elif "CANCEL" in st or "ERROR" in st:
                w = 1.0

            # map old->new
            p2 = p
            if proj == "1" and p in old2new:
                p2 = old2new[p]
            pop[p2] += w

    add_orders(new_orders, "0")
    add_orders(old_orders, "1")
    return pop


# =================== 4) ItemCF 训练（分站点） ===================
def train_itemcf(train_session_items):
    item_sim_by_proj = {}
    item_cnt_by_proj = {}

    for proj in ["0","1"]:
        sessions = [items for (p,_v), items in train_session_items.items() if p == proj]
        if not sessions:
            item_sim_by_proj[proj] = {}
            item_cnt_by_proj[proj] = Counter()
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

    return item_sim_by_proj, item_cnt_by_proj


# =================== 5) 分桶热门兜底 ===================
def build_popular_buckets(pv_train, visits_train):
    popular_by_proj = (
        pv_train.groupby("project_id")["product_id"]
                .value_counts()
                .groupby(level=0)
                .apply(lambda s: s.index.get_level_values(1).tolist())
                .to_dict()
    )

    popular_by_proj_src = {}
    popular_by_proj_src_dev = {}

    if "traffic_source" in visits_train.columns and "device_category" in visits_train.columns:
        vmeta = visits_train[["project_id","visit_id","traffic_source","device_category"]].copy()
        vmeta["project_id"] = vmeta["project_id"].map(safe_id)
        vmeta["visit_id"] = vmeta["visit_id"].map(safe_id)
        vmeta["traffic_source"] = vmeta["traffic_source"].fillna("NA").astype(str)
        vmeta["device_category"] = vmeta["device_category"].fillna("NA").astype(str)

        pv_meta = pv_train.merge(vmeta, on=["project_id","visit_id"], how="left")
        pv_meta["traffic_source"] = pv_meta["traffic_source"].fillna("NA").astype(str)
        pv_meta["device_category"] = pv_meta["device_category"].fillna("NA").astype(str)

        popular_by_proj_src = (
            pv_meta.groupby(["project_id","traffic_source"])["product_id"]
                  .value_counts()
                  .groupby(level=[0,1])
                  .apply(lambda s: s.index.get_level_values(2).tolist())
                  .to_dict()
        )

        popular_by_proj_src_dev = (
            pv_meta.groupby(["project_id","traffic_source","device_category"])["product_id"]
                  .value_counts()
                  .groupby(level=[0,1,2])
                  .apply(lambda s: s.index.get_level_values(3).tolist())
                  .to_dict()
        )

    return popular_by_proj, popular_by_proj_src, popular_by_proj_src_dev

def popular_fallback(proj, seen_set, popular_by_proj, popular_by_proj_src, popular_by_proj_src_dev,
                     k=6, traffic_source="NA", device_category="NA"):
    recs = []

    for pid in popular_by_proj_src_dev.get((proj, traffic_source, device_category), []):
        if pid not in seen_set and pid not in recs:
            recs.append(pid)
        if len(recs) == k:
            return recs

    for pid in popular_by_proj_src.get((proj, traffic_source), []):
        if pid not in seen_set and pid not in recs:
            recs.append(pid)
        if len(recs) == k:
            return recs

    for pid in popular_by_proj.get(proj, []):
        if pid not in seen_set and pid not in recs:
            recs.append(pid)
        if len(recs) == k:
            return recs

    pop2 = popular_by_proj.get("0", []) + popular_by_proj.get("1", [])
    for pid in pop2:
        if pid not in seen_set and pid not in recs:
            recs.append(pid)
        if len(recs) == k:
            return recs

    return recs[:k]


# =================== 6) 召回：对一个 session 产出候选 + CF分数等 ===================
def recall_candidates_for_session(proj, seen_items, item_sim_by_proj, item_cnt_by_proj,
                                  cand_n=120):
    seen = dedup_keep_order(seen_items)
    seen_set = set(seen)
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
            val = s * w
            if USE_POP_PENALTY:
                val *= (1.0 / np.sqrt(item_cnt.get(j, 1)))
            scores[j] += val

    # 返回 topN
    cand = [pid for pid, _ in scores.most_common(cand_n)]
    return cand, scores


# =================== 7) 构造 LGBM 训练数据 ===================
def make_rank_train_data(train_session_items, visits_train_meta, item_sim_by_proj, item_cnt_by_proj,
                         popular_by_proj, popular_by_proj_src, popular_by_proj_src_dev,
                         order_pop_newspace, old2new,
                         cand_per_session=120, neg_per_pos=6):
    """
    pointwise binary:
      - 正样本：leave-one-out 的最后一个商品（label=1）
      - 候选：对历史做 CF 召回 topN + 热门补齐
      - 负样本：候选中除正样本外随机采样 neg_per_pos 个
    特征尽量简单，稳定。
    """
    rows = []
    y = []

    keys = list(train_session_items.keys())
    np.random.shuffle(keys)

    # 只取一部分 session 训练（否则太大），你可以调大
    # 推荐先 50k～120k session
    MAX_SESS = 80000
    keys = keys[:min(MAX_SESS, len(keys))]

    for (proj, vid) in tqdm(keys, desc="Building ranker train data"):
        seq = train_session_items[(proj, vid)]
        if len(seq) < 2:
            continue

        # leave-one-out
        label_item = seq[-1]
        hist = seq[:-1]
        seen_set = set(hist)

        # session meta
        meta = visits_train_meta.get((proj, vid), ("NA", "NA"))
        traffic_source, device_category = meta

        # CF recall from hist
        cand, score_map = recall_candidates_for_session(proj, hist, item_sim_by_proj, item_cnt_by_proj,
                                                        cand_n=cand_per_session)

        # 补充热门，保证候选足够大
        if len(cand) < cand_per_session:
            fb = popular_fallback(proj, seen_set, popular_by_proj, popular_by_proj_src, popular_by_proj_src_dev,
                                  k=cand_per_session, traffic_source=traffic_source, device_category=device_category)
            for x in fb:
                if x not in cand and x not in seen_set:
                    cand.append(x)
                if len(cand) >= cand_per_session:
                    break

        # 确保正样本在候选里（如果不在，强行加入，给它 score=0）
        if label_item not in cand and label_item not in seen_set:
            cand = [label_item] + cand
        cand = cand[:cand_per_session]

        # 负采样池（排除正样本）
        neg_pool = [x for x in cand if x != label_item]
        if len(neg_pool) == 0:
            continue

        # 构建一个正样本 + 若干负样本
        chosen_negs = list(np.random.choice(neg_pool, size=min(neg_per_pos, len(neg_pool)), replace=False))

        def feat(item):
            # 基础：cf_score / rank / pop / order_pop
            cf = float(score_map.get(item, 0.0))
            # 排名（越小越好）；不在则给大数
            try:
                r = cand.index(item) + 1
            except ValueError:
                r = cand_per_session + 1

            pop_view = float(item_cnt_by_proj.get(proj, Counter()).get(item, 0))
            # 订单热度：统一映射到 new 空间
            item_newspace = project0ize_item(proj, item, old2new)
            pop_order = float(order_pop_newspace.get(item_newspace, 0.0))

            return {
                "project_id": int(proj),
                "traffic_source": traffic_source,
                "device_category": device_category,
                "session_len": len(hist),
                "cf_score": cf,
                "rank_in_cf": r,
                "view_pop": pop_view,
                "order_pop": pop_order,
            }

        # 正
        rows.append(feat(label_item)); y.append(1)
        # 负
        for nitem in chosen_negs:
            rows.append(feat(nitem)); y.append(0)

    X = pd.DataFrame(rows)
    y = np.array(y, dtype=np.int8)
    return X, y


# =================== 8) 训练 LGBM（pointwise） ===================
def train_lgbm(X, y):
    # 类别特征：直接用字符串，LightGBM 原生支持
    cat_cols = ["traffic_source", "device_category"]
    for c in cat_cols:
        X[c] = X[c].astype("category")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )

    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
    dval   = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": SEED,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(100)]
    )

    # quick val auc
    pred = model.predict(X_val)
    auc = roc_auc_score(y_val, pred)
    print("Ranker val AUC:", auc)
    return model


# =================== 9) 推理：召回候选 -> LGBM 打分重排 -> top6 ===================
def predict_submission(visits_test, test_session_items,
                       item_sim_by_proj, item_cnt_by_proj,
                       popular_by_proj, popular_by_proj_src, popular_by_proj_src_dev,
                       order_pop_newspace, old2new,
                       ranker_model):
    test_keys = visits_test[["project_id","visit_id"]].copy()
    test_keys["project_id"] = test_keys["project_id"].map(safe_id)
    test_keys["visit_id"] = test_keys["visit_id"].map(safe_id)

    test_keys["traffic_source"] = visits_test["traffic_source"].fillna("NA").astype(str) if "traffic_source" in visits_test.columns else "NA"
    test_keys["device_category"] = visits_test["device_category"].fillna("NA").astype(str) if "device_category" in visits_test.columns else "NA"

    test_ids = test_keys["visit_id"].tolist()
    test_set = set(test_ids)

    rows = []
    used_ranker = 0
    used_cf = 0

    for proj, vid, src, dev in tqdm(
        zip(test_keys["project_id"], test_keys["visit_id"], test_keys["traffic_source"], test_keys["device_category"]),
        total=len(test_keys),
        desc="Predicting"
    ):
        seen = test_session_items.get((proj, vid), [])
        seen_set = set(seen)

        # 无浏览：直接兜底（桶热门）
        if len(seen) == 0:
            recs = popular_fallback(proj, seen_set, popular_by_proj, popular_by_proj_src, popular_by_proj_src_dev,
                                    k=K_RECS, traffic_source=src, device_category=dev)
            rows.append((vid, " ".join(recs)))
            continue

        # CF 召回候选
        cand, score_map = recall_candidates_for_session(proj, seen, item_sim_by_proj, item_cnt_by_proj,
                                                        cand_n=CAND_PER_SESSION)
        if len(cand) > 0:
            used_cf += 1

        # 候选太少补齐
        if len(cand) < CAND_PER_SESSION:
            fb = popular_fallback(proj, seen_set, popular_by_proj, popular_by_proj_src, popular_by_proj_src_dev,
                                  k=CAND_PER_SESSION, traffic_source=src, device_category=dev)
            for x in fb:
                if x not in cand and x not in seen_set:
                    cand.append(x)
                if len(cand) >= CAND_PER_SESSION:
                    break
        cand = dedup_keep_order([c for c in cand if c not in seen_set])[:CAND_PER_SESSION]

        # 如果还是没候选（极端）：兜底
        if len(cand) == 0:
            recs = popular_fallback(proj, seen_set, popular_by_proj, popular_by_proj_src, popular_by_proj_src_dev,
                                    k=K_RECS, traffic_source=src, device_category=dev)
            rows.append((vid, " ".join(recs)))
            continue

        # 构造 ranker 特征表
        feats = []
        for item in cand:
            cf = float(score_map.get(item, 0.0))
            pop_view = float(item_cnt_by_proj.get(proj, Counter()).get(item, 0))
            item_newspace = project0ize_item(proj, item, old2new)
            pop_order = float(order_pop_newspace.get(item_newspace, 0.0))

            feats.append({
                "project_id": int(proj),
                "traffic_source": src,
                "device_category": dev,
                "session_len": len(seen),
                "cf_score": cf,
                "rank_in_cf": 1,      # 这里不强依赖 index，简单起见给常数
                "view_pop": pop_view,
                "order_pop": pop_order,
            })

        Xc = pd.DataFrame(feats)
        Xc["traffic_source"] = Xc["traffic_source"].astype("category")
        Xc["device_category"] = Xc["device_category"].astype("category")

        scores = ranker_model.predict(Xc)
        used_ranker += 1

        order = np.argsort(-scores)
        recs = []
        for idx in order:
            pid = cand[int(idx)]
            if pid not in recs and pid not in seen_set:
                recs.append(pid)
            if len(recs) == K_RECS:
                break

        # 补齐
        if len(recs) < K_RECS:
            fb = popular_fallback(proj, seen_set, popular_by_proj, popular_by_proj_src, popular_by_proj_src_dev,
                                  k=K_RECS, traffic_source=src, device_category=dev)
            for x in fb:
                if x not in recs and x not in seen_set:
                    recs.append(x)
                if len(recs) == K_RECS:
                    break

        rows.append((vid, " ".join(recs[:K_RECS])))

    sub = pd.DataFrame(rows, columns=["visit_id","product_ids"])

    # strict validation
    sub_set = set(sub["visit_id"].tolist())
    missing = test_set - sub_set
    extra = sub_set - test_set
    print("Key validation: missing =", len(missing), "extra =", len(extra))
    assert len(missing) == 0 and len(extra) == 0, "visit_id mismatch!"

    print("Diagnostics:")
    print("  used CF sessions:", used_cf, f"({used_cf/len(sub):.1%})")
    print("  used ranker sessions:", used_ranker, f"({used_ranker/len(sub):.1%})")
    print("  unique lists:", sub["product_ids"].nunique())
    print(sub.head(5))

    sub.to_csv(OUT_SUB, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
    print("Saved:", OUT_SUB)
    return sub


# =================== 10) 离线：简单 Recall@6 (leave-one-out) ===================
def offline_recall_at_6(train_session_items, item_sim_by_proj, item_cnt_by_proj,
                        popular_by_proj, popular_by_proj_src, popular_by_proj_src_dev,
                        visits_train_meta, ranker_model=None,
                        order_pop_newspace=None, old2new=None,
                        sample_sessions=20000):
    keys = list(train_session_items.keys())
    np.random.shuffle(keys)
    keys = keys[:min(sample_sessions, len(keys))]

    hit = 0
    total = 0

    for (proj, vid) in keys:
        seq = train_session_items[(proj, vid)]
        if len(seq) < 2:
            continue
        label = seq[-1]
        hist = seq[:-1]
        seen_set = set(hist)

        src, dev = visits_train_meta.get((proj, vid), ("NA", "NA"))
        # recall
        cand, score_map = recall_candidates_for_session(proj, hist, item_sim_by_proj, item_cnt_by_proj,
                                                        cand_n=CAND_PER_SESSION)
        # add popular
        if len(cand) < CAND_PER_SESSION:
            fb = popular_fallback(proj, seen_set, popular_by_proj, popular_by_proj_src, popular_by_proj_src_dev,
                                  k=CAND_PER_SESSION, traffic_source=src, device_category=dev)
            for x in fb:
                if x not in cand and x not in seen_set:
                    cand.append(x)
                if len(cand) >= CAND_PER_SESSION:
                    break
        cand = dedup_keep_order([c for c in cand if c not in seen_set])[:CAND_PER_SESSION]

        if len(cand) == 0:
            recs = popular_fallback(proj, seen_set, popular_by_proj, popular_by_proj_src, popular_by_proj_src_dev,
                                    k=K_RECS, traffic_source=src, device_category=dev)
        else:
            if ranker_model is None:
                # pure CF top6
                recs = cand[:K_RECS]
            else:
                feats = []
                for item in cand:
                    cf = float(score_map.get(item, 0.0))
                    pop_view = float(item_cnt_by_proj.get(proj, Counter()).get(item, 0))
                    item_newspace = project0ize_item(proj, item, old2new)
                    pop_order = float(order_pop_newspace.get(item_newspace, 0.0)) if order_pop_newspace else 0.0
                    feats.append({
                        "project_id": int(proj),
                        "traffic_source": src,
                        "device_category": dev,
                        "session_len": len(hist),
                        "cf_score": cf,
                        "rank_in_cf": 1,
                        "view_pop": pop_view,
                        "order_pop": pop_order,
                    })
                Xc = pd.DataFrame(feats)
                Xc["traffic_source"] = Xc["traffic_source"].astype("category")
                Xc["device_category"] = Xc["device_category"].astype("category")
                sc = ranker_model.predict(Xc)
                order = np.argsort(-sc)
                recs = []
                for idx in order:
                    pid = cand[int(idx)]
                    if pid not in recs:
                        recs.append(pid)
                    if len(recs) == K_RECS:
                        break

        total += 1
        if label in set(recs):
            hit += 1

    r = hit / max(total, 1)
    print(f"[Offline] Recall@6: {r:.4f} (sessions={total})")
    return r


def main():
    print("Loading files...")
    visits_train = read_csv_auto(TRAIN_VISITS)
    hits_train   = read_csv_auto(TRAIN_HITS)
    visits_test  = read_csv_auto(TEST_VISITS)
    hits_test    = read_csv_auto(TEST_HITS)

    old_prod = read_csv_auto(OLD_PROD)
    new_prod = read_csv_auto(NEW_PROD)
    old2new_df = read_csv_auto(OLD2NEW) if OLD2NEW.exists() else pd.DataFrame()

    old_orders = read_csv_auto(OLD_ORDERS) if OLD_ORDERS.exists() else pd.DataFrame()
    new_orders = read_csv_auto(NEW_ORDERS) if NEW_ORDERS.exists() else pd.DataFrame()

    # clean ids
    for df in [visits_train, visits_test]:
        df["project_id"] = df["project_id"].map(safe_id)
        df["visit_id"] = df["visit_id"].map(safe_id)
    for df in [hits_train, hits_test]:
        df["project_id"] = df["project_id"].map(safe_id)
        df["watch_id"] = df["watch_id"].map(safe_id)

    print("Building slug2pid...")
    slug2pid = build_slug2pid(old_prod, new_prod)

    print("Building interactions...")
    pv_train = build_pv(visits_train, hits_train, slug2pid)
    pv_test  = build_pv(visits_test, hits_test, slug2pid)

    train_session_items = build_session_items(pv_train)
    test_session_items  = build_session_items(pv_test)

    print("Train sessions:", len(train_session_items), "Test sessions with views:", len(test_session_items))

    # session meta for train
    visits_train_meta = {}
    if "traffic_source" in visits_train.columns and "device_category" in visits_train.columns:
        for _, r in visits_train[["project_id","visit_id","traffic_source","device_category"]].iterrows():
            visits_train_meta[(safe_id(r["project_id"]), safe_id(r["visit_id"]))] = (
                str(r["traffic_source"]) if pd.notna(r["traffic_source"]) else "NA",
                str(r["device_category"]) if pd.notna(r["device_category"]) else "NA",
            )
    else:
        for _, r in visits_train[["project_id","visit_id"]].iterrows():
            visits_train_meta[(safe_id(r["project_id"]), safe_id(r["visit_id"]))] = ("NA","NA")

    print("old->new mapping...")
    old2new = build_old2new_map(old2new_df)
    print("old->new pairs:", len(old2new))

    print("Order popularity (new-space)...")
    order_pop_newspace = build_order_popularity(old_orders, new_orders, old2new)
    print("Order-pop items:", len(order_pop_newspace))

    print("Training ItemCF...")
    item_sim_by_proj, item_cnt_by_proj = train_itemcf(train_session_items)
    print("project0 items:", len(item_cnt_by_proj.get("0",{})), "project1 items:", len(item_cnt_by_proj.get("1",{})))

    print("Building popularity buckets...")
    popular_by_proj, popular_by_proj_src, popular_by_proj_src_dev = build_popular_buckets(pv_train, visits_train)

    # baseline offline recall (pure CF)
    print("Offline recall - CF only:")
    offline_recall_at_6(train_session_items, item_sim_by_proj, item_cnt_by_proj,
                        popular_by_proj, popular_by_proj_src, popular_by_proj_src_dev,
                        visits_train_meta, ranker_model=None,
                        sample_sessions=20000)

    print("Building LGBM train data...")
    X, y = make_rank_train_data(
        train_session_items, visits_train_meta,
        item_sim_by_proj, item_cnt_by_proj,
        popular_by_proj, popular_by_proj_src, popular_by_proj_src_dev,
        order_pop_newspace, old2new,
        cand_per_session=CAND_PER_SESSION, neg_per_pos=NEG_PER_POS
    )
    print("Train rows:", len(X), "Pos rate:", y.mean())

    print("Training LightGBM ranker...")
    ranker = train_lgbm(X, y)

    print("Offline recall - CF + LGBM ranker:")
    offline_recall_at_6(train_session_items, item_sim_by_proj, item_cnt_by_proj,
                        popular_by_proj, popular_by_proj_src, popular_by_proj_src_dev,
                        visits_train_meta, ranker_model=ranker,
                        order_pop_newspace=order_pop_newspace, old2new=old2new,
                        sample_sessions=20000)

    print("Predicting submission...")
    predict_submission(visits_test, test_session_items,
                       item_sim_by_proj, item_cnt_by_proj,
                       popular_by_proj, popular_by_proj_src, popular_by_proj_src_dev,
                       order_pop_newspace, old2new,
                       ranker)

if __name__ == "__main__":
    main()
