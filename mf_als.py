import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============ 配置 ============
DATA_DIR = Path("C:/pycharm/DL_Project_1/rental-product-recommendation-system")

TRAIN_VISITS = DATA_DIR / "metrika_visits.csv"
TRAIN_HITS   = DATA_DIR / "metrika_hits.csv"
TEST_VISITS  = DATA_DIR / "metrika_visits_test.csv"

OLD_PROD = DATA_DIR / "old_site_products.csv"
NEW_PROD = DATA_DIR / "new_site_products.csv"

OUT_SUB  = DATA_DIR / "submission_mf_bpr.csv"

K_RECS = 6

# 训练超参（先保守）
EMB_DIM = 64
EPOCHS = 3
BATCH_SIZE = 4096
LR = 2e-3
WEIGHT_DECAY = 1e-6
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============ 工具函数 ============
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


# ============ 1) 抽 interactions：visit_id × product_id ============
print("Loading data...")
visits_train = read_csv_auto(TRAIN_VISITS)
hits_train   = read_csv_auto(TRAIN_HITS)
visits_test  = read_csv_auto(TEST_VISITS)
old_prod     = read_csv_auto(OLD_PROD)
new_prod     = read_csv_auto(NEW_PROD)

print("Building visit_id <-> watch_id mapping...")
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

print("Extracting PRODUCT page views...")
prod_views = hits_j[(hits_j["page_type"] == "PRODUCT") & (hits_j["is_page_view"] == 1)].copy()

print("Mapping slug -> product_id...")
old_map = old_prod[["id", "slug"]].rename(columns={"id": "product_id"}).copy()
old_map["project_id"] = 1
new_map = new_prod[["id", "slug"]].rename(columns={"id": "product_id"}).copy()
new_map["project_id"] = 0
slug2pid = pd.concat([old_map, new_map], ignore_index=True)

prod_views = prod_views.merge(slug2pid, on=["project_id", "slug"], how="left").dropna(subset=["product_id"])
prod_views["product_id"] = prod_views["product_id"].astype("int64")

interactions = prod_views[["visit_id", "product_id"]].drop_duplicates().copy()
print("Interactions:", interactions.shape)

# ============ 2) 建索引 ============
visit_ids = interactions["visit_id"].unique()
product_ids = interactions["product_id"].unique()

visit2idx = {v: i for i, v in enumerate(visit_ids)}
idx2visit = {i: v for v, i in visit2idx.items()}

prod2idx = {p: i for i, p in enumerate(product_ids)}
idx2prod = {i: p for p, i in prod2idx.items()}

interactions["u"] = interactions["visit_id"].map(visit2idx).astype("int32")
interactions["i"] = interactions["product_id"].map(prod2idx).astype("int32")

n_users = len(visit2idx)
n_items = len(prod2idx)
print("n_users:", n_users, "n_items:", n_items)

# user -> set(items)
user_pos = defaultdict(set)
for u, i in zip(interactions["u"].to_numpy(), interactions["i"].to_numpy()):
    user_pos[int(u)].add(int(i))

# 为每个 user 预先存 list，采样更快
user_pos_list = {u: np.array(list(items), dtype=np.int32) for u, items in user_pos.items()}
all_items = np.arange(n_items, dtype=np.int32)

# ============ 3) BPR-MF 模型 ============
class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, k):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, k)
        self.item_emb = nn.Embedding(n_items, k)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def score(self, u, i):
        ue = self.user_emb(u)
        ie = self.item_emb(i)
        return (ue * ie).sum(dim=1)

    def forward(self, u, i_pos, i_neg):
        s_pos = self.score(u, i_pos)
        s_neg = self.score(u, i_neg)
        return s_pos, s_neg

def bpr_loss(s_pos, s_neg):
    return -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-12).mean()

torch.manual_seed(SEED)
np.random.seed(SEED)

model = BPRMF(n_users, n_items, EMB_DIM).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ============ 4) 训练：在线负采样 ============
users_array = np.array(list(user_pos_list.keys()), dtype=np.int32)

def sample_batch(batch_size):
    u = np.random.choice(users_array, size=batch_size, replace=True)
    i_pos = np.empty(batch_size, dtype=np.int32)
    i_neg = np.empty(batch_size, dtype=np.int32)

    for idx, uu in enumerate(u):
        pos_items = user_pos_list[int(uu)]
        ip = np.random.choice(pos_items)
        # 负采样：直到采到不在正集合的 item
        while True:
            ineg = np.random.randint(0, n_items)
            if ineg not in user_pos[int(uu)]:
                break
        i_pos[idx] = ip
        i_neg[idx] = ineg
    return u, i_pos, i_neg

print("Training BPR-MF on", DEVICE)
steps_per_epoch = max(1000, len(interactions) // BATCH_SIZE)

for epoch in range(1, EPOCHS + 1):
    model.train()
    losses = []
    for _ in range(steps_per_epoch):
        u, ip, ineg = sample_batch(BATCH_SIZE)
        u = torch.tensor(u, dtype=torch.long, device=DEVICE)
        ip = torch.tensor(ip, dtype=torch.long, device=DEVICE)
        ineg = torch.tensor(ineg, dtype=torch.long, device=DEVICE)

        opt.zero_grad()
        s_pos, s_neg = model(u, ip, ineg)
        loss = bpr_loss(s_pos, s_neg)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print(f"Epoch {epoch}/{EPOCHS} - loss: {np.mean(losses):.4f}")

# ============ 5) 生成推荐（Top-6） ============
print("Generating submission...")

# Popular fallback（冷 visit / 没见过）
popular_products = interactions["product_id"].value_counts().index.tolist()

# seen set（用训练 interactions）
seen_by_visit = interactions.groupby("visit_id")["product_id"].apply(set).to_dict()

def popular_fallback(vid, k=6):
    seen = seen_by_visit.get(vid, set())
    recs = []
    for pid in popular_products:
        if pid not in seen:
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

# 计算每个 user 的 topN（用 dot product）
# 为了快：一次性取 item embedding
model.eval()
with torch.no_grad():
    item_emb = model.item_emb.weight.detach().cpu().numpy()  # (n_items, k)

def recommend_for_visit(vid, k=6, pool_top=2000):
    # vid 是字符串 visit_id
    if vid not in visit2idx:
        return popular_fallback(vid, k=k)

    uidx = visit2idx[vid]
    uvec = model.user_emb.weight[uidx].detach().cpu().numpy()  # (k,)

    # 候选池：先用热门前 pool_top 个 item（大幅加速）
    # 你也可以改大一些，例如 5000 或全量（但会慢）
    cand_pids = popular_products[:pool_top]
    cand_iidx = np.array([prod2idx[p] for p in cand_pids if p in prod2idx], dtype=np.int32)

    scores = item_emb[cand_iidx] @ uvec  # (num_cand,)
    order = np.argsort(-scores)

    seen = seen_by_visit.get(vid, set())
    recs = []
    for j in order:
        pid = idx2prod[int(cand_iidx[j])]
        if pid in seen:
            continue
        recs.append(pid)
        if len(recs) == k:
            break

    if len(recs) < k:
        extra = popular_fallback(vid, k=k)
        for pid in extra:
            if pid not in recs:
                recs.append(pid)
            if len(recs) == k:
                break
    return recs

visits_test["visit_id"] = visits_test["visit_id"].apply(to_int_str)

test_visits = visits_test["visit_id"].tolist()
seen = sum(v in visit2idx for v in test_visits)
print("Test visits:", len(test_visits))
print("Seen in train (visit2idx):", seen, f"({seen/len(test_visits):.2%})")


out = []
for vid in visits_test["visit_id"].tolist():
    recs = recommend_for_visit(vid, k=K_RECS, pool_top=3000)
    out.append((vid, " ".join(map(str, recs))))

sub = pd.DataFrame(out, columns=["visit_id", "product_ids"])
sub.to_csv(OUT_SUB, index=False)
print("Saved:", OUT_SUB)
print(sub.head(10))
