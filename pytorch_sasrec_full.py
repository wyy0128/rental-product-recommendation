# pytorch_sasrec_full.py
import os
import json
import math
import csv
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================
# Config
# =========================
DATA_DIR = Path(r"C:\pycharm\DL_Project_1\rental-product-recommendation-system")

TRAIN_VISITS = DATA_DIR / "metrika_visits.csv"
TRAIN_HITS   = DATA_DIR / "metrika_hits.csv"
TEST_VISITS  = DATA_DIR / "metrika_visits_test.csv"
TEST_HITS    = DATA_DIR / "metrika_hits_test.csv"
OLD_PROD     = DATA_DIR / "old_site_products.csv"
NEW_PROD     = DATA_DIR / "new_site_products.csv"

OUT_SUB      = DATA_DIR / "submission_sasrec.csv"

# Model / training
MAX_LEN = 30             # session 序列最大长度
EMB_DIM = 64
N_HEAD = 2
N_LAYERS = 2
DROPOUT = 0.2

BATCH_SIZE = 1024
EPOCHS = 8
LR = 1e-3
WEIGHT_DECAY = 1e-5

# sampling
NUM_NEG = 10           # 每个样本负采样数量（越大越慢但更准）
SEED = 42

# predict
TOPK_CAND = 200          # 从模型里拿 topK 候选，再过滤去重后取 6
K_RECS = 6

# cold-start buckets (fallback)
USE_BUCKET_FALLBACK = True

# avoid recommending already seen items
FILTER_SEEN = True

torch.manual_seed(SEED)
np.random.seed(SEED)


# =========================
# Utils: safe string ids
# =========================
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


# =========================
# 1) Build interactions: (project_id, visit_id) -> [product_id,...]
# =========================
def build_slug2pid(old_prod, new_prod):
    old_map = old_prod[["id","slug"]].rename(columns={"id":"product_id"}).copy()
    old_map["project_id"] = "1"
    new_map = new_prod[["id","slug"]].rename(columns={"id":"product_id"}).copy()
    new_map["project_id"] = "0"
    slug2pid = pd.concat([old_map, new_map], ignore_index=True)
    slug2pid["project_id"] = slug2pid["project_id"].map(safe_id)
    slug2pid["slug"] = slug2pid["slug"].astype(str)
    slug2pid["product_id"] = slug2pid["product_id"].map(safe_id)
    return slug2pid

def build_pv_from_visits_hits(visits_df, hits_df, slug2pid):
    # explode watch_ids -> watch_id
    v = visits_df[["project_id","visit_id","watch_ids"]].copy()
    v["project_id"] = v["project_id"].map(safe_id)
    v["visit_id"] = v["visit_id"].map(safe_id)
    v["watch_list"] = v["watch_ids"].apply(parse_watch_ids)
    v = v.explode("watch_list").rename(columns={"watch_list":"watch_id"}).dropna(subset=["watch_id"])
    v["watch_id"] = v["watch_id"].map(safe_id)

    h = hits_df.copy()
    h["project_id"] = h["project_id"].map(safe_id)
    h["watch_id"] = h["watch_id"].map(safe_id)

    # join
    hj = h.merge(v[["project_id","visit_id","watch_id"]], on=["project_id","watch_id"], how="inner")

    # product page views
    need = ["page_type","is_page_view","slug"]
    for c in need:
        if c not in hj.columns:
            raise ValueError(f"hits missing column: {c}")

    pv = hj[(hj["page_type"] == "PRODUCT") & (hj["is_page_view"] == "1")][["project_id","visit_id","slug"]].copy()
    pv["slug"] = pv["slug"].astype(str)

    pv = pv.merge(slug2pid, on=["project_id","slug"], how="left").dropna(subset=["product_id"])
    pv["product_id"] = pv["product_id"].map(safe_id)
    pv = pv.drop_duplicates(subset=["project_id","visit_id","product_id"])
    return pv

def build_session_items(pv: pd.DataFrame):
    sess = (
        pv.groupby(["project_id","visit_id"])["product_id"]
          .apply(lambda x: dedup_keep_order([safe_id(v) for v in x.tolist() if safe_id(v)]))
          .to_dict()
    )
    return sess


# =========================
# 2) Popularity buckets for fallback
# =========================
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

    if USE_BUCKET_FALLBACK:
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


# =========================
# 3) Build vocab (product_id -> index)
# =========================
def build_item_vocab(train_session_items):
    # only from train items; unseen in train cannot be predicted by DL model -> fallback will cover
    items = set()
    for (_p,_v), seq in train_session_items.items():
        for it in seq:
            if it:
                items.add(it)
    items = sorted(items)
    # 0: PAD
    item2idx = {it: (i+1) for i, it in enumerate(items)}
    idx2item = {i+1: it for i, it in enumerate(items)}
    return item2idx, idx2item


# =========================
# 4) Dataset for SASRec (next-item prediction with negative sampling)
# =========================
class SasRecDataset(Dataset):
    def __init__(self, session_items, item2idx, max_len=30, num_neg=20):
        self.max_len = max_len
        self.num_neg = num_neg
        self.item2idx = item2idx

        self.samples = []  # (proj, seq_idx_list, pos_idx)
        # build samples: for each session, for each position t, predict item[t]
        for (proj, vid), seq in session_items.items():
            # map to idx, drop OOV
            seq_idx = [item2idx[it] for it in seq if it in item2idx]
            if len(seq_idx) < 4:
                continue
            for t in range(1, len(seq_idx)):
                hist = seq_idx[:t]
                pos = seq_idx[t]
                self.samples.append((proj, hist, pos))

        # negatives pool
        self.all_items = np.array(list(item2idx.values()), dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        proj, hist, pos = self.samples[i]
        # truncate/pad
        hist = hist[-self.max_len:]
        x = np.zeros(self.max_len, dtype=np.int64)
        x[-len(hist):] = np.array(hist, dtype=np.int64)

        # negative sampling (avoid sampling positives in history)
        forbid = set(hist)
        forbid.add(pos)
        negs = []
        while len(negs) < self.num_neg:
            cand = int(np.random.choice(self.all_items))
            if cand not in forbid:
                negs.append(cand)
                forbid.add(cand)
        negs = np.array(negs, dtype=np.int64)

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(negs, dtype=torch.long),
        )


# =========================
# 5) SASRec model
# =========================
class SASRec(nn.Module):
    def __init__(self, num_items, max_len=30, emb_dim=64, n_heads=2, n_layers=2, dropout=0.2):
        super().__init__()
        self.num_items = num_items
        self.max_len = max_len
        self.emb_dim = emb_dim

        self.item_emb = nn.Embedding(num_items + 1, emb_dim, padding_idx=0)  # +1 for safety
        self.pos_emb = nn.Embedding(max_len, emb_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, seq):  # seq: [B, L]
        B, L = seq.shape
        device = seq.device

        item_e = self.item_emb(seq)  # [B, L, D]
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        pos_e = self.pos_emb(pos)

        x = item_e + pos_e
        x = self.dropout(self.layer_norm(x))

        # causal mask: prevent attending to future
        # PyTorch expects mask with True = disallow
        causal = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        x = self.encoder(x, mask=causal)  # [B, L, D]
        return x

    def predict_logits(self, seq, candidates):
        """
        seq: [B, L]
        candidates: [B, C] item indices
        returns logits: [B, C]
        """
        h = self.forward(seq)           # [B, L, D]
        last = h[:, -1, :]              # use last position representation
        cand_emb = self.item_emb(candidates)  # [B, C, D]
        logits = torch.einsum("bd,bcd->bc", last, cand_emb)
        return logits

    def full_sort_logits(self, seq):
        """
        seq: [B, L] -> logits for all items [B, num_items+1]
        """
        h = self.forward(seq)
        last = h[:, -1, :]
        # matmul with all item embeddings
        all_emb = self.item_emb.weight  # [num_items+1, D]
        logits = torch.matmul(last, all_emb.t())
        return logits


# =========================
# 6) Train loop (sampled softmax / BPR-like)
# =========================
from tqdm import tqdm
import time

def train_model(model, loader, device, epochs=8, lr=1e-3, weight_decay=1e-5):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        pbar = tqdm(
            loader,
            desc=f"Epoch {ep}/{epochs}",
            total=len(loader),
            ncols=100,
        )

        start_time = time.time()

        for batch_idx, (seq, pos, negs) in enumerate(pbar):
            seq = seq.to(device)
            pos = pos.to(device)
            negs = negs.to(device)

            cand = torch.cat([pos.unsqueeze(1), negs], dim=1)
            logits = model.predict_logits(seq, cand)
            labels = torch.zeros(seq.size(0), dtype=torch.long, device=device)

            loss = nn.CrossEntropyLoss()(logits, labels)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = seq.size(0)
            total_loss += float(loss.item()) * bs
            n += bs

            # === 更新进度条信息 ===
            avg_loss = total_loss / max(n, 1)
            elapsed = time.time() - start_time
            it_per_sec = (batch_idx + 1) / max(elapsed, 1e-6)

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "it/s": f"{it_per_sec:.2f}",
            })

        print(f"[Epoch {ep}] avg loss: {total_loss/max(n,1):.4f}")

    return model



# =========================
# 7) Offline eval: leave-one-out recall@6 on train sessions
# =========================
@torch.no_grad()
def offline_eval_recall_at_k(model, session_items, item2idx, device, k=6, max_len=30, sample_sessions=20000):
    keys = list(session_items.keys())
    np.random.shuffle(keys)
    keys = keys[:min(sample_sessions, len(keys))]

    hit = 0
    total = 0

    model.eval()
    for (proj, vid) in keys:
        seq = [item2idx[it] for it in session_items[(proj, vid)] if it in item2idx]
        if len(seq) < 2:
            continue
        label = seq[-1]
        hist = seq[:-1][-max_len:]

        x = np.zeros(max_len, dtype=np.int64)
        x[-len(hist):] = np.array(hist, dtype=np.int64)
        x = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)

        logits = model.full_sort_logits(x)[0]  # [num_items+1]
        # top-k excluding PAD
        top = torch.topk(logits[1:], k=k).indices.cpu().numpy() + 1
        if label in set(top.tolist()):
            hit += 1
        total += 1

    r = hit / max(total, 1)
    print(f"[Offline] Recall@{k}: {r:.4f}  (eval sessions={total})")
    return r


# =========================
# 8) Predict for test
# =========================
def encode_session(seq_items, item2idx, max_len=30):
    seq = [item2idx[it] for it in seq_items if it in item2idx]
    if len(seq) == 0:
        return None
    seq = seq[-max_len:]
    x = np.zeros(max_len, dtype=np.int64)
    x[-len(seq):] = np.array(seq, dtype=np.int64)
    return x

@torch.no_grad()
def recommend_with_model(model, proj, vid, seq_items, item2idx, idx2item,
                         device, topk_cand=200, k=6,
                         seen_filter=True, seen_set=None):
    x = encode_session(seq_items, item2idx, max_len=MAX_LEN)
    if x is None:
        return [], False

    x = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)
    logits = model.full_sort_logits(x)[0]  # [num_items+1]
    # get top candidates excluding PAD index 0
    top = torch.topk(logits[1:], k=topk_cand).indices.cpu().numpy() + 1
    recs = []
    for idx in top:
        pid = idx2item.get(int(idx))
        if pid is None:
            continue
        if seen_filter and seen_set is not None and pid in seen_set:
            continue
        if pid not in recs:
            recs.append(pid)
        if len(recs) == k:
            break
    return recs, True


# =========================
# Main
# =========================
def main():
    print("Loading data...")
    visits_train = read_csv_auto(TRAIN_VISITS)
    hits_train   = read_csv_auto(TRAIN_HITS)
    visits_test  = read_csv_auto(TEST_VISITS)
    hits_test    = read_csv_auto(TEST_HITS)
    old_prod     = read_csv_auto(OLD_PROD)
    new_prod     = read_csv_auto(NEW_PROD)

    # clean key cols
    for df in [visits_train, visits_test]:
        df["project_id"] = df["project_id"].map(safe_id)
        df["visit_id"] = df["visit_id"].map(safe_id)
    for df in [hits_train, hits_test]:
        df["project_id"] = df["project_id"].map(safe_id)
        df["watch_id"] = df["watch_id"].map(safe_id)

    print("Building slug2pid...")
    slug2pid = build_slug2pid(old_prod, new_prod)

    print("Building train pv...")
    pv_train = build_pv_from_visits_hits(visits_train, hits_train, slug2pid)
    print("Train interactions:", len(pv_train))

    print("Building test pv...")
    pv_test = build_pv_from_visits_hits(visits_test, hits_test, slug2pid)
    print("Test interactions:", len(pv_test))

    print("Building session items...")
    train_session_items = build_session_items(pv_train)
    test_session_items  = build_session_items(pv_test)

    # test keys
    test_keys_df = visits_test[["project_id","visit_id"]].copy()
    test_keys_df["project_id"] = test_keys_df["project_id"].map(safe_id)
    test_keys_df["visit_id"] = test_keys_df["visit_id"].map(safe_id)

    # extra context for fallback
    if "traffic_source" in visits_test.columns:
        test_keys_df["traffic_source"] = visits_test["traffic_source"].fillna("NA").astype(str)
    else:
        test_keys_df["traffic_source"] = "NA"
    if "device_category" in visits_test.columns:
        test_keys_df["device_category"] = visits_test["device_category"].fillna("NA").astype(str)
    else:
        test_keys_df["device_category"] = "NA"

    with_views = sum((p, v) in test_session_items for p, v in zip(test_keys_df["project_id"], test_keys_df["visit_id"]))
    print(f"Test visits: {len(test_keys_df)} | with product views: {with_views} ({with_views/len(test_keys_df):.1%})")

    print("Building popularity buckets...")
    popular_by_proj, popular_by_proj_src, popular_by_proj_src_dev = build_popular_buckets(pv_train, visits_train)
    print("Popular[0] top6:", popular_by_proj.get("0", [])[:6])
    print("Popular[1] top6:", popular_by_proj.get("1", [])[:6])

    print("Building item vocab from train...")
    item2idx, idx2item = build_item_vocab(train_session_items)
    num_items = len(item2idx)
    print("Vocab size (train items):", num_items)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| CUDA:", torch.cuda.is_available())

    print("Preparing dataset...")
    ds = SasRecDataset(train_session_items, item2idx, max_len=MAX_LEN, num_neg=NUM_NEG)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    print("Init model...")
    model = SASRec(
        num_items=num_items,
        max_len=MAX_LEN,
        emb_dim=EMB_DIM,
        n_heads=N_HEAD,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    print("Training...")
    model = train_model(model, loader, device, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY)

    print("Offline eval (train leave-one-out)...")
    offline_eval_recall_at_k(model, train_session_items, item2idx, device, k=6, max_len=MAX_LEN, sample_sessions=20000)

    print("Predicting for test and building submission...")
    rows = []
    used_dl = 0
    fallback_cnt = 0
    unique_lists = set()

    for proj, vid, src, devcat in zip(
        test_keys_df["project_id"],
        test_keys_df["visit_id"],
        test_keys_df["traffic_source"],
        test_keys_df["device_category"],
    ):
        seen = test_session_items.get((proj, vid), [])
        seen_set = set(seen)

        # DL recs if have sequence
        recs, ok = recommend_with_model(
            model, proj, vid,
            seq_items=seen,
            item2idx=item2idx, idx2item=idx2item,
            device=device,
            topk_cand=TOPK_CAND, k=K_RECS,
            seen_filter=FILTER_SEEN,
            seen_set=seen_set
        )
        if ok:
            used_dl += 1

        # if not enough, fallback fill
        if len(recs) < K_RECS:
            fb = popular_fallback(
                proj, seen_set,
                popular_by_proj, popular_by_proj_src, popular_by_proj_src_dev,
                k=K_RECS, traffic_source=src, device_category=devcat
            )
            # merge with recs, keep order, ensure no dup
            merged = []
            for x in recs + fb:
                if x not in merged and (not FILTER_SEEN or x not in seen_set):
                    merged.append(x)
                if len(merged) == K_RECS:
                    break
            recs = merged[:K_RECS]
            fallback_cnt += 1

        # final guarantee: 6 unique
        recs = dedup_keep_order(recs)[:K_RECS]
        if len(recs) < K_RECS:
            # ultimate pad from global popular
            fb2 = popular_by_proj.get("0", []) + popular_by_proj.get("1", [])
            for x in fb2:
                if x not in recs and (not FILTER_SEEN or x not in seen_set):
                    recs.append(x)
                if len(recs) == K_RECS:
                    break
        recs = recs[:K_RECS]

        rows.append((vid, " ".join(recs)))
        unique_lists.add(" ".join(recs))

    sub = pd.DataFrame(rows, columns=["visit_id","product_ids"])

    # strict key validation
    test_ids = set(test_keys_df["visit_id"].tolist())
    sub_ids = set(sub["visit_id"].tolist())
    missing = test_ids - sub_ids
    extra = sub_ids - test_ids
    print("Key validation | missing:", len(missing), "extra:", len(extra))
    assert len(missing) == 0 and len(extra) == 0, "visit_id mismatch! Do NOT submit."

    print("Diagnostics:")
    print(f"  used DL for: {used_dl}/{len(sub)} ({used_dl/len(sub):.1%})")
    print(f"  fallback used: {fallback_cnt}/{len(sub)} ({fallback_cnt/len(sub):.1%})")
    print(f"  unique recommendation lists: {len(unique_lists)}")
    print(sub.head(5))

    sub.to_csv(OUT_SUB, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
    print("Saved:", OUT_SUB)

    with open(OUT_SUB, "r", encoding="utf-8") as f:
        print("---- raw preview ----")
        for _ in range(3):
            print(f.readline().rstrip())


if __name__ == "__main__":
    main()
