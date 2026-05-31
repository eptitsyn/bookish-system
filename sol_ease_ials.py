"""
Kaggle Recommender System — EASE + iALS  (NDCG@20)
===================================================
Pipeline
  1. EASE  — embarrassingly-shallow item-item autoencoder, B = -(G+λI)^-1 / diag.
            Primary retriever; strongest single model on this data.
  2. iALS  — implicit weighted ALS (Hu/Koren). Complementary retriever +
            latent-factor dot feature for the reranker.
  3. Union candidate pool (EASE ∪ iALS ∪ popularity-fill).
  4. LightGBM LambdaRank reranker over EASE/iALS scores+ranks, ials_dot,
     popularity, impression CTR, and user/item/catalog stats.
  5. Temporal q90 split for honest NDCG@20 validation (no future leakage).

Target machine: 192 GB RAM server. n_items≈31k → dense 31k×31k EASE solve
(~7.8 GB f64) is cheap here. All test users are warm (no cold-start path needed,
but a popularity fallback is kept for safety).

Run:  python sol_ease_ials.py
Out:  ./output/submission.csv
"""

import warnings
import gc
import time
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path("./data")
OUT_DIR = Path("./output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CANDIDATES = 300          # final pool size per user
TOP_K = 20                  # submission length / eval cutoff

EASE_CANDIDATES = 220       # EASE is the primary retriever
IALS_CANDIDATES = 120       # iALS secondary retriever

# EASE L2 regularisation. Swept on the temporal-train model against val
# retrieval NDCG; the best value is then reused to fit the full-data model.
EASE_REGS = [100.0, 250.0, 500.0, 1000.0]

# iALS (implicit) hyper-params
IALS_FACTORS = 128
IALS_ITERS = 20
IALS_REG = 0.05
IALS_ALPHA = 1.0            # confidence scaling applied to the weighted matrix
BM25_K1 = 1.5
BM25_B = 0.75

# Interaction weights → confidence for iALS (EASE uses a binarised matrix)
W_VIEW = 1.0
W_RATING_BONUS = 0.5
W_PURCHASE = 5.0
W_PURCHASE_RAT = 1.0

RANDOM_SEED = 42
REC_BATCH = 4096            # users per retrieval batch
SCORE_BATCH = 4000          # users per reranker scoring batch
N_RANK_USERS = 30000        # warm val users used to train the ranker
N_CHECK_USERS = 3000        # val users for the honest NDCG@20 estimate

np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# CUDA (iALS only — EASE is dense linear algebra on CPU)
# ─────────────────────────────────────────────────────────────────────────────
def detect_cuda() -> bool:
    for check in [
        lambda: __import__("implicit.gpu", fromlist=["HAS_CUDA"]).HAS_CUDA,
        lambda: __import__("torch").cuda.is_available(),
    ]:
        try:
            if check():
                print("  ✓ CUDA found (iALS on GPU)")
                return True
        except Exception:
            pass
    print("  ✗ No CUDA — iALS on CPU")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. Loading data")
print("=" * 60)

train = pl.read_parquet(DATA_DIR / "train.pq")
items = pl.read_parquet(DATA_DIR / "items.pq")
test_u = pl.read_csv(DATA_DIR / "test_users.csv")

print(f"train {train.shape}  items {items.shape}  test {test_u.shape[0]:,}")
print(train.schema)

has_purchase = "is_purchased" in train.columns
has_rating = "rating" in train.columns
has_impressions = "impressions" in train.columns
print(f"has_purchase={has_purchase} has_rating={has_rating} "
      f"has_impressions={has_impressions}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. INTERACTION CONFIDENCE WEIGHTS (for iALS)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Interaction weights (iALS confidence)")
print("=" * 60)

if has_purchase and has_rating:
    train = train.with_columns(
        pl.when(pl.col("is_purchased") & (pl.col("rating") > 0))
        .then(W_PURCHASE + pl.col("rating").cast(pl.Float32) * W_PURCHASE_RAT)
        .when(pl.col("is_purchased"))
        .then(pl.lit(W_PURCHASE))
        .when(pl.col("rating") > 0)
        .then(W_VIEW + pl.col("rating").cast(pl.Float32) * W_RATING_BONUS)
        .otherwise(pl.lit(W_VIEW))
        .cast(pl.Float32)
        .alias("weight")
    )
elif has_purchase:
    train = train.with_columns(
        pl.when(pl.col("is_purchased")).then(pl.lit(W_PURCHASE))
        .otherwise(pl.lit(W_VIEW)).cast(pl.Float32).alias("weight")
    )
elif has_rating:
    train = train.with_columns(
        (W_VIEW + pl.col("rating").cast(pl.Float32) * W_RATING_BONUS).alias("weight")
    )
else:
    train = train.with_columns(pl.lit(1.0, dtype=pl.Float32).alias("weight"))

print(train["weight"].describe())

# ─────────────────────────────────────────────────────────────────────────────
# 3. INTEGER ENCODING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Integer encoding")
print("=" * 60)

all_users_s = train["user_id"].unique().sort()
all_items_s = train["item_id"].unique().sort()
n_users, n_items = len(all_users_s), len(all_items_s)

user_map = pl.DataFrame(
    {"user_id": all_users_s, "user_idx": pl.arange(n_users, eager=True, dtype=pl.Int32)}
)
item_map = pl.DataFrame(
    {"item_id": all_items_s, "item_idx": pl.arange(n_items, eager=True, dtype=pl.Int32)}
)
idx2item = all_items_s.to_numpy()

train = train.join(user_map, on="user_id", how="left").join(
    item_map, on="item_id", how="left"
)
print(f"n_users={n_users:,}  n_items={n_items:,}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. TEMPORAL SPLIT (q90)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Temporal train/val split")
print("=" * 60)

train = train.with_columns(pl.col("timestamp").cast(pl.Int64).alias("ts_int"))
split_val = train["ts_int"].quantile(0.9)
df_tr = train.filter(pl.col("ts_int") <= split_val)
df_val = train.filter(pl.col("ts_int") > split_val)
print(f"q90 split:  tr={len(df_tr):,}  val={len(df_val):,}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SPARSE MATRICES
# ─────────────────────────────────────────────────────────────────────────────
def build_sparse(df, nu, ni, weight_col="weight"):
    r = df["user_idx"].to_numpy().astype(np.int32)
    c = df["item_idx"].to_numpy().astype(np.int32)
    w = df[weight_col].to_numpy().astype(np.float32)
    m = sp.csr_matrix((w, (r, c)), shape=(nu, ni))
    m.sum_duplicates()
    return m


mat_full = build_sparse(train, n_users, n_items)   # weighted (iALS)
mat_tr = build_sparse(df_tr, n_users, n_items)
print(f"weighted matrix nnz={mat_full.nnz:,}  "
      f"density={mat_full.nnz / (n_users * n_items) * 100:.4f}%")


def binarize(mat):
    b = mat.copy().astype(np.float32)
    b.data[:] = 1.0
    return b.tocsr()


bin_full = binarize(mat_full)   # binary (EASE)
bin_tr = binarize(mat_tr)


# ─────────────────────────────────────────────────────────────────────────────
# 6. EASE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. EASE")
print("=" * 60)


def ease_gram(bin_mat):
    """Dense item-item Gram matrix G = Xᵀ X  (float64)."""
    t = time.time()
    G = (bin_mat.T @ bin_mat).toarray().astype(np.float64)
    print(f"  Gram {G.shape} built in {time.time() - t:.0f}s")
    return G


def ease_solve(G, reg):
    """Solve EASE weight matrix B for a given L2 reg. G is modified-then-restored."""
    t = time.time()
    di = np.diag_indices(G.shape[0])
    saved = G[di].copy()
    G[di] = saved + reg
    P = np.linalg.inv(G)
    G[di] = saved                       # restore for reuse across reg sweep
    d = np.diag(P).copy()
    B = P / (-d)
    B[di] = 0.0
    print(f"  EASE solve reg={reg:<6.0f} in {time.time() - t:.0f}s")
    return B.astype(np.float32)


def ease_recommend(B, X, uids, N, filter_seen=True):
    """Top-N EASE recommendations. Returns (ids[b,N], scores[b,N])."""
    uids = np.asarray(uids, np.int32)
    Xb = X[uids]
    S = (Xb @ B)                                   # dense (b, n_items)
    if filter_seen:
        Xb_coo = Xb.tocoo()
        S[Xb_coo.row, Xb_coo.col] = -np.inf
    part = np.argpartition(-S, kth=N - 1, axis=1)[:, :N]
    part_s = np.take_along_axis(S, part, axis=1)
    order = np.argsort(-part_s, axis=1)
    ids = np.take_along_axis(part, order, axis=1).astype(np.int32)
    sc = np.take_along_axis(part_s, order, axis=1).astype(np.float32)
    sc[~np.isfinite(sc)] = 0.0
    return ids, sc


# ── reg sweep on temporal-train Gram, scored against val retrieval NDCG ──────
def ndcg(actual, predicted, k=TOP_K):
    dcg = sum(1 / np.log2(i + 2) for i, p in enumerate(predicted[:k]) if p in actual)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return dcg / idcg if idcg > 0 else 0.0


val_gt = {
    r[0]: set(r[1])
    for r in df_val.group_by("user_idx").agg(pl.col("item_idx").alias("items")).rows()
}
tr_user_set = set(df_tr["user_idx"].unique().to_list())
check_pool = [u for u in val_gt.keys() if u in tr_user_set]
rng = np.random.default_rng(RANDOM_SEED + 1)
check_uids = rng.choice(
    check_pool, min(N_CHECK_USERS, len(check_pool)), replace=False
).tolist()

print(f"  Reg sweep on {len(check_uids):,} warm-val users")
G_tr = ease_gram(bin_tr)
best_reg, best_ndcg, B_tr = None, -1.0, None
for reg in EASE_REGS:
    B_candidate = ease_solve(G_tr, reg)
    ids, _ = ease_recommend(B_candidate, bin_tr, check_uids, TOP_K)
    sc_ndcg = np.mean([
        ndcg(val_gt[u], ids[i].tolist())
        for i, u in enumerate(check_uids) if u in val_gt
    ])
    print(f"    reg={reg:<6.0f}  val NDCG@20={sc_ndcg:.4f}")
    if sc_ndcg > best_ndcg:
        best_ndcg, best_reg, B_tr = sc_ndcg, reg, B_candidate
del G_tr
gc.collect()
print(f"  → best EASE reg={best_reg}  (val NDCG@20={best_ndcg:.4f})")

# Full-data EASE with the chosen reg
G_full = ease_gram(bin_full)
B_full = ease_solve(G_full, best_reg)
del G_full
gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# 7. iALS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. iALS (implicit)")
print("=" * 60)

USE_CUDA = detect_cuda()
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight


def train_ials(mat, label=""):
    t = time.time()
    model = AlternatingLeastSquares(
        factors=IALS_FACTORS,
        iterations=IALS_ITERS,
        regularization=IALS_REG,
        alpha=IALS_ALPHA,
        use_gpu=USE_CUDA,
        random_state=RANDOM_SEED,
    )
    model.fit(bm25_weight(mat, K1=BM25_K1, B=BM25_B).tocsr(), show_progress=False)
    print(f"  {label} done in {time.time() - t:.0f}s")
    return model


ials_full = train_ials(mat_full, "Full iALS")
ials_tr = train_ials(mat_tr, "Train iALS")


def to_cpu(x):
    return x.to_numpy() if hasattr(x, "to_numpy") else np.array(x)


U_f, V_f = to_cpu(ials_full.user_factors), to_cpu(ials_full.item_factors)
U_tr, V_tr = to_cpu(ials_tr.user_factors), to_cpu(ials_tr.item_factors)


def ials_recommend(model, mat, uids, N):
    arr = np.asarray(uids, np.int32)
    return model.recommend(arr, mat[arr], N=N, filter_already_liked_items=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. WARM TEST USERS + RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. Candidate generation")
print("=" * 60)

item_popularity = np.asarray(mat_full.sum(axis=0)).ravel().astype(np.float32)
pop_top_global = np.argsort(-item_popularity)[:N_CANDIDATES]

user_id_set = set(all_users_s.to_list())
test_list = test_u["user_id"].to_list()
warm_df = pl.DataFrame({"user_id": [u for u in test_list if u in user_id_set]}).join(
    user_map, on="user_id", how="left"
)
warm_uids = warm_df["user_idx"].to_list()
widx2uid = dict(zip(warm_df["user_idx"].to_list(), warm_df["user_id"].to_list()))
cold_users = [u for u in test_list if u not in user_id_set]
print(f"Warm: {len(warm_uids):,}  Cold: {len(cold_users):,}")


def pop_fill(mat, uids, primary_ids, extra_ids=None, n=N_CANDIDATES):
    """Union retriever outputs (dedup, drop seen/invalid), top up with popularity."""
    out = {}
    for i, uid in enumerate(uids):
        s, e = mat.indptr[uid], mat.indptr[uid + 1]
        seen = set(mat.indices[s:e].tolist())
        cands, cs = [], set()
        for ids_arr in (primary_ids, extra_ids):
            if ids_arr is None:
                continue
            for it in ids_arr[i].tolist():
                it = int(it)
                if 0 <= it < n_items and it not in cs:
                    cands.append(it)
                    cs.add(it)
                if len(cands) >= n:
                    break
            if len(cands) >= n:
                break
        for it in pop_top_global:
            if len(cands) >= n:
                break
            it = int(it)
            if it not in seen and it not in cs:
                cands.append(it)
                cs.add(it)
        out[uid] = cands[:n]
    return out


def batch_ease(B, X, uids, N):
    il, sl = [], []
    for s in range(0, len(uids), REC_BATCH):
        ib, sb = ease_recommend(B, X, uids[s:s + REC_BATCH], N)
        il.append(ib)
        sl.append(sb)
    return np.vstack(il), np.vstack(sl)


def batch_ials(model, mat, uids, N):
    il, sl = [], []
    for s in range(0, len(uids), REC_BATCH):
        ib, sb = ials_recommend(model, mat, uids[s:s + REC_BATCH], N)
        il.append(ib)
        sl.append(sb)
    return np.vstack(il), np.vstack(sl)


print("  EASE retrieval (warm test)...")
t0 = time.time()
ease_ids, ease_sc = batch_ease(B_full, bin_full, warm_uids, EASE_CANDIDATES)
print(f"    done in {time.time() - t0:.0f}s")

print("  iALS retrieval (warm test)...")
t0 = time.time()
ials_ids, ials_sc = batch_ials(ials_full, mat_full, warm_uids, IALS_CANDIDATES)
print(f"    done in {time.time() - t0:.0f}s")

candidates = pop_fill(mat_full, warm_uids, ease_ids, ials_ids)

# ─────────────────────────────────────────────────────────────────────────────
# 9. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. Feature engineering")
print("=" * 60)

max_ts = train["ts_int"].max()

# ── user stats ──
user_aggs = [
    pl.col("item_idx").n_unique().alias("user_n_items"),
    pl.col("item_idx").count().alias("user_n_interactions"),
]
if has_purchase:
    user_aggs += [
        pl.col("is_purchased").sum().cast(pl.Int32).alias("user_n_purchases"),
        pl.col("is_purchased").mean().cast(pl.Float32).alias("user_purchase_rate"),
    ]
if has_rating:
    user_aggs += [
        pl.col("rating").filter(pl.col("rating") > 0).mean().fill_null(0.0)
        .cast(pl.Float32).alias("user_avg_rating"),
        pl.col("rating").filter(pl.col("rating") > 0).count()
        .cast(pl.Int32).alias("user_n_ratings"),
    ]
user_aggs.append(
    ((max_ts - pl.col("ts_int")) / (3600 * 24 * 1_000_000))
    .min().cast(pl.Float32).alias("user_days_since_last")
)
user_stats_pl = train.group_by("user_idx").agg(user_aggs)
print(f"  user_stats {user_stats_pl.shape}")

# ── item stats ──
item_aggs = [
    pl.col("user_idx").n_unique().alias("item_n_users"),
    pl.col("user_idx").count().alias("item_n_interactions"),
]
if has_purchase:
    item_aggs += [
        pl.col("is_purchased").sum().cast(pl.Int32).alias("item_n_purchases"),
        pl.col("is_purchased").mean().cast(pl.Float32).alias("item_purchase_rate"),
    ]
if has_rating:
    item_aggs += [
        pl.col("rating").filter(pl.col("rating") > 0).mean().fill_null(0.0)
        .cast(pl.Float32).alias("item_avg_rating"),
        pl.col("rating").filter(pl.col("rating") > 0).count()
        .cast(pl.Int32).alias("item_n_ratings"),
    ]
item_stats_pl = train.group_by("item_idx").agg(item_aggs)

# ── CTR from impressions (strongest single rerank feature) ──
item_ctr_pl = None
if has_impressions:
    print("  CTR from impressions (streaming)...")
    t0 = time.time()
    n_shown = (
        train.lazy().select("impressions").explode("impressions")
        .rename({"impressions": "item_id"})
        .group_by("item_id").agg(pl.len().alias("n_shown"))
        .collect(engine="streaming")
    )
    n_clicks = train.lazy().group_by("item_id").agg(pl.len().alias("n_clicks")).collect()
    item_ctr_pl = (
        n_clicks.join(n_shown, on="item_id", how="left")
        .join(item_map, on="item_id", how="left")
        .filter(pl.col("item_idx").is_not_null())
        .with_columns([
            pl.col("n_shown").fill_null(0).cast(pl.Int32),
            (pl.col("n_clicks") / (pl.col("n_shown").fill_null(0) + 1.0))
            .cast(pl.Float32).alias("item_ctr"),
        ])
        .select(["item_idx", "n_shown", "item_ctr"])
    )
    print(f"    CTR done in {time.time() - t0:.0f}s  items={len(item_ctr_pl):,}")

# ── item catalog features ──
item_feat = items.join(item_map, on="item_id", how="left").filter(
    pl.col("item_idx").is_not_null()
)
exprs = []
for col, dtype in item_feat.schema.items():
    if col in ("item_id", "item_idx"):
        continue
    if isinstance(dtype, pl.List):
        exprs += [
            pl.col(col).list.len().cast(pl.Int32).alias(f"{col}_count"),
            pl.col(col).list.first().cast(pl.Int32).fill_null(-1).alias(f"{col}_first"),
        ]
    elif dtype in (pl.Utf8, pl.Categorical, pl.String):
        exprs.append(pl.col(col).cast(pl.Categorical).cast(pl.UInt32).cast(pl.Int32))
    elif dtype.is_numeric():
        exprs.append(pl.col(col).cast(pl.Float32))
if exprs:
    item_feat = item_feat.with_columns(exprs)
catalog_cols = [
    c for c in item_feat.columns
    if c not in ("item_id",) and item_feat[c].dtype.is_numeric() and c != "item_idx"
]
item_feat_slim = item_feat.select(["item_idx"] + catalog_cols).fill_null(0)
print(f"  catalog cols: {catalog_cols}")

item_all_feats = item_stats_pl
if item_ctr_pl is not None:
    item_all_feats = item_all_feats.join(item_ctr_pl, on="item_idx", how="left")
item_all_feats = item_all_feats.join(item_feat_slim, on="item_idx", how="left").fill_null(0)
print(f"  item_all_feats {item_all_feats.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. FEATURE FRAME BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_frame(uids, cands_dict, b_ease_ids, b_ease_sc,
                        b_ials_ids, b_ials_sc, label_pl=None, U=None, V=None):
    if U is None:
        U, V = U_f, V_f
    n_per = [len(cands_dict[u]) for u in uids]
    total = sum(n_per)
    u_col = np.repeat(np.array(uids, np.int32), n_per)
    i_col = np.concatenate([np.array(cands_dict[u], np.int32) for u in uids])
    rank_col = np.concatenate([np.arange(n, dtype=np.int32) for n in n_per])

    ease_sc_col = np.zeros(total, np.float32)
    ease_rank_col = np.full(total, N_CANDIDATES + 1, np.int32)
    from_ease_col = np.zeros(total, np.int32)
    ials_sc_col = np.zeros(total, np.float32)
    ials_rank_col = np.full(total, N_CANDIDATES + 1, np.int32)
    from_ials_col = np.zeros(total, np.int32)

    offset = 0
    for j in range(len(uids)):
        n = n_per[j]
        ease_lkp = {int(iid): (rk, float(sc)) for rk, (iid, sc)
                    in enumerate(zip(b_ease_ids[j].tolist(), b_ease_sc[j].tolist()))}
        ials_lkp = {int(iid): (rk, float(sc)) for rk, (iid, sc)
                    in enumerate(zip(b_ials_ids[j].tolist(), b_ials_sc[j].tolist()))}
        for k, iid in enumerate(i_col[offset:offset + n]):
            h = ease_lkp.get(int(iid))
            if h is not None:
                ease_rank_col[offset + k] = h[0]
                ease_sc_col[offset + k] = h[1]
                from_ease_col[offset + k] = 1
            h = ials_lkp.get(int(iid))
            if h is not None:
                ials_rank_col[offset + k] = h[0]
                ials_sc_col[offset + k] = h[1]
                from_ials_col[offset + k] = 1
        offset += n

    df = pl.DataFrame({
        "user_idx": pl.Series(u_col, dtype=pl.Int32),
        "item_idx": pl.Series(i_col, dtype=pl.Int32),
        "cand_rank": pl.Series(rank_col, dtype=pl.Int32),
        "ease_score": pl.Series(ease_sc_col, dtype=pl.Float32),
        "ease_rank": pl.Series(ease_rank_col, dtype=pl.Int32),
        "from_ease": pl.Series(from_ease_col, dtype=pl.Int32),
        "ials_score": pl.Series(ials_sc_col, dtype=pl.Float32),
        "ials_rank": pl.Series(ials_rank_col, dtype=pl.Int32),
        "from_ials": pl.Series(from_ials_col, dtype=pl.Int32),
        "pop_score": pl.Series(item_popularity[i_col], dtype=pl.Float32),
    })
    df = (df.join(item_all_feats, on="item_idx", how="left")
            .join(user_stats_pl, on="user_idx", how="left").fill_null(0))

    u_arr = u_col.clip(0, len(U) - 1)
    i_arr = i_col.clip(0, len(V) - 1)
    ials_dot = (U[u_arr] * V[i_arr]).sum(axis=1).astype(np.float32)
    df = df.with_columns(pl.Series("ials_dot", ials_dot, dtype=pl.Float32))

    if label_pl is not None:
        df = df.join(
            label_pl.with_columns(pl.lit(1).cast(pl.Int32).alias("label")),
            on=["user_idx", "item_idx"], how="left",
        ).with_columns(pl.col("label").fill_null(0).cast(pl.Int32))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 11. LightGBM LambdaRank
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("11. LightGBM LambdaRank")
print("=" * 60)

USE_LGBM = False
feature_cols = []
tr_s = []
try:
    import lightgbm as lgb

    rng = np.random.default_rng(RANDOM_SEED)
    val_users_warm = [u for u in df_val["user_idx"].unique().to_list() if u in tr_user_set]
    rng.shuffle(val_users_warm)
    val_users_warm = val_users_warm[:N_RANK_USERS]
    n_split = int(len(val_users_warm) * 0.8)
    tr_s, val_s = val_users_warm[:n_split], val_users_warm[n_split:]
    print(f"  ranker train={len(tr_s):,}  valid={len(val_s):,} — building candidates...")

    # Past-only models (B_tr / ials_tr / mat_tr) so val items are NOT filtered out.
    t_e_ids, t_e_sc = batch_ease(B_tr, bin_tr, tr_s, EASE_CANDIDATES)
    v_e_ids, v_e_sc = batch_ease(B_tr, bin_tr, val_s, EASE_CANDIDATES)
    t_a_ids, t_a_sc = batch_ials(ials_tr, mat_tr, tr_s, IALS_CANDIDATES)
    v_a_ids, v_a_sc = batch_ials(ials_tr, mat_tr, val_s, IALS_CANDIDATES)
    t_cands = pop_fill(mat_tr, tr_s, t_e_ids, t_a_ids)
    v_cands = pop_fill(mat_tr, val_s, v_e_ids, v_a_ids)

    val_gt_pl = df_val.select(["user_idx", "item_idx"]).unique()

    feat_tr_ = build_feature_frame(tr_s, t_cands, t_e_ids, t_e_sc, t_a_ids, t_a_sc,
                                   label_pl=val_gt_pl, U=U_tr, V=V_tr)
    feat_val = build_feature_frame(val_s, v_cands, v_e_ids, v_e_sc, v_a_ids, v_a_sc,
                                   label_pl=val_gt_pl, U=U_tr, V=V_tr)

    IGNORE = {"user_idx", "item_idx", "label"}
    feature_cols = [c for c in feat_val.columns
                    if c not in IGNORE and feat_val[c].dtype.is_numeric()]
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    feat_tr_ = feat_tr_.sort("user_idx")
    feat_val = feat_val.sort("user_idx")

    def to_lgb(df):
        X = df.select(feature_cols).to_numpy().astype(np.float32)
        y = df["label"].to_numpy().astype(np.int32)
        qid = df.group_by("user_idx", maintain_order=True).len()["len"].to_numpy()
        return X, y, qid

    X_tr, y_tr, q_tr = to_lgb(feat_tr_)
    X_val, y_val, q_val = to_lgb(feat_val)
    del feat_tr_, feat_val
    gc.collect()

    ds_tr = lgb.Dataset(X_tr, y_tr, group=q_tr, feature_name=feature_cols)
    ds_val = lgb.Dataset(X_val, y_val, group=q_val, reference=ds_tr)

    lgb_model = lgb.train(
        {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [20],
            "learning_rate": 0.05,
            "num_leaves": 127,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambdarank_truncation_level": 20,
            "verbose": -1,
            "n_jobs": -1,
            "random_state": RANDOM_SEED,
        },
        ds_tr,
        num_boost_round=1000,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(100)],
    )

    USE_LGBM = True
    fi = pl.DataFrame({"feature": feature_cols,
                       "gain": lgb_model.feature_importance("gain").tolist()}
                      ).sort("gain", descending=True)
    print(f"\n  Best iter: {lgb_model.best_iteration}")
    print(f"  Val NDCG@20: {lgb_model.best_score['valid_0']['ndcg@20']:.4f}")
    print(f"\n  Feature importance:\n{fi.head(20)}")

except Exception:
    import traceback
    traceback.print_exc()
    print("\n  LightGBM skipped → score blending fallback")

# ─────────────────────────────────────────────────────────────────────────────
# 12. HONEST VALIDATION NDCG@20
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("12. Validation NDCG@20")
print("=" * 60)

ranker_train_set = set(tr_s) if USE_LGBM else set()
vpool = [u for u in val_gt.keys() if u in tr_user_set and u not in ranker_train_set]
rng2 = np.random.default_rng(RANDOM_SEED + 7)
check2 = rng2.choice(vpool, min(N_CHECK_USERS, len(vpool)), replace=False).tolist()

c_e_ids, c_e_sc = batch_ease(B_tr, bin_tr, check2, EASE_CANDIDATES)
c_a_ids, c_a_sc = batch_ials(ials_tr, mat_tr, check2, IALS_CANDIDATES)
c_cands = pop_fill(mat_tr, check2, c_e_ids, c_a_ids)

ndcg_ease = np.mean([ndcg(val_gt[u], c_e_ids[i, :TOP_K].tolist())
                     for i, u in enumerate(check2) if u in val_gt])
ndcg_ials = np.mean([ndcg(val_gt[u], c_a_ids[i, :TOP_K].tolist())
                     for i, u in enumerate(check2) if u in val_gt])
ndcg_union = np.mean([ndcg(val_gt[u], c_cands[u]) for u in check2 if u in val_gt])
print(f"  NDCG@20 EASE-only : {ndcg_ease:.4f}")
print(f"  NDCG@20 iALS-only : {ndcg_ials:.4f}")
print(f"  NDCG@20 union@300 : {ndcg_union:.4f}  (recall ceiling)")

if USE_LGBM:
    feat_c = build_feature_frame(check2, c_cands, c_e_ids, c_e_sc, c_a_ids, c_a_sc,
                                 U=U_tr, V=V_tr)
    X_c = feat_c.select(feature_cols).to_numpy().astype(np.float32)
    feat_c = feat_c.with_columns(
        pl.Series("s", lgb_model.predict(X_c).astype(np.float32)))
    nd = []
    for uid in check2:
        if uid not in val_gt:
            continue
        pred = (feat_c.filter(pl.col("user_idx") == uid)
                .sort("s", descending=True)["item_idx"].to_list())
        nd.append(ndcg(val_gt[uid], pred))
    print(f"  NDCG@20 RERANK    : {np.mean(nd):.4f}  ({len(nd):,} users)")
    del feat_c
    gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# 13. SCORING → TOP-20
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("13. Batched scoring")
print("=" * 60)

n_warm = len(warm_uids)
top_u, top_i = [], []
t0 = time.time()

for bstart in range(0, n_warm, SCORE_BATCH):
    bend = min(bstart + SCORE_BATCH, n_warm)
    b_uids = warm_uids[bstart:bend]
    b_e_ids, b_e_sc = ease_ids[bstart:bend], ease_sc[bstart:bend]
    b_a_ids, b_a_sc = ials_ids[bstart:bend], ials_sc[bstart:bend]
    b_cands = {u: candidates[u] for u in b_uids}

    if USE_LGBM:
        feat_b = build_feature_frame(b_uids, b_cands, b_e_ids, b_e_sc, b_a_ids, b_a_sc)
        X_b = feat_b.select(feature_cols).to_numpy().astype(np.float32)
        feat_b = feat_b.with_columns(
            pl.Series("s", lgb_model.predict(X_b).astype(np.float32)))
        top_b = (feat_b.sort(["user_idx", "s"], descending=[False, True])
                 .group_by("user_idx", maintain_order=True).head(TOP_K))
        u_np = top_b["user_idx"].to_numpy()
        i_np = top_b["item_idx"].to_numpy()
        del feat_b, top_b, X_b
        gc.collect()
    else:
        # fallback: min-max blend of EASE + iALS + popularity
        u_np_l, i_np_l = [], []
        for j, uid in enumerate(b_uids):
            cands = b_cands[uid]
            e_lkp = dict(zip(b_e_ids[j].tolist(), b_e_sc[j].tolist()))
            a_lkp = dict(zip(b_a_ids[j].tolist(), b_a_sc[j].tolist()))
            se = np.array([e_lkp.get(it, 0.0) for it in cands], np.float32)
            sa = np.array([a_lkp.get(it, 0.0) for it in cands], np.float32)
            spop = item_popularity[cands]
            mm = lambda x: ((x - x.min()) / (x.max() - x.min() + 1e-9)
                            if x.max() > x.min() else x)
            order = np.argsort(-(0.7 * mm(se) + 0.2 * mm(sa) + 0.1 * mm(spop)))[:TOP_K]
            u_np_l.extend([uid] * TOP_K)
            i_np_l.extend(np.array(cands)[order])
        u_np = np.array(u_np_l, np.int32)
        i_np = np.array(i_np_l, np.int32)

    top_u.extend([widx2uid[int(u)] for u in u_np])
    top_i.extend(idx2item[i_np].tolist())

    elapsed = time.time() - t0
    eta = elapsed / bend * n_warm - elapsed
    print(f"  {bend:>{len(str(n_warm))}}/{n_warm}  {elapsed:.0f}s  ETA {eta:.0f}s", end="\r")
print()

# Cold-start fallback → global popularity (none expected here)
cold_pop = idx2item[pop_top_global[:TOP_K]].tolist()
for uid in cold_users:
    top_u.extend([uid] * TOP_K)
    top_i.extend(cold_pop)

# ─────────────────────────────────────────────────────────────────────────────
# 14. SAVE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("14. Saving submission")
print("=" * 60)

submission = pl.DataFrame({"user_id": pl.Series(top_u), "item_id": pl.Series(top_i)})
counts = submission.group_by("user_id").len()["len"]
print(f"Rows/user: min={counts.min()} max={counts.max()} mean={counts.mean():.1f}")
print(f"Total: {len(submission):,}  (expected {test_u.shape[0] * TOP_K:,})")

out_path = OUT_DIR / "submission.csv"
submission.write_csv(out_path)
print(f"\n✅  {out_path}")
print(submission.head(22))
