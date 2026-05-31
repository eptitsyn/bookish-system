"""
Kaggle Recommender System — v5
• Weighted ALS: is_purchased×5 + rating (instead of binary)
• Rich features: purchase_rate, avg_rating, CTR from impressions, recency
• NDCG fix: validation uses als_tr candidates (no val-data leakage)
• Batched OOM-safe scoring
• Polars + CUDA-aware
"""

import warnings
import gc
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path("./data")
OUT_DIR = Path("./output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CANDIDATES = 200
TOP_K = 20
TFIDF_K = 500  # item-item neighbourhood size — primary retriever
ALS_FACTORS = 64  # ALS kept only for the als_dot rerank feature
ALS_ITERS = 15
ALS_REG = 0.01
BM25_K1 = 1.5
BM25_B = 0.75
RANDOM_SEED = 42
ALS_BATCH = 4096
SCORE_BATCH = 2000

# Веса взаимодействий для ALS
W_VIEW = 1.0  # просмотр без оценки
W_RATING_BONUS = 0.5  # бонус за каждую звезду оценки
W_PURCHASE = 5.0  # покупка
W_PURCHASE_RAT = 1.0  # дополнительно за каждую звезду при покупке

np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# CUDA
# ─────────────────────────────────────────────────────────────────────────────
def detect_cuda() -> bool:
    for check in [
        lambda: __import__("cupy").cuda.runtime.getDeviceCount() > 0,
        lambda: getattr(__import__("implicit.gpu", fromlist=["HAS_CUDA"]), "HAS_CUDA"),
        lambda: __import__("torch").cuda.is_available(),
    ]:
        try:
            if check():
                print("  ✓ CUDA found")
                return True
        except Exception:
            pass
    print("  ✗ No CUDA — CPU")
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


# ─────────────────────────────────────────────────────────────────────────────
# 2. COLUMN DETECTION + RENAME
# ─────────────────────────────────────────────────────────────────────────────
def detect_col(schema, cands):
    for c in cands:
        if c in schema:
            return c
    for c in schema:
        for kw in cands:
            if kw.lower() in c.lower():
                return c
    return list(schema.keys())[0]


time_kws = ["timestamp", "time", "date", "event_time", "datetime"]
USER_COL = detect_col(train.schema, ["user_id", "userId", "user", "uid"])
ITEM_COL = detect_col(train.schema, ["item_id", "itemId", "item", "iid", "product_id"])
TIME_COL = next(
    (c for c in train.columns if any(k in c.lower() for k in time_kws)), None
)
PURCH_COL = next(
    (c for c in train.columns if "purch" in c.lower() or "bought" in c.lower()), None
)
RATING_COL = next(
    (c for c in train.columns if "rating" in c.lower() or "score" in c.lower()), None
)
IMP_COL = next(
    (c for c in train.columns if "impress" in c.lower() or "slate" in c.lower()), None
)

print(
    f"USER={USER_COL} ITEM={ITEM_COL} TIME={TIME_COL} "
    f"PURCH={PURCH_COL} RATING={RATING_COL} IMP={IMP_COL}"
)

ren = {USER_COL: "user_id", ITEM_COL: "item_id"}
if TIME_COL:
    ren[TIME_COL] = "timestamp"
if PURCH_COL:
    ren[PURCH_COL] = "is_purchased"
if RATING_COL:
    ren[RATING_COL] = "rating"
if IMP_COL:
    ren[IMP_COL] = "impressions"

train = train.rename(ren)
items = items.rename(
    {detect_col(items.schema, ["item_id", "itemId", "item", "iid"]): "item_id"}
)
test_u = test_u.rename(
    {detect_col(test_u.schema, ["user_id", "userId", "user", "uid"]): "user_id"}
)

has_purchase = "is_purchased" in train.columns
has_rating = "rating" in train.columns
has_impressions = "impressions" in train.columns
print(
    f"has_purchase={has_purchase}  has_rating={has_rating}  has_impressions={has_impressions}"
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. INTERACTION WEIGHTS  (is_purchased × 5 + rating bonus)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Computing interaction weights")
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
        pl.when(pl.col("is_purchased"))
        .then(pl.lit(W_PURCHASE))
        .otherwise(pl.lit(W_VIEW))
        .cast(pl.Float32)
        .alias("weight")
    )
elif has_rating:
    train = train.with_columns(
        (W_VIEW + pl.col("rating").cast(pl.Float32) * W_RATING_BONUS).alias("weight")
    )
else:
    train = train.with_columns(pl.lit(1.0, dtype=pl.Float32).alias("weight"))

w_stats = train["weight"].describe()
print(f"Weight stats:\n{w_stats}")
print(f"Purchases: {train['is_purchased'].sum():,}" if has_purchase else "")

# ─────────────────────────────────────────────────────────────────────────────
# 4. INTEGER ENCODING
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
# 5. TEMPORAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Train / Val split")
print("=" * 60)

if "timestamp" in train.columns:
    train = train.with_columns(pl.col("timestamp").cast(pl.Int64).alias("ts_int"))
    split_val = train["ts_int"].quantile(0.9)
    df_tr = train.filter(pl.col("ts_int") <= split_val)
    df_val = train.filter(pl.col("ts_int") > split_val)
    print(f"Temporal q90:  tr={len(df_tr):,}  val={len(df_val):,}")
else:
    n90 = int(len(train) * 0.9)
    sh = train.sample(fraction=1.0, shuffle=True, seed=RANDOM_SEED)
    df_tr, df_val = sh[:n90], sh[n90:]
    print(f"Random 90/10:  tr={len(df_tr):,}  val={len(df_val):,}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. WEIGHTED SPARSE MATRICES
# ─────────────────────────────────────────────────────────────────────────────
def build_sparse(df, nu, ni, weight_col="weight"):
    r = df["user_idx"].to_numpy().astype(np.int32)
    c = df["item_idx"].to_numpy().astype(np.int32)
    w = df[weight_col].to_numpy().astype(np.float32)
    m = sp.csr_matrix((w, (r, c)), shape=(nu, ni))
    m.sum_duplicates()
    return m


mat_full = build_sparse(train, n_users, n_items)
mat_tr = build_sparse(df_tr, n_users, n_items)
print(
    f"\nWeighted matrix: nnz={mat_full.nnz:,}  "
    f"density={mat_full.nnz / (n_users * n_items) * 100:.4f}%  "
    f"max_val={mat_full.data.max():.1f}"
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. iALS + CUDA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. CUDA + Weighted iALS")
print("=" * 60)

USE_CUDA = detect_cuda()

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight, TFIDFRecommender


# ── Primary retriever: TF-IDF item-item CF ──────────────────────────────────
# Offline (temporal q90, 2k warm val users) this beats weighted iALS by a wide
# margin: NDCG@20 0.0885 vs 0.0665, recall@200 0.40.  The reranker then lifts
# it further (→ ~0.11 on held-out users) with als_dot as the strongest feature.
def train_tfidf(mat, label=""):
    t = time.time()
    model = TFIDFRecommender(K=TFIDF_K)
    model.fit(mat, show_progress=False)
    print(f"  {label} done in {time.time() - t:.0f}s")
    return model


tfidf_full = train_tfidf(mat_full, "Full TFIDF")
tfidf_tr = train_tfidf(mat_tr, "Train TFIDF")


# ── ALS kept only to produce latent factors for the als_dot rerank feature ──
def train_als(mat, label=""):
    t = time.time()
    model = AlternatingLeastSquares(
        factors=ALS_FACTORS,
        iterations=ALS_ITERS,
        regularization=ALS_REG,
        use_gpu=USE_CUDA,
        random_state=RANDOM_SEED,
    )
    model.fit(bm25_weight(mat, K1=BM25_K1, B=BM25_B), show_progress=False)
    print(f"  {label} done in {time.time() - t:.0f}s")
    return model


als_full = train_als(mat_full, "Full ALS (dot feature)")
als_tr = train_als(mat_tr, "Train ALS (dot feature)")


# GPU factors → CPU numpy
def to_cpu(x):
    return x.to_numpy() if hasattr(x, "to_numpy") else np.array(x)


U_f = to_cpu(als_full.user_factors)  # (n_users, factors)
V_f = to_cpu(als_full.item_factors)  # (n_items, factors)
U_tr = to_cpu(als_tr.user_factors)
V_tr = to_cpu(als_tr.item_factors)

# ─────────────────────────────────────────────────────────────────────────────
# 8. CANDIDATE GENERATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. Candidate generation")
print("=" * 60)

item_popularity = np.asarray(mat_full.sum(axis=0)).ravel().astype(np.float32)
pop_top_global = np.argsort(-item_popularity)[:N_CANDIDATES]


def als_recommend(model, mat, uids, n=N_CANDIDATES):
    # Generic implicit recommend — works identically for TFIDF and ALS models.
    arr = np.array(uids, np.int32)
    return model.recommend(arr, mat[arr], N=n, filter_already_liked_items=True)


def pop_fill(mat, uids, als_ids_arr, n=N_CANDIDATES):
    out = {}
    for i, uid in enumerate(uids):
        s, e = mat.indptr[uid], mat.indptr[uid + 1]
        seen = set(mat.indices[s:e].tolist())
        # Dedupe retriever output and drop invalid/padding indices: degenerate
        # users (tiny history) can get repeated padding items from implicit.
        cands, cs = [], set()
        for it in als_ids_arr[i].tolist():
            it = int(it)
            if 0 <= it < n_items and it not in cs:
                cands.append(it)
                cs.add(it)
        for it in pop_top_global:
            if len(cands) >= n:
                break
            it = int(it)
            if it not in seen and it not in cs:
                cands.append(it)
                cs.add(it)
        out[uid] = cands[:n]
    return out


# Warm / cold test users
user_id_set = set(all_users_s.to_list())
test_list = test_u["user_id"].to_list()
warm_df = pl.DataFrame({"user_id": [u for u in test_list if u in user_id_set]}).join(
    user_map, on="user_id", how="left"
)
warm_uids = warm_df["user_idx"].to_list()
widx2uid = dict(zip(warm_df["user_idx"].to_list(), warm_df["user_id"].to_list()))
cold_users = [u for u in test_list if u not in user_id_set]
print(f"Warm: {len(warm_uids):,}  Cold: {len(cold_users):,}")

# Batched TFIDF recommend
ids_l, sc_l = [], []
t0 = time.time()
for s in range(0, len(warm_uids), ALS_BATCH):
    ib, sb = als_recommend(tfidf_full, mat_full, warm_uids[s : s + ALS_BATCH])
    ids_l.append(ib)
    sc_l.append(sb)
    if s % (ALS_BATCH * 8) == 0:
        print(f"  TFIDF recommend {s:,}/{len(warm_uids):,}", end="\r")

als_ids = np.vstack(ids_l)  # (n_warm, N_CANDIDATES)
als_sc_arr = np.vstack(sc_l)  # (n_warm, N_CANDIDATES)
print(f"\n  Done in {time.time() - t0:.1f}s")
candidates = pop_fill(mat_full, warm_uids, als_ids)

# ─────────────────────────────────────────────────────────────────────────────
# 9. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. Feature engineering")
print("=" * 60)

# ── 9a. User-level stats ─────────────────────────────────────────────────────
print("  User stats...")
max_ts = train["ts_int"].max() if "ts_int" in train.columns else None

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
        pl.col("rating")
        .filter(pl.col("rating") > 0)
        .mean()
        .fill_null(0.0)
        .cast(pl.Float32)
        .alias("user_avg_rating"),
        pl.col("rating")
        .filter(pl.col("rating") > 0)
        .count()
        .cast(pl.Int32)
        .alias("user_n_ratings"),
    ]
if max_ts is not None:
    user_aggs.append(
        ((max_ts - pl.col("ts_int")) / (3600 * 24 * 1_000_000))  # μs → days
        .min()
        .cast(pl.Float32)
        .alias("user_days_since_last")
    )

user_stats_pl = train.group_by("user_idx").agg(user_aggs)
print(f"  user_stats: {user_stats_pl.shape}  cols={user_stats_pl.columns}")

# ── 9b. Item-level stats ─────────────────────────────────────────────────────
print("  Item stats...")
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
        pl.col("rating")
        .filter(pl.col("rating") > 0)
        .mean()
        .fill_null(0.0)
        .cast(pl.Float32)
        .alias("item_avg_rating"),
        pl.col("rating")
        .filter(pl.col("rating") > 0)
        .count()
        .cast(pl.Int32)
        .alias("item_n_ratings"),
    ]

item_stats_pl = train.group_by("item_idx").agg(item_aggs)

# ── 9c. CTR from impressions ─────────────────────────────────────────────────
item_ctr_pl = None
if has_impressions:
    print("  Item CTR from impressions (full train, streaming)...")
    t0 = time.time()

    # CTR is the single strongest rerank feature — use ALL impressions (no
    # sampling). 227M exploded rows handled via polars streaming engine.
    n_shown = (
        train.lazy()
        .select("impressions")
        .explode("impressions")
        .rename({"impressions": "item_id"})
        .group_by("item_id")
        .agg(pl.len().alias("n_shown"))
        .collect(engine="streaming")
    )
    # n_clicks: how many times each item was the chosen item
    n_clicks = (
        train.lazy().group_by("item_id").agg(pl.len().alias("n_clicks")).collect()
    )
    item_ctr_pl = (
        n_clicks.join(n_shown, on="item_id", how="left")
        .join(item_map, on="item_id", how="left")
        .filter(pl.col("item_idx").is_not_null())
        .with_columns(
            [
                pl.col("n_shown").fill_null(0).cast(pl.Int32),
                (pl.col("n_clicks") / (pl.col("n_shown").fill_null(0) + 1.0))
                .cast(pl.Float32)
                .alias("item_ctr"),
            ]
        )
        .select(["item_idx", "n_shown", "item_ctr"])
    )
    print(
        f"  CTR done in {time.time() - t0:.1f}s  |  items with CTR: {len(item_ctr_pl):,}"
    )

# ── 9d. Item catalog features ────────────────────────────────────────────────
print("  Item catalog features...")
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
    c
    for c in item_feat.columns
    if c not in ("item_id",) and item_feat[c].dtype.is_numeric() and c != "item_idx"
]
item_feat_slim = item_feat.select(["item_idx"] + catalog_cols).fill_null(0)
print(f"  Catalog cols: {catalog_cols}")

# ── 9e. Merge all item features ──────────────────────────────────────────────
item_all_feats = item_stats_pl
if item_ctr_pl is not None:
    item_all_feats = item_all_feats.join(item_ctr_pl, on="item_idx", how="left")
item_all_feats = item_all_feats.join(
    item_feat_slim, on="item_idx", how="left"
).fill_null(0)
print(f"  item_all_feats: {item_all_feats.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. FEATURE FRAME BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_frame(uids, cands_dict, b_als_ids, b_als_sc, label_pl=None,
                        U=None, V=None):
    # U/V: ALS factors used for the als_dot feature. Pass the past-only model
    # (U_tr/V_tr) when building training/val frames, full model when scoring test.
    if U is None:
        U, V = U_f, V_f
    n_per = [len(cands_dict[u]) for u in uids]
    total = sum(n_per)
    u_col = np.repeat(np.array(uids, np.int32), n_per)
    i_col = np.concatenate([np.array(cands_dict[u], np.int32) for u in uids])
    rank_col = np.concatenate([np.arange(n, dtype=np.int32) for n in n_per])

    ret_sc_col = np.zeros(total, np.float32)
    from_ret_col = np.zeros(total, np.int32)
    offset = 0
    for j in range(len(uids)):
        n = n_per[j]
        lkp = dict(zip(b_als_ids[j].tolist(), b_als_sc[j].tolist()))
        for k, iid in enumerate(i_col[offset : offset + n]):
            v = lkp.get(int(iid))
            if v is not None:
                ret_sc_col[offset + k] = v
                from_ret_col[offset + k] = 1
        offset += n

    df = pl.DataFrame(
        {
            "user_idx": pl.Series(u_col, dtype=pl.Int32),
            "item_idx": pl.Series(i_col, dtype=pl.Int32),
            "cand_rank": pl.Series(rank_col, dtype=pl.Int32),
            "ret_score": pl.Series(ret_sc_col, dtype=pl.Float32),
            "from_ret": pl.Series(from_ret_col, dtype=pl.Int32),
            "pop_score": pl.Series(item_popularity[i_col], dtype=pl.Float32),
        }
    )

    df = (
        df.join(item_all_feats, on="item_idx", how="left")
        .join(user_stats_pl, on="user_idx", how="left")
        .fill_null(0)
    )

    # ALS embedding dot product (top rerank feature)
    u_arr = u_col.clip(0, len(U) - 1)
    i_arr = i_col.clip(0, len(V) - 1)
    als_dot = (U[u_arr] * V[i_arr]).sum(axis=1).astype(np.float32)
    df = df.with_columns(pl.Series("als_dot", als_dot, dtype=pl.Float32))

    if label_pl is not None:
        df = df.join(
            label_pl.with_columns(pl.lit(1).cast(pl.Int32).alias("label")),
            on=["user_idx", "item_idx"],
            how="left",
        ).with_columns(pl.col("label").fill_null(0).cast(pl.Int32))
    return df


def batch_als_fn(model, mat, uids, n=N_CANDIDATES):
    il, sl = [], []
    for s in range(0, len(uids), ALS_BATCH):
        ib, sb = als_recommend(model, mat, uids[s : s + ALS_BATCH], n)
        il.append(ib)
        sl.append(sb)
    return np.vstack(il), np.vstack(sl)


# ─────────────────────────────────────────────────────────────────────────────
# 11. LightGBM LambdaRank
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. LightGBM LambdaRank")
print("=" * 60)

USE_LGBM = False
feature_cols = []
try:
    import lightgbm as lgb

    rng = np.random.default_rng(RANDOM_SEED)

    # Train the ranker on users with a KNOWN future (val-period interactions),
    # generating candidates from the past-only model (tfidf_tr / mat_tr) so val
    # items are not filtered out. Split those users into ranker-train / -valid.
    tr_user_set = set(df_tr["user_idx"].unique().to_list())
    val_users_warm = [
        u for u in df_val["user_idx"].unique().to_list() if u in tr_user_set
    ]
    rng.shuffle(val_users_warm)
    n_rank = min(20000, len(val_users_warm))
    val_users_warm = val_users_warm[:n_rank]
    n_split = int(len(val_users_warm) * 0.8)
    tr_s = val_users_warm[:n_split]   # ranker-train users
    val_s = val_users_warm[n_split:]  # ranker-valid (early stopping)

    print(f"  ranker train={len(tr_s):,}  valid={len(val_s):,} — generating candidates...")
    # ── IMPORTANT: use tfidf_tr + mat_tr so val items are NOT filtered ──────
    v_ids, v_sc = batch_als_fn(tfidf_tr, mat_tr, val_s)
    t_ids, t_sc = batch_als_fn(tfidf_tr, mat_tr, tr_s)
    v_cands = pop_fill(mat_tr, val_s, v_ids)
    t_cands = pop_fill(mat_tr, tr_s, t_ids)

    val_gt_pl = df_val.select(["user_idx", "item_idx"]).unique()

    feat_val = build_feature_frame(
        val_s, v_cands, v_ids, v_sc, label_pl=val_gt_pl, U=U_tr, V=V_tr
    )
    feat_tr_ = build_feature_frame(
        tr_s, t_cands, t_ids, t_sc, label_pl=val_gt_pl, U=U_tr, V=V_tr
    )

    IGNORE = {"user_idx", "item_idx", "label"}
    feature_cols = [
        c
        for c in feat_val.columns
        if c not in IGNORE and feat_val[c].dtype.is_numeric()
    ]
    print(f"  Features ({len(feature_cols)}):\n    {feature_cols}")

    feat_val = feat_val.sort("user_idx")
    feat_tr_ = feat_tr_.sort("user_idx")

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
            "min_data_in_leaf": 20,
            "verbose": -1,
            "n_jobs": -1,
            "random_state": RANDOM_SEED,
        },
        ds_tr,
        num_boost_round=800,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    USE_LGBM = True
    fi = pl.DataFrame(
        {"feature": feature_cols, "gain": lgb_model.feature_importance("gain").tolist()}
    ).sort("gain", descending=True)
    print(f"\n  Best iter: {lgb_model.best_iteration}")
    print(f"  Val NDCG@20: {lgb_model.best_score['valid_0']['ndcg@20']:.4f}")
    print(f"\n  Feature importance:\n{fi.head(15)}")

except Exception as exc:
    import traceback

    traceback.print_exc()
    print(f"\n  LightGBM skipped → score blending")

# ─────────────────────────────────────────────────────────────────────────────
# 12. VALIDATION NDCG@20 (proper: als_tr candidates, val ground truth)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. Validation NDCG@20")
print("=" * 60)


def ndcg(actual, predicted, k=20):
    dcg = sum(1 / np.log2(i + 2) for i, p in enumerate(predicted[:k]) if p in actual)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return dcg / idcg if idcg > 0 else 0.0


# Use val candidates (generated with als_tr — no leakage)
val_gt = {
    r[0]: set(r[1])
    for r in df_val.group_by("user_idx").agg(pl.col("item_idx").alias("items")).rows()
}

# Generate val candidates for NDCG check (tfidf_tr → no filter of val items).
# Exclude users the ranker was trained on for an honest rerank estimate.
ranker_train_set = set(tr_s) if USE_LGBM else set()
check_pool = [u for u in val_gt.keys() if u not in ranker_train_set]
rng2 = np.random.default_rng(RANDOM_SEED + 1)
check_uids = rng2.choice(
    check_pool, min(2000, len(check_pool)), replace=False
).tolist()

ids_val_check, sc_val_check = batch_als_fn(tfidf_tr, mat_tr, check_uids)
cands_val_check = pop_fill(mat_tr, check_uids, ids_val_check)

# Retrieval NDCG
ndcg_retrieval = [
    ndcg(val_gt[u], cands_val_check[u]) for u in check_uids if u in val_gt
]
print(
    f"NDCG@20 retrieval ({len(ndcg_retrieval):,} val users): {np.mean(ndcg_retrieval):.4f}"
)

# Re-rank NDCG (if LightGBM trained)
if USE_LGBM:
    feat_check = build_feature_frame(
        check_uids, cands_val_check, ids_val_check, sc_val_check, U=U_tr, V=V_tr
    )
    X_check = feat_check.select(feature_cols).to_numpy().astype(np.float32)
    feat_check = feat_check.with_columns(
        pl.Series("lgbm_score", lgb_model.predict(X_check).astype(np.float32))
    )
    ndcg_rerank = []
    for uid in check_uids:
        if uid not in val_gt:
            continue
        pred = (
            feat_check.filter(pl.col("user_idx") == uid)
            .sort("lgbm_score", descending=True)["item_idx"]
            .to_list()
        )
        ndcg_rerank.append(ndcg(val_gt[uid], pred))
    print(
        f"NDCG@20 after re-rank ({len(ndcg_rerank):,} val users): {np.mean(ndcg_rerank):.4f}"
    )
    del feat_check
    gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
# 13. BATCHED SCORING → TOP-20  (OOM-safe)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("10. Batched scoring")
print("=" * 60)

n_warm = len(warm_uids)
print(f"  {n_warm:,} users × {N_CANDIDATES} cands → chunks of {SCORE_BATCH}")

top_u, top_i = [], []
t0 = time.time()

for bstart in range(0, n_warm, SCORE_BATCH):
    bend = min(bstart + SCORE_BATCH, n_warm)
    b_uids = warm_uids[bstart:bend]
    b_als_ids = als_ids[bstart:bend]
    b_als_sc = als_sc_arr[bstart:bend]
    b_cands = {u: candidates[u] for u in b_uids}

    if USE_LGBM:
        feat_b = build_feature_frame(b_uids, b_cands, b_als_ids, b_als_sc)
        X_b = feat_b.select(feature_cols).to_numpy().astype(np.float32)
        feat_b = feat_b.with_columns(
            pl.Series("lgbm_score", lgb_model.predict(X_b).astype(np.float32))
        )
        top_b = (
            feat_b.sort("lgbm_score", descending=True)
            .group_by("user_idx", maintain_order=False)
            .head(TOP_K)
        )
        u_np = top_b["user_idx"].to_numpy()
        i_np = top_b["item_idx"].to_numpy()
        del feat_b, top_b, X_b
        gc.collect()
    else:
        u_np_l, i_np_l = [], []
        for j, uid in enumerate(b_uids):
            cands = b_cands[uid]
            sc_als = b_als_sc[j, : len(cands)].copy()
            if len(sc_als) < len(cands):
                sc_als = np.pad(sc_als, (0, len(cands) - len(sc_als)))
            sc_pop = item_popularity[cands]
            mm = lambda x: (
                (x - x.min()) / (x.max() - x.min() + 1e-9) if x.max() > x.min() else x
            )
            order = np.argsort(-(0.8 * mm(sc_als) + 0.2 * mm(sc_pop)))[:TOP_K]
            u_np_l.extend([uid] * TOP_K)
            i_np_l.extend(np.array(cands)[order])
        u_np = np.array(u_np_l, np.int32)
        i_np = np.array(i_np_l, np.int32)

    top_u.extend([widx2uid[int(u)] for u in u_np])
    top_i.extend(idx2item[i_np].tolist())

    elapsed = time.time() - t0
    eta = elapsed / bend * n_warm - elapsed
    print(
        f"  {bend:>{len(str(n_warm))}}/{n_warm}  {elapsed:.0f}s  ETA {eta:.0f}s",
        end="\r",
    )

print()

# Cold-start → global top-K popularity
cold_pop = idx2item[pop_top_global[:TOP_K]].tolist()
for uid in cold_users:
    top_u.extend([uid] * TOP_K)
    top_i.extend(cold_pop)

# ─────────────────────────────────────────────────────────────────────────────
# 14. SAVE SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("11. Saving submission")
print("=" * 60)

submission = pl.DataFrame(
    {
        "user_id": pl.Series(top_u),
        "item_id": pl.Series(top_i),
    }
)

counts = submission.group_by("user_id").len()["len"]
print(f"Rows/user: min={counts.min()}  max={counts.max()}  mean={counts.mean():.1f}")
print(f"Total: {len(submission):,}  (expected {test_u.shape[0] * TOP_K:,})")

out_path = OUT_DIR / "submission.csv"
submission.write_csv(out_path)
print(f"\n✅  {out_path}")
print(submission.head(22))
