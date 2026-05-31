import gc
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp

warnings.filterwarnings("ignore")

DATA_DIR = Path("./data")
OUT_DIR = Path("./output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CANDIDATES = 200
TOP_K = 20
TFIDF_K = 500

ALS_FACTORS = 64
ALS_ITERS = 15
ALS_REG = 0.01
BM25_K1 = 1.5
BM25_B = 0.75

RANDOM_SEED = 42
ALS_BATCH = 4096
SCORE_BATCH = 2000

W_VIEW = 1.0
W_RATING_BONUS = 0.5
W_PURCHASE = 5.0
W_PURCHASE_RAT = 1.0

np.random.seed(RANDOM_SEED)


def detect_cuda():
    checks = [
        lambda: __import__("cupy").cuda.runtime.getDeviceCount() > 0,
        lambda: getattr(__import__("implicit.gpu", fromlist=["HAS_CUDA"]), "HAS_CUDA"),
        lambda: __import__("torch").cuda.is_available(),
    ]

    for check in checks:
        try:
            if check():
                print("CUDA found")
                return True
        except Exception:
            pass

    print("CUDA not found, use CPU")
    return False


def detect_col(schema, names):
    for name in names:
        if name in schema:
            return name

    for col in schema:
        for name in names:
            if name.lower() in col.lower():
                return col

    return list(schema.keys())[0]


def build_sparse(df, n_users, n_items, weight_col="weight"):
    rows = df["user_idx"].to_numpy().astype(np.int32)
    cols = df["item_idx"].to_numpy().astype(np.int32)
    vals = df[weight_col].to_numpy().astype(np.float32)

    mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
    mat.sum_duplicates()
    return mat


def train_tfidf(mat, name=""):
    start = time.time()
    model = TFIDFRecommender(K=TFIDF_K)
    model.fit(mat, show_progress=False)
    print(f"{name} TFIDF: {time.time() - start:.0f}s")
    return model


def train_als(mat, name=""):
    start = time.time()
    model = AlternatingLeastSquares(
        factors=ALS_FACTORS,
        iterations=ALS_ITERS,
        regularization=ALS_REG,
        use_gpu=USE_CUDA,
        random_state=RANDOM_SEED,
    )
    model.fit(bm25_weight(mat, K1=BM25_K1, B=BM25_B), show_progress=False)
    print(f"{name} ALS: {time.time() - start:.0f}s")
    return model


def to_cpu(x):
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    return np.array(x)


def recommend(model, mat, users, n=N_CANDIDATES):
    users = np.array(users, dtype=np.int32)
    return model.recommend(users, mat[users], N=n, filter_already_liked_items=True)


def add_popular_items(mat, users, ids, n=N_CANDIDATES):
    result = {}

    for j, uid in enumerate(users):
        left = mat.indptr[uid]
        right = mat.indptr[uid + 1]
        seen = set(mat.indices[left:right].tolist())

        cur_items = []
        used = set()

        for item in ids[j].tolist():
            item = int(item)
            if 0 <= item < n_items and item not in used:
                cur_items.append(item)
                used.add(item)

        for item in pop_top_global:
            if len(cur_items) >= n:
                break

            item = int(item)
            if item not in seen and item not in used:
                cur_items.append(item)
                used.add(item)

        result[uid] = cur_items[:n]

    return result


def batch_recommend(model, mat, users, n=N_CANDIDATES):
    all_ids = []
    all_scores = []

    for start in range(0, len(users), ALS_BATCH):
        ids, scores = recommend(model, mat, users[start : start + ALS_BATCH], n)
        all_ids.append(ids)
        all_scores.append(scores)

    return np.vstack(all_ids), np.vstack(all_scores)


def build_feature_frame(
    users, candidates, rec_ids, rec_scores, labels=None, U=None, V=None
):
    if U is None:
        U, V = U_full, V_full

    counts = [len(candidates[u]) for u in users]
    total = sum(counts)

    user_col = np.repeat(np.array(users, dtype=np.int32), counts)
    item_col = np.concatenate([np.array(candidates[u], dtype=np.int32) for u in users])
    rank_col = np.concatenate([np.arange(n, dtype=np.int32) for n in counts])

    rec_score_col = np.zeros(total, dtype=np.float32)
    from_rec_col = np.zeros(total, dtype=np.int32)

    pos = 0
    for j, uid in enumerate(users):
        n = counts[j]
        score_by_item = dict(zip(rec_ids[j].tolist(), rec_scores[j].tolist()))

        for k, item in enumerate(item_col[pos : pos + n]):
            score = score_by_item.get(int(item))
            if score is not None:
                rec_score_col[pos + k] = score
                from_rec_col[pos + k] = 1

        pos += n

    df = pl.DataFrame(
        {
            "user_idx": pl.Series(user_col, dtype=pl.Int32),
            "item_idx": pl.Series(item_col, dtype=pl.Int32),
            "cand_rank": pl.Series(rank_col, dtype=pl.Int32),
            "ret_score": pl.Series(rec_score_col, dtype=pl.Float32),
            "from_ret": pl.Series(from_rec_col, dtype=pl.Int32),
            "pop_score": pl.Series(item_popularity[item_col], dtype=pl.Float32),
        }
    )

    df = (
        df.join(item_features, on="item_idx", how="left")
        .join(user_features, on="user_idx", how="left")
        .fill_null(0)
    )

    user_arr = user_col.clip(0, len(U) - 1)
    item_arr = item_col.clip(0, len(V) - 1)
    als_dot = (U[user_arr] * V[item_arr]).sum(axis=1).astype(np.float32)
    df = df.with_columns(pl.Series("als_dot", als_dot, dtype=pl.Float32))

    if labels is not None:
        df = df.join(
            labels.with_columns(pl.lit(1).cast(pl.Int32).alias("label")),
            on=["user_idx", "item_idx"],
            how="left",
        ).with_columns(pl.col("label").fill_null(0).cast(pl.Int32))

    return df


def ndcg(actual, predicted, k=20):
    dcg = 0
    for i, item in enumerate(predicted[:k]):
        if item in actual:
            dcg += 1 / np.log2(i + 2)

    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(actual), k)))
    if idcg == 0:
        return 0.0
    return dcg / idcg


print("Load data")
train = pl.read_parquet(DATA_DIR / "train.pq")
items = pl.read_parquet(DATA_DIR / "items.pq")
test_users = pl.read_csv(DATA_DIR / "test_users.csv")

print(train.shape, items.shape, test_users.shape)

user_col = detect_col(train.schema, ["user_id", "userId", "user", "uid"])
item_col = detect_col(train.schema, ["item_id", "itemId", "item", "iid", "product_id"])

time_col = next(
    (
        c
        for c in train.columns
        if any(
            x in c.lower()
            for x in ["timestamp", "time", "date", "event_time", "datetime"]
        )
    ),
    None,
)
purchase_col = next(
    (c for c in train.columns if "purch" in c.lower() or "bought" in c.lower()), None
)
rating_col = next(
    (c for c in train.columns if "rating" in c.lower() or "score" in c.lower()), None
)
impressions_col = next(
    (c for c in train.columns if "impress" in c.lower() or "slate" in c.lower()), None
)

rename_cols = {user_col: "user_id", item_col: "item_id"}
if time_col:
    rename_cols[time_col] = "timestamp"
if purchase_col:
    rename_cols[purchase_col] = "is_purchased"
if rating_col:
    rename_cols[rating_col] = "rating"
if impressions_col:
    rename_cols[impressions_col] = "impressions"

train = train.rename(rename_cols)
items = items.rename(
    {detect_col(items.schema, ["item_id", "itemId", "item", "iid"]): "item_id"}
)
test_users = test_users.rename(
    {detect_col(test_users.schema, ["user_id", "userId", "user", "uid"]): "user_id"}
)

has_purchase = "is_purchased" in train.columns
has_rating = "rating" in train.columns
has_impressions = "impressions" in train.columns

print(f"purchase={has_purchase}, rating={has_rating}, impressions={has_impressions}")

print("Make weights")
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
        (W_VIEW + pl.col("rating").cast(pl.Float32) * W_RATING_BONUS)
        .cast(pl.Float32)
        .alias("weight")
    )
else:
    train = train.with_columns(pl.lit(1.0, dtype=pl.Float32).alias("weight"))

print(train["weight"].describe())

print("Encode ids")
all_users = train["user_id"].unique().sort()
all_items = train["item_id"].unique().sort()

n_users = len(all_users)
n_items = len(all_items)

user_map = pl.DataFrame(
    {"user_id": all_users, "user_idx": pl.arange(n_users, eager=True, dtype=pl.Int32)}
)
item_map = pl.DataFrame(
    {"item_id": all_items, "item_idx": pl.arange(n_items, eager=True, dtype=pl.Int32)}
)
idx_to_item = all_items.to_numpy()

train = train.join(user_map, on="user_id", how="left").join(
    item_map, on="item_id", how="left"
)
print(f"users={n_users}, items={n_items}")

print("Split train/val")
if "timestamp" in train.columns:
    train = train.with_columns(pl.col("timestamp").cast(pl.Int64).alias("ts_int"))
    split_value = train["ts_int"].quantile(0.9)
    train_part = train.filter(pl.col("ts_int") <= split_value)
    val_part = train.filter(pl.col("ts_int") > split_value)
else:
    split_n = int(len(train) * 0.9)
    shuffled = train.sample(fraction=1.0, shuffle=True, seed=RANDOM_SEED)
    train_part = shuffled[:split_n]
    val_part = shuffled[split_n:]

print(len(train_part), len(val_part))

mat_full = build_sparse(train, n_users, n_items)
mat_train = build_sparse(train_part, n_users, n_items)

print(
    f"matrix nnz={mat_full.nnz}, density={mat_full.nnz / (n_users * n_items) * 100:.4f}%"
)

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import TFIDFRecommender, bm25_weight

print("Train recommenders")
USE_CUDA = detect_cuda()

tfidf_full = train_tfidf(mat_full, "full")
tfidf_train = train_tfidf(mat_train, "train")

als_full = train_als(mat_full, "full")
als_train = train_als(mat_train, "train")

U_full = to_cpu(als_full.user_factors)
V_full = to_cpu(als_full.item_factors)
U_train = to_cpu(als_train.user_factors)
V_train = to_cpu(als_train.item_factors)

print("Make candidates")
item_popularity = np.asarray(mat_full.sum(axis=0)).ravel().astype(np.float32)
pop_top_global = np.argsort(-item_popularity)[:N_CANDIDATES]

known_users = set(all_users.to_list())
test_list = test_users["user_id"].to_list()

warm_df = pl.DataFrame({"user_id": [u for u in test_list if u in known_users]}).join(
    user_map, on="user_id", how="left"
)
warm_users = warm_df["user_idx"].to_list()
warm_idx_to_user_id = dict(
    zip(warm_df["user_idx"].to_list(), warm_df["user_id"].to_list())
)
cold_users = [u for u in test_list if u not in known_users]

print(f"warm={len(warm_users)}, cold={len(cold_users)}")

ids_list = []
scores_list = []
start = time.time()

for i in range(0, len(warm_users), ALS_BATCH):
    ids, scores = recommend(tfidf_full, mat_full, warm_users[i : i + ALS_BATCH])
    ids_list.append(ids)
    scores_list.append(scores)
    print(f"TFIDF {i}/{len(warm_users)}", end="\r")

rec_ids = np.vstack(ids_list)
rec_scores = np.vstack(scores_list)
print(f"\nDone: {time.time() - start:.1f}s")

candidates = add_popular_items(mat_full, warm_users, rec_ids)

print("Build features")
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
        ((max_ts - pl.col("ts_int")) / (3600 * 24 * 1_000_000))
        .min()
        .cast(pl.Float32)
        .alias("user_days_since_last")
    )

user_features = train.group_by("user_idx").agg(user_aggs)

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

item_stats = train.group_by("item_idx").agg(item_aggs)

item_ctr = None
if has_impressions:
    start = time.time()

    shown = (
        train.lazy()
        .select("impressions")
        .explode("impressions")
        .rename({"impressions": "item_id"})
        .group_by("item_id")
        .agg(pl.len().alias("n_shown"))
        .collect(engine="streaming")
    )

    clicks = train.lazy().group_by("item_id").agg(pl.len().alias("n_clicks")).collect()

    item_ctr = (
        clicks.join(shown, on="item_id", how="left")
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

    print(f"CTR: {time.time() - start:.1f}s")

item_catalog = items.join(item_map, on="item_id", how="left").filter(
    pl.col("item_idx").is_not_null()
)
exprs = []

for col, dtype in item_catalog.schema.items():
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
    item_catalog = item_catalog.with_columns(exprs)

catalog_cols = [
    c
    for c in item_catalog.columns
    if c != "item_id" and c != "item_idx" and item_catalog[c].dtype.is_numeric()
]

item_catalog = item_catalog.select(["item_idx"] + catalog_cols).fill_null(0)

item_features = item_stats
if item_ctr is not None:
    item_features = item_features.join(item_ctr, on="item_idx", how="left")
item_features = item_features.join(item_catalog, on="item_idx", how="left").fill_null(0)

print("Train ranker")
USE_LGBM = False
feature_cols = []

try:
    import lightgbm as lgb

    rng = np.random.default_rng(RANDOM_SEED)

    train_users = set(train_part["user_idx"].unique().to_list())
    val_users = [u for u in val_part["user_idx"].unique().to_list() if u in train_users]
    rng.shuffle(val_users)

    val_users = val_users[: min(20000, len(val_users))]
    split = int(len(val_users) * 0.8)

    rank_train_users = val_users[:split]
    rank_val_users = val_users[split:]

    val_ids, val_scores = batch_recommend(tfidf_train, mat_train, rank_val_users)
    train_ids, train_scores = batch_recommend(tfidf_train, mat_train, rank_train_users)

    val_candidates = add_popular_items(mat_train, rank_val_users, val_ids)
    train_candidates = add_popular_items(mat_train, rank_train_users, train_ids)

    val_labels = val_part.select(["user_idx", "item_idx"]).unique()

    val_frame = build_feature_frame(
        rank_val_users,
        val_candidates,
        val_ids,
        val_scores,
        labels=val_labels,
        U=U_train,
        V=V_train,
    )
    train_frame = build_feature_frame(
        rank_train_users,
        train_candidates,
        train_ids,
        train_scores,
        labels=val_labels,
        U=U_train,
        V=V_train,
    )

    ignored_cols = {"user_idx", "item_idx", "label"}
    feature_cols = [
        c
        for c in val_frame.columns
        if c not in ignored_cols and val_frame[c].dtype.is_numeric()
    ]

    train_frame = train_frame.sort("user_idx")
    val_frame = val_frame.sort("user_idx")

    def to_lgb(df):
        X = df.select(feature_cols).to_numpy().astype(np.float32)
        y = df["label"].to_numpy().astype(np.int32)
        group = df.group_by("user_idx", maintain_order=True).len()["len"].to_numpy()
        return X, y, group

    X_train, y_train, group_train = to_lgb(train_frame)
    X_val, y_val, group_val = to_lgb(val_frame)

    del train_frame, val_frame
    gc.collect()

    ds_train = lgb.Dataset(
        X_train, y_train, group=group_train, feature_name=feature_cols
    )
    ds_val = lgb.Dataset(X_val, y_val, group=group_val, reference=ds_train)

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
        ds_train,
        num_boost_round=800,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    USE_LGBM = True

    importance = pl.DataFrame(
        {"feature": feature_cols, "gain": lgb_model.feature_importance("gain").tolist()}
    ).sort("gain", descending=True)

    print("best iter:", lgb_model.best_iteration)
    print("val ndcg@20:", lgb_model.best_score["valid_0"]["ndcg@20"])
    print(importance.head(15))

except Exception:
    import traceback

    traceback.print_exc()
    print("LightGBM failed, use simple blending")

print("Validation")
val_gt = {
    row[0]: set(row[1])
    for row in val_part.group_by("user_idx")
    .agg(pl.col("item_idx").alias("items"))
    .rows()
}

ranker_train_set = set(rank_train_users) if USE_LGBM else set()
check_pool = [u for u in val_gt.keys() if u not in ranker_train_set]

rng = np.random.default_rng(RANDOM_SEED + 1)
check_users = rng.choice(check_pool, min(2000, len(check_pool)), replace=False).tolist()

check_ids, check_scores = batch_recommend(tfidf_train, mat_train, check_users)
check_candidates = add_popular_items(mat_train, check_users, check_ids)

base_scores = [ndcg(val_gt[u], check_candidates[u]) for u in check_users if u in val_gt]
print(f"retrieval ndcg@20: {np.mean(base_scores):.4f}")

if USE_LGBM:
    check_frame = build_feature_frame(
        check_users,
        check_candidates,
        check_ids,
        check_scores,
        U=U_train,
        V=V_train,
    )

    X_check = check_frame.select(feature_cols).to_numpy().astype(np.float32)
    check_frame = check_frame.with_columns(
        pl.Series("lgbm_score", lgb_model.predict(X_check).astype(np.float32))
    )

    scores = []
    for uid in check_users:
        if uid not in val_gt:
            continue

        pred = (
            check_frame.filter(pl.col("user_idx") == uid)
            .sort("lgbm_score", descending=True)["item_idx"]
            .to_list()
        )
        scores.append(ndcg(val_gt[uid], pred))

    print(f"rerank ndcg@20: {np.mean(scores):.4f}")

    del check_frame
    gc.collect()

print("Score test")
top_users = []
top_items = []
start = time.time()

for batch_start in range(0, len(warm_users), SCORE_BATCH):
    batch_end = min(batch_start + SCORE_BATCH, len(warm_users))

    batch_users = warm_users[batch_start:batch_end]
    batch_ids = rec_ids[batch_start:batch_end]
    batch_scores = rec_scores[batch_start:batch_end]
    batch_candidates = {u: candidates[u] for u in batch_users}

    if USE_LGBM:
        batch_frame = build_feature_frame(
            batch_users, batch_candidates, batch_ids, batch_scores
        )
        X_batch = batch_frame.select(feature_cols).to_numpy().astype(np.float32)
        batch_frame = batch_frame.with_columns(
            pl.Series("lgbm_score", lgb_model.predict(X_batch).astype(np.float32))
        )

        top_batch = (
            batch_frame.sort("lgbm_score", descending=True)
            .group_by("user_idx", maintain_order=False)
            .head(TOP_K)
        )

        user_np = top_batch["user_idx"].to_numpy()
        item_np = top_batch["item_idx"].to_numpy()

        del batch_frame, top_batch, X_batch
        gc.collect()

    else:
        user_list = []
        item_list = []

        for j, uid in enumerate(batch_users):
            cur_candidates = batch_candidates[uid]
            als_score = batch_scores[j, : len(cur_candidates)].copy()

            if len(als_score) < len(cur_candidates):
                als_score = np.pad(als_score, (0, len(cur_candidates) - len(als_score)))

            pop_score = item_popularity[cur_candidates]

            def norm(x):
                if x.max() > x.min():
                    return (x - x.min()) / (x.max() - x.min() + 1e-9)
                return x

            final_score = 0.8 * norm(als_score) + 0.2 * norm(pop_score)
            order = np.argsort(-final_score)[:TOP_K]

            user_list.extend([uid] * TOP_K)
            item_list.extend(np.array(cur_candidates)[order])

        user_np = np.array(user_list, dtype=np.int32)
        item_np = np.array(item_list, dtype=np.int32)

    top_users.extend([warm_idx_to_user_id[int(u)] for u in user_np])
    top_items.extend(idx_to_item[item_np].tolist())

    elapsed = time.time() - start
    print(f"{batch_end}/{len(warm_users)} users, {elapsed:.0f}s", end="\r")

print()

cold_top = idx_to_item[pop_top_global[:TOP_K]].tolist()
for uid in cold_users:
    top_users.extend([uid] * TOP_K)
    top_items.extend(cold_top)

submission = pl.DataFrame(
    {
        "user_id": pl.Series(top_users),
        "item_id": pl.Series(top_items),
    }
)

counts = submission.group_by("user_id").len()["len"]
print(
    f"rows per user: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}"
)
print(f"total rows: {len(submission)}, expected: {test_users.shape[0] * TOP_K}")

out_path = OUT_DIR / "submission.csv"
submission.write_csv(out_path)

print(out_path)
print(submission.head(22))
