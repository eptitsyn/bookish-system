"""Honest harness: ALL features past-only (df_tr). Tests metadata-overlap features
and candidate strategies. Train ranker on val-subset, eval on held-out val users."""
import time, sys, numpy as np, polars as pl, scipy.sparse as sp
from implicit.nearest_neighbours import TFIDFRecommender
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
import lightgbm as lgb

SEED = 42; TOPK = 20; NCAND = 200
np.random.seed(SEED)
def log(*a): print(*a, flush=True)

t0 = time.time()
train = pl.read_parquet("data/train.pq")
items = pl.read_parquet("data/items.pq")
train = train.with_columns(
    pl.when(pl.col("is_purchased") & (pl.col("rating") > 0)).then(5.0 + pl.col("rating").cast(pl.Float32))
    .when(pl.col("is_purchased")).then(pl.lit(5.0))
    .when(pl.col("rating") > 0).then(1.0 + 0.5 * pl.col("rating").cast(pl.Float32))
    .otherwise(pl.lit(1.0)).cast(pl.Float32).alias("weight"))
users = train["user_id"].unique().sort(); items_s = train["item_id"].unique().sort()
nu, ni = len(users), len(items_s)
umap = pl.DataFrame({"user_id": users, "u": pl.arange(nu, eager=True, dtype=pl.Int32)})
imap = pl.DataFrame({"item_id": items_s, "i": pl.arange(ni, eager=True, dtype=pl.Int32)})
train = train.join(umap, on="user_id").join(imap, on="item_id")
train = train.with_columns(pl.col("timestamp").cast(pl.Int64).alias("ts"))
split = train["ts"].quantile(0.9)
tr = train.filter(pl.col("ts") <= split); val = train.filter(pl.col("ts") > split)
log(f"nu={nu} ni={ni} tr={len(tr)} val={len(val)}  {time.time()-t0:.0f}s")

def csr(df):
    r = df["u"].to_numpy().astype(np.int32); c = df["i"].to_numpy().astype(np.int32)
    w = df["weight"].to_numpy().astype(np.float32)
    m = sp.csr_matrix((w, (r, c)), shape=(nu, ni)); m.sum_duplicates(); return m
mat_tr = csr(tr)

tr_users = set(tr["u"].unique().to_list())
val_gt = {}
for u, it in val.group_by("u").agg(pl.col("i")).rows():
    if u in tr_users: val_gt[u] = set(it)
allv = list(val_gt.keys()); np.random.shuffle(allv)
N_RANK_TR, N_EVAL = 16000, 4000
rank_tr_u = allv[:N_RANK_TR]; eval_u = allv[N_RANK_TR:N_RANK_TR+N_EVAL]
log(f"val warm={len(val_gt)} rank_tr={len(rank_tr_u)} eval={len(eval_u)}")

def ndcg(actual, predicted, k=TOPK):
    dcg = sum(1/np.log2(i+2) for i,p in enumerate(predicted[:k]) if p in actual)
    idcg = sum(1/np.log2(i+2) for i in range(min(len(actual),k)))
    return dcg/idcg if idcg>0 else 0.0

# ── retriever + ALS (past only) ──
t=time.time()
tfidf = TFIDFRecommender(K=500); tfidf.fit(mat_tr, show_progress=False)
als = AlternatingLeastSquares(factors=64, iterations=15, regularization=0.01, use_gpu=False, random_state=SEED)
als.fit(bm25_weight(mat_tr, K1=1.5, B=0.75), show_progress=False)
U, V = als.user_factors, als.item_factors
log(f"models {time.time()-t:.0f}s")

# ── past-only stats ──
pop = np.asarray(mat_tr.sum(0)).ravel().astype(np.float32)
it_stats = tr.group_by("i").agg(
    pl.len().alias("it_n"), pl.col("is_purchased").mean().cast(pl.Float32).alias("it_prate"),
    pl.col("rating").filter(pl.col("rating")>0).mean().fill_null(0).cast(pl.Float32).alias("it_arat"),
).to_pandas().set_index("i").reindex(range(ni)).fillna(0).to_numpy().astype(np.float32)
u_stats = tr.group_by("u").agg(
    pl.len().alias("u_n"), pl.col("is_purchased").mean().cast(pl.Float32).alias("u_prate"),
    pl.col("rating").filter(pl.col("rating")>0).mean().fill_null(0).cast(pl.Float32).alias("u_arat"),
).to_pandas().set_index("u").reindex(range(nu)).fillna(0).to_numpy().astype(np.float32)
# CTR from impressions (df_tr)
n_shown = tr.lazy().select("impressions").explode("impressions").rename({"impressions":"item_id"}).group_by("item_id").agg(pl.len().alias("ns")).collect()
n_click = tr.lazy().group_by("item_id").agg(pl.len().alias("nc")).collect()
ctr = (n_click.join(n_shown,on="item_id",how="left").join(imap,on="item_id",how="left")
       .filter(pl.col("i").is_not_null()).with_columns((pl.col("nc")/(pl.col("ns").fill_null(0)+1.0)).alias("ctr"))
       .select(["i","ctr"]).to_pandas().set_index("i").reindex(range(ni)).fillna(0)["ctr"].to_numpy().astype(np.float32))

# ── item metadata: series/author/first-category arrays (object dtype) ──
im = items.join(imap, on="item_id", how="inner").sort("i")
series_of = [set() for _ in range(ni)]; author_of = [set() for _ in range(ni)]
for i, sids, aids in im.select(["i","series_id","author_ids"]).iter_rows():
    series_of[i] = set(sids) if sids else set()
    author_of[i] = set(aids) if aids else set()
# user history series/author sets (from tr)
u_hist = {}  # u -> (series_set, author_set, n_items)
hist_df = tr.group_by("u").agg(pl.col("i")).rows()
need_u = set(rank_tr_u) | set(eval_u)
for u, hi in hist_df:
    if u not in need_u: continue
    ss=set(); aset=set()
    for it in hi:
        ss |= series_of[it]; aset |= author_of[it]
    u_hist[u] = (ss, aset)
log(f"features ready {time.time()-t0:.0f}s")

def candidates(uids):
    uarr=np.array(uids,np.int32)
    ids,sc = tfidf.recommend(uarr, mat_tr[uarr], N=NCAND, filter_already_liked_items=True)
    return ids, sc

FEATS = ["rank","tfidf","als_dot","pop","it_n","it_prate","it_arat","ctr",
         "u_n","u_prate","u_arat","ser_ov","auth_ov","ser_any","auth_any"]
def build(uids, ids, sc):
    F=[]; Y=[]; G=[]; C=[]
    for k,u in enumerate(uids):
        cand=ids[k]; cs=sc[k]; n=len(cand); gt=val_gt[u]
        dot=(U[u]*V[cand]).sum(1)
        sshist, ahist = u_hist.get(u,(set(),set()))
        ser_ov=np.array([len(series_of[c]&sshist) for c in cand],np.float32)
        auth_ov=np.array([len(author_of[c]&ahist) for c in cand],np.float32)
        f=np.column_stack([np.arange(n),cs,dot,pop[cand],it_stats[cand,0],it_stats[cand,1],it_stats[cand,2],
            ctr[cand],np.repeat(u_stats[u][None],n,0),ser_ov,auth_ov,(ser_ov>0).astype(np.float32),(auth_ov>0).astype(np.float32)]).astype(np.float32)
        F.append(f); Y.append(np.array([1 if int(c) in gt else 0 for c in cand],np.int32)); G.append(n); C.append(cand)
    return np.vstack(F), np.concatenate(Y), np.array(G), C

t=time.time()
ids_tr,sc_tr = candidates(rank_tr_u); ids_ev,sc_ev = candidates(eval_u)
Xtr,ytr,gtr,_ = build(rank_tr_u, ids_tr, sc_tr)
Xev,yev,gev,Cev = build(eval_u, ids_ev, sc_ev)
log(f"built {Xtr.shape} {time.time()-t:.0f}s")

def run(cols):
    idx=[FEATS.index(c) for c in cols]
    ds=lgb.Dataset(Xtr[:,idx],ytr,group=gtr); dv=lgb.Dataset(Xev[:,idx],yev,group=gev,reference=ds)
    m=lgb.train(dict(objective="lambdarank",metric="ndcg",ndcg_eval_at=[20],learning_rate=0.05,
        num_leaves=63,min_data_in_leaf=50,verbose=-1,random_state=SEED),ds,num_boost_round=500,
        valid_sets=[dv],callbacks=[lgb.early_stopping(40,verbose=False)])
    pred=m.predict(Xev[:,idx]); off=0; nd=[]
    for k,u in enumerate(eval_u):
        n=gev[k]; order=np.argsort(-pred[off:off+n])[:TOPK]; off+=n
        nd.append(ndcg(val_gt[u], Cev[k][order].tolist()))
    return float(np.mean(nd)), m

# retrieval baseline
raw=[ndcg(val_gt[u], ids_ev[k][:TOPK].tolist()) for k,u in enumerate(eval_u)]
log(f"RETRIEVAL TFIDF: {np.mean(raw):.4f}")
base_cols=["rank","tfidf","als_dot","pop","it_n","it_prate","it_arat","ctr","u_n","u_prate","u_arat"]
nd0,_=run(base_cols); log(f"RERANK base ({len(base_cols)}f): {nd0:.4f}")
nd1,m1=run(FEATS); log(f"RERANK +metadata ({len(FEATS)}f): {nd1:.4f}")
imp=pl.DataFrame({"f":FEATS,"g":m1.feature_importance('gain').tolist()}).sort("g",descending=True)
log(imp)
