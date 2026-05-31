"""Push experiment: candidate-union recall + richer reranker (multi-retriever
scores + per-user impression features). Honest past-only setup."""
import time, sys, numpy as np, polars as pl, scipy.sparse as sp
from implicit.nearest_neighbours import TFIDFRecommender, CosineRecommender
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
import lightgbm as lgb

SEED=42; TOPK=20
np.random.seed(SEED)
def log(*a): print(*a, flush=True)

t0=time.time()
train=pl.read_parquet("data/train.pq")
train=train.with_columns(
    pl.when(pl.col("is_purchased") & (pl.col("rating")>0)).then(5.0+pl.col("rating").cast(pl.Float32))
    .when(pl.col("is_purchased")).then(pl.lit(5.0))
    .when(pl.col("rating")>0).then(1.0+0.5*pl.col("rating").cast(pl.Float32))
    .otherwise(pl.lit(1.0)).cast(pl.Float32).alias("weight"))
users=train["user_id"].unique().sort(); items_s=train["item_id"].unique().sort()
nu,ni=len(users),len(items_s)
umap=pl.DataFrame({"user_id":users,"u":pl.arange(nu,eager=True,dtype=pl.Int32)})
imap=pl.DataFrame({"item_id":items_s,"i":pl.arange(ni,eager=True,dtype=pl.Int32)})
train=train.join(umap,on="user_id").join(imap,on="item_id").with_columns(pl.col("timestamp").cast(pl.Int64).alias("ts"))
split=train["ts"].quantile(0.9)
tr=train.filter(pl.col("ts")<=split); val=train.filter(pl.col("ts")>split)
log(f"nu={nu} ni={ni} {time.time()-t0:.0f}s")

def csr(df):
    r=df["u"].to_numpy().astype(np.int32);c=df["i"].to_numpy().astype(np.int32);w=df["weight"].to_numpy().astype(np.float32)
    m=sp.csr_matrix((w,(r,c)),shape=(nu,ni));m.sum_duplicates();return m
mat_tr=csr(tr)

tr_users=set(tr["u"].unique().to_list())
val_gt={}
for u,it in val.group_by("u").agg(pl.col("i")).rows():
    if u in tr_users: val_gt[u]=set(it)
allv=list(val_gt.keys()); np.random.shuffle(allv)
rank_tr_u=allv[:16000]; eval_u=allv[16000:20000]
log(f"val warm={len(val_gt)} rank_tr={len(rank_tr_u)} eval={len(eval_u)}")

def ndcg(a,p,k=TOPK):
    dcg=sum(1/np.log2(i+2) for i,x in enumerate(p[:k]) if x in a)
    idcg=sum(1/np.log2(i+2) for i in range(min(len(a),k)))
    return dcg/idcg if idcg>0 else 0.0

# retrievers
t=time.time()
tfidf=TFIDFRecommender(K=500); tfidf.fit(mat_tr,show_progress=False)
cos=CosineRecommender(K=500); cos.fit(mat_tr,show_progress=False)
als=AlternatingLeastSquares(factors=128,iterations=20,regularization=0.01,use_gpu=False,random_state=SEED)
als.fit(bm25_weight(mat_tr,K1=1.5,B=0.75),show_progress=False)
U,V=als.user_factors,als.item_factors
log(f"models {time.time()-t:.0f}s")

def rec(model,uids,n):
    ua=np.array(uids,np.int32)
    ids,sc=model.recommend(ua,mat_tr[ua],N=n,filter_already_liked_items=True)
    return ids,sc

# ---- recall of unions ----
def dedup_topn(arr_list, n):
    out=[]; seen=set()
    for a in arr_list:
        for x in a:
            x=int(x)
            if 0<=x<ni and x not in seen:
                out.append(x); seen.add(x)
                if len(out)>=n: return out
    return out

def recall_union(uids, models_n, k):
    per=[rec(m,uids,nn)[0] for m,nn in models_n]
    s=[]
    for j,u in enumerate(uids):
        cand=dedup_topn([per[mi][j] for mi in range(len(per))], k)
        s.append(len(set(cand)&val_gt[u])/len(val_gt[u]))
    return float(np.mean(s))

ev=eval_u
log("--- recall@K ceilings (eval) ---")
for name, mn in [("TFIDF200",[(tfidf,200)]),("TFIDF500",[(tfidf,500)]),
                 ("TFIDF+ALS",[(tfidf,150),(als,150)]),
                 ("TFIDF+Cos",[(tfidf,150),(cos,150)]),
                 ("TFIDF+ALS+Cos",[(tfidf,120),(als,120),(cos,120)])]:
    for k in (200,300):
        log(f"  {name} recall@{k}: {recall_union(ev,mn,k):.4f}")

# ---- per-user impression history counts (from tr) ----
t=time.time()
uimpr={}  # u -> dict(item_idx->count)
imap_d=dict(zip(items_s.to_list(), range(ni)))
need=set(rank_tr_u)|set(eval_u)
sub=tr.filter(pl.col("u").is_in(list(need))).select(["u","impressions"])
for u,impr in sub.group_by("u").agg(pl.col("impressions").flatten()).rows():
    d={}
    for it in impr:
        ii=imap_d.get(it)
        if ii is not None: d[ii]=d.get(ii,0)+1
    uimpr[u]=d
log(f"user impr hist {time.time()-t:.0f}s")

# ---- build candidates (TFIDF+ALS union, 200) with multi-retriever feats ----
pop=np.asarray(mat_tr.sum(0)).ravel().astype(np.float32)
it_stats=tr.group_by("i").agg(pl.len().alias("n"),pl.col("is_purchased").mean().cast(pl.Float32).alias("pr"),
    pl.col("rating").filter(pl.col("rating")>0).mean().fill_null(0).cast(pl.Float32).alias("ar")
    ).to_pandas().set_index("i").reindex(range(ni)).fillna(0).to_numpy().astype(np.float32)
u_stats=tr.group_by("u").agg(pl.len().alias("n"),pl.col("is_purchased").mean().cast(pl.Float32).alias("pr"),
    pl.col("rating").filter(pl.col("rating")>0).mean().fill_null(0).cast(pl.Float32).alias("ar")
    ).to_pandas().set_index("u").reindex(range(nu)).fillna(0).to_numpy().astype(np.float32)
ns=tr.lazy().select("impressions").explode("impressions").rename({"impressions":"item_id"}).group_by("item_id").agg(pl.len().alias("ns")).collect(engine="streaming")
nc=tr.lazy().group_by("item_id").agg(pl.len().alias("nc")).collect()
ctr=(nc.join(ns,on="item_id",how="left").join(imap,on="item_id",how="left").filter(pl.col("i").is_not_null())
     .with_columns((pl.col("nc")/(pl.col("ns").fill_null(0)+1.0)).alias("c")).select(["i","c"])
     .to_pandas().set_index("i").reindex(range(ni)).fillna(0)["c"].to_numpy().astype(np.float32))

NC=200
FEATS=["t_rank","t_sc","a_rank","a_sc","als_dot","pop","it_n","it_pr","it_ar","ctr",
       "u_n","u_pr","u_ar","impr_cnt","impr_any","n_ret"]
def build(uids):
    ti,tsc=rec(tfidf,uids,NC); ai,asc=rec(als,uids,NC)
    F=[];Y=[];G=[];C=[]
    for j,u in enumerate(uids):
        tmap={int(x):(r,s) for r,(x,s) in enumerate(zip(ti[j],tsc[j])) if 0<=int(x)<ni}
        amap={int(x):(r,s) for r,(x,s) in enumerate(zip(ai[j],asc[j])) if 0<=int(x)<ni}
        cand=dedup_topn([ti[j],ai[j]],NC)
        n=len(cand); gt=val_gt[u]; impr=uimpr.get(u,{})
        rows=[]
        for c in cand:
            tr_,ts_=tmap.get(c,(NC,0.0)); ar_,as_=amap.get(c,(NC,0.0))
            dot=float(U[u]@V[c]); ic=impr.get(c,0)
            nret=(c in tmap)+(c in amap)
            rows.append([tr_,ts_,ar_,as_,dot,pop[c],it_stats[c,0],it_stats[c,1],it_stats[c,2],ctr[c],
                         u_stats[u,0],u_stats[u,1],u_stats[u,2],ic,float(ic>0),nret])
        F.append(np.array(rows,np.float32)); Y.append(np.array([1 if c in gt else 0 for c in cand],np.int32))
        G.append(n); C.append(cand)
    return np.vstack(F),np.concatenate(Y),np.array(G),C

t=time.time()
Xtr,ytr,gtr,_=build(rank_tr_u); Xev,yev,gev,Cev=build(eval_u)
log(f"built {Xtr.shape} {time.time()-t:.0f}s")

def run(cols,label):
    idx=[FEATS.index(c) for c in cols]
    ds=lgb.Dataset(Xtr[:,idx],ytr,group=gtr); dv=lgb.Dataset(Xev[:,idx],yev,group=gev,reference=ds)
    m=lgb.train(dict(objective="lambdarank",metric="ndcg",ndcg_eval_at=[20],learning_rate=0.05,
        num_leaves=127,min_data_in_leaf=50,verbose=-1,random_state=SEED),ds,num_boost_round=600,
        valid_sets=[dv],callbacks=[lgb.early_stopping(50,verbose=False)])
    pred=m.predict(Xev[:,idx]);off=0;nd=[]
    for k,u in enumerate(eval_u):
        n=gev[k];order=np.argsort(-pred[off:off+n])[:TOPK];off+=n
        nd.append(ndcg(val_gt[u],Cev[k][order].tolist()))
    log(f"{label}: {np.mean(nd):.4f}  (best_iter {m.best_iteration})")
    return m

base=["t_rank","t_sc","als_dot","pop","it_n","it_pr","it_ar","ctr","u_n","u_pr","u_ar"]
run(base,"base TFIDF-only feats")
run(base+["a_rank","a_sc","n_ret"],"+ALS-union feats")
m=run(FEATS,"+impr feats (ALL)")
imp=pl.DataFrame({"f":FEATS,"g":m.feature_importance('gain').tolist()}).sort("g",descending=True)
log(imp)
