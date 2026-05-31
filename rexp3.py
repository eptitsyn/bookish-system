"""Impressions as candidate SOURCE + feature. Honest past-only."""
import time, numpy as np, polars as pl, scipy.sparse as sp
from implicit.nearest_neighbours import TFIDFRecommender
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
import lightgbm as lgb
SEED=42; TOPK=20; np.random.seed(SEED)
def log(*a): print(*a, flush=True)
t0=time.time()
train=pl.read_parquet("data/train.pq")
train=train.with_columns(
    pl.when(pl.col("is_purchased")&(pl.col("rating")>0)).then(5.0+pl.col("rating").cast(pl.Float32))
    .when(pl.col("is_purchased")).then(pl.lit(5.0)).when(pl.col("rating")>0).then(1.0+0.5*pl.col("rating").cast(pl.Float32))
    .otherwise(pl.lit(1.0)).cast(pl.Float32).alias("weight"))
users=train["user_id"].unique().sort(); items_s=train["item_id"].unique().sort()
nu,ni=len(users),len(items_s)
umap=pl.DataFrame({"user_id":users,"u":pl.arange(nu,eager=True,dtype=pl.Int32)})
imap=pl.DataFrame({"item_id":items_s,"i":pl.arange(ni,eager=True,dtype=pl.Int32)})
train=train.join(umap,on="user_id").join(imap,on="item_id").with_columns(pl.col("timestamp").cast(pl.Int64).alias("ts"))
split=train["ts"].quantile(0.9); tr=train.filter(pl.col("ts")<=split); val=train.filter(pl.col("ts")>split)
def csr(df):
    r=df["u"].to_numpy().astype(np.int32);c=df["i"].to_numpy().astype(np.int32);w=df["weight"].to_numpy().astype(np.float32)
    m=sp.csr_matrix((w,(r,c)),shape=(nu,ni));m.sum_duplicates();return m
mat_tr=csr(tr)
tr_users=set(tr["u"].unique().to_list())
val_gt={u:set(it) for u,it in val.group_by("u").agg(pl.col("i")).rows() if u in tr_users}
allv=list(val_gt.keys()); np.random.shuffle(allv)
rank_tr_u=allv[:16000]; eval_u=allv[16000:20000]
log(f"setup {time.time()-t0:.0f}s warm={len(val_gt)}")
def ndcg(a,p,k=TOPK):
    dcg=sum(1/np.log2(i+2) for i,x in enumerate(p[:k]) if x in a); idcg=sum(1/np.log2(i+2) for i in range(min(len(a),k)))
    return dcg/idcg if idcg>0 else 0.0
t=time.time()
tfidf=TFIDFRecommender(K=500); tfidf.fit(mat_tr,show_progress=False)
als=AlternatingLeastSquares(factors=64,iterations=15,regularization=0.01,use_gpu=False,random_state=SEED)
als.fit(bm25_weight(mat_tr,K1=1.5,B=0.75),show_progress=False); U,V=als.user_factors,als.item_factors
log(f"models {time.time()-t:.0f}s")
# per-user impression counts (tr only)
imap_d=dict(zip(items_s.to_list(),range(ni)))
need=set(rank_tr_u)|set(eval_u)
uimpr={}
for u,impr in tr.filter(pl.col("u").is_in(list(need))).group_by("u").agg(pl.col("impressions").flatten()).rows():
    d={}
    for it in impr:
        ii=imap_d.get(it)
        if ii is not None: d[ii]=d.get(ii,0)+1
    uimpr[u]=d
log(f"impr hist {time.time()-t0:.0f}s")
pop=np.asarray(mat_tr.sum(0)).ravel().astype(np.float32)
it_n=tr.group_by("i").agg(pl.len().alias("n")).to_pandas().set_index("i").reindex(range(ni)).fillna(0)["n"].to_numpy().astype(np.float32)
u_n=tr.group_by("u").agg(pl.len().alias("n")).to_pandas().set_index("u").reindex(range(nu)).fillna(0)["n"].to_numpy().astype(np.float32)
ns=tr.lazy().select("impressions").explode("impressions").rename({"impressions":"item_id"}).group_by("item_id").agg(pl.len().alias("ns")).collect(engine="streaming")
nc=tr.lazy().group_by("item_id").agg(pl.len().alias("nc")).collect()
ctr=(nc.join(ns,on="item_id",how="left").join(imap,on="item_id",how="left").filter(pl.col("i").is_not_null())
     .with_columns((pl.col("nc")/(pl.col("ns").fill_null(0)+1.0)).alias("c")).select(["i","c"])
     .to_pandas().set_index("i").reindex(range(ni)).fillna(0)["c"].to_numpy().astype(np.float32))

N_TF=150; N_IMP=100; NC=200
def rec(model,uids,n):
    ua=np.array(uids,np.int32); ids,sc=model.recommend(ua,mat_tr[ua],N=n,filter_already_liked_items=True); return ids,sc
def build(uids, add_impr_cand):
    ti,tsc=rec(tfidf,uids,N_TF)
    F=[];Y=[];G=[];C=[]; rec_hit=[]
    for j,u in enumerate(uids):
        tmap={int(x):(r,s) for r,(x,s) in enumerate(zip(ti[j],tsc[j])) if 0<=int(x)<ni}
        seen=set(mat_tr.indices[mat_tr.indptr[u]:mat_tr.indptr[u+1]].tolist())
        impr=uimpr.get(u,{})
        cand=[];cs=set()
        for x in ti[j]:
            x=int(x)
            if 0<=x<ni and x not in cs: cand.append(x);cs.add(x)
        if add_impr_cand:
            for x,_ in sorted(impr.items(),key=lambda kv:-kv[1]):
                if len(cand)>=NC: break
                if x not in cs and x not in seen: cand.append(x);cs.add(x)
        # pad pop
        gt=val_gt[u]
        rows=[]
        for c in cand:
            tr_,ts_=tmap.get(c,(N_TF,0.0)); dot=float(U[u]@V[c]); ic=impr.get(c,0)
            rows.append([tr_,ts_,dot,pop[c],it_n[c],ctr[c],u_n[u],ic,float(c in tmap)])
        F.append(np.array(rows,np.float32)); Y.append(np.array([1 if c in gt else 0 for c in cand],np.int32))
        G.append(len(cand)); C.append(np.array(cand,np.int32))
        rec_hit.append(len(set(cand)&gt)/len(gt))
    return np.vstack(F),np.concatenate(Y),np.array(G),C,float(np.mean(rec_hit))
FE=["t_rank","t_sc","als_dot","pop","it_n","ctr","u_n","impr_cnt","from_tf"]
def go(add):
    Xtr,ytr,gtr,_,_=build(rank_tr_u,add); Xev,yev,gev,Cev,rcl=build(eval_u,add)
    ds=lgb.Dataset(Xtr,ytr,group=gtr); dv=lgb.Dataset(Xev,yev,group=gev,reference=ds)
    m=lgb.train(dict(objective="lambdarank",metric="ndcg",ndcg_eval_at=[20],learning_rate=0.05,num_leaves=127,
        min_data_in_leaf=50,verbose=-1,random_state=SEED),ds,num_boost_round=600,valid_sets=[dv],
        callbacks=[lgb.early_stopping(50,verbose=False)])
    pred=m.predict(Xev);off=0;nd=[]
    for k,u in enumerate(eval_u):
        n=gev[k];order=np.argsort(-pred[off:off+n])[:TOPK];off+=n; nd.append(ndcg(val_gt[u],Cev[k][order].tolist()))
    log(f"add_impr_cand={add}: recall@200={rcl:.4f}  RERANK NDCG={np.mean(nd):.4f}  iter={m.best_iteration}")
    return m
go(False)
m=go(True)
log(pl.DataFrame({"f":FE,"g":m.feature_importance('gain').tolist()}).sort("g",descending=True))
