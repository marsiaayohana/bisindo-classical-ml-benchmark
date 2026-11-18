# Supplementary (not part of the primary protocol). Reads only saved outputs in A_outputs/.
import pandas as pd, numpy as np, json, sys
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
D=sys.argv[1]; OUT=sys.argv[2]
SIG=["S1","S2","S3","S4","S5","S6","S7"]; CL=["Apa Kabar","Halo","Maaf","Sama-sama","Terima kasih","Tolong"]
names={}
for pos in range(0,6300,30):
    v=(pos%1050)//30; names[f"{SIG[v//5]}_{CL[pos//1050].replace(' ','')}{v%5+1}"]=pos//30
rep=[]
# ---- Part 1: McNemar pairing check
g=pd.read_csv(D+"A_grouped_cv_oof_predictions.csv"); dl=pd.read_csv(D+"A_dl_cv_oof_predictions.csv")
g=g[g.features=="frame285"].copy(); g["blk"]=g.old_row//30
dl["blk"]=dl.video.map(names)
rep.append(f"DL video names unmapped: {int(dl.blk.isna().sum())}")
vote={}; truth={}; tied={}
for m,h in g.groupby("model"):
    pred={}; tie={}
    for b,x in h.groupby("blk"):
        c=np.bincount(x.y_pred,minlength=6); pred[b]=int(c.argmax()); tie[b]=int((c==c.max()).sum()>1); truth.setdefault(b,int(x.y_true.iloc[0]))
        assert x.y_true.nunique()==1 and x.fold.nunique()==1
    vote[m]=pd.Series(pred); tied[m]=pd.Series(tie)
T=pd.Series(truth).sort_index(); rep.append(f"classical blocks: {len(T)}; each block predicted once per model: {all(len(v)==210 for v in vote.values())}")
rows=[]
for dm,h in dl.groupby("model"):
    h=h.set_index("blk").sort_index(); 
    assert h.index.is_unique and len(h)==210 and (h.index==T.index).all()
    assert (h.y_true.values==T.values).all(), "label mismatch between DL and classical for same video"
    dok=(h.y_pred.values==h.y_true.values)
    for cm,p in vote.items():
        cok=(p.sort_index().values==T.values); b=int((dok&~cok).sum()); c=int((~dok&cok).sum())
        # sensitivity: classical ties counted all right / all wrong
        t=tied[cm].sort_index().values
        out=dict(dl=dm,classical=cm,dl_acc=dok.mean()*100,cls_acc=cok.mean()*100,dl_only_right=b,cls_only_right=c,both_right=int((dok&cok).sum()),both_wrong=int((~dok&~cok).sum()),p_exact=binomtest(b,b+c,.5).pvalue if b+c else 1.0,classical_tied_videos=int(t.sum()))
        for tag,fix in(("ties_right",lambda ok:ok|t.astype(bool)),("ties_wrong",lambda ok:ok&~t.astype(bool))):
            ck=fix(cok); bb=int((dok&~ck).sum()); cc=int((~dok&ck).sum()); out["p_"+tag]=binomtest(bb,bb+cc,.5).pvalue if bb+cc else 1.0
        rows.append(out)
P=pd.DataFrame(rows); P.to_csv(OUT+"A_supp_mcnemar_pairing_check.csv",index=False)
rep.append("McNemar rows saved; max p across tie sensitivities per row:"); rep.append(P[["dl","classical","p_exact","p_ties_right","p_ties_wrong"]].round(4).to_string(index=False))
# fold structure note
dlf=dl.groupby("model").apply(lambda h:h.groupby("fold").blk.apply(set))
gf=g[g.model=="RF"].groupby("fold").blk.apply(set)
same=all(any(s==t for t in gf) for s in dlf.loc["1D-CNN"])
rep.append(f"DL folds identical to classical grouped folds (as video sets): {same}  -> training partitions differ; pairing is by video, OOF in both")
# ---- Part 2: length-only shortcut check (supplementary; does not replace any primary result)
L=pd.read_csv(D+"A_dl_sequence_lengths.csv"); L["blk"]=L.video.map(names); assert L.blk.notna().all() and L.blk.is_unique and len(L)==210
L["y"]=L.label.map({c:i for i,c in enumerate(CL)}); L=L.set_index("blk").loc[T.index]; assert (L.y.values==T.values).all()
fold_of=dl[dl.model=="1D-CNN"].set_index("blk").fold.reindex(T.index).values   # same 5 folds as the DL cross-validation
X=L[["n_real_frames"]].values.astype(float); y=L.y.values
mk={"majority class (reference)":None,
    "DecisionTree(max_depth=3)":lambda:DecisionTreeClassifier(max_depth=3,random_state=42),
    "RandomForest(100)":lambda:RandomForestClassifier(100,random_state=42),
    "KNN(5)":lambda:KNeighborsClassifier(5),
    "Logistic regression":lambda:LogisticRegression(max_iter=1000)}
res=[]
for n,f in mk.items():
    accs=[]
    for k in sorted(set(fold_of)):
        te=fold_of==k; tr=~te
        if f is None: p=np.full(te.sum(),np.bincount(y[tr]).argmax())
        else: m=f().fit(X[tr],y[tr]); p=m.predict(X[te])
        accs.append((p==y[te]).mean()*100)
    res.append(dict(model=n,mean=np.mean(accs),sd_ddof0=np.std(accs),folds=";".join(f"{a:.2f}" for a in accs)))
R=pd.DataFrame(res); R.insert(0,"eval_unit","video level | DL stratified folds | feature = n_real_frames only"); R.to_csv(OUT+"A_supp_length_only_check.csv",index=False)
rep.append("LENGTH-ONLY (chance = 6 balanced classes -> 16.67%):"); rep.append(R.round(2).to_string(index=False))
# descriptive: length by class and signer
rep.append("n_real_frames by class: "+str(L.groupby("label").n_real_frames.agg(["mean","std","min","max"]).round(2).to_dict("index")))
rep.append("n_real_frames by signer: "+str(L.groupby("signer").n_real_frames.agg(["mean","std","min","max"]).round(2).to_dict("index")))
# signer-only reference (is length mostly a signer proxy?)
sg=L.signer.astype("category").cat.codes.values.reshape(-1,1)
open(OUT+"A_supp_report.txt","w").write("\n".join(rep)); print("\n".join(rep))
