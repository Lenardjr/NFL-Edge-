"""Walk-forward market residual model; exploratory, not a trading strategy."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,log_loss

def run(df):
    required=['season','gameday','home_win','model_home_prob','market_home_prob']
    df=df.dropna(subset=required).copy().sort_values('gameday')
    for c in ['model_home_prob','market_home_prob']:
        if not df[c].between(0,1).all():raise ValueError('invalid probabilities')
    if not df.home_win.isin([0,1]).all():raise ValueError('invalid labels')
    eps=1e-6
    def logit(x):return np.log(np.clip(x,eps,1-eps)/np.clip(1-x,eps,1-eps))
    X=np.column_stack([logit(df.market_home_prob.to_numpy()),logit(df.model_home_prob.to_numpy())-logit(df.market_home_prob.to_numpy())])
    y=df.home_win.to_numpy();seasons=df.season.to_numpy()
    rows=[];preds=[]
    for s in sorted(df.season.unique()):
        train=np.flatnonzero(seasons<s);test=np.flatnonzero(seasons==s)
        if len(train)<200:continue
        # Fixed regularization; train exclusively on completed earlier seasons.
        clf=LogisticRegression(C=.1,max_iter=2000)
        clf.fit(X[train],y[train])
        corrected=clf.predict_proba(X[test])[:,1]
        baseline=df.market_home_prob.to_numpy()[test]
        old=df.model_home_prob.to_numpy()[test]
        rows.append({'season':int(s),'games':len(test),'market_brier':float(brier_score_loss(y[test],baseline)),
            'corrected_brier':float(brier_score_loss(y[test],corrected)),
            'model_brier':float(brier_score_loss(y[test],old)),
            'market_log_loss':float(log_loss(y[test],baseline)),
            'corrected_log_loss':float(log_loss(y[test],corrected)),
            'market_coef':float(clf.coef_[0,0]),'model_residual_coef':float(clf.coef_[0,1])})
        for i,p in zip(test,corrected):preds.append({'season':int(s),'game_id':str(df.iloc[i].get('game_id','')),
            'home_win':int(y[i]),'market_home_prob':float(df.iloc[i].market_home_prob),
            'corrected_home_prob':float(p)})
    p=pd.DataFrame(preds)
    if p.empty:return {'status':'insufficient training seasons','seasons':rows},p
    summary={'status':'exploratory market-relative forecast; not tradable ROI','games':len(p),
      'market_brier':float(brier_score_loss(p.home_win,p.market_home_prob)),
      'corrected_brier':float(brier_score_loss(p.home_win,p.corrected_home_prob)),
      'market_log_loss':float(log_loss(p.home_win,p.market_home_prob)),
      'corrected_log_loss':float(log_loss(p.home_win,p.corrected_home_prob)),
      'seasons':rows}
    return summary,p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--predictions',required=True);ap.add_argument('--output-dir',default='research_outputs');a=ap.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    summary,p=run(pd.read_csv(a.predictions))
    (out/'market_residual_summary.json').write_text(json.dumps(summary,indent=2))
    p.to_csv(out/'market_residual_predictions.csv',index=False)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
