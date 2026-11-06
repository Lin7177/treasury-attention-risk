"""Fixed metrics and convex baseline implementation for the target-centric study."""
import numpy as np
import pandas as pd

def sigmoid(z):return np.exp(-np.logaddexp(0.,-np.asarray(z)))

def metrics(p,d):
    p=np.asarray(p,float);y=d.direction.to_numpy();pred=np.where(p>=.5,1,-1);nf=y!=0
    up=float((pred[y==1]==1).mean()) if (y==1).any() else None
    down=float((pred[y==-1]==-1).mean()) if (y==-1).any() else None
    q=np.clip(p[nf],1e-12,1-1e-12);truth=(y[nf]==1).astype(float)
    result=dict(n=len(y),nonflat_n=int(nf.sum()),accuracy=float((pred==y).mean()),balanced_accuracy=(up+down)/2 if up is not None and down is not None else None,up_recall=up,down_recall=down,predicted_up_share=float((pred==1).mean()),brier=float(np.mean((q-truth)**2)) if len(q) else None,log_loss=float(-np.mean(truth*np.log(q)+(1-truth)*np.log1p(-q))) if len(q) else None)
    for key,value in [('p_mean',p.mean()),('p_std',p.std()),('p05',np.quantile(p,.05)),('p50',np.quantile(p,.5)),('p95',np.quantile(p,.95))]:result[key]=float(value)
    return result

def macro_metrics(p,d):
    p=np.asarray(p);vals=[]
    for _,indices in d.groupby(d.date.str[:4]).indices.items():
        m=metrics(p[indices],d.iloc[indices]);vals.append(m['balanced_accuracy'])
    return None if any(v is None for v in vals) else float(np.mean(vals))

def fit_logit(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if set(np.unique(y))!={0.,1.}:raise ValueError('Logit requires two classes')
    mu=x.mean(0);sd=x.std(0);sd[sd==0]=1;z=np.c_[np.ones(len(x)),(x-mu)/sd];w=np.zeros(z.shape[1]);trace=[]
    def objective(w):return float(np.mean(np.logaddexp(0,z@w)-y*(z@w))+.005*np.dot(w[1:],w[1:]))
    for iteration in range(101):
        prob=sigmoid(z@w);g=z.T@(prob-y)/len(y);g[1:]+=.01*w[1:]
        loss=objective(w);norm=float(np.abs(g).max());trace.append(dict(iteration=iteration,objective=loss,gradient_inf=norm))
        if not np.isfinite(loss) or not np.isfinite(g).all():raise ArithmeticError('Nonfinite convex optimization')
        if norm<=1e-8:return dict(mean=mu.tolist(),scale=sd.tolist(),weights=w.tolist(),trace=trace)
        if iteration==100:break
        h=z.T@((prob*(1-prob))[:,None]*z)/len(y);h[1:,1:]+=.01*np.eye(len(w)-1);step=np.linalg.solve(h,g);rate=1.
        for _ in range(30):
            proposal=w-rate*step
            if objective(proposal)<=loss-1e-4*rate*float(g@step):w=proposal;break
            rate/=2
        else:raise ArithmeticError('Logit line search failed')
    raise ArithmeticError('Logit iteration limit')

def predict_logit(model,x):
    z=np.c_[np.ones(len(x)),(x-np.array(model['mean']))/np.array(model['scale'])]
    return sigmoid(z@np.array(model['weights']))
