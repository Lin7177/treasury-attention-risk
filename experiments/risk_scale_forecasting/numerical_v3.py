"""Convex log-link HAR QLIKE benchmark and observed-energy metrics."""
import numpy as np
FLOOR=1e-8

class NumericalFailure(ArithmeticError):
    def __init__(self,message,trace=None,diagnostics=None):
        super().__init__(message);self.trace=[] if trace is None else trace;self.diagnostics={} if diagnostics is None else diagnostics

def positive_guard(raw,role,context=None):
    raw=np.asarray(raw,float);finite=np.isfinite(raw);floor=finite & (raw<=FLOOR)
    if not finite.all() or floor.any():
        diagnostic=dict(role=role,n=raw.size,raw_floor_hits=int(floor.sum()),nonfinite_raw=int((~finite).sum()),minimum_finite_raw=float(raw[finite].min()) if finite.any() else None,**({} if context is None else context))
        raise NumericalFailure('Supervised raw forecast violates numerical contract',diagnostics=diagnostic)

def qlike(q,h):
    q=np.asarray(q,float);h=np.asarray(h,float)
    if q.shape!=h.shape or not np.isfinite(q).all() or not np.isfinite(h).all() or (q<0).any() or (h<=0).any():raise ValueError('Finite q>=0 and aligned h>0 required')
    return np.log(h)+q/h

def metrics(q,h):
    q=np.asarray(q,float);h=np.asarray(h,float);v=qlike(q,h)
    return dict(n=len(q),qlike=float(v.mean()),mae=float(np.abs(h-q).mean()),mse=float(np.square(h-q).mean()),mean_actual=float(q.mean()),mean_forecast=float(h.mean()),zero_targets=int((q==0).sum()),forecast_p05=float(np.quantile(h,.05)),forecast_p50=float(np.quantile(h,.5)),forecast_p95=float(np.quantile(h,.95)))

def har_design(x,own,target_scale=5.):
    x=np.asarray(x,float)
    if x.ndim!=3 or x.shape[1:]!=(20,5):raise ValueError('Expected ordered 20x5 raw changes')
    if not np.isfinite(target_scale) or target_scale<=0:raise ValueError('Positive training target scale required')
    sq=x*x/(target_scale/5.)
    cols=[3] if own else list(range(5))
    return np.log1p(np.concatenate([sq[:,-width:,cols].mean(1) for width in [1,5,20]],axis=1))

def derivatives(w,z,q):
    eta=z@w
    with np.errstate(over='ignore',invalid='ignore'):
        v=q*np.exp(-eta)
        loss=float(np.mean(eta+v)+.005*np.dot(w[1:],w[1:]))
    gradient=z.T@(1-v)/len(q);gradient[1:]+=.01*w[1:]
    hessian=z.T@(v[:,None]*z)/len(q);hessian[1:,1:]+=.01*np.eye(len(w)-1)
    return loss,gradient,hessian

def fit_har(x,target):
    x=np.asarray(x,float);target=np.asarray(target,float);trace=[]
    try:
        if len(x)!=len(target) or not np.isfinite(x).all() or not np.isfinite(target).all() or (target<0).any() or target.mean()<=0:raise ValueError('Invalid HAR inputs or no positive target mean')
        mu=x.mean(0);sd=x.std(0);sd[sd==0]=1;ys=float(target.mean());q=target/ys;z=np.c_[np.ones(len(x)),(x-mu)/sd];w=np.zeros(z.shape[1])
        for iteration in range(101):
            with np.errstate(over='ignore'):raw=np.exp(z@w)
            positive_guard(raw,'accepted_har_iterate',{'iteration':iteration})
            loss,g,h=derivatives(w,z,q);norm=float(np.max(np.abs(g)));trace.append(dict(iteration=iteration,penalized_normalized_qlike=loss,gradient_inf=norm))
            if not np.isfinite(loss) or not np.isfinite(g).all() or not np.isfinite(h).all():raise NumericalFailure('Nonfinite HAR objective/derivatives',diagnostics={'role':'har_optimizer','iteration':iteration})
            if norm<=1e-8:return dict(mean=mu.tolist(),scale=sd.tolist(),beta=w.tolist(),target_scale=ys,trace=trace)
            if iteration==100:break
            step=np.linalg.solve(h,g);rate=1.
            for _ in range(40):
                candidate=w-rate*step;candidate_loss=derivatives(candidate,z,q)[0]
                if np.isfinite(candidate_loss) and candidate_loss<=loss-1e-4*rate*float(g@step):w=candidate;break
                rate/=2
            else:raise NumericalFailure('HAR Armijo line search failed',diagnostics={'role':'har_optimizer','iteration':iteration})
        raise NumericalFailure('HAR Newton exhausted frozen 100-iteration cap',diagnostics={'role':'har_optimizer','iteration':100})
    except NumericalFailure as error:
        error.trace=list(trace);raise
    except np.linalg.LinAlgError as error:
        raise NumericalFailure(str(error),trace=list(trace),diagnostics={'role':'har_optimizer'}) from error
    except ValueError:
        raise
    except Exception as error:
        raise NumericalFailure(str(error),trace=list(trace),diagnostics={'role':'har_optimizer'}) from error

def predict_har(state,x,role="unspecified"):
    x=np.asarray(x,float);z=np.c_[np.ones(len(x)),(x-np.asarray(state['mean']))/np.asarray(state['scale'])];eta=z@np.asarray(state['beta'])
    raw=np.exp(eta)
    positive_guard(raw,role)
    return raw*state['target_scale']
