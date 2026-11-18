"""V4 owned zero-mean GARCH/DCC numerical baseline. No optimizer runs at import."""
from __future__ import annotations
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import softmax, ndtri
from scipy.stats import qmc

class FitError(RuntimeError):
    def __init__(self,message,trace=None,details=None):
        super().__init__(message);self.trace=trace or [];self.details=details or {}

def ab_transform(u):
    p=softmax(np.r_[np.asarray(u,dtype=float),0.0])[:2]
    ab=.999*p
    jac=np.diag(ab)-np.outer(ab,ab)/.999
    return ab,jac

def garch_filter(r,omega,alpha,beta,initial=1.0,derivatives=False):
    r=np.asarray(r,dtype=float);n=len(r)
    h=np.empty(n);grad=np.empty((n,3)) if derivatives else None
    value=float(initial);dv=np.zeros(3)
    for t in range(n):
        if value<=0 or not np.isfinite(value):raise FloatingPointError("GARCH invalid variance")
        h[t]=value
        if derivatives:grad[t]=dv
        if np.isfinite(r[t]):
            rv=r[t]*r[t]
            nextdv=np.array([1.,rv,value])+beta*dv
            value=omega+alpha*rv+beta*value
        else:
            nextdv=np.array([1.,value,value])+(alpha+beta)*dv
            value=omega+(alpha+beta)*value
        dv=nextdv
    return (h,grad) if derivatives else h

def garch_objective(u,r):
    omega=float(np.exp(u[0]));(alpha,beta),jac=ab_transform(u[1:])
    h,dh=garch_filter(r,omega,alpha,beta,derivatives=True)
    observed=np.isfinite(r)
    if observed.sum()<30:raise ValueError("Insufficient observed GARCH data")
    x=np.asarray(r)[observed];v=h[observed]
    loss=float(np.mean(np.log(v)+x*x/v))
    dtheta=np.mean((1/v-x*x/v**2)[:,None]*dh[observed],axis=0)
    gradient=np.r_[dtheta[0]*omega,jac.T@dtheta[1:]]
    if not np.isfinite(loss) or not np.isfinite(gradient).all():raise FloatingPointError("Nonfinite GARCH objective")
    return loss,gradient

def projected_gradient(u,gradient,bounds):
    g=np.array(gradient,copy=True)
    for k,(lo,hi) in enumerate(bounds):
        if u[k]<=lo+1e-8 and g[k]>0:g[k]=0
        if u[k]>=hi-1e-8 and g[k]<0:g[k]=0
    return float(np.max(np.abs(g)))

def fit_garch(r,deadline=None):
    r=np.asarray(r,dtype=float)
    bounds=[(-20.,10.),(-14.,14.),(-14.,14.)]
    start=np.array([np.log(.05),0.,np.log(18.)])
    trace=[]
    def objective(u):
        if deadline is not None and time.perf_counter()>deadline:raise FitError("Market compute deadline exceeded",trace)
        value=garch_objective(u,r)
        if deadline is not None and time.perf_counter()>deadline:raise FitError("Market compute deadline exceeded",trace)
        return value
    def callback(u):
        f,g=objective(u);trace.append(dict(iteration=len(trace)+1,loss=f,projected_gradient=projected_gradient(u,g,bounds)))
    try:
        f,g=objective(start);trace.append(dict(iteration=0,loss=f,projected_gradient=projected_gradient(start,g,bounds)))
        result=minimize(objective,start,jac=True,method="L-BFGS-B",bounds=bounds,callback=callback,
            options=dict(maxiter=500,maxls=50,gtol=1e-7,ftol=1e-12))
        f,g=objective(result.x);pg=projected_gradient(result.x,g,bounds)
        if not result.success or not np.isfinite(f) or not np.isfinite(g).all() or not np.isfinite(pg) or pg>1e-4:
            raise FitError("GARCH optimizer acceptance failed",trace,dict(message=str(result.message),projected_gradient=pg))
        (a,b),_=ab_transform(result.x[1:])
        return dict(parameters=dict(omega=float(np.exp(result.x[0])),alpha=float(a),beta=float(b)),
                    objective=f,projected_gradient=pg,trace=trace,iterations=int(result.nit),
                    observations=int(np.isfinite(r).sum()),missing=int((~np.isfinite(r)).sum()))
    except FitError:raise
    except Exception as exc:raise FitError(str(exc),trace) from exc

def correlation(q):
    diagonal=np.diagonal(q,axis1=-2,axis2=-1)
    if np.any(diagonal<=0):raise FloatingPointError("Invalid Q diagonal")
    roots=np.sqrt(diagonal)
    return q/(roots[..., :,None]*roots[...,None,:])

def correlation_derivative(q,dq):
    roots=np.sqrt(np.diag(q));r=q/np.outer(roots,roots)
    dd=np.diag(dq)/np.diag(q)
    return dq/np.outer(roots,roots)-.5*r*(dd[:,None]+dd[None,:])

def dcc_filter(z,qbar,a,b,with_gradient=False):
    z=np.asarray(z,dtype=float);n,p=z.shape
    q=np.asarray(qbar,dtype=float).copy()
    dq=np.zeros((2,p,p))
    qs=np.empty((n,p,p));rs=np.empty_like(qs)
    total=0.;deriv=np.zeros(2);observed=0
    for t in range(n):
        r=correlation(q)
        if with_gradient:dr=np.stack([correlation_derivative(q,v) for v in dq])
        qs[t]=q;rs[t]=r
        complete=np.isfinite(z[t]).all()
        if complete:
            chol=np.linalg.cholesky(r);inv=np.linalg.solve(r,np.eye(p))
            v=np.linalg.solve(r,z[t])
            total+=2*np.log(np.diag(chol)).sum()+z[t]@v
            if with_gradient:
                score=inv-np.outer(v,v)
                deriv+=np.einsum("ij,kij->k",score,dr)
            observed+=1
            innovation=np.outer(z[t],z[t])
            dinnovation=np.zeros_like(dq)
        else:
            # Discard partial joint information; prior-moment approximation, not exact partial-data inference.
            innovation=r
            dinnovation=dr if with_gradient else np.zeros_like(dq)
        if with_gradient:
            nextdq=b*dq+a*dinnovation
            nextdq[0]+=innovation-qbar
            nextdq[1]+=q-qbar
            dq=nextdq
        q=(1-a-b)*qbar+a*innovation+b*q
    if observed<30:raise ValueError("Insufficient complete DCC data")
    return dict(q=qs,r=rs,loss=float(total/observed),gradient=deriv/observed,
                complete=observed,missing=n-observed)

def dcc_objective(u,z,qbar):
    (a,b),jac=ab_transform(u)
    values=dcc_filter(z,qbar,a,b,True)
    gradient=jac.T@values["gradient"]
    if not np.isfinite(values["loss"]) or not np.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite DCC objective")
    return values["loss"],gradient

def fit_dcc(z,deadline=None):
    z=np.asarray(z,dtype=float);complete=np.isfinite(z).all(axis=1)
    raw=z[complete].T@z[complete]/complete.sum()
    qbar=.99*raw+.01*np.eye(z.shape[1])
    bounds=[(-14.,14.),(-14.,14.)];start=np.array([0.,np.log(18.)]);trace=[]
    def objective(u):
        if deadline is not None and time.perf_counter()>deadline:raise FitError("Market compute deadline exceeded",trace)
        value=dcc_objective(u,z,qbar)
        if deadline is not None and time.perf_counter()>deadline:raise FitError("Market compute deadline exceeded",trace)
        return value
    def callback(u):
        f,g=objective(u);trace.append(dict(iteration=len(trace)+1,loss=f,projected_gradient=projected_gradient(u,g,bounds)))
    try:
        f,g=objective(start);trace.append(dict(iteration=0,loss=f,projected_gradient=projected_gradient(start,g,bounds)))
        result=minimize(objective,start,jac=True,method="L-BFGS-B",bounds=bounds,callback=callback,
            options=dict(maxiter=500,maxls=50,gtol=1e-7,ftol=1e-12))
        f,g=objective(result.x);pg=projected_gradient(result.x,g,bounds)
        if not result.success or not np.isfinite(f) or not np.isfinite(g).all() or not np.isfinite(pg) or pg>1e-4:
            raise FitError("DCC optimizer acceptance failed",trace,dict(message=str(result.message),projected_gradient=pg))
        (a,b),_=ab_transform(result.x)
        return dict(parameters=dict(alpha=float(a),beta=float(b)),qbar=qbar.tolist(),
                    objective=f,projected_gradient=pg,trace=trace,iterations=int(result.nit),
                    observations=int(complete.sum()),missing=int((~complete).sum()),
                    prior_shrinkage=.01)
    except FitError:raise
    except Exception as exc:raise FitError(str(exc),trace) from exc

def forecast_dcc_five(h,q,garch,dcc,exponent=12,seed=202611054):
    """QMC expectation of five future conditional Gaussian second moments, exact recursion on each path.
    Four future innovations suffice; the fifth conditional moment is integrated analytically.
    Forecast integration is approximate; nested exponent-13 sensitivity must be reported.
    """
    h=np.asarray(h,dtype=float);q=np.asarray(q,dtype=float)
    if h.ndim!=2 or q.shape!=(len(h),h.shape[1],h.shape[1]):raise ValueError("Batched current h/Q required")
    n,p=h.shape;sim=2**exponent
    design=qmc.Sobol(d=4*p,scramble=True,seed=seed).random_base2(exponent)
    normals=ndtri(np.clip(design,np.finfo(float).eps,1-np.finfo(float).eps)).reshape(sim,4,p)
    omega=np.array([v["omega"] for v in garch]);a=np.array([v["alpha"] for v in garch]);b=np.array([v["beta"] for v in garch])
    ca,cb=dcc["parameters"]["alpha"],dcc["parameters"]["beta"];bar=np.asarray(dcc["qbar"])
    hv=np.broadcast_to(h[:,None,:],(n,sim,p)).copy()
    qv=np.broadcast_to(q[:,None,:,:],(n,sim,p,p)).copy()
    prediction=np.zeros((n,p,p))
    for step in range(5):
        rv=correlation(qv)
        sv=np.sqrt(hv)
        conditional=rv*(sv[..., :,None]*sv[...,None,:])
        prediction+=conditional.mean(axis=1)
        if step<4:
            chol=np.linalg.cholesky(rv)
            epsilon=np.einsum("nsij,sj->nsi",chol,normals[:,step])
            values=epsilon*sv
            hv=omega+a*values**2+b*hv
            outer=epsilon[..., :,None]*epsilon[...,None,:]
            qv=(1-ca-cb)*bar+ca*outer+cb*qv
    if not np.isfinite(prediction).all():raise FloatingPointError("Nonfinite DCC forecast")
    # Each univariate conditional second moment has an analytic five-step mean.
    # Rescaling the simulated aggregate correlation removes known marginal Monte Carlo noise.
    expected=np.zeros_like(h);current=h.copy()
    for _ in range(5):
        expected+=current
        current=omega+(a+b)*current
    root=np.sqrt(expected)
    return correlation(prediction)*(root[:,:,None]*root[:,None,:])

def integration_discrepancy(coarse,fine):
    p=coarse.shape[-1]
    stein=(np.trace(np.linalg.solve(coarse,fine)+np.linalg.solve(fine,coarse),axis1=-2,axis2=-1)-2*p)/(2*p)
    frobenius=np.linalg.norm(fine-coarse,axis=(1,2))/np.linalg.norm(fine,axis=(1,2))
    if not np.isfinite(stein).all() or not np.isfinite(frobenius).all() or (stein < -1e-12).any():
        raise FloatingPointError("Invalid integration discrepancy or materially negative Stein difference")
    return stein,frobenius

def controlled_dcc_five(h,q,garch,dcc,seed=202611054):
    if not 1<=len(h)<=8:
        raise ValueError("Frozen DCC request batch must contain one to eight rows")
    coarse=forecast_dcc_five(h,q,garch,dcc,12,seed)
    fine=forecast_dcc_five(h,q,garch,dcc,13,seed)
    stein,frob=integration_discrepancy(coarse,fine)
    bad=(stein>1e-4)|(frob>.01)
    paths=np.full(len(h),8192,dtype=int)
    if bad.any():
        highest=forecast_dcc_five(h[bad],q[bad],garch,dcc,14,seed)
        nextstein,nextfrob=integration_discrepancy(fine[bad],highest)
        if (nextstein>1e-4).any() or (nextfrob>.01).any():
            raise FitError("DCC registered quadrature precision cap failed",details=dict(
                attempted_highest_paths=16384,unresolved=int(((nextstein>1e-4)|(nextfrob>.01)).sum()),
                max_stein=float(nextstein.max()),max_relative_frobenius=float(nextfrob.max())))
        fine[bad]=highest;stein[bad]=nextstein;frob[bad]=nextfrob;paths[bad]=16384
    return fine,dict(paths=paths,stein=stein,relative_frobenius=frob,upgraded=int(bad.sum()),
                     coarse_paths=4096,default_paths=8192,max_paths=16384)

def har_features(x,scale):
    x=np.asarray(x,dtype=float)
    return np.column_stack([np.log1p(np.mean(x[:,-window:]**2,axis=1)/(scale/5)) for window in (1,5,20)])

def fit_har(x,target,deadline=None):
    target=np.asarray(target,dtype=float);scale=float(target.mean())
    if not scale>0:raise FitError("Nonpositive HAR training target mean")
    features=har_features(x,scale);mean=features.mean(axis=0);std=features.std(axis=0)
    zero=std==0;std[zero]=1
    design=np.column_stack([np.ones(len(target)),(features-mean)/std])
    q=target/scale;penalty=np.r_[0.,np.full(3,.01)];beta=np.zeros(4);trace=[]
    def values(v):
        eta=design@v
        with np.errstate(over="raise",invalid="raise"):
            term=q*np.exp(-eta)
            loss=float(np.mean(eta+term)+.5*np.dot(penalty*v,v))
        gradient=design.T@(1-term)/len(q)+penalty*v
        hessian=(design.T*term)@design/len(q)+np.diag(penalty)
        return loss,gradient,hessian
    try:
        for iteration in range(101):
            if deadline is not None and time.perf_counter()>deadline:raise FitError("Market compute deadline exceeded",trace)
            loss,g,h=values(beta)
            trace.append(dict(iteration=iteration,loss=loss,gradient_inf=float(np.max(np.abs(g)))))
            if np.max(np.abs(g))<=1e-8:
                return dict(beta=beta.tolist(),scale=scale,mean=mean.tolist(),std=std.tolist(),
                    zero_std=zero.tolist(),trace=trace,objective=loss,iterations=iteration,
                    gradient_inf=float(np.max(np.abs(g))))
            if iteration==100:raise FitError("HAR iteration limit",trace)
            direction=np.linalg.solve(h,g);step=1.
            for _ in range(40):
                candidate=beta-step*direction
                try:
                    trial=values(candidate)[0]
                    if not np.isfinite(trial):trial=float("inf")
                except (FloatingPointError,OverflowError):
                    trial=float("inf")
                if trial<=loss-1e-4*step*np.dot(g,direction):
                    raw=np.exp(design@candidate)
                    if np.any(raw<=1e-8) or not np.isfinite(raw).all():raise FitError("HAR forecast floor",trace)
                    beta=candidate;break
                step*=.5
            else:raise FitError("HAR Armijo limit",trace)
    except FitError:raise
    except Exception as exc:raise FitError(str(exc),trace) from exc

def predict_har(x,state):
    f=har_features(x,state["scale"])
    design=np.column_stack([np.ones(len(f)),(f-np.asarray(state["mean"]))/np.asarray(state["std"])])
    raw=np.exp(design@np.asarray(state["beta"]))
    if np.any(raw<=1e-8) or not np.isfinite(raw).all():raise FloatingPointError("HAR inference floor")
    return raw*state["scale"]

def matrix_qlike(target,forecast):
    target=np.asarray(target,dtype=float);forecast=np.asarray(forecast,dtype=float)
    if target.shape!=forecast.shape or target.ndim!=3 or target.shape[1]!=target.shape[2] or not len(target):
        raise ValueError("Expected matched nonempty [N,p,p] matrix arrays")
    if not np.isfinite(target).all() or not np.isfinite(forecast).all():
        raise ValueError("Nonfinite target or forecast matrix")
    if not np.allclose(target,target.swapaxes(-1,-2),atol=1e-10,rtol=1e-10):
        raise ValueError("Nonsymmetric target matrix")
    if not np.allclose(forecast,forecast.swapaxes(-1,-2),atol=1e-10,rtol=1e-10):
        raise ValueError("Nonsymmetric forecast matrix")
    tolerance=1e-10*np.maximum(1,np.max(np.abs(target),axis=(-1,-2)))
    if (np.linalg.eigvalsh(target)[:,0]<-tolerance).any():
        raise ValueError("Target matrix is not positive semidefinite")
    chol=np.linalg.cholesky(forecast)
    logdet=2*np.log(np.diagonal(chol,axis1=-2,axis2=-1)).sum(axis=-1)
    trace=np.trace(np.linalg.solve(forecast,target),axis1=-2,axis2=-1)
    result=(logdet+trace)/forecast.shape[-1]
    if not np.isfinite(result).all():raise FloatingPointError("Nonfinite matrix QLIKE")
    return result
