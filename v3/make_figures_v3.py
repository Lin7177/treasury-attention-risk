"""Figures for integrated V3 manuscript; all sources frozen result products."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':12,'xtick.labelsize':11,'ytick.labelsize':11})
ROOT=Path(__file__).resolve().parent;R=ROOT/'outputs/q5_analysis_001';OUT=ROOT/'deliverables/submission/figures'
MODELS=['Mean','SMA20','EWMA','OwnHAR','MultiHAR','Fixed','Uniform','Content']
def main():
 OUT.mkdir(parents=True,exist_ok=True);m=pd.read_csv(R/'model_metrics.csv');m=m[m.aggregation=='macro'].set_index('model');y=pd.read_csv(R/'year_metrics.csv').set_index(['model','year']);pairs=pd.read_csv(R/'paired_intervals.csv');tail=pd.read_csv(R/'tail_sensitivity.csv')
 fig,ax=plt.subplots(1,2,figsize=(10.6,3.5),layout='constrained');idx=np.arange(8);values=m.loc[MODELS].qlike.to_numpy();ax[0].barh(idx,values,color=['#aab9c4']*3+['#244b65']*2+['#a98761','#6a8f85','#9a4539']);ax[0].set_yticks(idx,MODELS);ax[0].invert_yaxis();ax[0].set_xlim(3.82,4.12);ax[0].set_xlabel('Year-equal physical QLIKE (lower is better)')
 for i,v in enumerate(values):ax[0].text(v+.003,i,f'{v:.4f}',va='center',fontsize=11)
 a=np.array([[y.loc[(n,yr),'qlike']-y.loc[('EWMA',yr),'qlike'] for yr in [2022,2023,2024]] for n in MODELS]);im=ax[1].imshow(a,vmin=-.4,vmax=.4,cmap='RdBu_r',aspect='auto');ax[1].set_yticks(idx,MODELS);ax[1].set_xticks(range(3),['2022','2023','2024\nH1'],fontsize=11)
 for i in range(8):
  for j in range(3):ax[1].text(j,i,f'{a[i,j]:+.3f}',ha='center',va='center',fontsize=11)
 fig.colorbar(im,ax=ax[1],label='QLIKE minus EWMA',shrink=.85);fig.savefig(OUT/'risk_results.png',dpi=220);plt.close(fig)
 p=pairs[(pairs.block==10)&(pairs.statistic=='macro')&(pairs.confidence==.95)]
 fig,ax=plt.subplots(figsize=(8.8,3.1),layout='constrained');idx=np.arange(len(p));v=p.estimate.to_numpy();ax.errorbar(v,idx,xerr=np.vstack([v-p.lower.to_numpy(),p.upper.to_numpy()-v]),fmt='o',capsize=3,color='#244b65');ax.set_yticks(idx,[a+' - '+b for a,b in zip(p.model,p.baseline)]);ax.invert_yaxis();ax.axvline(0,color='gray',lw=.8);ax.set_xlabel('Year-equal paired QLIKE difference, 95% block CI; negative favors model');fig.savefig(OUT/'paired_risk.png',dpi=220);plt.close(fig)
 tr=pd.read_csv(ROOT/'outputs/q5_main_001/training_trace.csv');fig,ax=plt.subplots(1,2,figsize=(9.4,3.1),layout='constrained')
 for name,color in [('Fixed','#244b65'),('Uniform','#6a8f85'),('Content','#9a4539')]:
  mean=tr[tr.model==name].groupby('epoch').mean(numeric_only=True);ax[0].plot(mean.index,mean.fixed_train_normalized_qlike,color=color,label=name);ax[1].plot(mean.index,mean.validation_normalized_qlike,color=color,label=name)
 for a,title in zip(ax,['Fixed full-training loss','Earlier checkpoint-validation loss']):a.set_xlabel('Epoch');a.set_ylabel('Normalized QLIKE');a.set_title(title,fontsize=12);a.legend(fontsize=11)
 ax[1].get_legend().remove();fig.savefig(OUT/'risk_losses.png',dpi=220);plt.close(fig)
 t=tail.groupby(['model','baseline'],sort=False)[['full_delta','without_top1pct_delta']].mean();idx=np.arange(len(t));fig,ax=plt.subplots(figsize=(8.8,3.1),layout='constrained');ax.barh(idx-.16,t.full_delta,.3,color='#244b65',label='All requests (primary)');ax.barh(idx+.16,t.without_top1pct_delta,.3,color='#a98761',label='Exclude fixed top 1% Q (ex post)');ax.set_yticks(idx,[a+' - '+b for a,b in t.index]);ax.invert_yaxis();ax.axvline(0,color='gray',lw=.8);ax.set_xlabel('Year-equal QLIKE difference (negative favors model)');ax.legend(fontsize=11,loc='best');fig.savefig(OUT/'risk_tail.png',dpi=220);plt.close(fig)
if __name__=='__main__':main()
