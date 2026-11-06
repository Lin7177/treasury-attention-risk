"""Publication figures from frozen outputs; does not train or select models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'deliverables/submission/figures'
RUN=ROOT/'outputs/target_main_001'
ANALYSIS=ROOT/'outputs/analysis_verified_001'
NAMES=['F0_Own20Logit','F1_FinancialLogit','F2_PCALogit','N0_OwnMaskedFixed','N1_Fixed','N2_UniformCross','N3_ContentCross','N4_SelfTarget']
SHORT=['F0 Own LR','F1 Financial LR','F2 PCA LR','N0 Own Fixed','N1 Fixed','N2 Uniform','N3 Cross','N4 Self Target']
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(RUN/'predictions_ensemble.csv');tr=pd.read_csv(RUN/'training_trace.csv')
    from analyze_target import metrics,macro_ba
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.5),layout='constrained')
    table=[]
    for name in NAMES:
        q=p[p.model==name];table.append([metrics(q.probability_up,q.direction)['balanced_accuracy'],macro_ba(q.probability_up,q.direction,q.date.str[:4])])
    a=np.array(table)*100;y=np.arange(len(NAMES));axes[0].barh(y-.18,a[:,0],.34,label='Pooled BA',color='#8ba6bc');axes[0].barh(y+.18,a[:,1],.34,label='Year-equal macro BA',color='#234b66');axes[0].set_yticks(y,SHORT);axes[0].invert_yaxis();axes[0].set_xlim(35,65);axes[0].axvline(50,color='gray',lw=.7);axes[0].set_xlabel('Balanced accuracy (%)');axes[0].legend(fontsize=8)
    annual=np.array([[metrics((q:=p[(p.model==n)&(p.year==yr)]).probability_up,q.direction)['balanced_accuracy']*100 for yr in [2022,2023,2024]] for n in NAMES])
    im=axes[1].imshow(annual,vmin=40,vmax=60,cmap='RdBu');axes[1].set_yticks(y,SHORT);axes[1].set_xticks(range(3),['2022','2023','2024\nH1'],fontsize=8)
    for i in range(8):
        for j in range(3):axes[1].text(j,i,f'{annual[i,j]:.1f}',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=axes[1],label='BA (%)',shrink=.8);fig.savefig(OUT/'new_results.png',dpi=220);plt.close(fig)
    intervals=pd.read_csv(ANALYSIS/'paired_intervals.csv')
    q=intervals[(intervals.block_length==10)&(intervals.metric=='macro_balanced_accuracy')&(intervals.confidence_level==.95)].copy()
    fig,ax=plt.subplots(figsize=(8.7,3.7),layout='constrained');y=np.arange(len(q));delta=q.point_estimate.to_numpy()*100
    ax.errorbar(delta,y,xerr=np.vstack([delta-q.low.to_numpy()*100,q.high.to_numpy()*100-delta]),fmt='o',color='#234b66',capsize=3);ax.set_yticks(y,[a.split('_')[0]+' - '+('Old T1' if b.startswith('OLD') else b.split('_')[0]) for a,b in zip(q.left,q.right)]);ax.invert_yaxis();ax.axvline(0,color='gray',lw=1);ax.set_xlabel('Year-equal macro BA difference (percentage points), 95% paired block CI');fig.savefig(OUT/'paired_macro.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.2),layout='constrained')
    for name,color in [('N1_Fixed','#234b66'),('N2_UniformCross','#8a745e'),('N3_ContentCross','#a14335'),('N4_SelfTarget','#4f855c')]:
        avg=tr[tr.model==name].groupby('epoch').mean(numeric_only=True)
        axes[0].plot(avg.index,avg.fixed_train_bce,label=name[:2],color=color)
        axes[1].plot(avg.index,avg.validation_bce,label=name[:2],color=color)
    for ax,title in zip(axes,['Fixed full-training BCE','Checkpoint-validation BCE']):ax.set_title(title,fontsize=11);ax.set_xlabel('Epoch');ax.set_ylabel('BCE');ax.legend(ncol=2,fontsize=8)
    fig.savefig(OUT/'losses.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(9.4,2.7),layout='constrained')
    for ax,yr in zip(axes,[2022,2023,2024]):
        for n,c in [('N2_UniformCross','#8a745e'),('N3_ContentCross','#a14335'),('N4_SelfTarget','#4f855c')]:ax.hist(p[(p.model==n)&(p.year==yr)].probability_up,bins=np.linspace(0,1,17),histtype='step',label=n[:2],color=c,lw=1.4)
        ax.axvline(.5,color='gray',lw=.6);ax.set_title(str(yr)+(' H1' if yr==2024 else ''),fontsize=10);ax.set_xlabel('Probability of yield rise');ax.set_ylabel('Requests');ax.legend(fontsize=8)
    fig.savefig(OUT/'probabilities.png',dpi=220);plt.close(fig)
if __name__=='__main__':main()
