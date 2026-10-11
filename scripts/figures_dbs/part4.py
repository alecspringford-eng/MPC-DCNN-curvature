import os; os.environ['PART']='4'
from common import *
A=P5+'analysis/'
v=pd.read_csv(A+'validation_all.csv'); m=pd.read_csv(A+'per_seed_metrics.csv').rename(columns={'arm':'label'})
d=v.merge(m,on=['label','seed'])
d['family']=np.where(d.label.str.contains('both'),'both-branch g+h (smoothhess_directional)','h-branch only (smoothhess_directional)')
d['lam_name']=d.label.str.extract(r'lam(\w+)$')[0]
LAM={'both-branch g+h (smoothhess_directional)':{'0':0,'LOW':0.250,'HIGH':0.788},'h-branch only (smoothhess_directional)':{'0':0,'LOW':0.384,'HIGH':1.21}}
d['lam']=[LAM[f][l] for f,l in zip(d.family,d.lam_name)]
d['val_ratio']=np.where(d.label.str.contains('both'),d.ratio_vs_both_lam0,d.ratio_vs_h_lam0)
p3=pd.read_csv(R+'phase3_dbs_sh_directional/per_seed_results.csv'); p3['lam_name']=p3.label.str.extract(r'lam(\w+)$')[0]; p3['family']='h-branch only (smoothhess_directional)'
d=d.merge(p3[['family','lam_name','seed','h_frobenius']].rename(columns={'h_frobenius':'h3'}),on=['family','lam_name','seed'],how='left')
d['h_frobenius']=d.h_frobenius.fillna(d.h3)
d['stall_count']=d.n_stall
base=d[d.lam==0].set_index(['family','seed'])
for c in ['h_frobenius','g_frobenius','absgap_mean','Dg_mean','Dh_mean','n_iter_mean','n_stall','best_val_loss']:
    d['rel_'+c]=[100*(r[c]/base.loc[(r.family,r.seed),c]-1) for _,r in d.iterrows()]
d.to_pickle(O+'.dbs_sweep.pkl')
cols=['family','label','lam','seed','best_val_loss','val_ratio','h_frobenius','g_frobenius','Dg_mean','Dh_mean','absgap_mean','n_iter_mean','n_iter_p95','n_stall','rel_h_frobenius','rel_absgap_mean','rel_n_iter_mean','rel_n_stall']
tab(d[cols],'DBS_complete_regularisation_sweep',A+'validation_all.csv + per_seed_metrics.csv','Per (family, λ, seed): validation, curvature (Hessian Frobenius), directional Bregman gap, CCP iterations, stalls')
agg=d.groupby(['family','lam']).agg(val_ratio=('val_ratio','mean'),h_frob=('h_frobenius','mean'),g_frob=('g_frobenius','mean'),absgap=('absgap_mean','mean'),Dg=('Dg_mean','mean'),Dh=('Dh_mean','mean'),n_iter_mean=('n_iter_mean','mean'),stalls_total=('n_stall','sum'),rel_h=('rel_h_frobenius','mean'),rel_gap=('rel_absgap_mean','mean')).reset_index()
tab(agg,'DBS_summary',A+'validation_all.csv + per_seed_metrics.csv','DBS seed-averaged sweep summary'); print(agg.round(3).to_string())
SRC='phase5_dbs_joint_curvature/analysis/{validation_all,per_seed_metrics}.csv (8 seeds × 500 replay states per arm)'
EXP='DBS, λ=0 baseline vs LOW/HIGH; two families: h-branch-only and both-branch g+h (smoothhess_directional) penalty; N=1 surrogate plant'
FAM=list(LAM)
FC={FAM[0]:'#1f77b4',FAM[1]:'#ff7f0e'}
def sweep(ax,y,yl,log=False,ref=None):
    for f in FAM:
        s=d[d.family==f]
        for l,g in s.groupby('lam'): ax.scatter([l]*len(g),g[y],c=FC[f],alpha=.35,s=18)
        mm=s.groupby('lam')[y].mean(); ax.plot(mm.index,mm.values,'o-',c=FC[f],label=f,lw=2,ms=8)
    ax.scatter([0],[d[d.lam==0][y].mean()],marker='*',s=250,c='k',zorder=5,label='unregularised baseline (mean)')
    ax.set_xlabel('regularisation strength λ');ax.set_ylabel(yl)
    if ref is not None: ax.axhline(ref,c='k',ls='--',lw=1)
    if log: ax.set_yscale('log')
fig,ax=plt.subplots(2,3,figsize=(17,9))
sweep(ax[0,0],'best_val_loss','validation loss (MSE; lower=better)');sweep(ax[0,1],'val_ratio','validation ratio vs same-family λ=0',ref=1);sweep(ax[0,2],'h_frobenius','curvature: ‖H_h‖_F of learned h',log=False)
sweep(ax[1,0],'absgap_mean','directional Bregman gap |Dg−Dh| (mean over CCP steps)',log=True);sweep(ax[1,1],'n_iter_mean','mean CCP iterations (incl. stalls at cap 15)');sweep(ax[1,2],'n_stall','# stalled solves per seed (of 500)')
ax[0,0].legend(fontsize=7);fig.suptitle('DBS regularisation-strength sweep (dots = seeds, line = seed mean, star = unregularised baseline)')
save(fig,'02_curvature_accuracy','DBS_sweep_vs_regularisation_strength',SRC,EXP,'Validation loss/ratio, curvature, directional gap, iterations and stalls against λ, per seed.','Curvature ‖H_h‖_F falls 34–58%; |Dg−Dh| falls 85–92% for the both-branch model but only 13–39% for the h-only model (whose same-direction gap falls ~88–90% in phase 3); validation ratio 0.97–0.98 (not worse). Stalls drop for HIGH but rise for both-branch LOW.','Only two nonzero λ per family (no dense sweep); both families share the same seeds but not the same λ0 checkpoint; gap here is |Dg−Dh|.')
for y,yl,nm,log in (('val_ratio','validation ratio (1 = baseline)','validation_ratio_vs_lambda',False),('h_frobenius','curvature ‖H_h‖_F','curvature_vs_lambda',False),('absgap_mean','directional Bregman gap |Dg−Dh|','bregman_gap_vs_lambda',True),('n_stall','stalled solves per seed (of 500)','stalls_vs_lambda',False)):
    fig,ax=plt.subplots(figsize=(7,4.5));sweep(ax,y,yl,log,ref=1 if y=='val_ratio' else None);ax.legend(fontsize=7);ax.set_title(f'DBS: {yl} vs λ')
    save(fig,'02_curvature_accuracy',f'DBS_{nm}',SRC,EXP,f'{yl} against regularisation strength.','See combined sweep figure.','Two λ values only.')
fig,ax=plt.subplots(2,3,figsize=(17,9))
def rel(ax_,x,y,xl,yl,logx=False,logy=False,ymax=None):
    for f in FAM:
        s=d[d.family==f]
        if ymax is not None: s=s[s[y]<ymax]
        for l,mk in zip(sorted(s.lam.unique()),['*','o','s']):
            g=s[s.lam==l]; ax_.scatter(g[x],g[y],c=FC[f],marker=mk,s=120 if mk=='*' else 40,alpha=.7,edgecolor='k' if mk=='*' else 'none',label=f'{f.split(" ")[0]} λ={l}')
    ax_.set_xlabel(xl);ax_.set_ylabel(yl)
    if logx:ax_.set_xscale('log')
    if logy:ax_.set_yscale('log')
rel(ax[0,0],'h_frobenius','best_val_loss','curvature ‖H_h‖_F','validation loss');rel(ax[0,1],'absgap_mean','best_val_loss','directional Bregman gap |Dg−Dh|','validation loss',True)
rel(ax[0,2],'absgap_mean','n_iter_mean','directional gap |Dg−Dh|','mean CCP iterations',True,ymax=6);ax[0,2].text(.02,.97,'1 outlier omitted (both-branch λ=0.25, seed 1: 9.5 mean iterations)',transform=ax[0,2].transAxes,fontsize=7,va='top');rel(ax[1,0],'rel_absgap_mean','val_ratio','relative gap change (%) vs baseline','validation ratio')
rel(ax[1,1],'rel_h_frobenius','val_ratio','relative curvature change (%)','validation ratio');rel(ax[1,2],'h_frobenius','absgap_mean','curvature ‖H_h‖_F','directional gap',False,True)
ax[0,0].legend(fontsize=6,ncol=2);ax[1,0].axhline(1,c='k',ls='--',lw=1);ax[1,1].axhline(1,c='k',ls='--',lw=1);ax[1,0].axvline(0,c='k',lw=.7);ax[1,1].axvline(0,c='k',lw=.7)
fig.suptitle('DBS trade-offs per seed×λ (stars = unregularised baselines; reading: lower-left = more accurate & flatter)')
save(fig,'02_curvature_accuracy','DBS_validation_curvature_gap_relations',SRC,EXP,'Six relation plots: val loss vs curvature/gap, iterations vs gap, relative changes vs val ratio, curvature vs gap.','Large gap/curvature reductions come at no validation cost in DBS (validation ratio stays ≈0.97–0.98).','Seed-level n=8 per λ; baseline checkpoints are different per family.')
fig,ax=plt.subplots(1,2,figsize=(12,4.8))
for a,(y,yl) in zip(ax,[('rel_h_frobenius','curvature'),('rel_absgap_mean','directional Bregman gap')]):
    for f in FAM:
        s=d[(d.family==f)&(d.lam>0)]
        for l,g in s.groupby('lam'): a.scatter(100*(g.val_ratio-1),-g[y],c=FC[f],s=50,alpha=.7,label=f'{f.split(" ")[0]} λ={l}')
    a.scatter([0],[0],marker='*',s=250,c='k',label='baseline');a.axvline(0,c='k',lw=.7);a.axhline(0,c='k',lw=.7);a.axvspan(0,20,color='#eee',zorder=0)
    a.set_xlabel('relative validation degradation (%)  [<0 = better than baseline]');a.set_ylabel(f'relative {yl} reduction (%)  [higher = flatter]');a.set_title(f'Pareto: accuracy cost vs {yl} reduction (best = upper-left)')
ax[0].legend(fontsize=7)
save(fig,'02_curvature_accuracy','DBS_pareto_accuracy_vs_geometry',SRC,EXP,'Pareto-style plot: relative validation degradation vs relative curvature / gap reduction, per seed and λ.','All regularised points sit upper-left of baseline: geometry improves a lot and validation does not degrade.','Grey band marks the 0–20% degradation tolerance (the project’s ≤1.2 admissibility ratio).')
pc=pd.read_csv(A+'pct_change_vs_control.csv'); pcs=pc[pc.metric.isin(['Dg_mean','Dh_mean','absgap_mean','n_iter_mean','n_iter_p95','stall'])]
tab(pc,'DBS_pct_change_vs_control',A+'pct_change_vs_control.csv','Seed-mean % changes with bootstrap CIs and seeds-lower counts')
fig,ax=plt.subplots(figsize=(12,5)); mets=['Dg_mean','Dh_mean','absgap_mean','n_iter_mean','n_iter_p95','stall']; wd=.18; x=np.arange(len(mets))
for i,(cmp_,lam,c,lab) in enumerate([('A_h_vs_lam0','LOW','#9ecae1','h-only LOW'),('A_h_vs_lam0','HIGH','#1f77b4','h-only HIGH'),('B_both_vs_lam0','LOW','#fdd0a2','both LOW'),('B_both_vs_lam0','HIGH','#e6550d','both HIGH')]):
    s=pcs[(pcs.comparison==cmp_)&(pcs.lam==lam)].set_index('metric').loc[mets]; ax.bar(x+(i-1.5)*wd,s.pct_mean.clip(-200,400),wd,yerr=[(s.pct_mean-s.lo).clip(0,400).values,(s.hi-s.pct_mean).clip(0,400).values],color=c,label=lab,capsize=2)
ax.axhline(0,c='k');ax.set_xticks(x,mets);ax.set_ylabel('% change vs λ=0 (seed mean, 95% CI; clipped to [−200,400])');ax.legend();ax.set_title('DBS: relative changes vs unregularised control (n=8 seeds)')
save(fig,'02_curvature_accuracy','DBS_relative_changes_vs_control',SRC,EXP,'Seed-mean relative change in gap metrics, iteration stats and stall count vs λ=0, with CIs.','Dg/Dh fall strongly in all configs (Dh −91…−93%; Dg −89…−94% only for both-branch); the stall count falls for HIGH and rises for both-branch LOW (CIs wide).','Clipped axis; percent changes on small bases (stalls) are noisy.')
