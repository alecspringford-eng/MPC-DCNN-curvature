import os; os.environ['PART']='3'
from common import *
from scipy import stats
w=pd.read_pickle(O+'.w.pkl'); DU=0.002
SF=R+'phase5b_stall_mechanism/tables/solve_features.csv.gz'
SRC='phase5b_stall_mechanism/tables/solve_features.csv.gz; phase5_dbs_joint_curvature/{current,safeguarded}/solves_*.csv.gz, current/iters_current.csv.gz'
EXP='DBS both-branch; baseline λ=0 vs HIGH λ=0.788 (also LOW 0.25 for iterations/stalls); 4000 paired states'
key=['seed','bank_idx']
f=pd.read_csv(SF); f['arm']=f.arm.str.replace('dbs_both_lam','').replace({'0':'base'})
keep=['n_iter','agap_max_first2','agap_mean_all','Dg_mean_all','Dh_mean_all','Dg_mean_first','Dh_mean_first','gap_mean_all','step_mean_first','step_mean_all','act_gh_all_rep_f2','kink_frac','dJtrue_first2_maxpos']
for a in ['base','LOW','HIGH']:
    x=f[f.arm==a][key+keep].rename(columns={c:f'{c}_{a}' for c in keep if c!='n_iter'}).drop(columns='n_iter'); w=w.merge(x,on=key)
it=pd.read_csv(P5+'current/iters_current.csv.gz'); it['arm']=it.arm.str.replace('dbs_both_lam','').replace({'0':'base'})
it['acurv_h']=it.secant_curv_h.abs(); it['acurv_g']=it.secant_curv_g.abs()
ag=it.groupby(['arm']+key).agg(acurv_h=('acurv_h','mean'),acurv_g=('acurv_g','mean'),frac_rate=('rate_active','mean'),frac_umin=('at_umin','mean')).reset_index()
for a in ['base','LOW','HIGH']:
    x=ag[ag.arm==a].drop(columns='arm').rename(columns={c:f'{c}_{a}' for c in ['acurv_h','acurv_g','frac_rate','frac_umin']}); w=w.merge(x,on=key,how='left')
for a in ['base','LOW','HIGH']:
    lo=np.maximum(0,w.u_prev-DU); hi=w.u_prev+DU; u=w[f'u_sol_{a}']
    w[f'rate_dist_{a}']=np.minimum(u-lo,hi-u)/DU
w['kk']=w.k_base
w.to_pickle(O+'.w2.pkl')
EPS=1e-16
def lg(x): return np.log10(np.maximum(x,EPS))
CAT=['D_persistent_conv','A_rescued','B_persistent_stall','C_induced']; NAME={'A_rescued':'A rescued','B_persistent_stall':'B persistent stall','C_induced':'C new stall','D_persistent_conv':'D converge both'}
def jit(a,b,ax,**kw): ax.scatter(a,b,**kw)
METS=[('agap_max_first2','max |Dg−Dh| over first 2 CCP transitions (pre-registered E1 geometry metric)'),('Dg_mean_all','mean directional Bregman gap of g (Dg) over iterates'),('Dh_mean_all','mean directional Bregman gap of h (Dh) over iterates')]
rows=[]
for m,desc in METS:
    b,h=w[f'{m}_base'],w[f'{m}_HIGH']
    pos=(b>1e-12)
    fig,ax=plt.subplots(2,3,figsize=(17,9))
    a=ax[0,0]; a.scatter(b,h,s=5,alpha=.3,c=COL['reg']); mx=max(b.max(),h.max()); mn=max(EPS*1e4,min(b[b>0].min(),h[h>0].min()));
    a.set_xscale('log');a.set_yscale('log');a.plot([mn,mx],[mn,mx],'k--',label='y = x (no change)'); a.set_xlim(mn,mx);a.set_ylim(mn,mx);a.set_xlabel('baseline (λ=0)');a.set_ylabel('HIGH (λ=0.788)');a.set_title(f'Per-state baseline vs regularised (n={len(w)})');a.legend()
    z0,z1=100*(b<=1e-10).mean(),100*(h<=1e-10).mean(); bn,hn=b[b>1e-10],h[h>1e-10]
    bins=np.linspace(min(lg(bn).min(),lg(hn).min()),max(lg(bn).max(),lg(hn).max()),50)
    ax[0,1].hist(lg(bn),bins,alpha=.6,color=COL['base'],label=f'baseline (n={len(bn)}; {z0:.0f}% of states ≈0 excluded)');ax[0,1].hist(lg(hn),bins,alpha=.6,color=COL['reg'],label=f'HIGH (n={len(hn)}; {z1:.0f}% ≈0 excluded)');ax[0,1].set_xlabel('log10(metric), states with metric>1e-10');ax[0,1].set_ylabel('# states');ax[0,1].legend(fontsize=7);ax[0,1].set_title('Distribution of non-zero values')
    for v,c,l in ((b,COL['base'],'baseline'),(h,COL['reg'],'HIGH')): x=np.sort(v); ax[0,2].plot(np.maximum(x,1e-12),np.arange(1,len(x)+1)/len(x),c=c,label=l)
    ax[0,2].set_xscale('log');ax[0,2].set_xlabel(m+' (zeros plotted at 1e-12)');ax[0,2].set_ylabel('ECDF');ax[0,2].legend();ax[0,2].set_title('ECDF incl. zeros (plateau = fraction ≈0)')
    ax[1,0].boxplot([lg(bn),lg(hn)],tick_labels=['baseline','HIGH']); ax[1,0].set_ylabel('log10(metric)'); ax[1,0].set_title('Box plot, non-zero states')
    ax[1,1].violinplot([lg(bn),lg(hn)],showmedians=True); ax[1,1].set_xticks([1,2],['baseline','HIGH']); ax[1,1].set_ylabel('log10(metric)'); ax[1,1].set_title('Violin, non-zero states')
    red=100*(1-h[pos]/b[pos]); ax[1,2].hist(red.clip(-300,100),60,color='#2ca02c'); ax[1,2].axvline(0,c='k'); ax[1,2].axvline(red.median(),c='r',ls='--',label=f'median {red.median():.0f}%'); ax[1,2].set_xlabel('% reduction per state (clipped at −300%)'); ax[1,2].set_ylabel('# states'); ax[1,2].legend(); ax[1,2].set_title(f'% reduction, states with baseline>1e-12 (n={pos.sum()})')
    fig.suptitle(f'DBS {m}: {desc}',fontsize=12)
    save(fig,'03_bregman_geometry',f'DBS_bregman_{m}_baseline_vs_regularised',SRC,EXP,f'Baseline vs HIGH-regularised per-state distribution of {m}.',f'{z0:.0f}% of baseline and {z1:.0f}% of HIGH states have gap≈0 (smooth/linear region); among states with non-zero baseline gap the median reduction is {red.median():.0f}%.','Gap is zero for most states, so means are dominated by the non-zero minority; metric is measured on iterates visited, which differ between arms; many HIGH values reduced to ≈0 (median 100% is a ceiling).')
    rows.append(dict(metric=m,median_base=b.median(),median_HIGH=h.median(),mean_base=b.mean(),mean_HIGH=h.mean(),pct_states_lower=100*(h<b).mean(),median_pct_reduction=red.median(),wilcoxon_p=stats.wilcoxon(b,h).pvalue))
tab(pd.DataFrame(rows),'DBS_bregman_summary',SRC,'Per-state Bregman metric: baseline vs HIGH')
mm='agap_max_first2'
fig,ax=plt.subplots(2,4,figsize=(19,8.5))
for j,(xv,xl) in enumerate([('y','reference y_ref'),('u_prev','previous control u_prev'),('kk','MPC timestep k'),('rate_dist_base','distance of final u to rate/box limit (units of du_max)')]):
    for r_,(arm,c) in enumerate((('base',COL['base']),('HIGH',COL['reg']))):
        ax[r_,j].scatter(w[xv].values if xv!='rate_dist_base' else w[f'rate_dist_{arm}'],lg(w[f'{mm}_{arm}']),s=4,alpha=.25,c=c); ax[r_,j].set_xlabel(xl.replace('final u',f'final u ({arm})')); ax[r_,j].set_ylabel(f'log10 {mm} ({arm})')
fig.suptitle('DBS Bregman gap (E1 metric) vs state / control / timestep / rate-limit distance, baseline (top) and HIGH (bottom)')
save(fig,'03_bregman_geometry','DBS_bregman_vs_state_control_ratelimit',SRC,EXP,'Per-state gap against y_ref, u_prev, k and distance to the rate-limit bound.','Shows where in the (scalar) state space the gap is large and whether regularisation shifts it uniformly.','Rate-limit distance is derived (min(u−lo, hi−u)/du_max, du_max=0.002 read from iterate steps, not config).')
fig,ax=plt.subplots(1,3,figsize=(17,4.8))
for arm,c in (('base',COL['base']),('HIGH',COL['reg'])):
    ax[0].scatter(w[f'n_iter_{arm}']+np.random.default_rng(1).uniform(-.25,.25,len(w)),lg(w[f'{mm}_{arm}']),s=4,alpha=.2,c=c,label=arm)
ax[0].set_xlabel('CCP iterations (jittered; 15 = cap)');ax[0].set_ylabel('log10 agap_max_first2');ax[0].legend();ax[0].set_title('Gap vs CCP iterations')
data=[lg(w[w.cat==c][f'{mm}_base']) for c in CAT]; ax[1].boxplot(data,tick_labels=[NAME[c].replace(' ','\n') for c in CAT],showfliers=False); ax[1].set_ylabel('log10 gap (baseline model)'); ax[1].set_title('Baseline gap by stall transition group')
data=[lg(w[w.cat==c][f'{mm}_HIGH']) for c in CAT]; ax[2].boxplot(data,tick_labels=[NAME[c].replace(' ','\n') for c in CAT],showfliers=False); ax[2].set_ylabel('log10 gap (HIGH model)'); ax[2].set_title('HIGH gap by stall transition group')
save(fig,'03_bregman_geometry','DBS_bregman_vs_iterations_and_stall_status',SRC,EXP,'Gap vs CCP iteration count and vs stall transition group.','Stalled solves have systematically large first-two-transition gaps (partly circular: a 2-cycle is itself a large-gap event).','Circularity noted in PHASE5B_REPORT: first-2-transition features of a stall partly are the cycle.')
fig,ax=plt.subplots(1,3,figsize=(17,4.8))
for arm,c in (('base',COL['base']),('HIGH',COL['reg'])):
    ax[0].scatter(lg(w[f'acurv_h_{arm}']),lg(w[f'{mm}_{arm}']),s=4,alpha=.2,c=c,label=arm)
    ax[1].scatter(w[f'n_iter_{arm}']+np.random.default_rng(2).uniform(-.25,.25,len(w)),lg(w[f'acurv_h_{arm}']),s=4,alpha=.2,c=c)
ax[0].set_xlabel('log10 mean |secant curvature of h|');ax[0].set_ylabel('log10 agap_max_first2');ax[0].legend();ax[0].set_title('Curvature vs Bregman gap')
ax[1].set_xlabel('CCP iterations (jittered)');ax[1].set_ylabel('log10 mean |secant curvature h|');ax[1].set_title('Curvature vs CCP iterations')
for arm,c in (('base',COL['base']),('HIGH',COL['reg'])):
    q=pd.qcut(w[f'acurv_h_{arm}'].rank(method='first'),12); pr=w.groupby(q,observed=True).apply(lambda d:(d[f'n_iter_{arm}']>=15).mean()); xs=w.groupby(q,observed=True)[f'acurv_h_{arm}'].median()
    ax[2].plot(xs,100*pr.values,'o-',c=c,label=arm)
ax[2].set_xscale('symlog',linthresh=1e-9);ax[2].set_xlabel('median |secant curvature h| in decile bin');ax[2].set_ylabel('P(stall) %');ax[2].legend();ax[2].set_title('Curvature vs probability of stall (deciles)')
save(fig,'03_bregman_geometry','DBS_secant_curvature_vs_gap_iters_stall',SRC,EXP,'Per-state secant curvature of h along the CCP step (a state-level curvature proxy) against gap, iterations and stall probability.','Uses the logged secant curvature along the iterate path; this is a different definition from the model-level Hessian-Frobenius curvature in 02.','Secant curvature is measured on the visited iterates, so it is path-dependent; exploratory.')
def st(x): return pd.Series(dict(mean=x.mean(),median=x.median(),P90=x.quantile(.9),P95=x.quantile(.95),P99=x.quantile(.99),max=x.max(),std=x.std(),IQR=x.quantile(.75)-x.quantile(.25),n=len(x)))
g=pd.read_csv(P5+'safeguarded/solves_safeguarded.csv.gz'); g['arm']=g.arm.str.replace('dbs_both_lam','').replace({'0':'base'})
conds={'baseline':w.n_iter_base,'LOW 0.25':w.n_iter_LOW,'HIGH 0.788':w.n_iter_HIGH,'baseline+safeguard':w.sg_n_iter_base}
T=pd.DataFrame({k:st(v) for k,v in conds.items()}).T.reset_index().rename(columns={'index':'condition'}); T['scope']='all solves (cap=15 included as 15)'
Tc=pd.DataFrame({k:st(v[v<15]) for k,v in conds.items()}).T.reset_index().rename(columns={'index':'condition'}); Tc['scope']='converged-only (n_iter<15)'
tab(pd.concat([T,Tc]),'DBS_ccp_iteration_statistics',SRC,'CCP iteration statistics per condition (cap-censored at 15)')
CL=['#555555','#9ecae1','#1f77b4','#d62728']; ks=list(conds)
cf5={'baseline':('n_iter_base','converged_base'),'LOW 0.25':('n_iter_LOW','converged_LOW'),'HIGH 0.788':('n_iter_HIGH','converged_HIGH'),'baseline+safeguard':('sg_n_iter_base','sg_converged_base'),'HIGH 0.788+safeguard':('sg_n_iter_HIGH','sg_converged_HIGH')}
ks5=list(cf5); N=len(w)
cv={c:w[b].to_numpy().astype(bool) for c,(a,b) in cf5.items()}; nc={c:int((~cv[c]).sum()) for c in ks5}
conv={c:w[cf5[c][0]].to_numpy()[cv[c]] for c in ks5}; plotx={c:np.where(cv[c],w[cf5[c][0]].to_numpy(),16) for c in ks5}
assert N==4000 and list(nc.values())==[165,373,56,81,55] and all(conv[c].min()>=2 for c in ks5) and all(((w[cf5[c][0]]>=15)==(~cv[c])).all() for c in ks5[:3])
CL5=['#555555','#9ecae1','#1f77b4','#d62728','#6a3d9a']; lab={c:f'{c} (NC={nc[c]})' for c in ks5}
tl=[c.replace(' ','\n')+f'\nNC {nc[c]}' for c in ks5]
fig,ax=plt.subplots(2,3,figsize=(17,9))
for c,col in zip(ks5,CL5): ax[0,0].hist(plotx[c],bins=np.arange(1.5,17.5,1),alpha=.45,color=col,label=lab[c])
ax[0,0].set_yscale('log');ax[0,0].legend(fontsize=7);ax[0,0].set_xticks([2,4,6,8,10,12,14,16],['2','4','6','8','10','12','14','NC']);ax[0,0].set_xlabel('CCP iterations (NC = not converged)');ax[0,0].set_ylabel('# states (log)');ax[0,0].set_title('Histogram')
XMAX=15
for c,col in zip(ks5,CL5): x=np.sort(conv[c]); y=np.arange(1,len(x)+1)/N; ax[0,1].step(np.append(x,XMAX),np.append(y,y[-1]),c=col,label=lab[c])
ax[0,1].set_ylim(.9,1.001);ax[0,1].set_xlabel('CCP iterations (converged runs; ECDF over all 4000, NC never counted)');ax[0,1].set_ylabel('fraction of 4000 converged (zoom 0.9–1)');ax[0,1].legend(fontsize=7,loc='lower right');ax[0,1].set_title('ECDF: fraction of all 4000 instances converged by x iterations (upper tail)',fontsize=10)
ax[0,2].boxplot([conv[c] for c in ks5],tick_labels=tl);ax[0,2].set_ylabel('CCP iterations (converged runs only)');ax[0,2].set_title('Box plot (with outliers)')
ax[1,0].violinplot([conv[c] for c in ks5],showmedians=True);ax[1,0].set_xticks(range(1,6),tl);ax[1,0].set_ylabel('CCP iterations (converged runs only)');ax[1,0].set_title('Violin')
for a_ in (ax[0,2],ax[1,0]): a_.tick_params(axis='x',labelsize=8)
rj=np.random.default_rng(3);ax[1,1].scatter(w.n_iter_base+rj.uniform(-.25,.25,len(w)),w.n_iter_HIGH+rj.uniform(-.25,.25,len(w)),s=4,alpha=.25,c=COL['reg']);ax[1,1].plot([2,15],[2,15],'k--',label='y=x');ax[1,1].set_xlabel('baseline CCP iterations (15 = cap = NC)');ax[1,1].set_ylabel('HIGH CCP iterations (15 = cap = NC)');ax[1,1].legend();ax[1,1].set_title('Paired states (jittered): below line = improvement')
di=w.n_iter_HIGH-w.n_iter_base;ax[1,2].hist(di,bins=np.arange(-13.5,13.5,1),color=COL['reg']);ax[1,2].set_yscale('log');ax[1,2].axvline(0,c='k');ax[1,2].set_xlabel('Δ iterations = HIGH − baseline');ax[1,2].set_ylabel('# states (log)');ax[1,2].set_title(f'Per-state difference (improved {100*(di<0).mean():.1f}%, same {100*(di==0).mean():.1f}%, worse {100*(di>0).mean():.1f}%)')
fig.suptitle(f'DBS CCP iteration distributions, n={N} states per condition (same states in every condition; safeguard = proximal + backtracking)')
save(fig,'04_ccp_iterations','DBS_ccp_iterations_distributions',SRC,EXP,'Histogram/ECDF/box/violin/paired scatter/Δ histogram for CCP iterations in five conditions (baseline, LOW, HIGH, baseline+safeguard, HIGH+safeguard); NC = not converged (converged flag), shown as a separate bar/count.','Not-converged counts of 4000 states: baseline 165, LOW 373, HIGH 56, baseline+safeguard 81, HIGH+safeguard 55; converged runs have median 2 iterations in every condition. Safeguarded NC runs are mostly backtrack_exhausted at 1-5 iterations, not cap hits.','NC uses the converged flag (cap and NC coincide only without the safeguard). Box/violin/ECDF use converged runs only. Panels 5-6 (HIGH vs baseline, no safeguard) unchanged.')
fig,ax=plt.subplots(1,3,figsize=(17,4.8))
pi_=100*(w.n_iter_base-w.n_iter_HIGH)/w.n_iter_base;ax[0].hist(pi_,60,color='#2ca02c');ax[0].set_yscale('log');ax[0].axvline(0,c='k');ax[0].set_xlabel('% iteration improvement (base→HIGH)');ax[0].set_ylabel('# states (log)');ax[0].set_title('Percentage improvement per state')
w['kbin']=pd.cut(w.kk,10);gk=w.groupby('kbin',observed=True)
xs=[i.mid for i in gk.size().index]
for lab,col,c in (('baseline','n_iter_base',COL['base']),('HIGH','n_iter_HIGH',COL['reg'])):
    ax[1].plot(xs,gk[col].mean(),'o-',c=c,label=f'{lab} mean');ax[1].plot(xs,gk[col].quantile(.95),'s--',c=c,alpha=.6,label=f'{lab} P95')
ax[1].set_xlabel('MPC timestep k (bin centre)');ax[1].set_ylabel('CCP iterations');ax[1].legend(fontsize=8);ax[1].set_title('Iterations by MPC timestep')
w.boxplot(column='n_iter_base',by='seed',ax=ax[2],positions=np.arange(8)-.18,widths=.3,showfliers=False,grid=False);w.boxplot(column='n_iter_HIGH',by='seed',ax=ax[2],positions=np.arange(8)+.18,widths=.3,showfliers=False,grid=False,color='#1f77b4');ax[2].set_title('Iterations by seed (trajectory set; grey=baseline, blue=HIGH)');ax[2].set_xlabel('seed');ax[2].set_ylabel('CCP iterations');ax[2].set_xticks(range(8)); fig.suptitle('')
save(fig,'04_ccp_iterations','DBS_ccp_iterations_by_timestep_seed_improvement',SRC,EXP,'Improvement %, iterations by MPC timestep and by seed.','Iterations are a weak function of k; the stall tail is spread across seeds with some seed concentration.','Each seed/scenario is a fixed state bank of 500 replayed states, not a closed-loop rollout.')
fig,ax=plt.subplots(1,2,figsize=(11,4.4)); lab=['baseline','LOW','HIGH','base+SG']
ax[0].bar(lab,[conds[k].mean() for k in ks],color=CL);ax[0].set_ylabel('mean CCP iterations');ax[0].set_title('incl. stalls (as 15)')
ax[1].bar(lab,[conds[k][conds[k]<15].mean() for k in ks],color=CL);ax[1].set_ylabel('mean CCP iterations');ax[1].set_title('converged solves only (conditional)')
for a,vals in zip(ax,([conds[k].mean() for k in ks],[conds[k][conds[k]<15].mean() for k in ks])):
    for i,v in enumerate(vals): a.text(i,v+.03,f'{v:.2f}',ha='center')
save(fig,'04_ccp_iterations','DBS_mean_iterations_with_without_stalls',SRC,EXP,'Mean CCP iterations with stalls counted at the cap vs among converged solves only.','Most of the apparent mean-iteration gain is the removed stalls; conditional on converging the means barely differ.','Conditioning on convergence compares different subsets of states; the cap treats a stall as 15, an arbitrary censor.')
feat=[('y','reference y_ref'),('u_prev','u_prev'),('kk','timestep k'),('agap_max_first2_base','baseline gap E1'),('Dg_mean_first_base','baseline Dg (1st it)'),('Dh_mean_first_base','baseline Dh (1st it)'),('step_mean_first_base','baseline mean |step| first'),('acurv_h_base','baseline |secant curv h|'),('acurv_g_base','baseline |secant curv g|'),('frac_rate_base','baseline frac iters rate-limit active'),('rate_dist_base','dist to limit (baseline u_sol)'),('n_iter_base','baseline CCP iters'),('n_iter_LOW','LOW CCP iters'),('agap_max_first2_HIGH','HIGH gap E1'),('acurv_h_HIGH','HIGH |secant curv h|'),('n_iter_HIGH','HIGH CCP iters'),('act_gh_all_rep_f2_base','baseline ReLU-unit repeat flips (first 2)'),('kink_frac_base','baseline kink fraction')]
feat=[(c,l) for c,l in feat if c in w]
rows=[]
def auc(a,b):
    a=a.dropna();b=b.dropna()
    if len(a)<3 or len(b)<3: return np.nan,np.nan
    u=stats.mannwhitneyu(a,b); return u.statistic/(len(a)*len(b)),u.pvalue
for c,l in feat:
    r=dict(feature=c,desc=l)
    for g1,g2 in (('C_induced','D_persistent_conv'),('A_rescued','D_persistent_conv'),('C_induced','A_rescued'),('B_persistent_stall','A_rescued')):
        A,p=auc(w[w.cat==g1][c],w[w.cat==g2][c]); r[f'AUC_{g1[0]}_vs_{g2[0]}']=A; r[f'p_{g1[0]}_vs_{g2[0]}']=p
    for g1 in CAT: r[f'median_{g1[0]}']=w[w.cat==g1][c].median()
    rows.append(r)
GT=pd.DataFrame(rows); tab(GT,'DBS_stall_group_feature_comparison',SRC+'; scenario_categories.csv','Median per group and Mann-Whitney AUC (0.5=no separation) for C-vs-D, A-vs-D, C-vs-A, B-vs-A (exploratory, uncorrected p)')
fn=[c for c,l in feat]; nc=4; nr=int(np.ceil(len(feat)/nc))
fig,ax=plt.subplots(nr,nc,figsize=(19,3.6*nr))
for a,(c,l) in zip(ax.ravel(),feat):
    d=[w[w.cat==k][c].dropna() for k in CAT]
    if 'agap' in c or 'acurv' in c or c.startswith('D') or 'step' in c: d=[lg(x) for x in d]; l='log10 '+l
    a.boxplot(d,tick_labels=['D','A','B','C'],showfliers=False);a.set_title(l,fontsize=9)
for a in ax.ravel()[len(feat):]: a.axis('off')
fig.suptitle('DBS: features by stall-transition group (D=converge both n=3791, A=rescued n=153, B=persistent n=12, C=new stall n=44)')
save(fig,'10_additional_diagnostics','DBS_group_feature_boxplots',SRC,EXP,'All logged state/geometry/iterate features split by A/B/C/D transition group.','See AUC table: rescued (A) states are clearly different from D; the question is whether C separates from D and A.','B and C are small (12, 44); p-values exploratory/uncorrected; many features are outcome-correlated (circular for stall-derived ones).')
print(GT[['feature','AUC_C_vs_D','AUC_A_vs_D','AUC_C_vs_A','AUC_B_vs_A']].round(2).to_string())
fig,ax=plt.subplots(2,3,figsize=(17,8))
for a,(c,l) in zip(ax.ravel(),[('y','reference y_ref'),('u_prev','u_prev'),('kk','timestep k'),('agap_max_first2_base','baseline gap E1'),('acurv_h_base','baseline |secant curv h|'),('rate_dist_base','dist to rate/box limit (base)')]):
    for k in CAT:
        x=np.sort(w[w.cat==k][c].dropna()); a.step(x,np.arange(1,len(x)+1)/len(x),c=COL[k[0]],label=f'{NAME[k]} (n={len(x)})')
    if 'agap' in c or 'acurv' in c: a.set_xscale('symlog',linthresh=1e-9)
    a.set_xlabel(l);a.set_ylabel('ECDF')
ax[0,0].legend(fontsize=7)
save(fig,'10_additional_diagnostics','DBS_new_stall_group_ecdfs',SRC,EXP,'ECDFs of key features for D/A/B/C groups.','Compares helped (A) vs harmed (C) states; overlapping ECDFs = no separation.','Small B and C groups.')
fig,ax=plt.subplots(2,3,figsize=(17,9)); tm={'baseline':w.wall_s_base,'LOW 0.25':w.wall_s_LOW,'HIGH 0.788':w.wall_s_HIGH,'baseline+safeguard':w.sg_wall_s_base}
bins=np.logspace(-3,0,60)
for (k,v),c in zip(tm.items(),CL): ax[0,0].hist(v,bins,alpha=.45,color=c,label=k); x=np.sort(v); ax[0,1].plot(x,np.arange(1,len(x)+1)/len(x),c=c,label=k)
ax[0,0].set_xscale('log');ax[0,0].set_xlabel('wall time per solve (s)');ax[0,0].set_ylabel('# solves');ax[0,0].legend(fontsize=8);ax[0,0].set_title('Solve-time histogram')
ax[0,1].set_xscale('log');ax[0,1].set_xlabel('wall time (s)');ax[0,1].set_ylabel('ECDF');ax[0,1].set_title('ECDF')
for arm,c in (('base',COL['base']),('HIGH',COL['reg'])): ax[0,2].scatter(w[f'n_iter_{arm}']+np.random.default_rng(4).uniform(-.25,.25,len(w)),w[f'wall_s_{arm}'],s=4,alpha=.25,c=c,label=arm)
ax[0,2].set_yscale('log');ax[0,2].set_xlabel('CCP iterations (jittered)');ax[0,2].set_ylabel('wall time (s)');ax[0,2].legend();ax[0,2].set_title('Solve time vs CCP iterations')
ax[1,0].scatter(w.wall_s_base,w.wall_s_HIGH,s=4,alpha=.25,c=COL['reg']);m=[1e-3,.5];ax[1,0].plot(m,m,'k--',label='y=x');ax[1,0].set_xscale('log');ax[1,0].set_yscale('log');ax[1,0].set_xlabel('baseline wall (s)');ax[1,0].set_ylabel('HIGH wall (s)');ax[1,0].legend();ax[1,0].set_title('Paired baseline vs regularised')
ax[1,1].scatter(w.wall_s_base,w.sg_wall_s_base,s=4,alpha=.25,c=COL['sg']);ax[1,1].plot(m,m,'k--',label='y=x');ax[1,1].set_xscale('log');ax[1,1].set_yscale('log');ax[1,1].set_xlabel('baseline wall (s)');ax[1,1].set_ylabel('baseline + safeguard wall (s)');ax[1,1].legend();ax[1,1].set_title('Paired baseline vs safeguard')
ax[1,2].boxplot([w[w.cat==k].wall_s_base for k in CAT],tick_labels=['D','A','B','C'],showfliers=False);ax[1,2].set_ylabel('baseline wall time (s)');ax[1,2].set_title('Baseline solve time by group (stalls = A,B)')
fig.suptitle('DBS wall-clock per solve (instrumented single-machine run; not a controlled benchmark)')
save(fig,'10_additional_diagnostics','DBS_solve_time_overview',SRC,EXP,'Solve-time distribution, vs iterations, and paired baseline vs HIGH / safeguard.','Time tracks iteration count; the safeguard costs more per solve (backtracking rebuilds), HIGH is not cheaper per solve except via fewer stall iterations.','Single run on one machine; phase-6b timing benchmark is marked INVALID (load gate), so no speed claim is made. No MPC deadline is defined in the setup, so no deadline plots.')
tt=pd.DataFrame({k:st(v) for k,v in tm.items()}).T.reset_index().rename(columns={'index':'condition'}); tab(tt,'DBS_solve_time_statistics',SRC,'Wall-time statistics (s)')
g0=g[g.arm=='base']
fig,ax=plt.subplots(1,2,figsize=(11,4.4)); ax[0].scatter(g0.n_backtrack,g0.wall_s,s=5,alpha=.3,c=COL['sg']);ax[0].set_yscale('log');ax[0].set_xlabel('# backtracking steps (safeguard, baseline arm)');ax[0].set_ylabel('wall (s)');ax[0].set_title('Backtracks vs solve time')
ax[1].scatter(g0.n_backtrack,g0.build_s/g0.wall_s,s=5,alpha=.3,c=COL['sg']);ax[1].set_xlabel('# backtracking steps');ax[1].set_ylabel('build_s / wall_s');ax[1].set_title('Model (re)build share of solve time')
save(fig,'10_additional_diagnostics','DBS_safeguard_backtracks_vs_time',SRC,EXP,'Number of backtracking steps against safeguarded solve time and rebuild share.','Overhead scales with backtracks/rebuilds.','Per-solve timers from one run.')
w['d_it']=w.n_iter_HIGH-w.n_iter_base; w['d_gap']=w.agap_max_first2_HIGH-w.agap_max_first2_base; w['d_time']=w.wall_s_HIGH-w.wall_s_base
w['d_it_rel']=100*w.d_it/w.n_iter_base; w['d_time_rel']=100*w.d_time/w.wall_s_base
fig,ax=plt.subplots(2,3,figsize=(17,9))
ax[0,0].hist(w.d_it,np.arange(-13.5,13.5),color=COL['reg']);ax[0,0].set_yscale('log');ax[0,0].set_xlabel('Δ iterations');ax[0,0].set_title('Δ iterations (HIGH−base)')
ax[0,1].hist(np.sign(w.d_gap)*np.log10(1+np.abs(w.d_gap)/1e-6),60,color=COL['reg']);ax[0,1].set_xlabel('signed log10(1+|Δ gap|/1e-6)');ax[0,1].set_title('Δ gap E1 (HIGH−base), signed-log')
ax[0,2].hist(w.d_time.clip(-.1,.1),80,color=COL['reg']);ax[0,2].set_xlabel('Δ wall (s), clipped ±0.1');ax[0,2].set_title('Δ solve time')
ax[1,0].scatter(w.d_gap,w.d_it+np.random.default_rng(5).uniform(-.2,.2,len(w)),s=4,alpha=.25,c=COL['reg']);ax[1,0].set_xscale('symlog',linthresh=1e-6);ax[1,0].axhline(0,c='k',lw=.7);ax[1,0].axvline(0,c='k',lw=.7);ax[1,0].set_xlabel('Δ gap E1');ax[1,0].set_ylabel('Δ iterations (jittered)');ax[1,0].set_title('Δ iterations vs Δ Bregman gap')
ax[1,1].scatter(w.d_it+np.random.default_rng(6).uniform(-.2,.2,len(w)),w.d_time,s=4,alpha=.25,c=COL['reg']);ax[1,1].axhline(0,c='k',lw=.7);ax[1,1].axvline(0,c='k',lw=.7);ax[1,1].set_xlabel('Δ iterations');ax[1,1].set_ylabel('Δ wall (s)');ax[1,1].set_title('Δ time vs Δ iterations')
sc=ax[1,2].scatter(w.y,w.u_prev,c=w.d_it.clip(-5,5),cmap='RdBu_r',s=8,vmin=-5,vmax=5);fig.colorbar(sc,ax=ax[1,2],label='Δ iterations (clipped ±5; blue = fewer under HIGH)');ax[1,2].set_xlabel('y_ref');ax[1,2].set_ylabel('u_prev');ax[1,2].set_title('Per-state improvement map')
fig.suptitle(f'DBS per-state deltas (HIGH − baseline), n={len(w)} states')
save(fig,'10_additional_diagnostics','DBS_per_state_deltas',SRC,EXP,'Distributions of per-state Δ iterations/Δ gap/Δ time and their relations; improvement map over (y_ref,u_prev).','The mean hides a split: a few hundred states improve a lot (stall rescues) and some regress (new stalls); the vast majority are unchanged.','Δ gap compares metrics measured on different iterate paths.')
dd=pd.DataFrame({'metric':['d_it','d_gap','d_time'],'mean':[w.d_it.mean(),w.d_gap.mean(),w.d_time.mean()],'median':[w.d_it.median(),w.d_gap.median(),w.d_time.median()],'pct_improved':[100*(w.d_it<0).mean(),100*(w.d_gap<0).mean(),100*(w.d_time<0).mean()],'pct_worse':[100*(w.d_it>0).mean(),100*(w.d_gap>0).mean(),100*(w.d_time>0).mean()]})
tab(dd,'DBS_per_state_delta_summary',SRC,'Δ(HIGH−base) summaries'); print(dd)
w[key+['cat','d_it','d_gap','d_time','d_it_rel','d_time_rel']].pipe(lambda d:tab(d,'DBS_per_state_deltas',SRC,'Per-state deltas, 4000 rows keyed by (seed,bank_idx)'))
rows=[];fig,ax=plt.subplots(2,3,figsize=(17,9))
def binned(a,x,y,xl,yl,title,logx=True):
    q=pd.qcut(x.rank(method='first'),15);gb=pd.DataFrame({'x':x,'y':y}).groupby(q,observed=True)
    xs=gb.x.median();ys=gb.y.mean();se=gb.y.sem()
    a.errorbar(xs,ys,2*se,fmt='o-',c='k');a.set_xlabel(xl);a.set_ylabel(yl);a.set_title(title,fontsize=10)
    if logx:a.set_xscale('symlog',linthresh=1e-9)
for a,(xc,yc,xl,yl) in zip(ax.ravel(),[('agap_max_first2_base','n_iter_base','baseline gap E1','mean CCP iters (decile bins)'),('acurv_h_base','n_iter_base','baseline |sec. curv h|','mean CCP iters'),('agap_max_first2_base','wall_s_base','baseline gap E1','mean wall (s)'),('agap_max_first2_base','stall_base','baseline gap E1','P(stall)'),('rate_dist_base','stall_base','dist to limit (units du_max)','P(stall)'),('agap_max_first2_base','rate_dist_base','baseline gap E1','mean dist to limit')]):
    binned(a,w[xc],w[yc].astype(float),xl,yl,f'{yl} vs {xl}',logx=xc.startswith(('agap','acurv')))
    r,p=stats.spearmanr(w[xc],w[yc].astype(float),nan_policy='omit');rows.append(dict(x=xc,y=yc,spearman_rho=r,p=p,n=len(w)))
fig.suptitle('DBS geometry vs solver difficulty (binned means ±2 s.e., 15 equal-count bins); association, not causation')
save(fig,'10_additional_diagnostics','DBS_geometry_vs_difficulty_binned',SRC,EXP,'Binned relations between geometry metrics/rate-limit distance and CCP iterations, wall time, P(stall).','Reports whether larger gaps/curvature go with more iterations or stall probability (Spearman table in tables/).','Baseline-model quantities measured on the solve itself (partly circular for stalls); correlation only.')
for arm in ['HIGH']:
    for xc,yc in [('agap_max_first2_HIGH','n_iter_HIGH'),('acurv_h_HIGH','n_iter_HIGH'),('agap_max_first2_HIGH','wall_s_HIGH')]:
        r,p=stats.spearmanr(w[xc],w[yc],nan_policy='omit');rows.append(dict(x=xc,y=yc,spearman_rho=r,p=p,n=len(w)))
CR=pd.DataFrame(rows);tab(CR,'DBS_correlations_spearman',SRC,'Spearman rank correlations');print(CR.round(3).to_string())
fig,ax=plt.subplots(1,3,figsize=(17,4.8))
for a,(xc,yc,yl) in zip(ax,[('agap_max_first2_base','n_iter_base','CCP iters'),('acurv_h_base','n_iter_base','CCP iters'),('agap_max_first2_base','wall_s_base','wall (s)')]):
    hb=a.hexbin(lg(w[xc]),w[yc],gridsize=40,bins='log',cmap='viridis',mincnt=1,yscale='log' if 'wall' in yc else 'linear');a.set_xlabel('log10 '+xc);a.set_ylabel(yl);fig.colorbar(hb,ax=a,label='log10 count')
save(fig,'10_additional_diagnostics','DBS_geometry_difficulty_hexbin',SRC,EXP,'Hexbin density: gap/curvature vs iterations/time (baseline).','Dense core shows most states are low-gap, low-iteration; stalls form a separate high-iteration band.','Exploratory.')
