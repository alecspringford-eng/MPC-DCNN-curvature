import os; os.environ['PART']='2'
from common import *
w=pd.read_pickle(O+'.w.pkl'); mech=pd.read_pickle(O+'.mech.pkl')
SRC=P5+'{current/iters_current,safeguarded/iters_safeguarded}.csv.gz + solves_current (u_prev)'
EXP='DBS both-branch; baseline λ=0 vs HIGH λ=0.788 vs baseline+proximal/backtracking safeguard; rate limit du=±0.002 about u_prev, u_min=0'
it=pd.read_csv(P5+'current/iters_current.csv.gz'); its=pd.read_csv(P5+'safeguarded/iters_safeguarded.csv.gz')
for d in (it,its): d['arm']=d.arm.str.replace('dbs_both_lam','').replace({'0':'base'})
def tr(df,arm,sd,bi): return df[(df.arm==arm)&(df.seed==sd)&(df.bank_idx==bi)].sort_values('it')
DU=0.002
def panel_u(ax,t,up,title,col):
    lo=max(0,up-DU); hi=up+DU
    ax.axhspan(hi,hi+.0008,color='#ddd'); ax.axhspan(lo-.0008 if lo>0 else -.0008,lo,color='#ddd')
    ax.axhline(hi,c='k',ls='--',lw=1,label=f'rate limit u_prev+{DU}'); ax.axhline(lo,c='k',ls=':',lw=1,label='lower bound (rate limit / u_min=0)')
    ax.plot(t.it,t.u_next,'o-',c=col,ms=5,label='u after iteration'); ax.set_xlabel('CCP iteration'); ax.set_ylabel('control u'); ax.set_title(title,fontsize=10)
def case_fig(r,name,note=''):
    sd,bi=int(r.seed),int(r.bank_idx); up=w[(w.seed==sd)&(w.bank_idx==bi)].u_prev.iloc[0]; t=tr(it,'base',sd,bi)
    fig,ax=plt.subplots(2,3,figsize=(16,8))
    panel_u(ax[0,0],t,up,'A↔B alternation: control vs CCP iteration',COL['base']); ax[0,0].legend(fontsize=7)
    ax[0,1].bar(t.it,t.d,color=np.where(t.d>0,'#d62728','#1f77b4')); ax[0,1].axhline(0,c='k'); ax[0,1].set_xlabel('CCP iteration'); ax[0,1].set_ylabel('step d = u_next − u_t'); ax[0,1].set_title('Step-to-step change (sign flips each iteration)',fontsize=10)
    ax[0,2].plot(t.it,t.J_surr,'o-',c='purple'); ax[0,2].set_xlabel('CCP iteration'); ax[0,2].set_ylabel('surrogate objective J_surr'); ax[0,2].set_title('Objective alternates between two values',fontsize=10)
    ax[1,0].plot(t.u_t,t.u_next,'o-',c=COL['base'],alpha=.7); ax[1,0].plot([t.u_t.min(),t.u_t.max()],[t.u_t.min(),t.u_t.max()],'k--',lw=.8,label='u_next = u_t (fixed point)')
    ax[1,0].annotate('',xy=(t.u_next.iloc[1],t.u_next.iloc[2]),xytext=(t.u_next.iloc[0],t.u_next.iloc[1]),arrowprops=dict(arrowstyle='->',color='r'))
    ax[1,0].set_xlabel('u_t'); ax[1,0].set_ylabel('u_next'); ax[1,0].set_title('Trajectory through control space (2 points = 2-cycle)',fontsize=10); ax[1,0].legend(fontsize=7)
    ax[1,1].plot(t.it,t.Dg,'o-',label='Dg'); ax[1,1].plot(t.it,t.Dh,'s-',label='Dh'); ax[1,1].set_xlabel('CCP iteration'); ax[1,1].set_ylabel('directional Bregman gap'); ax[1,1].legend(); ax[1,1].set_title('Bregman gaps of g and h along the step',fontsize=10)
    flags=np.vstack([t.rate_active,t.at_umin,t.at_umax]).astype(float); ax[1,2].imshow(flags,aspect='auto',cmap='Greys',extent=[t.it.min()-.5,t.it.max()+.5,3,0]); ax[1,2].grid(False); ax[1,2].set_yticks([.5,1.5,2.5],['rate limit active','at u_min','at u_max']); ax[1,2].set_xlabel('CCP iteration'); ax[1,2].set_title('Active constraints per iteration (black=active)',fontsize=10)
    fig.suptitle(f'DBS baseline stall, seed {sd}, bank_idx {bi}, u_prev={up:.4f} ({r.rescue_group}) {note}',fontsize=13)
    save(fig,'06_oscillation_cases',name,SRC,EXP,'Baseline CCP iterates for one stalled state: control, step, objective, gaps, constraint activity.','Control alternates between two vertices every iteration, the step flips sign and J_surr takes only two values: a two-vertex limit cycle.','Single state; Dg/Dh are the logged per-iterate directional Bregman gaps of the learned g,h.')
good=mech[mech.two_cycle_exact].copy(); good['key']=list(zip(good.seed,good.bank_idx))
picks=[]
for sd in range(8):
    c=good[(good.seed==sd)&(good.rescue_group=='both')].sort_values('Jamp')
    if len(c): picks.append(c.iloc[len(c)//2])
picks=picks[:6]
for r in picks: case_fig(r,f'DBS_two_vertex_oscillation_case_s{int(r.seed)}_b{int(r.bank_idx)}')
fig,axs=plt.subplots(2,3,figsize=(16,7),sharey=False)
for a,r in zip(axs.ravel(),picks):
    up=w[(w.seed==r.seed)&(w.bank_idx==r.bank_idx)].u_prev.iloc[0]; panel_u(a,tr(it,'base',int(r.seed),int(r.bank_idx)),up,f'seed {int(r.seed)}, bank {int(r.bank_idx)}',COL['base'])
fig.suptitle('Six baseline stalls (λ=0): each is a two-vertex alternation A→B→A→B between the rate-limit bounds');
save(fig,'06_oscillation_cases','DBS_two_vertex_oscillation_gallery',SRC,EXP,'Six representative baseline stalls, control vs CCP iteration, with rate-limit bounds.','All six are exact period-2 cycles between u_prev−0.002 (or 0) and u_prev+0.002, never settling in 15 iterations.','Cases are median-amplitude picks per seed, not random draws; 150/152 + 12/12 of rescued stalls pass the same exact test.')
cand=good[good.rescue_group=='both'].copy()
def hero_ok(r):
    return len(tr(it,'HIGH',r.seed,r.bank_idx))>0 and len(tr(its,'base',r.seed,r.bank_idx))>0
hero=None
for _,r in cand.sort_values('Jamp',ascending=False).iterrows():
    if hero_ok(r): hero=r;break
def hero_fig(r,name,title):
    sd,bi=int(r.seed),int(r.bank_idx); up=w[(w.seed==sd)&(w.bank_idx==bi)].u_prev.iloc[0]
    conds=[('Baseline (λ=0)',tr(it,'base',sd,bi),COL['base']),('Joint geometry reg. (HIGH 0.788)',tr(it,'HIGH',sd,bi),COL['reg']),('Baseline + proximal/backtracking safeguard',tr(its,'base',sd,bi),COL['sg'])]
    fig,ax=plt.subplots(2,3,figsize=(16,7.5),sharex=True)
    for j,(nm,t,c) in enumerate(conds):
        panel_u(ax[0,j],t,up,f'{nm}\n{len(t)} CCP iterations',c)
        ax[1,j].plot(t.it,t.J_surr,'o-',c=c); ax[1,j].set_xlabel('CCP iteration'); ax[1,j].set_ylabel('J_surr' if j==0 else '')
    ax[0,0].legend(fontsize=7); fig.suptitle(title+f'  (seed {sd}, bank_idx {bi}, same initial state)',fontsize=13)
    save(fig,'06_oscillation_cases',name,SRC,EXP,'Same initial state solved with baseline, joint geometry regularisation and the safeguard.','Baseline alternates for 15 iterations; both interventions break the cycle and converge in a few iterations.','One state chosen as a clear example (largest objective amplitude among "rescued by both"), so it illustrates the mechanism and is not an estimate of typical behaviour.')
if hero is not None:
    hero_fig(hero,'DBS_two_vertex_oscillation_hero_comparison','Same state: baseline stall vs geometry regularisation vs safeguard')
    for r in cand.sort_values('Jamp',ascending=False).iloc[1:4].itertuples():
        if hero_ok(r): hero_fig(r,f'DBS_comparison_case_s{int(r.seed)}_b{int(r.bank_idx)}','Same state under three conditions')
ex=mech[(~mech.two_cycle_exact)|(mech.rescue_group=='geometry_only')]
print('exceptional',ex[['seed','bank_idx','rescue_group','two_cycle_exact']].to_dict('records'))
for r in ex.itertuples():
    sd,bi=int(r.seed),int(r.bank_idx); up=w[(w.seed==sd)&(w.bank_idx==bi)].u_prev.iloc[0]
    fig,ax=plt.subplots(1,3,figsize=(16,4))
    for a,(nm,t,c) in zip(ax,[('baseline',tr(it,'base',sd,bi),COL['base']),('HIGH geometry',tr(it,'HIGH',sd,bi),COL['reg']),('safeguard',tr(its,'base',sd,bi),COL['sg'])]):
        panel_u(a,t,up,f'{nm}: {len(t)} its',c)
    fig.suptitle(f'Exceptional case: s{sd} b{bi}, group={r.rescue_group}, exact 2-cycle={r.two_cycle_exact}')
    save(fig,'07_regularizer_vs_safeguard',f'DBS_exceptional_case_s{sd}_b{bi}',SRC,EXP,'Exception to "rescued by both / exact two-cycle".','Shows how the exceptional state behaves under baseline, geometry reg. and safeguard.','Single case.')
g_=mech.groupby('rescue_group').two_cycle_exact.agg(['sum','count']).reset_index(); g_['pct']=100*g_['sum']/g_['count']
tab(g_,'DBS_two_cycle_by_rescue_group',SRC,'Number of baseline stalls with exact two-vertex cycle by rescue group')
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
b=ax[0].bar(g_.rescue_group,g_['count'],color='#ccc',label='all'); ax[0].bar(g_.rescue_group,g_['sum'],color=COL['sg'],label='exact two-vertex cycle'); ax[0].legend(); ax[0].set_ylabel('baseline stalls'); ax[0].set_title('Two-vertex mechanism in each rescue group')
for i,r in g_.iterrows(): ax[0].text(i,r['count']+3,f"{r['sum']}/{r['count']}",ha='center')
ax[1].hist(mech.amp_u,bins=30,color=COL['base']); ax[1].set_xlabel('peak-to-peak u over last 6 iterations'); ax[1].set_ylabel('# baseline stalls'); ax[1].set_title('Cycle amplitude (≈0.004 = 2·du_max)')
save(fig,'07_regularizer_vs_safeguard','DBS_overlap_shared_mechanism',SRC,EXP,'Do cases fixed by both methods share the two-vertex mechanism?','Yes: 150/152 "both", 12/12 "safeguard-only" are exact two-cycles; overall 162/165. The large overlap is explained by a single shared failure mode that both a trust-region step and a flatter-curvature model suppress.','The test is an operational criterion (last 6 iterates period-2, |d|>1e-3); "why" is inferred from the logs, not proven causally. The 1 geometry-only case is not a two-cycle.')
