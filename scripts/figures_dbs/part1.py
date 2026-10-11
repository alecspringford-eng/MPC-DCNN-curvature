import os; os.environ['PART']='1'
from common import *
from matplotlib.patches import Circle
s,g,w=load()
SRC='phase5_dbs_joint_curvature/current/solves_current.csv.gz; phase5b_stall_mechanism/tables/{scenario_categories,solve_features}'
EXP='DBS, both-branch geometry reg. (lambda=0 vs HIGH 0.788; LOW 0.25), N=1 surrogate plant, 8 seeds x 500 states = 4000 paired states'
bs=w.stall_base
w['sg_rescued']=bs & w.sg_rescued_flag
w['geo_rescued']=bs & ~w.stall_HIGH
assert bs.sum()==165 and w.geo_rescued.sum()==153 and w.sg_rescued.sum()==164
assert (w.cat=='C_induced').sum()==44 and (w.cat=='B_persistent_stall').sum()==12
both=(w.geo_rescued&w.sg_rescued).sum(); assert both==152, both
assert (w.geo_rescued&~w.sg_rescued).sum()==1 and (~w.geo_rescued&w.sg_rescued&bs).sum()==12
NAME={'A_rescued':'A rescued (base stall→conv)','B_persistent_stall':'B stalls under both','C_induced':'C new stall (conv→stall)','D_persistent_conv':'D converges under both'}
n=len(w)
tab(pd.DataFrame({'arm':['baseline','LOW 0.25','HIGH 0.788'],'stalls':[w.stall_base.sum(),w.stall_LOW.sum(),w.stall_HIGH.sum()],'n':n,
  'stall_rate_pct':[100*w.stall_base.mean(),100*w.stall_LOW.mean(),100*w.stall_HIGH.mean()],
  'abs_reduction':[0,w.stall_base.sum()-w.stall_LOW.sum(),w.stall_base.sum()-w.stall_HIGH.sum()],
  'rel_reduction_pct':[0,100*(1-w.stall_LOW.sum()/165),100*(1-w.stall_HIGH.sum()/165)]}),'DBS_stall_statistics',SRC,'Stall counts/rates per arm (stall = n_iter>=15 CCP cap)')
tr=w.cat.value_counts().rename_axis('category').reset_index(name='count'); tr['pct_of_4000']=100*tr['count']/n
tab(tr,'DBS_matched_stall_transitions',SRC,'4-way matched-state classification baseline vs HIGH')
cols=['seed','bank_idx','cat','ref_base','u_prev_base','k_base','n_iter_base','n_iter_HIGH','n_iter_LOW','sg_n_iter_base' if 'sg_n_iter_base' in w else 'sg_n_iter_base']
cols=['seed','bank_idx','cat','y','u_prev','k_base','scen_base','n_iter_base','n_iter_LOW','n_iter_HIGH','sg_n_iter_base','wall_s_base','wall_s_HIGH','wall_s_sg_dummy'] if False else ['seed','bank_idx','cat','y','u_prev','k_base','scen_base','n_iter_base','n_iter_LOW','n_iter_HIGH','sg_n_iter_base','wall_s_base','wall_s_HIGH']
for c,nm in (('A_rescued','rescued_stall_cases'),('C_induced','newly_introduced_stall_cases'),('B_persistent_stall','persistent_stall_cases')):
    tab(w[w.cat==c][cols],'DBS_'+nm,SRC,nm.replace('_',' ')+' (seed,bank_idx trace to solves files)')
ov=pd.DataFrame({'group':['both','geometry only','safeguard only','neither'],
 'count':[both,(w.geo_rescued&~w.sg_rescued).sum(),(~w.geo_rescued&w.sg_rescued&bs).sum(),(bs&~w.geo_rescued&~w.sg_rescued).sum()]})
ov['pct_of_165']=100*ov['count']/165
tab(ov,'DBS_regulariser_safeguard_overlap',SRC+'; safeguard rescue = sg_converged','Overlap of baseline stalls rescued by geometry reg. and by proximal/backtracking safeguard')
w['rescue_group']=np.select([w.geo_rescued&w.sg_rescued,w.geo_rescued,w.sg_rescued&bs,bs],['both','geometry_only','safeguard_only','neither'],'not_baseline_stall')
tab(w[w.stall_base][['seed','bank_idx','rescue_group','cat','y','u_prev','n_iter_base','n_iter_HIGH','sg_n_iter_base']],'DBS_rescue_cases_by_group',SRC,'Per-stall rescue group, 165 rows')
w.to_pickle(O+'.w.pkl')
fig,ax=plt.subplots(1,2,figsize=(10,4.2))
lab=['baseline\n(λ=0)','LOW\n(0.25)','HIGH\n(0.788)']; v=[w.stall_base.sum(),w.stall_LOW.sum(),w.stall_HIGH.sum()]
b=ax[0].bar(lab,v,color=[COL['base'],'#9ecae1',COL['reg']]); ax[0].bar_label(b,[f'{x}\n({100*x/n:.1f}%)' for x in v]); ax[0].set_ylabel('stalled solves (n_iter ≥ 15)'); ax[0].set_title(f'DBS CCP stalls, n={n} matched states')
ax[0].set_ylim(0,max(v)*1.2)
ax[1].bar(['HIGH vs base','LOW vs base'],[100*(1-v[2]/165),100*(1-v[1]/165)],color=['#1f77b4','#9ecae1']); ax[1].axhline(0,c='k'); ax[1].set_ylabel('relative change in stall count (%)\n(positive = fewer stalls)'); ax[1].set_title('Net stall reduction')
for i,x in enumerate([100*(1-v[2]/165),100*(1-v[1]/165)]): ax[1].text(i,x+(2 if x>=0 else -8),f'{x:.0f}%',ha='center')
save(fig,'05_stalls','DBS_stall_counts_bar',SRC,EXP,'Stall counts/rates for baseline, LOW, HIGH.','HIGH cuts stalls 165→56 (−66%) but LOW *raises* them 165→373: the benefit is not monotone in λ.','Net count hides gross flows (rescued vs newly induced); LOW worse than baseline.')
M=np.array([[(w.cat=='D_persistent_conv').sum(),(w.cat=='C_induced').sum()],[(w.cat=='A_rescued').sum(),(w.cat=='B_persistent_stall').sum()]])
fig,ax=plt.subplots(1,2,figsize=(11,4.4))
for a,(Z,fmt,tt) in zip(ax,[(M,'{:d}','count'),(100*M/M.sum(1,keepdims=True),'{:.1f}%','% of baseline row (conditional)')]):
    a.imshow(Z if tt=='count' else Z,cmap='Blues',norm=matplotlib.colors.LogNorm(1,M.max()) if tt=='count' else None); a.grid(False)
    for i in range(2):
        for j in range(2): a.text(j,i,fmt.format(int(Z[i,j]) if tt=='count' else Z[i,j]),ha='center',va='center',fontsize=15,color='k' if Z[i,j]<(M.max()/2 if tt=='count' else 50) else 'w')
    a.set_xticks([0,1],['HIGH converges','HIGH stalls']); a.set_yticks([0,1],['baseline converges','baseline stalls']); a.set_title(f'Baseline vs HIGH (λ=0.788): {tt}')
fig.suptitle(f'DBS matched-state stall transitions (n={n}); log colour on counts')
save(fig,'05_stalls','DBS_stall_transition_heatmap',SRC,EXP,'2×2 matched-state contingency of stall status, baseline vs HIGH, as counts and row-normalised %.','153/165 baseline stalls rescued (92.7%), 12 persist, and 44/3835 (1.1%) previously-converging states become stalls.','Row % hides the base rate: new stalls are rare per converged state but add up to 44.')
fig,ax=plt.subplots(figsize=(9,5)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,10)
sc=8/4000
bl={'stall':(165,'baseline stalls',COL['base']),'conv':(3835,'baseline converges',COL['D'])}
hh={'stall':(56,'HIGH stalls',COL['C']),'conv':(3944,'HIGH converges',COL['A'])}
def band(x0,y0a,y0b,x1,y1a,y1b,c):
    xs=np.linspace(0,1,30);sm=3*xs**2-2*xs**3
    top=y0b+(y1b-y0b)*sm; bot=y0a+(y1a-y0a)*sm; X=x0+(x1-x0)*xs
    ax.fill(np.r_[X,X[::-1]],np.r_[top,bot[::-1]],color=c,alpha=.5,lw=0)
vh=lambda n:max(n*sc,0.35)
yb={'stall':(9,9-vh(165)),'conv':(9-vh(165)-.4,9-vh(165)-.4-3835*sc)}
yh={'stall':(9,9-vh(56)),'conv':(9-vh(56)-.4,9-vh(56)-.4-3944*sc)}
off_b={'stall':yb['stall'][0],'conv':yb['conv'][0]}; off_h={'stall':yh['stall'][0],'conv':yh['conv'][0]}
flows=[('conv','conv',3791,COL['D']),('stall','conv',153,COL['A']),('conv','stall',44,COL['C']),('stall','stall',12,COL['B'])]
LG=[]
for a_,b_,cnt,c in flows:
    hb=cnt/ (165 if a_=='stall' else 3835)*(yb[a_][0]-yb[a_][1]); hh_=cnt/(56 if b_=='stall' else 3944)*(yh[b_][0]-yh[b_][1])
    band(2,off_b[a_]-hb,off_b[a_],8,off_h[b_]-hh_,off_h[b_],c); off_b[a_]-=hb; off_h[b_]-=hh_
    LG.append(matplotlib.patches.Patch(color=c,alpha=.6,label={153:'A rescued: 153',44:'C new stall: 44',12:'B persistent stall: 12',3791:'D converge both: 3791'}[cnt]))
for k,(nn,l,c) in bl.items(): ax.add_patch(plt.Rectangle((1.8,yb[k][1]),.2,yb[k][0]-yb[k][1],color=c)); ax.text(1.7,np.mean(yb[k]),f'{l}\n{nn}',ha='right',va='center')
for k,(nn,l,c) in hh.items(): ax.add_patch(plt.Rectangle((8,yh[k][1]),.2,yh[k][0]-yh[k][1],color=c)); ax.text(8.3,np.mean(yh[k]),f'{l}\n{nn}',ha='left',va='center')
ax.legend(handles=LG,loc='lower center',ncol=4,fontsize=9,frameon=False,bbox_to_anchor=(.5,-.08))
ax.set_title('DBS stall flow, baseline → HIGH (A=153 rescued, B=12 persistent, C=44 new, D=3791); stall boxes enlarged for visibility')
save(fig,'05_stalls','DBS_stall_flow_sankey',SRC,EXP,'Flow diagram of matched states between stall status under baseline and HIGH.','Almost all flow mass is D (converge both); the interesting thin flows are A (153), C (44), B (12).','Small groups drawn with a minimum height, so box heights for stall groups are not to scale.')
fig,ax=plt.subplots(1,3,figsize=(16,4.8))
order=['D_persistent_conv','A_rescued','B_persistent_stall','C_induced']
for a,(xv,yv,xl,yl) in zip(ax,[('y','u_prev','reference y_ref','previous control u_prev'),('k_base','u_prev','MPC timestep k','previous control u_prev'),('k_base','y','MPC timestep k','reference y_ref')]):
    for c in order:
        d=w[w.cat==c]; a.scatter(d[xv]+np.random.default_rng(0).normal(0,.01*(w[xv].max()-w[xv].min()),len(d)),d[yv],s=6 if c[0]=='D' else 28,c=COL[c[0]],alpha=.25 if c[0]=='D' else .8,label=f'{NAME[c]} (n={len(d)})',edgecolor='none' if c[0]=='D' else 'k',lw=.3,zorder=1 if c[0]=='D' else 3)
    a.set_xlabel(xl); a.set_ylabel(yl)
ax[0].legend(fontsize=7,loc='upper right'); fig.suptitle('DBS stall locations in available state variables (state is scalar tracking output; ref, u_prev, k) - slight jitter on x')
save(fig,'05_stalls','DBS_stall_locations',SRC,EXP,'Where rescued/persistent/new stalls sit in (ref,u_prev,k) coordinates.','Stalls cluster at specific u_prev/ref levels rather than uniformly; see group differences in 05/10 diagnostics.','The DBS state here is low-dimensional (placeholder plant, not a physical state vector); no PCA needed.')
fig,ax=plt.subplots(1,3,figsize=(16,4.6))
cats=['both','geometry only','safeguard only','neither']
b=ax[0].bar(cats,ov['count'],color=[COL['A'],COL['reg'],COL['sg'],COL['D']]); ax[0].bar_label(b,[f'{c}\n{p:.1f}%' for c,p in zip(ov['count'],ov.pct_of_165)]); ax[0].set_ylabel('baseline stalls (of 165)'); ax[0].set_title('Rescue category (rescued = leaves stall status)'); ax[0].set_ylim(0,185)
Zc=np.array([[0,12],[1,152]]); ax[1].imshow(Zc,cmap='Blues',norm=matplotlib.colors.SymLogNorm(1,vmax=152)); ax[1].grid(False)
for i in range(2):
    for j in range(2): ax[1].text(j,i,Zc[i,j],ha='center',va='center',fontsize=16,color='w' if Zc[i,j]>50 else 'k')
ax[1].set_xticks([0,1],['safeguard\nnot rescued','safeguard\nrescued']); ax[1].set_yticks([0,1],['geometry not\nrescued','geometry\nrescued']); ax[1].set_title('Contingency (165 baseline stalls)')
ax[2].axis('off'); ax[2].set_xlim(0,10); ax[2].set_ylim(0,6)
rg_,rs_=153,164; ra,rb=1.9,2.0
ax[2].add_patch(Circle((3.3,3),ra*1.0,color=COL['reg'],alpha=.4)); ax[2].add_patch(Circle((5.2,3),rb,color=COL['sg'],alpha=.4))
ax[2].text(2.6,3,'1',ha='center',fontsize=14); ax[2].text(4.25,3,'152',ha='center',fontsize=16,fontweight='bold'); ax[2].text(6.1,3,'12',ha='center',fontsize=14)
ax[2].text(3,5.3,'Geometry reg. (153)',ha='center',color=COL['reg']); ax[2].text(5.8,5.3,'Safeguard (164)',ha='center',color=COL['sg']); ax[2].text(5,.3,'neither: 0 | areas schematic, not to scale',ha='center',fontsize=9); ax[2].set_title('Venn of rescued baseline stalls'); ax[2].set_aspect('equal')
save(fig,'07_regularizer_vs_safeguard','DBS_rescue_overlap_summary',SRC,EXP,'How the 165 baseline stalls split between geometry-regularisation rescue and proximal/backtracking safeguard rescue.','Verified: 152 rescued by both, 1 geometry-only, 12 safeguard-only, 0 neither; the safeguard rescues 164/165 (99.4%), geometry 153/165 (92.7%).','Safeguard "rescue" = sg_converged (one such solve reaches the cap on iteration 15 yet is flagged converged). Venn areas are schematic.')
fig=plt.figure(figsize=(7,4.5)); gs=fig.add_gridspec(2,1,height_ratios=[3,1.2],hspace=.05); a1=fig.add_subplot(gs[0]); a2=fig.add_subplot(gs[1],sharex=a1)
cmb=[('both',[1,1],152),('safeguard only',[0,1],12),('geometry only',[1,0],1)]
for i,(nm,m,cn) in enumerate(cmb):
    a1.bar(i,cn,color='#444'); a1.text(i,cn+3,cn,ha='center')
    for j in range(2): a2.scatter(i,j,s=120,c='k' if m[j] else '#ddd',zorder=3)
    a2.plot([i,i],[j for j in range(2) if m[j]][::1] if sum(m)==2 else [0,0],c='k')
a2.set_yticks([0,1],['geometry reg.','safeguard']); a2.set_xticks(range(3),[c[0] for c in cmb]); a1.set_ylabel('# baseline stalls'); a1.set_title('UpSet-style: rescue-set intersections (165 stalls; neither=0)'); plt.setp(a1.get_xticklabels(),visible=False); a2.grid(False)
save(fig,'07_regularizer_vs_safeguard','DBS_rescue_upset',SRC,EXP,'UpSet-style intersection bars for the two rescue methods.','Intersection is overwhelmingly the "both" set.','Two sets only, so equivalent to the Venn.')
it=pd.read_csv(P5+'current/iters_current.csv.gz'); its=pd.read_csv(P5+'safeguarded/iters_safeguarded.csv.gz')
for d in (it,its): d['arm']=d.arm.str.replace('dbs_both_lam','').replace({'0':'base'})
def traj(df,arm,sd,bi): return df[(df.arm==arm)&(df.seed==sd)&(df.bank_idx==bi)].sort_values('it')
def twocycle(t):
    if len(t)<6: return False
    u=t.u_next.values[-6:]; d=t.d.values[-6:]
    return bool(np.all(np.abs(u[2:]-u[:-2])<1e-5) and np.all(d[1:]*d[:-1]<0) and np.all(np.abs(d)>1e-3))
rows=[]
for _,r in w[w.stall_base].iterrows():
    t=traj(it,'base',r.seed,r.bank_idx); rows.append(dict(seed=r.seed,bank_idx=r.bank_idx,rescue_group=r.rescue_group,two_cycle_exact=twocycle(t),n_it=len(t),
       frac_rate_active=t.rate_active.mean(),frac_at_umin=t.at_umin.mean(),frac_at_umax=t.at_umax.mean(),amp_u=np.ptp(t.u_next.values[-6:]),Jamp=np.ptp(t.J_surr.values[-6:])))
mech=pd.DataFrame(rows); tab(mech,'DBS_baseline_stall_mechanism',P5+'current/iters_current.csv.gz','Per baseline stall: exact two-vertex-cycle flag (last 6 iterates alternate by ±0.002 rate step, u repeats every 2 its)')
print(mech.groupby('rescue_group').two_cycle_exact.agg(['sum','count']))
mech.to_pickle(O+'.mech.pkl')
