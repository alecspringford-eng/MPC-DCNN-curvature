import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=os.environ.get('DBS_RESULTS_ROOT','results/private/')
O=os.environ.get('DBS_FIGURE_ROOT','figures/dbs_regen/')
P5=R+'phase5_dbs_joint_curvature/'
plt.rcParams.update({'font.size':11,'axes.grid':True,'grid.alpha':.3,'figure.dpi':100,'figure.constrained_layout.use':True,'axes.titlesize':12})
COL={'base':'#555555','reg':'#1f77b4','sg':'#d62728','A':'#2ca02c','B':'#ff7f0e','C':'#d62728','D':'#9e9e9e'}
INDEX=[]
def save(fig,d,name,src,exp,shows,interp,cav):
    for e,kw in (('png',{'dpi':300}),('pdf',{})):
        fig.savefig(f'{O}{d}/{name}.{e}',bbox_inches='tight',**kw)
    plt.close(fig)
    INDEX.append(dict(kind='figure',file=f'{d}/{name}.png',src=src,exp=exp,shows=shows,interp=interp,cav=cav))
    json.dump(INDEX,open(O+'.index_part_%s.json'%os.environ.get('PART','x'),'w'))
def tab(df,name,src,shows):
    df.to_csv(f'{O}tables/{name}.csv',index=False)
    INDEX.append(dict(kind='table',file=f'tables/{name}.csv',src=src,exp='DBS phase5 both-branch',shows=shows,interp='',cav=''))
    json.dump(INDEX,open(O+'.index_part_%s.json'%os.environ.get('PART','x'),'w'))
def load():
    s=pd.read_csv(P5+'current/solves_current.csv.gz'); g=pd.read_csv(P5+'safeguarded/solves_safeguarded.csv.gz')
    for d in (s,g): d['arm']=d.arm.str.replace('dbs_both_lam','').replace({'0':'base'})
    s['stall']=s.n_iter>=15; g['stall']=g.n_iter>=15
    key=['seed','bank_idx']
    w=s.pivot(index=key,columns='arm')
    w.columns=[f'{a}_{b}' for a,b in w.columns]
    w=w.reset_index()
    gw=g.pivot(index=key,columns='arm'); gw.columns=[f'sg_{a}_{b}' for a,b in gw.columns]
    w=w.merge(gw.reset_index(),on=key)
    cat=pd.read_csv(R+'phase5b_stall_mechanism/tables/scenario_categories.csv')
    w=w.merge(cat[key+['cat']],on=key)
    f=pd.read_csv(R+'phase5b_stall_mechanism/tables/solve_features.csv.gz',usecols=['arm','seed','bank_idx','sg_converged'])
    f=f[f.arm=='dbs_both_lam0'].rename(columns={'sg_converged':'sg_rescued_flag'})[key+['sg_rescued_flag']]
    w=w.merge(f,on=key)
    w['y']=w['ref_base']; w['u_prev']=w['u_prev_base']
    return s,g,w
