"""Build README visuals from retained results only; no prediction fitting.

Run from an installed reproduction environment: python scripts/build_readme_assets.py
"""
from pathlib import Path
import csv, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/readme'
OUT.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((ROOT/'results/spatial/curves.csv').open()))
budgets=[1,2,4,5]
keys=['D1','D1_PAIR_CALIBRATED','D1_DXY']
curves={k:[next(float(r['mean_target_pearson']) for r in rows if r['candidate']==k and r['policy_arm']=='selected' and int(r['budget'])==b) for b in budgets] for k in keys}
gains=[a-b for a,b in zip(curves['D1_DXY'],curves['D1_PAIR_CALIBRATED'])]
colors={'D1':'#6B7280','D1_PAIR_CALIBRATED':'#306BB3','D1_DXY':'#B85C25'}
labels={'D1':'Direct correlations','D1_PAIR_CALIBRATED':'Shared map','D1_DXY':'Map + in-plane distance'}
styles={'D1':('--','o'),'D1_PAIR_CALIBRATED':('-.','s'),'D1_DXY':('-','D')}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#192D42','axes.labelcolor':'#33475B','xtick.color':'#506174','ytick.color':'#506174','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B5BEC8','axes.linewidth':0.8,'svg.fonttype':'none','figure.facecolor':'white','axes.facecolor':'white'})
fig=plt.figure(figsize=(13,5.65))
gs=fig.add_gridspec(1,2,left=.075,right=.98,bottom=.16,top=.72,wspace=.24,width_ratios=[1.25,1])
ax=fig.add_subplot(gs[0]); bx=fig.add_subplot(gs[1])
fig.text(.075,.94,'A matched test of spatial information',fontsize=22,fontweight='bold')
fig.text(.075,.88,'6,641 recording identities · 10 scans · 1 mouse · selected regularization',fontsize=11,color='#506174')
for k in keys:
 ls,mark=styles[k]
 ax.plot(budgets,curves[k],color=colors[k],ls=ls,marker=mark,lw=2,ms=6,label=labels[k])
ax.set(ylim=(.19,.33),ylabel='Mean target Pearson r')
ax.set_yticks([.20,.24,.28,.32])
ax.set_title('A  Prediction accuracy',loc='left',pad=12,fontweight='bold',fontsize=13)
ax.legend(frameon=False,loc='lower right',fontsize=10,handlelength=2.5,labelspacing=.6)
bx.plot(budgets,gains,color=colors['D1_DXY'],marker='D',lw=2,ms=6)
bx.fill_between(budgets,0,gains,color=colors['D1_DXY'],alpha=.055)
bx.set(ylim=(0,.0365),ylabel='Change in Pearson r')
bx.set_yticks([0,.01,.02,.03]);bx.set_yticklabels(['0.00','+0.01','+0.02','+0.03'])
bx.set_title('B  Position gain beyond the map',loc='left',pad=12,fontweight='bold',fontsize=13)
for b,g in zip(budgets,gains):
 bx.annotate(f'+{g:.5f}',(b,g),xytext=(0,11),textcoords='offset points',ha='center',fontsize=10,fontweight='bold',color=colors['D1_DXY'])
for a in [ax,bx]:
 a.set_xscale('log',base=2);a.set_xlim(.92,5.5)
 a.xaxis.set_major_locator(FixedLocator(budgets));a.xaxis.set_major_formatter(FixedFormatter(['1','2','4','5']))
 a.minorticks_off();a.set_xlabel('Repeats per training movie (log₂ scale)',labelpad=10)
 a.grid(axis='y',color='#DDE3EA',lw=.6);a.set_axisbelow(True);a.tick_params(length=3,pad=5)
fig.text(.075,.035,'Source: results/spatial/curves.csv. Equal-scan means after nested aggregation; four observed budgets, no new fits.',fontsize=9,color='#506174')
fig.savefig(OUT/'main-results.svg',metadata={'Creator':'','Date':None})
fig.savefig(OUT/'main-results.png',dpi=180,metadata={'Software':''})
plt.close(fig)

svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="810" viewBox="0 0 1400 810" role="img" aria-labelledby="title desc">
<title id="title">Matched few-shot neural prediction pipeline</title>
<desc id="desc">Target calibration and context training data support direct correlations, a shared map, and the map plus in-plane soma distance. A common covariance-decoding framework uses held-out-movie context activity to predict target activity. Target test responses enter only scoring.</desc>
<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="none" stroke="#8796A7" stroke-width="1.5"/></marker></defs>
<rect width="1400" height="810" rx="20" fill="#F6F8FB"/>
<g font-family="Arial, Helvetica, sans-serif">
<text x="56" y="58" fill="#5D7187" font-size="16" font-weight="700" letter-spacing="2.5">THE EXPERIMENT</text>
<text x="56" y="104" fill="#192D42" font-size="32" font-weight="700">One prediction task. Three matched estimators.</text>
<text x="56" y="138" fill="#506174" font-size="20">Measure the contribution of position beyond a learned correlation map.</text>
<rect x="56" y="174" width="1288" height="130" rx="12" fill="#E8EEF6"/>
<text x="82" y="208" fill="#506174" font-size="14" font-weight="700" letter-spacing="1.4">TRAINING INPUTS</text>
<text x="82" y="244" fill="#192D42" font-size="23" font-weight="700">Limited target calibration</text>
<text x="82" y="275" fill="#506174" font-size="19">1, 2, 4, or 5 repeats × four training movies</text>
<line x1="696" y1="196" x2="696" y2="282" stroke="#C7D2DF"/>
<text x="730" y="244" fill="#192D42" font-size="23" font-weight="700">Context-neuron responses</text>
<text x="730" y="275" fill="#506174" font-size="19">Fit shared maps and context covariance</text>
<path d="M700 304V326H258M700 326H1142" fill="none" stroke="#8796A7" stroke-width="2"/>
<path d="M258 326V349" fill="none" stroke="#8796A7" stroke-width="2" marker-end="url(#arrow)"/>
<path d="M700 326V349" fill="none" stroke="#8796A7" stroke-width="2" marker-end="url(#arrow)"/>
<path d="M1142 326V349" fill="none" stroke="#8796A7" stroke-width="2" marker-end="url(#arrow)"/>
<rect x="56" y="360" width="404" height="164" rx="12" fill="#FFFFFF" stroke="#CED6E0"/>
<rect x="498" y="360" width="404" height="164" rx="12" fill="#FFFFFF" stroke="#B6CDE8"/>
<rect x="940" y="360" width="404" height="164" rx="12" fill="#FFF9F3" stroke="#D6B193"/>
<text x="82" y="395" fill="#6B7280" font-size="14" font-weight="700" letter-spacing="1.4">01 / DIRECT</text>
<text x="82" y="434" fill="#192D42" font-size="24" font-weight="700">Empirical correlations</text>
<text x="82" y="470" fill="#506174" font-size="19">Target–context relationships</text>
<text x="82" y="498" fill="#506174" font-size="19">from the calibration responses</text>
<text x="524" y="395" fill="#306BB3" font-size="14" font-weight="700" letter-spacing="1.4">02 / MAPPED</text>
<text x="524" y="434" fill="#192D42" font-size="24" font-weight="700">Shared correlation map</text>
<text x="524" y="470" fill="#506174" font-size="19">Calibrate relationships using</text>
<text x="524" y="498" fill="#506174" font-size="19">context–context training pairs</text>
<text x="966" y="395" fill="#A74E1D" font-size="14" font-weight="700" letter-spacing="1.4">03 / MAPPED + POSITION</text>
<text x="966" y="434" fill="#192D42" font-size="24" font-weight="700">Add in-plane distance</text>
<text x="966" y="470" fill="#506174" font-size="19">The same map, augmented</text>
<text x="966" y="498" fill="#506174" font-size="19">with released x/y soma separation</text>
<path d="M258 524V550H700V577M1142 524V550H700M700 524V577" fill="none" stroke="#8796A7" stroke-width="2" marker-end="url(#arrow)"/>
<rect x="56" y="588" width="1288" height="154" rx="12" fill="#192D42"/>
<text x="82" y="625" fill="#B8C9DA" font-size="14" font-weight="700" letter-spacing="1.4">COMMON COVARIANCE-DECODING FRAMEWORK</text>
<text x="82" y="668" fill="#FFFFFF" font-size="24" font-weight="700">Test-movie context activity</text>
<path d="M433 661H473" fill="none" stroke="#B8C9DA" stroke-width="2" marker-end="url(#arrow)"/>
<text x="490" y="668" fill="#FFFFFF" font-size="24" font-weight="700">Predicted target activity</text>
<path d="M802 661H842" fill="none" stroke="#B8C9DA" stroke-width="2" marker-end="url(#arrow)"/>
<text x="860" y="668" fill="#FFFFFF" font-size="24" font-weight="700">Pearson r on held-out targets</text>
<text x="82" y="711" fill="#B8C9DA" font-size="19">Target test responses enter only scoring. Position gain = arm 03 − arm 02.</text>
<text x="56" y="781" fill="#63758A" font-size="17">Schematic of the included estimator and split design. No target test responses are used to fit or select models.</text>
</g></svg>'''
(OUT/'pipeline.svg').write_text(svg)
# Machine-readable values kept with the graphics for convenient review.
summary={'source':'results/spatial/curves.csv','policy_arm':'selected','budgets':budgets,'metric':'mean target Pearson r','curves':curves,'position_gain':gains,'aggregation':'Equal means over calibration subsets, allocations, folds, movies, and scans after target averaging','refit_performed':False}
(OUT/'chart-data.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Generated pipeline.svg, main-results.svg/png, chart-data.json')
