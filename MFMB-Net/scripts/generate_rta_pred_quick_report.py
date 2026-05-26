#!/usr/bin/env python3
import argparse
import os
import pandas as pd


def sel_official(df):
    if {'train_drop_last','eval_drop_last','test_drop_last'}.issubset(df.columns):
        d = df[(df['train_drop_last']==1)&(df['eval_drop_last']==0)&(df['test_drop_last']==0)].copy()
        if not d.empty:
            df = d
    return df


def to_metrics(df, mode_col='mode'):
    out=df[['missing',mode_col,'MAE_mean','Corr_mean','Non0_acc_2_mean','Non0_F1_score_mean','Mult_acc_5_mean','Mult_acc_7_mean']].copy()
    out=out.rename(columns={
        mode_col:'method','MAE_mean':'MAE','Corr_mean':'Corr','Non0_acc_2_mean':'Non0_acc_2','Non0_F1_score_mean':'Non0_F1_score','Mult_acc_5_mean':'Mult_acc_5','Mult_acc_7_mean':'Mult_acc_7'
    })
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output_dir', default='results/auto_anchor_runs_rta_pred')
    args=ap.parse_args()
    out_dir=args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    base=pd.read_csv('results/auto_anchor_runs_fulltest/anchor_experiment_agg.csv')
    base=sel_official(base)
    base=base[base['missing'].isin([0.4,0.5])]
    best_fixed=base[base['mode'].isin(['text','audio','vision'])].copy()

    dyn_soft=to_metrics(base[base['mode']=='dynamic_soft'])

    ta=pd.read_csv('results/auto_anchor_runs_taskrouter_final/tuned_dynamic_task_soft_0.1_0.5.csv')
    ta=ta[ta['missing'].isin([0.4,0.5])][['missing','MAE','Corr','Non0_acc_2','Non0_F1_score','Mult_acc_5','Mult_acc_7']].copy()
    ta['method']='dynamic_task_soft_tuned'

    rta_f=pd.read_csv('results/auto_anchor_runs_rta/anchor_experiment_agg.csv')
    rta_f=sel_official(rta_f)
    rta_f=to_metrics(rta_f[(rta_f['mode']=='dynamic_rta') & (rta_f['missing'].isin([0.4,0.5]))])
    rta_f['method']='dynamic_rta_feature'

    rta_p=pd.read_csv(os.path.join(out_dir,'anchor_experiment_agg.csv'))
    rta_p=sel_official(rta_p)
    rta_p=to_metrics(rta_p[(rta_p['mode']=='dynamic_rta_pred') & (rta_p['missing'].isin([0.4,0.5]))])
    rta_p['method']='dynamic_rta_pred'

    fixed_rows=[]
    for miss,g in best_fixed.groupby('missing'):
        gg=to_metrics(g)
        i=gg['MAE'].idxmin()
        r=gg.loc[i].to_dict()
        r['method']='best_fixed'
        r['best_fixed_from']=gg.loc[i,'method']
        fixed_rows.append(r)
    best_fixed_df=pd.DataFrame(fixed_rows)

    table=pd.concat([
        dyn_soft[['missing','method','MAE','Corr','Non0_acc_2','Non0_F1_score','Mult_acc_5','Mult_acc_7']],
        ta[['missing','method','MAE','Corr','Non0_acc_2','Non0_F1_score','Mult_acc_5','Mult_acc_7']],
        rta_f[['missing','method','MAE','Corr','Non0_acc_2','Non0_F1_score','Mult_acc_5','Mult_acc_7']],
        rta_p[['missing','method','MAE','Corr','Non0_acc_2','Non0_F1_score','Mult_acc_5','Mult_acc_7']],
        best_fixed_df[['missing','method','MAE','Corr','Non0_acc_2','Non0_F1_score','Mult_acc_5','Mult_acc_7']],
    ],ignore_index=True)
    order={'dynamic_soft':0,'dynamic_task_soft_tuned':1,'dynamic_rta_feature':2,'dynamic_rta_pred':3,'best_fixed':4}
    table=table.sort_values(['missing','method'],key=lambda s:s.map(order) if s.name=='method' else s).reset_index(drop=True)
    out_csv=os.path.join(out_dir,'rta_pred_quick_comparison.csv')
    table.to_csv(out_csv,index=False,float_format='%.6f')

    # diagnostics from rta_pred_analysis
    diag_path=os.path.join(out_dir,'rta_pred_analysis')
    gate_match=[]
    if os.path.isdir(diag_path):
        for fn in os.listdir(diag_path):
            if not fn.endswith('_dynamic_rta_pred.csv'):
                continue
            miss=float(fn.split('_missing')[1].split('_seed')[0])
            d=pd.read_csv(os.path.join(diag_path,fn))
            if 'gate_oracle_match' in d.columns:
                gate_match.append({'missing':miss,'gate_oracle_match':pd.to_numeric(d['gate_oracle_match'],errors='coerce').mean()})
    gdf=pd.DataFrame(gate_match)
    gagg=gdf.groupby('missing',as_index=False).mean() if not gdf.empty else pd.DataFrame(columns=['missing','gate_oracle_match'])

    # report
    rpt=os.path.join(out_dir,'rta_pred_quick_report.md')
    lines=['# RTA Prediction-level Quick Report','','## Comparison (missing=0.4,0.5)','']
    lines.append('```csv')
    lines.append(table.to_csv(index=False, float_format='%.6f').strip())
    lines.append('```')
    lines.append('')

    def getv(miss,method,col):
        s=table[(table['missing']==miss)&(table['method']==method)]
        return float(s.iloc[0][col]) if len(s) else float('nan')

    lines.append('## Key Answers')
    for miss in [0.4,0.5]:
        mr=getv(miss,'dynamic_rta_pred','MAE')
        mf=getv(miss,'dynamic_rta_feature','MAE')
        ms=getv(miss,'dynamic_soft','MAE')
        mt=getv(miss,'dynamic_task_soft_tuned','MAE')
        mb=getv(miss,'best_fixed','MAE')
        lines.append(f'- missing={miss:.1f}: MAE rta_pred={mr:.4f}, rta_feature={mf:.4f}, dynamic_soft={ms:.4f}, task_tuned={mt:.4f}, best_fixed={mb:.4f}')

    pred_mae = table[table['method'] == 'dynamic_rta_pred'][['missing', 'MAE']].rename(columns={'MAE': 'MAE_pred'})
    feat_mae = table[table['method'] == 'dynamic_rta_feature'][['missing', 'MAE']].rename(columns={'MAE': 'MAE_feat'})
    cmp_df = pred_mae.merge(feat_mae, on='missing', how='inner')
    b_pred_vs_feat = int((cmp_df['MAE_pred'] < cmp_df['MAE_feat']).sum()) if not cmp_df.empty else 0
    lines.append(f'- dynamic_rta_pred better MAE than dynamic_rta_feature: {b_pred_vs_feat}/2 missing rates.')
    if not gagg.empty:
        lines.append('- gate_oracle_match by missing:')
        for _,r in gagg.iterrows():
            lines.append(f"  - missing={float(r['missing']):.1f}: {float(r['gate_oracle_match']):.4f}")
        lines.append('  - random baseline is 0.5 for two-route gate oracle match.')

    # acceptance checks
    lines.append('')
    lines.append('## Acceptance Check')
    ok1 = getv(0.4,'dynamic_rta_pred','MAE') <= getv(0.4,'dynamic_soft','MAE') + 0.03
    ok2 = getv(0.5,'dynamic_soft','MAE') - getv(0.5,'dynamic_rta_pred','MAE') >= 0.03
    ok3 = (getv(0.5,'dynamic_rta_pred','Corr') >= getv(0.5,'dynamic_task_soft_tuned','Corr') - 0.01) and (getv(0.5,'dynamic_rta_pred','Non0_F1_score') >= getv(0.5,'dynamic_task_soft_tuned','Non0_F1_score') - 0.01)
    lines.append(f'- Check1 (0.4 MAE close to dynamic_soft): {ok1}')
    lines.append(f'- Check2 (0.5 MAE >=0.03 better than dynamic_soft): {ok2}')
    lines.append(f'- Check3 (0.5 Corr/F1 not obviously below task_tuned): {ok3}')
    lines.append(f'- Recommendation: {"proceed to 0.1~0.5 three-seed" if (ok1 and ok2 and ok3) else "do not scale yet; revise gate/oracle"}')

    with open(rpt,'w',encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('Saved:', out_csv)
    print('Saved:', rpt)


if __name__=='__main__':
    main()
