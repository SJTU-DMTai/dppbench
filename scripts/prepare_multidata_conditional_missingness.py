"""MAR/MNAR extension; original MCAR script and baseline files stay unchanged."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import prepare_multidata_missingness as base

MAR_FIELDS = {'fraud_detection':'TransactionDT', 'home_credit':'DAYS_ID_PUBLISH',
              'polish_bankruptcy':'Attr3'}


def probabilities(values, observed, calibration, rate, strength):
    """High-value group has larger probability; missing covariate is z=0."""
    finite = np.isfinite(values)
    fit = calibration & observed
    if not fit.any(): raise ValueError('No observed target values in common training calibration set')
    threshold_set = calibration & finite
    if not threshold_set.any(): raise ValueError('No finite condition values in calibration set')
    threshold = float(np.median(values[threshold_set]))
    z = np.where(finite, np.where(values > threshold, 1., -1.), 0.)
    def sigmoid(v): return 1/(1+np.exp(-np.clip(v,-40,40)))
    lo, hi = -40., 40.
    for _ in range(100):
        mid = (lo+hi)/2
        if sigmoid(mid+strength*z[fit]).mean() < rate: lo = mid
        else: hi = mid
    alpha = (lo+hi)/2
    prob = sigmoid(alpha+strength*z) if rate else np.zeros(len(values))
    return prob, {'threshold':threshold,'alpha':alpha,'strength':strength,
                  'calibration_observed_target_count':int(fit.sum()),
                  'expected_rate_observed_calibration':float(prob[fit].mean()),
                  'rule':'condition > training median: z=+1; otherwise z=-1; missing/nonfinite condition: z=0'}


def conditional_mask(frame, columns, rate, seed, task, mechanism, calibration_positions, strength):
    calibration = np.zeros(len(frame),dtype=bool)
    calibration[np.asarray(calibration_positions,dtype=int)] = True
    dirty, mask = frame.copy(deep=True), pd.DataFrame(False,index=frame.index,columns=columns)
    audit = {}
    for i,col in enumerate(columns):
        if col not in frame or not pd.api.types.is_numeric_dtype(frame[col]):
            raise ValueError(f'Required numeric target field: {task}.{col}')
        condition = MAR_FIELDS[task] if mechanism=='MAR' else col
        if condition not in frame or not pd.api.types.is_numeric_dtype(frame[condition]):
            raise ValueError(f'Required numeric condition field: {task}.{condition}')
        if mechanism=='MAR' and condition in columns: raise ValueError('MAR condition must remain unmasked')
        observed = frame[col].notna().to_numpy()
        prob, info = probabilities(frame[condition].to_numpy(dtype=float),observed,calibration,rate,strength)
        rng = np.random.default_rng(np.random.SeedSequence([seed,list(base.TASKS).index(task),i]))
        selected = (rng.random(len(frame)) < prob) & observed
        mask[col] = selected
        if selected.any(): dirty.loc[selected,col] = np.nan
        info.update(condition_field=condition,mechanism=mechanism,target_rate=rate,
                    condition_kept_unmasked=mechanism=='MAR')
        audit[col] = info
    base.check_only_masks(frame,dirty,mask,columns)
    return dirty,mask,audit


def original_common_training_positions(frame,source,task,seeds):
    config=base.TASKS[task]
    pool=pd.read_parquet(source/task/'std_test/train_frozen.parquet')
    if config['id']:
        positions=pd.Series(np.arange(len(frame)),index=frame[config['id']]).loc[pool[config['id']]].to_numpy()
    else:
        positions,_=train_test_split(np.arange(len(frame)),test_size=.2,random_state=42,
                                    stratify=frame[config['target']])
    ratio=(base.yaml_config(task).get('train') or {}).get('val_ratio',.2)
    sets=[]
    for seed in seeds:
        tr,_=train_test_split(np.arange(len(pool)),test_size=ratio,random_state=seed,
                              stratify=pool[config['target']])
        sets.append(set(positions[tr].tolist()))
    common=sorted(set.intersection(*sets))
    if not common: raise ValueError('Empty intersection of original training records')
    return common


def prepare(args):
    source,output=Path(args.source_root).resolve(),Path(args.output_root).resolve()
    base.disjoint(source,output);base.disjoint(base.REPO,output)
    if output.exists(): raise FileExistsError(f'Use a new directory: {output}')
    output.mkdir(parents=True);base.dump(output/'INCOMPLETE.json',{'status':'preparing'})
    # Only the data-preparation process uses this callback. Original dataset
    # classes, evaluators and baseline implementations are never patched.
    original_mask=base.mask_frame
    def mask_callback(frame,columns,rate,seed,task):
        common=original_common_training_positions(frame,source,task,args.run_seeds)
        dirty,mask,audit=conditional_mask(frame,columns,rate,seed,task,args.mechanism,common,args.strength)
        folder=output/'condition_audits';folder.mkdir(exist_ok=True)
        base.dump(folder/f'{task}.json',{'calibration':'intersection_of_original_training_records',
                  'run_seeds':args.run_seeds,'calibration_raw_positions':common,'conditions':audit})
        return dirty,mask
    try:
        base.mask_frame=mask_callback
        for task in args.tasks:base.prepare_task(source,output,task,args.rate,args.mask_seed)
    finally:
        base.mask_frame=original_mask
    base.dump(output/'missingness_manifest.json',{
        'version':'original_protocol_primary_conditional_v1','repo':str(base.REPO),
        'source':str(source),'tasks':args.tasks,'mechanism':args.mechanism,
        'rate':args.rate,'mask_seed':args.mask_seed,'run_seeds':args.run_seeds,
        'columns':{task:base.TASKS[task]['cols'] for task in args.tasks},
        'protocol':'original_loader_frozen_membership_and_seed_rules_unchanged',
        'code_hashes':base.code_fingerprints(),
        'input_hashes':{str(f.relative_to(output)):base.sha(f) for task in args.tasks
                        for folder in ('data','std_test') for f in (output/task/folder).iterdir() if f.is_file()}})
    (output/'INCOMPLETE.json').unlink();base.verify(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--tasks',nargs='+',choices=list(base.TASKS),default=list(base.TASKS))
    p.add_argument('--mechanism',choices=['MAR','MNAR'],required=True)
    p.add_argument('--rate',type=float,default=.3);p.add_argument('--mask-seed',type=int,default=1001)
    p.add_argument('--strength',type=float,default=2.)
    p.add_argument('--run-seeds',nargs='+',type=int,default=[42,43,44])
    a=p.parse_args()
    if not 0<=a.rate<1 or a.mask_seed<0 or not np.isfinite(a.strength) or a.strength<=0:
        p.error('Require 0<=rate<1, nonnegative mask seed, finite positive strength')
    if len(set(a.tasks))!=len(a.tasks) or len(set(a.run_seeds))!=len(a.run_seeds):p.error('Duplicate tasks/seeds')
    prepare(a)


if __name__=='__main__':main()
