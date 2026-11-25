"""Original-loader data-only MCAR copies for three additional datasets.

Only input/output roots change in exp1. Labels, IDs, original split membership,
row/column order, baseline configurations and HPO are preserved. No downloads.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.default_credit_missingness import disjoint, sha, dump

TASKS = {
    "fraud_detection": {
        "class": "dppbench.tasks.fraud_detection.fraud_detection_data.FraudDetectionData",
        "files": ["train_transaction.csv", "train_identity.csv", "test_transaction.csv", "test_identity.csv"],
        "raw": "train_transaction.csv", "id": "TransactionID", "target": "isFraud",
        "cols": ["TransactionAmt", "C1", "C2", "C5", "C6", "C9"],
    },
    "home_credit": {
        "class": "dppbench.tasks.home_credit.home_credit_data.HomeCreditData",
        "files": ["application_train.csv", "application_test.csv", "bureau.csv", "bureau_balance.csv",
                  "previous_application.csv", "POS_CASH_balance.csv", "credit_card_balance.csv", "installments_payments.csv"],
        "raw": "application_train.csv", "id": "SK_ID_CURR", "target": "TARGET",
        "cols": ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "DAYS_BIRTH", "DAYS_EMPLOYED"],
    },
    "polish_bankruptcy": {
        "class": "dppbench.tasks.polish_bankruptcy.polish_bankruptcy_data.PolishBankruptcyData",
        "files": [f"{i}year.arff" for i in range(1,6)], "raw": "5year.arff", "id": None, "target": "TARGET",
        "cols": ["Attr1", "Attr2", "Attr5", "Attr6", "Attr10", "Attr12"],
    },
}


def load_class(dotted):
    module, name = dotted.rsplit('.',1)
    return getattr(importlib.import_module(module),name)


def mask_frame(frame, columns, rate, seed, task):
    masked = frame.copy(deep=True)
    mask = pd.DataFrame(False,index=frame.index,columns=columns)
    for i,col in enumerate(columns):
        if col not in frame or not pd.api.types.is_numeric_dtype(frame[col]):
            raise ValueError(f"Required numeric field unavailable: {task}.{col}")
        # Stable streams, nested across rates; do not use labels or importance.
        rng = np.random.default_rng(np.random.SeedSequence([seed,list(TASKS).index(task),i]))
        chosen = (rng.random(len(frame)) < rate) & frame[col].notna().to_numpy()
        mask[col] = chosen
        if chosen.any(): masked.loc[chosen,col] = np.nan
    return masked,mask


def check_only_masks(original, changed, mask, columns):
    if list(original.columns)!=list(changed.columns) or len(original)!=len(changed):
        raise ValueError("Row/column schema changed")
    for col in original:
        if col not in columns:
            pd.testing.assert_series_equal(original[col],changed[col],check_names=True)
        else:
            selected = mask[col].to_numpy()
            if not changed.loc[selected,col].isna().all(): raise ValueError(f"Missing mask not applied: {col}")
            pd.testing.assert_series_equal(original.loc[~selected,col],changed.loc[~selected,col],check_dtype=False)
            if not original.loc[selected,col].notna().all(): raise ValueError("Mask includes naturally missing values")


def rewrite_csv(source,target,frame,mask,config):
    """Preserve all unmasked field tokens; do not round-trip CSV floats."""
    positions = {int(value):i for i,value in enumerate(frame[config['id']])}
    if len(positions)!=len(frame): raise ValueError("Raw primary IDs not unique")
    seen=set(); flags=mask.to_numpy(dtype=bool)
    with source.open(newline='',encoding='utf-8-sig') as src,target.open('w',newline='',encoding='utf-8') as dst:
        reader=csv.reader(src);writer=csv.writer(dst);header=next(reader);writer.writerow(header)
        key=header.index(config['id']);cols={col:header.index(col) for col in config['cols']}
        for fields in reader:
            if not fields: continue
            value=int(fields[key]);i=positions[value]
            if value in seen: raise ValueError("Repeated raw CSV ID")
            seen.add(value)
            for j,(col,index) in enumerate(cols.items()):
                if flags[i,j]: fields[index]=''
            writer.writerow(fields)
    if len(seen)!=len(frame): raise ValueError("Raw CSV rows do not match original loader rows")


def rewrite_arff(source,target,mask,columns):
    """Change only selected dense ARFF tokens to '?'; preserve headers."""
    names=[];in_data=False;row=0;flags=mask.to_numpy(dtype=bool)
    with source.open(encoding='utf-8') as src,target.open('w',encoding='utf-8',newline='') as dst:
        for line in src:
            stripped=line.strip()
            if not in_data:
                match=re.match(r"@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))",stripped,re.I)
                if match:names.append(next(x for x in match.groups() if x is not None))
                dst.write(line)
                if stripped.lower()=='@data':in_data=True
                continue
            if not stripped or stripped.startswith('%'):dst.write(line);continue
            if stripped.startswith('{'):raise ValueError("Sparse ARFF unsupported; no output should be used")
            fields=next(csv.reader([stripped]))
            if len(fields)!=len(names):raise ValueError("ARFF attribute count mismatch")
            for j,col in enumerate(columns):
                if flags[row,j]:fields[names.index(col)]='?'
            dst.write(','.join(fields)+'\n');row+=1
    if row!=len(mask):raise ValueError("ARFF row count mismatch")


def partition_positions(frame,pool,test,config):
    """Prove original frozen membership before modifying frozen values."""
    key=config['id']
    if key:
        if not frame[key].is_unique or frame[key].isna().any():raise ValueError("Invalid IDs")
        index=pd.Series(np.arange(len(frame)),index=frame[key])
        positions=[index.loc[part[key]].to_numpy() for part in (pool,test)]
    else:
        # Original build_binary uses this exact stratified holdout. No new split
        # is substituted: abort unless original frozen rows match this ordering.
        from sklearn.model_selection import train_test_split
        positions=list(train_test_split(np.arange(len(frame)),test_size=.2,random_state=42,
                                        stratify=frame[config['target']]))
    for part,idx in zip((pool,test),positions):
        actual=part.drop(columns='__split__',errors='ignore').reset_index(drop=True)
        expected=frame.iloc[idx].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected[actual.columns],actual,check_dtype=False,check_exact=True)
    if len(set(positions[0]) & set(positions[1])) or len(set(positions[0]) | set(positions[1]))!=len(frame):
        raise ValueError("Original frozen membership is not a disjoint complete partition")
    return positions


def code_fingerprints():
    return {str(p.relative_to(REPO)):sha(p) for folder in ('baselines','dppbench','scripts')
            for p in (REPO/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.yml')
            and not any(part.startswith('.') or part in ('data','std_test','__pycache__','ckpts') for part in p.relative_to(REPO).parts)}


def prepare_task(source,output,task,rate,seed):
    config=TASKS[task];src=source/task
    frozen_names=['train_frozen.parquet','std_test.parquet','meta.json']
    inputs=[*(src/'data'/name for name in config['files']),*(src/'std_test'/name for name in frozen_names)]
    for path in inputs:
        if not path.is_file() or path.is_symlink():raise ValueError(f"Require existing regular source file, no download/extract: {path}")
    source_hashes={str(path):sha(path) for path in inputs}
    # All required files were checked first: these three original loaders only
    # read CSV/ARFF here, without downloading, extracting, or writing caches.
    cls=load_class(config['class']);original=cls(data_dir=str(src/'data'));original.load_data()
    frame=original.train_df
    pool=pd.read_parquet(src/'std_test/train_frozen.parquet')
    test=pd.read_parquet(src/'std_test/std_test.parquet')
    positions=partition_positions(frame,pool,test,config)
    dirty,mask=mask_frame(frame,config['cols'],rate,seed,task)
    check_only_masks(frame,dirty,mask,config['cols'])
    dest=output/task;data=dest/'data';std=dest/'std_test';audit=dest/'audit'
    data.mkdir(parents=True);std.mkdir();audit.mkdir()
    for name in config['files']:shutil.copy2(src/'data'/name,data/name)
    if task=='polish_bankruptcy':rewrite_arff(src/'data'/config['raw'],data/config['raw'],mask,config['cols'])
    else:rewrite_csv(src/'data'/config['raw'],data/config['raw'],frame,mask,config)
    for part,idx,name in zip((pool,test),positions,frozen_names[:2]):
        changed=part.copy(deep=True)
        for col in config['cols']:
            selected=mask.iloc[idx][col].to_numpy()
            if selected.any():changed.loc[selected,col]=np.nan
        changed.to_parquet(std/name,index=False)
        # Preserve exact original records, labels, nonpolluted features and order.
        check_only_masks(part,changed,mask.iloc[idx].reset_index(drop=True),config['cols'])
    shutil.copy2(src/'std_test/meta.json',std/'meta.json')
    source_view=frame[[*([config['id']] if config['id'] else []),*config['cols']]].copy()
    source_view.insert(0,'__audit_raw_row__',np.arange(len(frame)))
    source_view.to_parquet(audit/'original_target_fields.parquet',index=False)
    saved_mask=mask.copy();saved_mask.insert(0,'__audit_raw_row__',np.arange(len(frame)))
    if config['id']:saved_mask[config['id']]=frame[config['id']]
    saved_mask.to_parquet(audit/'injected_mask.parquet',index=False)
    dump(audit/'frozen_raw_positions.json',{'train_frozen':positions[0].tolist(),'std_test':positions[1].tolist()})
    report={}
    for col in config['cols']:
        report[col]={}
        for label,part,idx in [('all',frame,np.arange(len(frame))),('train_frozen',pool,positions[0]),('std_test',test,positions[1])]:
            natural=int(part[col].isna().sum());new=int(mask.iloc[idx][col].sum());observed=len(part)-natural
            report[col][label]={'n_rows':len(part),'natural_missing':natural,'injected_missing':new,
                                'injected_fraction_of_observed':new/observed if observed else None,
                                'total_missing_fraction':(natural+new)/len(part)}
    dump(audit/'missingness_rates.json',report)
    # Actually load the copies through the unchanged original loader.
    loaded=cls(data_dir=str(data));loaded.load_data()
    pd.testing.assert_frame_equal(loaded.train_df,dirty,check_exact=True)
    for name,aux in original.auxiliary_dfs.items():
        pd.testing.assert_frame_equal(loaded.auxiliary_dfs[name],aux,check_exact=True)
    if original.test_df is not None:pd.testing.assert_frame_equal(loaded.test_df,original.test_df,check_exact=True)
    # Check seed42's original stratified train/val boundary. Same labels/order
    # guarantee other original seed rules remain unchanged as well.
    from sklearn.model_selection import train_test_split
    cfg=yaml_config(task)
    ratio=(cfg.get('train') or {}).get('val_ratio',.2)
    expected_indices=train_test_split(np.arange(len(pool)),test_size=ratio,random_state=42,stratify=pool[config['target']])
    loaded.load_std_test_frozen();parts=loaded.split(val_ratio=ratio,seed=42)
    changed_pool=pd.read_parquet(std/'train_frozen.parquet').drop(columns='__split__',errors='ignore')
    for name,idx in zip(('train','val'),expected_indices):
        pd.testing.assert_frame_equal(parts[name],changed_pool.iloc[idx].reset_index(drop=True),check_exact=True)
    if source_hashes!={str(path):sha(path) for path in inputs}:raise RuntimeError("Source data changed")
    dump(audit/'source_hashes.json',source_hashes)
    print(f"[READY] {task}: source untouched; original loader and partitions verified",flush=True)


def yaml_config(task):
    import yaml
    return yaml.safe_load((REPO/'dppbench/tasks'/task/'model.yaml').read_text(encoding='utf-8'))


def verify(output):
    output=Path(output).resolve()
    if (output/'INCOMPLETE.json').exists():raise ValueError("Incomplete preparation")
    manifest=json.loads((output/'missingness_manifest.json').read_text())
    for rel,value in manifest['input_hashes'].items():
        if sha(output/rel)!=value:raise ValueError(f"Corrupted copy changed: {rel}")
    for task in manifest['tasks']:
        for path,value in json.loads((output/task/'audit/source_hashes.json').read_text()).items():
            if sha(path)!=value:raise ValueError(f"Original input changed: {path}")
    for rel,value in manifest['code_hashes'].items():
        if sha(Path(manifest['repo'])/rel)!=value:raise ValueError(f"Code/config changed since preparation: {rel}")
    print('[VERIFIED] Original inputs, copied inputs and code/config hashes',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root');p.add_argument('--output-root',required=True)
    p.add_argument('--tasks',nargs='+',choices=list(TASKS),default=list(TASKS))
    p.add_argument('--rate',type=float,default=.3);p.add_argument('--mask-seed',type=int,default=1001)
    p.add_argument('--verify',action='store_true');a=p.parse_args();output=Path(a.output_root).resolve()
    if a.verify:verify(output);return
    if not a.source_root or not 0<=a.rate<1 or a.mask_seed<0:p.error('Require source-root,0<=rate<1,nonnegative mask-seed')
    source=Path(a.source_root).resolve();disjoint(source,output);disjoint(REPO,output)
    if output.exists():raise FileExistsError(f'Use a new output root: {output}')
    if len(set(a.tasks))!=len(a.tasks):p.error('Duplicate tasks')
    output.mkdir(parents=True);dump(output/'INCOMPLETE.json',{'status':'preparing'})
    for task in a.tasks:prepare_task(source,output,task,a.rate,a.mask_seed)
    dump(output/'missingness_manifest.json',{'version':'original_protocol_primary_MCAR_v1','repo':str(REPO),
         'source':str(source),'tasks':a.tasks,'rate':a.rate,'mask_seed':a.mask_seed,
         'columns':{task:TASKS[task]['cols'] for task in a.tasks},
         'protocol':'original_loader_frozen_membership_and_seed_rules_unchanged',
         'code_hashes':code_fingerprints(),
         'input_hashes':{str(f.relative_to(output)):sha(f) for task in a.tasks for folder in ('data','std_test')
                         for f in (output/task/folder).iterdir() if f.is_file()}})
    (output/'INCOMPLETE.json').unlink();verify(output)


if __name__=='__main__':main()
