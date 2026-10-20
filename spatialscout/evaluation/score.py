"""Compute the eight benchmark IoU metrics for newly generated model runs.

A valid trajectory ends with finish_mapping and result.finished=true.
Recoverable intermediate tool errors do not disqualify a task.
"""
import argparse
import csv
import json
from pathlib import Path
from spatialscout.paths import CATALOG_PATH
from spatialscout.evaluation.iou import score_scene

KEYS=['all_mean_iou','all_gt0','all_gt50','all_gt90','submitted_mean_iou','submitted_gt0','submitted_gt50','submitted_gt90']

def finished(path):
    last=None
    try:
        with path.open() as stream:
            for line in stream:
                row=json.loads(line)
                if 'tool' in row:last=row
    except (OSError,ValueError):return False
    return bool(last and last['tool']=='finish_mapping' and last.get('result',{}).get('finished') is True)

def metrics(records,macro):
    rows=[]
    for r in records:
        numerators=[r['iou_sum']]+[r['hits'][str(t)] for t in (0.0,0.5,0.9)]
        rows.append([n/d if d else 0.0 for d in (r['gt_count'],r['submitted']) for n in numerators])
    if macro:return dict(zip(KEYS,[sum(r[i] for r in rows)/len(rows) for i in range(8)]))
    nums=[sum(r['iou_sum'] for r in records)]+[sum(r['hits'][str(t)] for r in records) for t in (0.0,0.5,0.9)]
    return dict(zip(KEYS,[n/d if d else 0.0 for d in (sum(r['gt_count'] for r in records),sum(r['submitted'] for r in records)) for n in nums]))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir',type=Path)
    p.add_argument('--batch',choices=['prepared_80','original_10','additional_50','procthor_30','all'],default='prepared_80')
    p.add_argument('--include-unfinished',action='store_true',help='Explicitly score saved predictions even if the trajectory did not finish normally')
    p.add_argument('--output',type=Path)
    a=p.parse_args();run=a.run_dir.resolve()
    catalog=json.loads(CATALOG_PATH.read_text())['scenes']
    batches={'additional_50','procthor_30'} if a.batch=='prepared_80' else {a.batch}
    scenes=[s['name'] for s in catalog if a.batch=='all' or s.get('batch','original_10') in batches]
    records=[];excluded=[]
    for name in scenes:
        d=run/name
        if not all((d/f).is_file() for f in ['ground_truth.json','predictions.json']):
            excluded.append({'scene':name,'reason':'missing GT or predictions'});continue
        if not a.include_unfinished and not finished(d/'agent_trace.jsonl'):
            excluded.append({'scene':name,'reason':'final executed tool is not finish_mapping with finished=true'});continue
        records.append(score_scene(d))
    if not records:raise SystemExit('No eligible tasks found; no scores written.')
    report={'run_dir':str(run),'batch':a.batch,'eligible_tasks':len(records),'expected_tasks':len(scenes),'primary_aggregation':'macro','value_unit':'fraction, multiply by 100 for percent','strict_thresholds':[0,.5,.9],'macro':metrics(records,True),'micro':metrics(records,False),'excluded_tasks':excluded,'scenes':records}
    out=a.output or run/'eight_metrics.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    with out.with_suffix('.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=['aggregation','eligible_tasks',*KEYS]);w.writeheader()
        for mode in ['macro','micro']:w.writerow({'aggregation':mode,'eligible_tasks':len(records),**report[mode]})
    print(f'{len(records)}/{len(scenes)} eligible tasks. Values below are percentages.')
    for mode in ['macro','micro']:
        print(mode+': '+', '.join(f'{k}={100*v:.2f}' for k,v in report[mode].items()))
    print(out)

if __name__=='__main__':main()
