"""Run review families sequentially in fresh processes, with durable queue evidence."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import argparse

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/family_review_20260913'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--file',type=Path,default=OUT/'pilot.csv')
    args=parser.parse_args()
    rows=json.loads(args.file.with_name(args.file.stem+'_expanded.json').read_text())
    status_path=OUT/('queue_status.json' if args.file.stem=='pilot' else args.file.stem+'_queue_status.json')
    families={r['service_id'].split('__')[0] for r in rows}
    priority=['review_be8ad4aa26ae','review_b306040de80f','review_80d83900c0e1','review_b644f8992353',
              'review_f34e7b070979','review_5c5b13bdf534','review_06980a939764','review_78256bd138d4',
              'review_8d6aaf072a61','review_8f1154c018bb']
    events=[]
    for family in sorted(families,key=lambda f:priority.index(f) if f in priority else 99):
        expected={r['service_id'] for r in rows if r['service_id'].startswith(family+'__')}
        def terminal():
            if not (OUT/'pilot.db').exists():return set()
            with sqlite3.connect((OUT/'pilot.db').resolve().as_uri()+'?mode=ro',uri=True) as c:
                return {r[0] for r in c.execute("select service_id from services where status in ('completed','failed')")}&expected
        if terminal()==expected:
            events.append({'family':family,'status':'already_terminal'});continue
        log=OUT/f'{family if args.file.stem=="pilot" else args.file.stem+"_"+family}.log'
        timeout=max(float(r.get('timeout_s') or 1800) for r in rows if r['service_id'] in expected)
        started=time.time()
        event={'family':family,'status':'running','started_at_unix':started,'timeout_s':timeout,'log':str(log)}
        events.append(event)
        status_path.write_text(json.dumps(events,indent=2))
        with log.open('a',encoding='utf-8') as output:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/run_family_review.py'),'--file',str(args.file),'--family-id',family],
                                   cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
            event['pid']=child.pid
            status_path.write_text(json.dumps(events,indent=2))
            try:
                event['exit_code']=child.wait(timeout=timeout)
                event['status']='terminal' if terminal()==expected else 'incomplete_process_exit'
            except subprocess.TimeoutExpired:
                # Windows venv Python is a launcher with a worker child.
                if os.name=='nt':
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,check=False)
                else:child.kill()
                child.wait()
                event['status']='preparation_or_run_budget_exceeded'
        event['elapsed_s']=time.time()-started
        event['terminal_services']=len(terminal())
        status_path.write_text(json.dumps(events,indent=2))
        print(f'{family}: {event["status"]}; {event["terminal_services"]}/3 terminal',flush=True)


if __name__=='__main__':main()
