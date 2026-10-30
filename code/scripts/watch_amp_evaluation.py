"""Continuously evaluate the latest immutable AMP snapshots, without altering learners."""
import argparse,datetime,json,os,subprocess,time
from pathlib import Path
from amp_gait_metrics import analyze_video_directory,compact_summary


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--out',type=Path,default=Path('runs/amp_supervision'))
 p.add_argument('--runs',nargs='+',default=['amp_walk_h8_control','amp_walk_h16'])
 p.add_argument('--interval',type=float,default=30);p.add_argument('--tag',default='');p.add_argument('--video-commands',default='0.5,0,0;0.8,0,0');args=p.parse_args()
 root=args.root.resolve();out=(root/args.out).resolve();out.mkdir(parents=True,exist_ok=True)
 prefix='evaluation_worker'+('_'+args.tag if args.tag else '');status=out/(prefix+'_status.json');events=out/(prefix+'_events.jsonl')
 python=str(root/'.venv/bin/python')
 env=os.environ.copy();env.update(PYTHONPATH=str(root/'code'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',MUJOCO_GL='egl',PYTHONUNBUFFERED='1')
 last_gait={}
 def write(state,**fields):
  if 'gait_metrics' in fields:
   last_gait.update(name=fields.get('name'),iteration=fields.get('iteration'),gait_summary=fields.get('gait_summary'),metrics=fields['gait_metrics'])
  record=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),state=state,**fields)
  if last_gait:record['last_gait_inspection']=last_gait
  temporary=status.with_suffix('.tmp.json');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(status)
  with events.open('a') as f:f.write(json.dumps(record)+'\n')
  print(json.dumps(record),flush=True)
 def run(command,log):
  with log.open('ab') as f:
   child=subprocess.Popen(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
   try:return child.wait(timeout=1200)
   except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=60);return 124
 write('started',runs=args.runs)
 while True:
  jobs=[]
  for name in args.runs:
   candidates=sorted((root/'runs'/name).glob('ckpt_*.pt'))
   # Prioritize current behavior over a backlog of obsolete checkpoints.
   for ckpt in reversed(candidates):
    iteration=int(ckpt.stem.split('_')[-1])
    if iteration%2000:continue
    report=out/f'eval_{name}_{iteration:07d}_perturbed.json'
    if not report.exists():jobs.append((iteration,name,ckpt,report));break
  if not jobs:write('waiting');time.sleep(args.interval);continue
  for iteration,name,ckpt,report in sorted(jobs,key=lambda item:item[0],reverse=True):
   write('evaluating',name=name,iteration=iteration,checkpoint=str(ckpt))
   cmd=['bash','scripts/evaluate_amp_walk.sh',str(ckpt),'--seconds','20','--num-envs','16','--commands','0.3,0,0;0.5,0,0;0.8,0,0','--diagnostics','--reset-perturbation','1','--reset-seed','17','--out',str(report)]
   result=run(cmd,report.with_suffix('.log'))
   if result or not report.exists():write('evaluation_failed',name=name,iteration=iteration,returncode=result);time.sleep(args.interval);continue
   data=json.loads(report.read_text());metrics=[]
   for c in data['commands']:
    d=c.get('gait_diagnostics_alive',{})
    metrics.append({'command':c['command'],'survival_rate':c['survival_rate'],'rmse':c['planar_rmse_mps_alive'],'yaw_mae':c['mean_abs_yaw_error_radps_alive'],'fixed_style_distance':d.get('fixed_expert_pair_distance',{}).get('mean'),'gait':d})
   candidate=all(c['survival_rate']>=.9 and c['rmse']<.25 and c['yaw_mae']<.35 for c in metrics)
   write('evaluated',name=name,iteration=iteration,tracking_candidate=candidate,metrics=metrics,requires_visual_style_review=True,report=str(report))
   video=out/f'video_{name}_{iteration:07d}'
   if not (video/'summary.json').exists():
    write('rendering',name=name,iteration=iteration)
    command=[python,'code/scripts/render_amp_walk.py','--ckpt',str(ckpt),'--commands',args.video_commands,'--seconds','10','--diagnostics','--out',str(video)]
    result=run(command,out/f'video_{name}_{iteration:07d}.log')
    write('rendered' if result==0 else 'render_failed',name=name,iteration=iteration,returncode=result,path=str(video))
   if (video/'summary.json').exists():
    try:
     gait=analyze_video_directory(video)
     write('gait_inspected',name=name,iteration=iteration,path=str(video),gait_summary=str(video/'gait_summary.json'),gait_metrics=compact_summary(gait),tracking_candidate=candidate,requires_visual_style_review=True)
    except Exception as exc:
     write('gait_inspection_failed',name=name,iteration=iteration,path=str(video),reason=str(exc),requires_visual_style_review=True)
  time.sleep(args.interval)

if __name__=='__main__':main()
