"""Hash-bound final-checkpoint policy diagnostics; no training or outcome selection."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def eligible(config):
    return config['ds_mode']=='none' or (config['ds_mode']=='adapter'
        and config.get('ds_adapter_speed_squash')=='tanh'
        and config.get('ds_adapter_speed_interpretation')=='log'
        and config.get('ds_speed_bound')=='sphere')


def contract(report,tasks,source_commit):
    expected={}
    for task in tasks['tasks']:
        cfg=dict(tasks['common']);cfg.update(task)
        for seed in tasks['seeds']:
            key=(task['name'],seed)
            if key in expected:raise ValueError('Duplicate expected identity')
            expected[key]=cfg
    if report['source_commit']!=source_commit or report['expected_runs']!=len(expected):raise ValueError('Upstream cohort/source mismatch')
    records={}
    for r in report['records']:
        key=(r['name'],r['seed'])
        if key not in expected or key in records:raise ValueError('Unexpected or duplicate receipt identity')
        records[key]=r
    if sum(r['status']=='complete' for r in records.values())!=report['complete_runs']:raise ValueError('Upstream completion count mismatch')
    return expected,records


def bind_run(record,expected,source_commit,run_root):
    if record['status']!='complete' or record.get('returncode')!=0 or record.get('source_commit')!=source_commit:
        raise ValueError('Only validated successful runs may supply checkpoints')
    root=Path(record['run_dir']).resolve();root.relative_to(Path(run_root).resolve())
    flags_path=root/'flags.json';cfg_path=root/'effective_agent_config.json'
    flags=json.loads(flags_path.read_text());config=json.loads(cfg_path.read_text())
    if flags['seed']!=record['seed'] or record['budget']!=flags['online_steps']:raise ValueError('Run seed/budget mismatch')
    for key,value in expected.items():
        if key in ('name','entry','save_dir'):continue
        actual=config.get('action_chunking') if key=='action_chunking' else config.get('target_entropy') if key=='target_entropy' else flags.get(key)
        if actual!=value:raise ValueError('Frozen task mismatch: '+key)
    if config!=record['effective_config'] or digest(cfg_path)!=record['effective_config_sha256']:
        raise ValueError('Effective configuration changed after validation')
    if config['env_action_dim']!=5 or config['horizon_length']!=1:raise ValueError('Expected the frozen H1 five-dimensional driver')
    if flags['ds_mode']=='adapter' and (config.get('adapter_representation')!='cartesian'
        or config.get('adapter_speed_squash')!='tanh' or config.get('adapter_speed_interpretation')!='log'
        or config.get('ds_speed_bound')!='sphere'):raise ValueError('Wrong posthoc recipe')
    checkpoint=root/f"params_{flags['online_steps']}.pkl"
    if digest(checkpoint)!=record['checkpoint_sha256']:raise ValueError('Checkpoint changed after validation')
    hashes={str(p):digest(p) for p in [flags_path,cfg_path]};hashes[str(checkpoint)]=record['checkpoint_sha256']
    return root,flags,config,checkpoint,hashes


def probe_run(record,expected,source_commit,plan,panels,helper):
    import jax
    import jax.numpy as jnp
    from agents.acrlpd import ACRLPDAgent
    from direction_speed.adapter import DirectionSpeedAdapter
    from utils.flax_utils import restore_agent_with_file
    for name in ['agents.acrlpd','direction_speed.adapter','utils.flax_utils']:
        Path(sys.modules[name].__file__).resolve().relative_to(Path(plan['source_root']).resolve())
    root,flags,config,checkpoint,hashes=bind_run(record,expected,source_commit,plan['run_root'])
    panel=panels[flags['env_name']];observations=panel['observations'];noise=panel['noise']
    posthoc=flags['ds_mode']=='adapter';adapter=DirectionSpeedAdapter(speed_squash='tanh',speed_interpretation='log',speed_bound='sphere')
    example=adapter.decompose(np.zeros(5,dtype='float32')) if posthoc and config['adapter_critic_space']!='env' else np.zeros(5,dtype='float32')
    agent=ACRLPDAgent.create(record['seed'],observations[0],example,copy.deepcopy(config))
    agent=restore_agent_with_file(agent,str(checkpoint));before=np.asarray(agent.rng).copy()
    updates=max(0,flags['online_steps']-max(1,flags['start_training'])+1)*flags['utd_ratio']
    if int(agent.network.step)!=updates+1 or int(agent.network.step)!=record['optimizer_step']:
        raise ValueError('Restored optimizer counter mismatch')
    if record['actual_joint_optimizer_updates']!=updates:raise ValueError('Receipt update count mismatch')
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves((agent.network.params,agent.network.opt_state))):
        raise ValueError('Nonfinite restored state')
    dist=agent.network.select('actor')(jnp.asarray(observations))
    loc=np.asarray(dist.raw_distribution.mean());std=np.asarray(dist.raw_distribution.stddev())
    transform=lambda raw:np.asarray(dist.raw_to_policy(jnp.asarray(raw)))
    cases=[]
    for scale in plan['noise_scales']:
        if posthoc:cases.append(helper.compare_forward_maps(loc,std,noise,adapter.compose,transform,scale))
        else:
            raw=loc[:,None,:]+scale*std[:,None,:]*noise[:,:,[0,1,2,4,5]]
            executed=transform(raw.reshape(-1,5)).reshape(*noise.shape[:2],5)
            cases.append(dict(noise_scale=scale,native=helper.conditional_moments(executed)))
    np.testing.assert_array_equal(agent.rng,before)
    if int(agent.network.step)!=updates+1:raise ValueError('Analysis updated model')
    for path,value in hashes.items():
        if digest(path)!=value:raise ValueError('Run artifact changed during analysis')
    return dict(name=record['name'],seed=record['seed'],status='policy_probe_complete',method='posthoc' if posthoc else 'native',
                source_commit=source_commit,checkpoint_sha256=record['checkpoint_sha256'],identity_hashes=hashes,
                optimizer_step=int(agent.network.step),actual_joint_optimizer_updates=updates,panel=panel['identity'],cases=cases,
                scope='Trained checkpoint conditional actions and same-head mapping counterfactual; no policy update or task-performance attribution.')


def enforce_plan(expected,plan):
    selected={key for key,cfg in expected.items() if eligible(cfg)}
    if 'expected_runs' in plan and len(expected)!=plan['expected_runs']:raise ValueError('Registered cohort size mismatch')
    if 'eligible_runs' in plan and len(selected)!=plan['eligible_runs']:raise ValueError('Registered subset size mismatch')
    if 'eligible_identities' in plan:
        identities=[(r['name'],r['seed']) for r in plan['eligible_identities']]
        if len(identities)!=len(set(identities)) or set(identities)!=selected:raise ValueError('Registered checkpoint identities mismatch')
    if 'checkpoint_step' in plan and any(expected[key]['online_steps']!=plan['checkpoint_step'] for key in selected):
        raise ValueError('Registered checkpoint budget mismatch')


def analyze(report,tasks,source_commit,plan,operation):
    expected,records=contract(report,tasks,source_commit);enforce_plan(expected,plan);output=[];seen_dirs=set()
    for key,config in expected.items():
        record=records.get(key)
        if not eligible(config):
            output.append(dict(name=key[0],seed=key[1],status='not_applicable',upstream_status=record.get('status') if record else 'missing'));continue
        if record is None or record['status']!='complete':
            output.append(dict(name=key[0],seed=key[1],status='upstream_unavailable',upstream_status=record.get('status') if record else 'missing'));continue
        try:
            if record.get('returncode')!=0 or record.get('source_commit')!=source_commit:
                raise ValueError('Receipt source or exit-code mismatch')
            path=record['run_dir']
            if path in seen_dirs:raise ValueError('One run directory assigned to multiple identities')
            seen_dirs.add(path);value=operation(record,config)
            if (value.get('name'),value.get('seed'))!=key or value.get('source_commit')!=source_commit:
                raise ValueError('Probe result identity mismatch')
            output.append(value)
        except Exception as error:output.append(dict(name=key[0],seed=key[1],status='needs_review',error=repr(error)))
    groups=[]
    for task in tasks['tasks']:
        cfg=dict(tasks['common']);cfg.update(task)
        if not eligible(cfg):continue
        rows=[r for r in output if r['name']==task['name']];valid=[r for r in rows if r['status']=='policy_probe_complete']
        group=dict(name=task['name'],expected_seeds=tasks['seeds'],available_seeds=[r['seed'] for r in valid],status='complete' if len(valid)==len(tasks['seeds']) else 'incomplete')
        if group['status']=='complete':
            values=[next(c for c in r['cases'] if c['noise_scale']==1.)['posthoc' if r['method']=='posthoc' else 'native'] for r in valid]
            for metric in ['radius_mean','mean_within_state_radius_std','mean_within_state_xyz_covariance_trace']:
                group['per_seed_'+metric]=[r[metric] for r in values];group['mean_'+metric]=float(np.mean(group['per_seed_'+metric]))
        groups.append(group)
    count=sum(r['status']=='policy_probe_complete' for r in output);eligible_count=sum(eligible(c) for c in expected.values())
    return dict(status='complete' if count==eligible_count and report['status']=='complete' else 'needs_review',
                source_commit=source_commit,expected_runs=len(expected),eligible_runs=eligible_count,policy_probe_complete_runs=count,
                records=output,groups=groups,interpretation='Descriptive final-policy geometry. Same-head execution counterfactual does not identify the effect of retraining or the cause of task success.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','helper','source-root','release-manifest','cohort-report','task-config','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();paths=[args.plan,args.helper,args.release_manifest,args.cohort_report,args.task_config]
    hashes={str(x):digest(x) for x in paths};plan=json.loads(args.plan.read_text());manifest=json.loads(args.release_manifest.read_text())
    if Path(plan['source_root']).resolve()!=args.source_root.resolve():raise ValueError('Source root differs from plan')
    if plan['states']<2 or plan['draws']<2 or 1. not in plan['noise_scales']:raise ValueError('Invalid registered sampling plan')
    if digest(args.helper)!=plan['helper_sha256']:raise ValueError('Policy helper changed')
    if manifest['source_commit']!=plan['training_source_commit']:raise ValueError('Wrong frozen release')
    if digest(args.task_config)!=manifest['tasks_sha256']:raise ValueError('Task config is not bound to training release')
    for row in manifest['python_sources']:
        if digest(args.source_root/row['path'])!=row['sha256']:raise ValueError('Source changed: '+row['path'])
    spec=importlib.util.spec_from_file_location('posthoc_policy_helper',args.helper);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    sys.path.insert(0,str(args.source_root.resolve()))
    import jax
    if jax.default_backend()!='cpu':raise ValueError('Use JAX_PLATFORMS=cpu for this diagnostic')
    panels={}
    for env,item in plan['panels'].items():
        path=Path(item['path'])
        if digest(path)!=item['sha256']:raise ValueError('Prior dataset changed')
        with np.load(path) as data:
            observations=data['observations'];valid=np.flatnonzero(data['terminals']==0)
            indices=np.random.default_rng(plan['panel_seed']).choice(valid,size=plan['states'],replace=False);observations=np.asarray(observations[indices],dtype='float32')
        if not np.isfinite(observations).all():raise ValueError('Nonfinite panel')
        noise=np.random.default_rng(plan['noise_seed']).standard_normal((plan['states'],plan['draws'],6)).astype('float32')
        panels[env]=dict(observations=observations,noise=noise,identity=dict(dataset_sha256=item['sha256'],indices=indices.tolist(),panel_sha256=hashlib.sha256(observations.tobytes()+indices.astype('int64').tobytes()).hexdigest(),noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest()))
    result=analyze(json.loads(args.cohort_report.read_text()),json.loads(args.task_config.read_text()),manifest['source_commit'],plan,
                   lambda r,c:probe_run(r,c,manifest['source_commit'],plan,panels,helper))
    if result['expected_runs']!=plan['expected_runs'] or result['eligible_runs']!=plan['eligible_runs']:raise ValueError('Wrong cohort/subset size')
    for row in manifest['python_sources']:
        if digest(args.source_root/row['path'])!=row['sha256']:raise ValueError('Source changed during analysis')
    if any(digest(Path(path))!=value for path,value in hashes.items()):raise ValueError('Analysis contract changed during execution')
    result.update(tool_sha256=digest(Path(__file__)),input_hashes=hashes,backend=jax.default_backend(),optimizer_updates_performed=0)
    args.output.parent.mkdir(parents=True,exist_ok=True);temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(args.output)
    print(json.dumps({k:result[k] for k in ['status','expected_runs','eligible_runs','policy_probe_complete_runs']}))
    if result['status']!='complete':raise SystemExit(2)


if __name__=='__main__':main()
