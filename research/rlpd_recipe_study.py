"""Prospective full factorial recipe contract and complete-panel paired analysis.

This module does not submit jobs, train models or arm a continuation. The new
study never substitutes outcomes from the already frozen native/posthoc cohorts.
"""

from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import re

import numpy as np

COHORT = 'rlpd_recipe_factorial_20261020'
TASKS = {'triple2': 'cube-triple-play-singletask-task2-v0',
         'double2': 'cube-double-play-singletask-task2-v0'}
SEEDS = tuple(range(20310, 20320))
CAPACITY = {'small': [256, 256], 'large': [512, 512, 512, 512]}
FACTORS = ('entropy_backup', 'target_reduction', 'capacity')
LEVELS = {'entropy_backup': (True, False), 'target_reduction': ('min', 'mean'),
          'capacity': ('small', 'large')}
BASE = (True, 'min', 'small')
GRID = list(range(0, 1000001, 100000))
COMPONENTS = ('actor', 'critic', 'target_critic', 'temp', 'rng')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def case_digest(case):
    return hashlib.sha256(canonical(case).encode()).hexdigest()


def flags(argv):
    result = {}
    for arg in argv[2:]:
        if not arg.startswith('--') or '=' not in arg:
            raise ValueError('Expected explicit key=value flags')
        key, value = arg[2:].split('=', 1)
        if key in result:
            raise ValueError('Duplicate flag: '+key)
        result[key] = value
    return result


def _recipe(bits):
    return tuple(LEVELS[name][bit] for name, bit in zip(FACTORS, bits, strict=True))


def build_plan(source_commit, source_binding_sha256, data, registered_at):
    if (re.fullmatch('[0-9a-f]{40}', source_commit) is None
            or re.fullmatch('[0-9a-f]{64}', source_binding_sha256) is None):
        raise ValueError('Source identity must be concrete before registration')
    cases = []
    cells = list(itertools.product((1, 20), (0, 1), (0, 1), (0, 1)))
    for seed_index, seed in enumerate(SEEDS):
        tasks = list(TASKS) if seed_index % 2 == 0 else list(reversed(TASKS))
        for task in tasks:
            shift = (seed_index*5 + list(TASKS).index(task)*3) % 16
            for utd, *bits in cells[shift:] + cells[:shift]:
                backup, reduction, capacity = _recipe(bits)
                group = f'rrf_{task}_u{utd}_b{int(backup)}_q{reduction}_c{capacity}_20261020'
                opts = {'agent': 'agents/rlpd_recipe_ablation.py',
                    'agent.backup_entropy': str(backup), 'agent.recipe_target_reduction': reduction,
                    'agent.actor_hidden_dims': str(tuple(CAPACITY[capacity])),
                    'agent.value_hidden_dims': str(tuple(CAPACITY[capacity])),
                    'agent.num_qs': '10', 'agent.num_min_qs': '2',
                    'agent.target_entropy': '-2.5', 'agent.actor_lr': '0.0003',
                    'agent.critic_lr': '0.0003', 'agent.temp_lr': '0.0003',
                    'agent.layer_norm': 'True', 'agent.batch_size': '256',
                    'agent.discount': '0.99', 'agent.tau': '0.005',
                    'agent.bc_alpha': '0.0', 'agent.fixed_alpha': '-1.0',
                    'env_name': TASKS[task], 'seed': str(seed), 'utd_ratio': str(utd),
                    'horizon_length': '1', 'online_steps': '1000000',
                    'start_training': '5000', 'buffer_size': '1000000',
                    'eval_at_start': 'True', 'eval_interval': '100000',
                    'eval_episodes': '50', 'eval_seed': '20261030', 'video_episodes': '0',
                    'save_interval': '1000000', 'log_interval': '5000',
                    'dataset_proportion': '1.0', 'sparse': 'False',
                    'stage_balanced_sampling': 'False', 'ds_mode': 'none',
                    'actor_update_mode': 'sac', 'agent.action_chunking': 'True',
                    'save_dir': '/workspace/results/'+COHORT+'/science', 'run_group': group}
                index = len(cases)
                cases.append(dict(index=index, lane=index % 4, task=task, seed=seed,
                    utd_ratio=utd, entropy_backup=backup, target_reduction=reduction,
                    capacity=capacity, run_group=group,
                    argv=['/workspace/venv/bin/python', 'main_rlpd_recipe_ablation.py',
                          *['--'+key+'='+value for key, value in opts.items()]]))
    primary = [dict(task=task, factor=factor, utd_ratio=20,
                    baseline=dict(zip(FACTORS, BASE, strict=True)),
                    intervention={**dict(zip(FACTORS, BASE, strict=True)),
                                  factor: LEVELS[factor][1]})
               for task in TASKS for factor in FACTORS]
    return dict(format='rlpd-recipe-factorial-v1', cohort=COHORT,
        registered_at=registered_at, status='protocol_frozen_not_armed',
        timing='Post-native-outcome design on two development tasks; fresh-seed paired interventions, not held-out-task confirmation.',
        training_source_commit=source_commit, source_binding_sha256=source_binding_sha256,
        intended_remote_release='/workspace/releases/rlpd-recipe-'+source_commit[:12],
        tasks=TASKS, training_seeds=list(SEEDS), expected_runs=320, cases=cases,
        factors=dict(utd_ratio=[1, 20], **{k:list(v) for k,v in LEVELS.items()}),
        interpretation=dict(entropy='Toggle TD entropy backup only; actor entropy and temperature learning stay enabled.',
            target='Minimum versus mean of the same two randomly sampled target heads from ten; not a ten-head mean or single-Q comparison.',
            capacity='Joint actor/critic architecture package, 256x2 versus 512x4; does not separate width, depth, initialization or actor versus critic capacity.',
            utd='Critic updates per collected transition; actor and temperature update once per update call.'),
        data=deepcopy(data), prior_online_ratio=[.5,.5], environment_steps_per_run=1000000,
        start_training=5000, expected_actor_temperature_updates=995001,
        expected_critic_updates={'1':995001, '20':19900020},
        total_training_environment_steps=320000000,
        evaluation=dict(steps=GRID, actual_episodes_per_point=50, seed=20261030,
                        primary='normalized success AUC on actual 0-1M grid, percentage points',
                        secondary=['endpoint','tail3','right-censored time-to-50%']),
        primary_analysis=dict(comparisons=primary, paired_seeds_required=10,
            interval=dict(method='paired percentile bootstrap; approximate descriptive coverage',
                          draws=20000, seed=20261031, coverage=1-.05/6,
                          family='six local one-factor repair contrasts at UTD20'),
            report=['all ten paired differences','mean','median','positive/zero/negative counts','interval']),
        secondary_analysis=dict(factorial_effects='All seven E/Q/C effects at UTD1 and UTD20, with difference of each effect across UTD; no added intervals.',
            combinations='All four multi-factor repairs versus the baseline at UTD20; no added intervals.',
            inference='No exact p value, post-hoc equivalence claim or unique-root-cause rule.'),
        pairing=dict(actual_constructor_components=list(COMPONENTS),
            initialization='Within each task/seed/architecture, components must match across all eight UTD/entropy/target cells; no parameter equality across architectures.',
            warmup='All sixteen task/seed cells must have identical actual first-5000 raw-action hashes.',
            simulator='Parameter and action-prefix hashes do not certify complete simulator-state or execution-distribution matching.'),
        integrity=dict(read_outcomes_only_after_all_320_actual_terminal=True,
            full_task_panel_required_before_group_estimates=160,
            scientific_retries=False, overwrite=False, replace_seeds=False,
            preserve_failed_and_zero_cases=True),
        execution=dict(job_id='a21566001929515008267857', gpus=4,
            existing_queue_unchanged=True, intended_after='existing matched-map 60-run queue',
            armed=False, performance_gate=None,
            prerequisites=['freshness audit','exact-argv parser audit','numerical/target checks',
                'isolated frozen source and data binding','all sixteen factor cells in deferred real 4090 engineering',
                'actual initialization, warmup and independent checkpoint restoration checks',
                'separate runtime/exit-observation acceptance before arm']),
        limits=['No cohort is launched by generating this plan.',
                'No old native/posthoc outcome or seed is reused as a cell.',
                'This design tests the declared recipe levels on native Cube policies; a repaired-recipe posthoc comparison would be a separate matched study.',
                'A factorial interaction or several positive repairs need not imply one unique cause.',
                'This is not an exhaustive search over all Q targets or network architectures.'])


def verify_plan(plan):
    rebuilt = build_plan(plan['training_source_commit'], plan['source_binding_sha256'],
                         plan['data'], plan['registered_at'])
    if canonical(plan) != canonical(rebuilt):
        raise ValueError('Frozen factorial plan differs from its declared contract')
    return plan


def load_plan(path, expected_sha256):
    if digest(path) != expected_sha256:
        raise ValueError('Registered plan bytes changed')
    return verify_plan(json.loads(Path(path).read_text()))


def contrast_weights(factors, fixed=None):
    """Finite differences, averaging over the unfixed nuisance factors.

    A main effect averages four pairs; a pair interaction averages two
    differences-in-differences; the triple interaction is an eight-cell contrast.
    """
    fixed = fixed or {}
    if not set(factors) <= set(FACTORS) or not factors or len(set(factors)) != len(factors):
        raise ValueError('Invalid factorial contrast')
    if set(factors) & set(fixed) or not set(fixed) <= set(FACTORS):
        raise ValueError('An intervention factor cannot also be fixed')
    rows = []
    for bits in itertools.product((0,1), repeat=3):
        recipe = _recipe(bits); values = dict(zip(FACTORS,recipe,strict=True))
        if any(values[k] != v for k,v in fixed.items()):
            continue
        sign = np.prod([(-1,1)[bits[FACTORS.index(f)]] for f in factors])
        weight = float(sign / 2**(3-len(factors)-len(fixed)))
        rows.append((recipe,weight))
    if not rows:
        raise ValueError('Unknown fixed level')
    return rows


def _actual_exit(record, case, kind, source_commit, used):
    path=Path(record[kind+'_exit_path']).resolve()
    if path in used or digest(path) != record[kind+'_exit_sha256']:
        raise ValueError('Missing, changed or reused actual process receipt')
    used.add(path); value=json.loads(path.read_text())
    if (value.get('case_sha256') != case_digest(case) or value.get('case_index') != case['index']
            or value.get('source_commit') != source_commit
            or type(value.get('returncode')) is not int
            or value['returncode'] != record['returncode' if kind=='driver' else 'validation_returncode']
            or not np.isfinite([value['started'],value['ended']]).all()
            or value['ended'] < value['started']
            or type(value.get('pid')) is not int or value['pid']<=0
            or type(value.get('observed_by_pid')) is not int or value['observed_by_pid']<=0):
        raise ValueError('Actual source-bound process exit differs')


def verify_terminal(plan, records):
    """Require every actual terminal receipt before reading any outcome."""
    verify_plan(plan)
    if (len(records)!=320 or {r['index'] for r in records}!=set(range(320))):
        raise ValueError('All 320 case identities must be present before unsealing')
    cases={c['index']:c for c in plan['cases']}; used=set()
    for r in records:
        c=cases[r['index']]
        if (r.get('source_commit')!=plan['training_source_commit']
                or r.get('case_sha256')!=case_digest(c)
                or r.get('source_binding_sha256')!=plan['source_binding_sha256']
                or r.get('status') not in ('complete','failed','needs_review')
                or type(r.get('returncode')) is not int):
            raise ValueError('Case is not an identity-bound actual terminal')
        _actual_exit(r,c,'driver',plan['training_source_commit'],used)
        if r['returncode']==0:
            _actual_exit(r,c,'validator',plan['training_source_commit'],used)
        if r['status']=='complete' and (r['returncode']!=0 or r['validation_returncode']!=0):
            raise ValueError('Failed process marked complete')


def verify_pairing(cases, records):
    """All sixteen methods per task/seed; initialization is architecture scoped."""
    by_index={r['index']:r for r in records}
    for seed in SEEDS:
        panel=[c for c in cases if c['seed']==seed]
        if len(panel)!=16:
            raise ValueError('Incomplete sixteen-cell seed panel')
        rows=[by_index[c['index']] for c in panel]
        if any(re.fullmatch('[0-9a-f]{64}',r.get('warmup_actions_sha256','')) is None for r in rows):
            raise ValueError('Missing actual warmup hashes')
        if len({r['warmup_actions_sha256'] for r in rows})!=1:
            raise ValueError('Warmup commands differ across recipe cells')
        for capacity in CAPACITY:
            subset=[by_index[c['index']] for c in panel if c['capacity']==capacity]
            for component in COMPONENTS:
                hashes=[r.get('initial_component_sha256',{}).get(component,'') for r in subset]
                if any(re.fullmatch('[0-9a-f]{64}',h) is None for h in hashes) or len(set(hashes))!=1:
                    raise ValueError('Actual initial components differ within architecture')


def bound_metrics(record, case, plan):
    """Read the complete curve from source-bound files after the global gate."""
    import csv
    root=Path(record['run_dir'])
    required={'flags.json','effective_agent_config.json','env.csv','online_agent.csv','eval.csv','params_1000000.pkl'}
    if set(record['identity_hashes'])!=required:
        raise ValueError('Missing bound result inputs')
    for name,sha in record['identity_hashes'].items():
        if digest(root/name)!=sha:
            raise ValueError('A bound scientific input changed')
    config=json.loads((root/'effective_agent_config.json').read_text())
    expected=dict(agent_name='rlpd_recipe_ablation',backup_entropy=case['entropy_backup'],
        recipe_target_reduction=case['target_reduction'],actor_hidden_dims=CAPACITY[case['capacity']],
        value_hidden_dims=CAPACITY[case['capacity']],num_qs=10,num_min_qs=2,target_entropy=-2.5)
    if any(config.get(k)!=v for k,v in expected.items()):
        raise ValueError('Effective factors differ')
    if (record['actual_environment_steps']!=1000000
            or record['actual_optimizer_updates']!=dict(actor=995001,temp=995001,
                critic=995001*case['utd_ratio'],target_critic=0)
            or record.get('independent_checkpoint_restore') is not True):
        raise ValueError('Budget/update/restore acceptance differs')
    with (root/'eval.csv').open() as stream:
        evaluations=list(csv.DictReader(stream))
    if ([int(r['step']) for r in evaluations]!=GRID
            or any(int(r['evaluation_episodes_observed'])!=50 for r in evaluations)):
        raise ValueError('Evaluation grid or actual episode counts differ')
    success=[]
    for r in evaluations:
        count=float(r['evaluation_successes_observed']); value=float(r['success'])
        if (not count.is_integer() or not 0<=count<=50 or not np.isfinite(value)
                or not np.isclose(value,count/50,rtol=0,atol=1e-7)):
            raise ValueError('Observed successes differ from reported rate')
        success.append(value)
    return dict(eval_steps=GRID,success=success,auc_percentage=float(np.trapezoid(success,GRID)/1000000*100),
                endpoint_percentage=success[-1]*100,tail3_percentage=float(np.mean(success[-3:])*100))


def analyze(plan, records, metric_reader=bound_metrics):
    verify_terminal(plan,records)
    by_index={r['index']:r for r in records};out=dict(status='complete',primary=[],secondary=[],tasks=[],records=[],cells=[])
    for task in TASKS:
        cases=[c for c in plan['cases'] if c['task']==task]
        rs=[by_index[c['index']] for c in cases]
        missing=[c['index'] for c in cases if by_index[c['index']]['status']!='complete']
        if missing:
            out['status']='needs_review';out['tasks'].append(dict(task=task,status='incomplete_panel',unavailable_indices=missing));continue
        verify_pairing(cases,rs)
        values={}
        for c in cases:
            m=metric_reader(by_index[c['index']],c,plan)
            rates=np.asarray(m['success'],dtype=float)
            if (m['eval_steps']!=GRID or rates.shape!=(11,) or not np.isfinite(rates).all()
                    or np.any((rates<0)|(rates>1))
                    or not np.isclose(m['auc_percentage'],np.trapezoid(rates,GRID)/1000000*100,rtol=0,atol=1e-7)):
                raise ValueError('Metric reader returned an invalid complete curve')
            values[c['seed'],c['utd_ratio'],tuple(c[f] for f in FACTORS)]=m['auc_percentage']
            reached=[step for step,rate in zip(GRID,rates,strict=True) if rate>=.5]
            out['records'].append(dict(index=c['index'],task=task,seed=c['seed'],utd_ratio=c['utd_ratio'],
                **{f:c[f] for f in FACTORS},eval_steps=GRID,success=rates.tolist(),
                auc_percentage=m['auc_percentage'],endpoint_percentage=float(rates[-1]*100),
                tail3_percentage=float(rates[-3:].mean()*100),
                time_to_50=reached[0] if reached else None,time_to_50_censored=not bool(reached)))
        for utd,*bits in itertools.product((1,20),(0,1),(0,1),(0,1)):
            recipe=_recipe(bits)
            rows=[r for r in out['records'] if r['task']==task and r['utd_ratio']==utd
                  and tuple(r[f] for f in FACTORS)==recipe]
            out['cells'].append(dict(task=task,utd_ratio=utd,recipe=list(recipe),n=10,
                mean_auc_percentage=float(np.mean([r['auc_percentage'] for r in rows])),
                mean_endpoint_percentage=float(np.mean([r['endpoint_percentage'] for r in rows])),
                mean_tail3_percentage=float(np.mean([r['tail3_percentage'] for r in rows]))))
        def difference(weights,utd):
            return np.asarray([sum(w*values[seed,utd,recipe] for recipe,w in weights) for seed in SEEDS])
        def describe(delta):
            return dict(seeds=list(SEEDS),paired_differences=delta.tolist(),mean=float(delta.mean()),
                median=float(np.median(delta)),positive=int((delta>0).sum()),zero=int((delta==0).sum()),negative=int((delta<0).sum()))
        for factor in FACTORS:
            fixed={k:v for k,v in zip(FACTORS,BASE,strict=True) if k!=factor}
            delta=difference(contrast_weights((factor,),fixed),20)
            rng=np.random.default_rng(20261031);indices=rng.integers(0,10,(20000,10));coverage=1-.05/6
            out['primary'].append(dict(task=task,factor=factor,**describe(delta),
                descriptive_interval=np.quantile(delta[indices].mean(axis=1),[(1-coverage)/2,(1+coverage)/2]).tolist()))
        for n in (1,2,3):
            for factors in itertools.combinations(FACTORS,n):
                weights=contrast_weights(factors);low=difference(weights,1);high=difference(weights,20)
                for label,delta in [('utd1',low),('utd20',high),('utd20_minus_utd1',high-low)]:
                    out['secondary'].append(dict(task=task,factors=list(factors),contrast=label,**describe(delta)))
        for bits in itertools.product((0,1),repeat=3):
            if sum(bits)<2:
                continue
            recipe=_recipe(bits);delta=difference([(recipe,1.),(BASE,-1.)],20)
            out['secondary'].append(dict(task=task,contrast='combined_repair_utd20',recipe=list(recipe),**describe(delta)))
        out['tasks'].append(dict(task=task,status='complete_160_case_panel',seeds=list(SEEDS)))
    out['scope']='Fresh-seed paired native recipe interventions on development tasks; approximate descriptive intervals, no unique-root-cause or held-out generalization guarantee.'
    return out
