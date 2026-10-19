import copy
import json

import numpy as np
import pytest

from research import gac_comparisons as analysis


def fixture_design():
    rows, contrasts = [], []
    for task, env in analysis.ENVIRONMENTS.items():
        def arm(profile, protocol):
            return dict(cohort=analysis.GAC, task=task, profile=profile, protocol=protocol)
        for protocol in ('online_only', 'prior_balanced'):
            contrasts.append(dict(name=f'{task}_{protocol}_radius', left=arm('r25', protocol),
                                  right=arm('r1', protocol), question='radius and execution package'))
        for profile in ('r1', 'r25'):
            contrasts.append(dict(name=f'{task}_{profile}_prior', left=arm(profile, 'prior_balanced'),
                                  right=arm(profile, 'online_only'), question='prior access'))
            for utd in (1, 20):
                for cohort, prefix in ((analysis.NATIVE, 'aa_reference'), (analysis.POSTHOC, 'aa_posthoc_reference')):
                    contrasts.append(dict(name=f'{task}_{cohort}_{utd}_{profile}',
                        left=dict(cohort=cohort, group=f'{prefix}_{task}_utd{utd}'),
                        right=arm(profile, 'prior_balanced'), question='full recipe comparison'))
        for profile in ('r1', 'r25'):
            for protocol in ('online_only', 'prior_balanced'):
                for seed in analysis.SEEDS:
                    rows.append(dict(name=f'gac_{task}_{profile}_{protocol}_s{seed}', seed=seed,
                                     task=task, profile=profile, protocol=protocol, env_name=env,
                                     plan_sha256='a'*64))
    return dict(contrasts=contrasts), rows


def curve(endpoint):
    rates = np.linspace(0, endpoint, 11)
    reached = [step for step, rate in zip(analysis.GRID, rates) if rate >= .5]
    return dict(eval_steps=list(analysis.GRID), success=rates.tolist(), auc=float(endpoint/2),
                endpoint=float(endpoint), tail3=float(rates[-3:].mean()),
                time_to_50=reached[0] if reached else None, time_to_50_censored=not bool(reached))


def fixture_reports(rows, *, zero=False):
    reports = {}
    for name in analysis.SOURCES:
        definitions = rows if name == analysis.GAC else analysis.reference_rows(name)
        records = []
        for definition in definitions:
            row = dict(name=definition['name'], seed=definition['seed'], source_commit=analysis.SOURCES[name],
                       status='complete', returncode=0, started=100., ended=200.,
                       effective_config=dict(algorithm=name, batch_size=256))
            if name == analysis.GAC:
                endpoint = (.2 if definition['profile'] == 'r1' else .4) + (.2 if definition['protocol'] == 'prior_balanced' else 0)
                row.update(validation_returncode=0, plan_sha256=definition['plan_sha256'], online_steps=1_000_000,
                           actual_updates=dict(analysis.UPDATES), exact_checkpoint_fields=179,
                           env_name=definition['env_name'], replay_protocol=definition['protocol'],
                           radius=1. if definition['profile'] == 'r1' else 2.5,
                           action_execution='identity' if definition['profile'] == 'r1' else 'clip',
                           runtime={'device':'cuda'}, action_summary={'environment_steps':1_000_000},
                           metrics=curve(0 if zero else endpoint))
                row['effective_config'].update(radius=row['radius'], protocol=row['replay_protocol'])
            else:
                endpoint = (.6 if name == analysis.NATIVE else .8) + (.1 if definition['utd'] == 20 else 0)
                row.update(curve(0 if zero else endpoint), budget=1_000_000, observed_eval_episode_counts='verified',
                           cuda_visible_devices=str(analysis.SEEDS.index(row['seed'])),
                           actual_optimizer_updates=dict(actor=995001, temp=995001,
                                                         critic=995001*definition['utd'], target_critic=0))
                row['effective_config']['utd'] = definition['utd']
            records.append(row)
        reports[name] = dict(source_commit=analysis.SOURCES[name], status='complete', expected_runs=len(records),
                             complete_runs=len(records), records=records)
    return reports


def test_all_twenty_four_contrasts_use_the_same_four_registered_pairs():
    cohort, rows = fixture_design(); reports = fixture_reports(rows)
    before = copy.deepcopy(reports)
    result = analysis.summarize(cohort, rows, reports)
    assert result['status'] == 'complete' and result['contrast_count'] == 24
    assert result['available_complete_runs'] == {analysis.GAC:32, analysis.NATIVE:16, analysis.POSTHOC:16}
    radius = result['comparisons'][0]
    assert radius['summary']['auc']['mean'] == pytest.approx(.1)
    assert radius['summary']['auc']['descriptive_bootstrap_95'] == pytest.approx([.1, .1])
    assert [p['seed'] for p in radius['pairs']] == list(analysis.SEEDS)
    assert radius['pairs'][0]['actual_updates']['left'] == analysis.UPDATES
    assert radius['pairs'][0]['effective_config_differences']['radius'] == [2.5, 1.]
    assert radius['pairs'][0]['time_to_50']['left'] == {'time_to_50':None, 'time_to_50_censored':True}
    assert reports == before


def test_failed_seed_keeps_its_identity_and_prevents_every_affected_group_estimate():
    cohort, rows = fixture_design(); reports = fixture_reports(rows)
    failed = reports[analysis.GAC]['records'][0]
    failed.update(status='failed', returncode=1, error='preserved fixture failure')
    reports[analysis.GAC].update(status='needs_review', complete_runs=31)
    result = analysis.summarize(cohort, rows, reports)
    incomplete = [r for r in result['comparisons'] if r['status'] == 'incomplete']
    assert result['status'] == 'incomplete' and incomplete
    for row in incomplete:
        assert len(row['pairs']) == 3 and 'summary' not in row
        assert row['unavailable_seeds'][0]['seed'] == 10
        assert row['unavailable_seeds'][0]['reasons'][0]['status'] == 'failed'
        assert row['unavailable_seeds'][0]['reasons'][0]['error'] == 'preserved fixture failure'


def test_missing_reference_report_does_not_remove_any_registered_contrast():
    cohort, rows = fixture_design(); reports = fixture_reports(rows)
    reports[analysis.NATIVE] = None
    result = analysis.summarize(cohort, rows, reports)
    assert len(result['comparisons']) == 24
    missing = [r for r in result['comparisons'] if r['status'] == 'incomplete']
    assert len(missing) == 8
    assert all(not r['pairs'] and len(r['unavailable_seeds']) == 4 and 'summary' not in r for r in missing)
    assert all(r['unavailable_seeds'][0]['reasons'][0]['status'] == 'report_missing' for r in missing)


def test_all_missing_reports_have_no_scientific_estimates():
    cohort, rows = fixture_design(); result = analysis.summarize(cohort, rows, {})
    assert result['status'] == 'incomplete' and len(result['comparisons']) == 24
    assert all(not r['pairs'] and 'summary' not in r for r in result['comparisons'])


def test_all_zero_outcomes_are_valid_and_keep_censoring():
    cohort, rows = fixture_design(); result = analysis.summarize(cohort, rows, fixture_reports(rows, zero=True))
    assert result['status'] == 'complete'
    for row in result['comparisons']:
        assert row['summary']['auc']['zero_pairs'] == 4
        assert row['summary']['auc']['descriptive_bootstrap_95'] == [0., 0.]
        assert all(p['time_to_50']['left']['time_to_50_censored'] for p in row['pairs'])


def test_adverse_posthoc_results_are_reported_without_a_performance_gate():
    cohort, rows = fixture_design(); reports = fixture_reports(rows)
    for row in reports[analysis.POSTHOC]['records']: row.update(curve(0))
    result = analysis.summarize(cohort, rows, reports)
    adverse = [r for r in result['comparisons'] if r['left']['cohort'] == analysis.POSTHOC]
    assert result['status'] == 'complete' and len(adverse) == 8
    assert all(r['summary']['auc']['negative_pairs'] == 4 and r['summary']['auc']['mean'] < 0 for r in adverse)


@pytest.mark.parametrize('change', ['source','row_source','count','status','duplicate','missing','wrong_seed','exit','boolean_exit',
    'gac_validation_exit','gac_plan','gac_budget','gac_updates','gac_fields','gac_env','gac_protocol','gac_radius',
    'gac_execution','gac_cpu','gac_actions','rlpd_budget','rlpd_counts','rlpd_episode_evidence','rlpd_gpu',
    'curve_grid','curve_range','curve_nan','curve_metric','curve_threshold','curve_censor','elapsed'])
def test_falsified_or_incomplete_complete_evidence_is_rejected(change):
    cohort, rows = fixture_design(); reports = fixture_reports(rows)
    report = reports[analysis.GAC]; gac = report['records'][0]
    native = reports[analysis.NATIVE]['records'][0]
    if change == 'source': report['source_commit'] = 'wrong'
    elif change == 'row_source': gac['source_commit'] = 'wrong'
    elif change == 'count': report['complete_runs'] = 31
    elif change == 'status': report['status'] = 'needs_review'
    elif change == 'duplicate': report['records'][1] = copy.deepcopy(gac)
    elif change == 'missing': report['records'].pop()
    elif change == 'wrong_seed': gac['seed'] = 14
    elif change == 'exit': gac['returncode'] = None
    elif change == 'boolean_exit': gac['returncode'] = False
    elif change == 'gac_validation_exit': gac['validation_returncode'] = None
    elif change == 'gac_plan': gac['plan_sha256'] = 'b'*64
    elif change == 'gac_budget': gac['online_steps'] -= 1
    elif change == 'gac_updates': gac['actual_updates']['actor'] -= 1
    elif change == 'gac_fields': gac['exact_checkpoint_fields'] -= 1
    elif change == 'gac_env': gac['env_name'] = 'wrong'
    elif change == 'gac_protocol': gac['replay_protocol'] = 'prior_balanced'
    elif change == 'gac_radius': gac['radius'] = 2.5
    elif change == 'gac_execution': gac['action_execution'] = 'clip'
    elif change == 'gac_cpu': gac['runtime']['device'] = 'cpu'
    elif change == 'gac_actions': gac['action_summary']['environment_steps'] -= 1
    elif change == 'rlpd_budget': native['budget'] -= 1
    elif change == 'rlpd_counts': native['actual_optimizer_updates']['critic'] -= 1
    elif change == 'rlpd_episode_evidence': native['observed_eval_episode_counts'] = 'flags_only'
    elif change == 'rlpd_gpu': native['cuda_visible_devices'] = ''
    elif change == 'curve_grid': gac['metrics']['eval_steps'][0] = 1
    elif change == 'curve_range': gac['metrics']['success'][0] = -1
    elif change == 'curve_nan': gac['metrics']['success'][0] = float('nan')
    elif change == 'curve_metric': gac['metrics']['auc'] += .01
    elif change == 'curve_threshold': gac['metrics']['time_to_50'] = 1_000_000
    elif change == 'curve_censor': gac['metrics']['time_to_50_censored'] = False
    else: gac['ended'] = 99.
    with pytest.raises(ValueError): analysis.summarize(cohort, rows, reports)


def test_empirical_bootstrap_enumerates_the_registered_four_seed_distribution():
    values = [-.2, 0, .1, .5]; result = analysis.paired_summary(values)
    means = sorted((a+b+c+d)/4 for a in values for b in values for c in values for d in values)
    assert result['mean'] == pytest.approx(.1)
    assert result['negative_pairs'] == 1 and result['positive_pairs'] == 2 and result['zero_pairs'] == 1
    assert result['descriptive_bootstrap_95'] == pytest.approx(np.quantile(means, [.025, .975]))
    with pytest.raises(ValueError): analysis.paired_summary(values[:3])


def test_boolean_optimizer_count_is_not_an_observed_zero_update_count():
    cohort, rows = fixture_design(); reports = fixture_reports(rows)
    reports[analysis.NATIVE]['records'][0]['actual_optimizer_updates']['target_critic'] = False
    with pytest.raises(ValueError): analysis.summarize(cohort, rows, reports)


def test_boolean_threshold_time_is_not_an_observed_step_zero():
    metrics = curve(.8); metrics.update(success=[.8]*11, auc=.8, endpoint=.8, tail3=.8,
                                        time_to_50=False, time_to_50_censored=False)
    with pytest.raises(ValueError): analysis.recompute_curve(metrics, False)


def test_report_writer_keeps_incomplete_rows_and_refuses_overwrite(tmp_path):
    cohort, rows = fixture_design(); result = analysis.summarize(cohort, rows, {})
    output = tmp_path/'report.json'; analysis.write_report(output, result)
    before = output.read_bytes()
    assert output.with_suffix('.md').read_text().count('not estimated') == 24
    with pytest.raises(FileExistsError): analysis.write_report(output, result)
    assert output.read_bytes() == before


def test_existing_markdown_alone_is_also_preserved(tmp_path):
    output = tmp_path/'report.json'; output.with_suffix('.md').write_text('prior attempt')
    with pytest.raises(FileExistsError): analysis.write_report(output, {})
    assert not output.exists()


def save_science_fixture(root):
    cohort, rows = fixture_design()
    positions = {}
    for row in rows:
        gpu = analysis.SEEDS.index(row['seed']); row['gpu'] = gpu
        row['position'] = positions.get(gpu, 0); positions[gpu] = row['position']+1
        row['plan'] = 'plans/'+row['name']+'.json'
        plan = dict(seed=row['seed'], env_name=row['env_name'], replay_protocol=row['protocol'], online_steps=1_000_000,
                    eval_interval=100000, eval_episodes=50, eval_seed=20261019, source_commit=analysis.SOURCES[analysis.GAC],
                    device='cuda', action_execution='identity' if row['profile'] == 'r1' else 'clip',
                    learner={'radius':1. if row['profile'] == 'r1' else 2.5})
        path=root/row['plan']; path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(plan))
        row['plan_sha256'] = analysis.digest(path)
    hashes={r['plan']:r['plan_sha256'] for r in rows}
    cohort.update(cohort=analysis.GAC, source_commit=analysis.SOURCES[analysis.GAC], seeds=list(analysis.SEEDS),
                  expected_runs=32, budget=1_000_000, evaluation_steps=analysis.GRID, evaluation_episodes=50,
                  evaluation_seed=20261019, expected_updates=analysis.UPDATES, contrast_count=24, plan_hashes=hashes)
    (root/'cohort.json').write_text(json.dumps(cohort)); (root/'tasks.json').write_text(json.dumps(rows))
    (root/'freeze-binding.json').write_text(json.dumps(dict(plans=hashes,cohort_sha256=analysis.digest(root/'cohort.json'),
                                                          tasks_sha256=analysis.digest(root/'tasks.json'))))
    return cohort, rows


def test_science_reader_checks_the_real_export_layout_and_rejects_a_changed_plan(tmp_path):
    cohort, rows = save_science_fixture(tmp_path)
    assert analysis.verify_science(tmp_path) == (cohort, rows)
    plan = tmp_path/rows[0]['plan']; plan.write_text(plan.read_text()+' ')
    with pytest.raises(ValueError, match='plan identity'): analysis.verify_science(tmp_path)
