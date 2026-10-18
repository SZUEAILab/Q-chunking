"""Historical evidence must retain zeros, raw-file identities and actual grid."""

import copy
import csv
import json

import numpy as np
import pytest

from research.action_adapter_legacy import EXPECTED, STEPS, analyze, curve_metrics, digest


@pytest.fixture
def cohort(tmp_path):
    records = []
    for task, horizon, method, seed in sorted(EXPECTED):
        alias = "posthoc" if method == "adapter" else "none"
        directory = tmp_path / task / f"h{horizon}" / "rlpd" / alias / f"seed{seed}"
        directory.mkdir(parents=True)
        flags = dict(seed=seed, horizon_length=horizon, online_steps=1_000_000, eval_episodes=100,
                     env_name=f"cube-triple-play-singletask-{task}-v0", ds_mode=alias,
                     allow_posthoc_direction_speed_rlpd=method == "adapter", num_qs=10)
        (directory / "flags.json").write_text(json.dumps(flags))
        success = np.full(10, 0 if task == "task5" else .2 + .1*(method == "adapter"))
        with (directory / "eval.csv").open("w") as stream:
            writer=csv.DictWriter(stream,fieldnames=["step","success"]);writer.writeheader()
            writer.writerows(dict(step=int(step),success=float(value)) for step,value in zip(STEPS,success,strict=True))
        records.append(dict(task=task,horizon=horizon,method=method,seed=seed,issues=[],
                            path=str(directory.relative_to(tmp_path)),flags=flags,
                            flags_sha256=digest(directory/"flags.json"),eval_sha256=digest(directory/"eval.csv"),
                            metrics=curve_metrics(STEPS,success)))
    index=dict(records=records,data_git_commit="prior-input-revision",limitations=["Lifecycle unverified"])
    return tmp_path,index


def test_all_seeds_and_task5_zero_results_remain_in_analysis(cohort):
    root,index=cohort
    report=analyze(index,root,bootstrap_samples=100)
    assert report["verified_runs"]==200 and report["verified_files"]==400
    assert report["prior_index_data_git_commit"]=="prior-input-revision"
    assert not report["has_step_zero"] and report["observed_auc_window"]==[100000,1000000]
    assert len(report["comparisons"])==10
    for row in report["comparisons"]:
        metric=row["metrics"]["observed_auc"]
        if row["task"]=="task5":
            assert metric["mean_paired_difference"]==0 and metric["zero_pairs"]==10
        else:
            assert metric["mean_paired_difference"]==pytest.approx(.1)
            assert metric["positive_pairs"]==10


def test_auc_uses_only_observed_window_and_tail3_is_separate():
    values=np.linspace(0,1,10)
    metric=curve_metrics(STEPS,values)
    assert metric["observed_auc"]==pytest.approx(.5)
    assert metric["endpoint"]==1
    assert metric["tail3"]==pytest.approx(8/9)


@pytest.mark.parametrize("kind",["missing_step","extra_step_zero","nan","out_of_range"])
def test_incomplete_or_invalid_curves_are_rejected(kind):
    steps,values=STEPS.copy(),np.full(10,.5)
    if kind=="missing_step":steps,values=steps[1:],values[1:]
    if kind=="extra_step_zero":steps,values=np.r_[0,steps],np.r_[0,values]
    if kind=="nan":values[2]=np.nan
    if kind=="out_of_range":values[2]=1.1
    with pytest.raises(ValueError):curve_metrics(steps,values)


def test_changed_raw_file_is_not_recertified_by_old_index(cohort):
    root,index=cohort
    path=root/index["records"][0]["path"]/"eval.csv"
    path.write_text(path.read_text().replace("0.30000000000000004","0.4"))
    with pytest.raises(ValueError,match="file changed"):analyze(index,root,bootstrap_samples=10)


def test_duplicate_alias_is_not_selected_silently(cohort):
    root,index=cohort
    row=next(r for r in index["records"] if r["method"]=="adapter")
    directory=root/row["path"]
    alias=directory.parent.parent/"adapter"/directory.name;alias.mkdir(parents=True)
    (alias/"flags.json").write_bytes((directory/"flags.json").read_bytes())
    with pytest.raises(ValueError,match="file set changed"):analyze(index,root,bootstrap_samples=10)


def test_missing_seed_identity_prevents_aggregate(cohort):
    root,index=cohort
    index=copy.deepcopy(index);index["records"].pop()
    with pytest.raises(ValueError,match="200 unique"):analyze(index,root,bootstrap_samples=10)


def test_unmatched_hyperparameter_prevents_pair_comparison(cohort):
    root,index=cohort
    row=next(r for r in index["records"] if r["method"]=="adapter")
    row["flags"]["num_qs"]=2
    path=root/row["path"]/"flags.json";path.write_text(json.dumps(row["flags"]))
    row["flags_sha256"]=digest(path)
    with pytest.raises(ValueError,match="Unmatched paired settings"):analyze(index,root,bootstrap_samples=10)
