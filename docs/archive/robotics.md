# 机器人任务：历史协议

[历史归档索引](index.md) · [文档入口](../index.md)

> 历史归档：各节保留原文的日期、实验身份和结论范围。旧文中的“当前”“推荐”“待做”不代表今天的执行状态；日常规则以 [实验](../current/experiments.md)、[运行与存储](../current/runbook.md) 为准。

## 内容

- [Robot replay-supported actor and stage-balanced replay pilot](#robot-awr-balanced-protocol)
- [Robot direction-only replay-support pilot](#robot-direction-support-protocol)
- [Robot manifold-support discovery protocol](#robot-manifold-support-protocol)
- [Task1 300k manifold-support discovery protocol](#robot-manifold-task1-300k-protocol)
- [Task1 executed-moment-control pilot protocol](#robot-moment-control-task1-pilot-protocol)
- [Fixed-alpha perturbation-control confirmation](#robot-moment-fixed-alpha-task1-confirmation-protocol)
- [Task1 perturbation-only control with fixed SAC temperature](#robot-moment-fixed-alpha-task1-protocol)
- [Fixed-alpha full-moment-matched counterfactual](#robot-moment-matched-fixed-alpha-protocol)
- [Task1 perturbation-only moment-control experiment](#robot-moment-noise-only-task1-pilot-protocol)
- [Task1 bounded residual moment-control pilot](#robot-moment-residual-task1-pilot-protocol)
- [Fixed-alpha perturbation-matched counterfactual](#robot-perturbation-matched-fixed-alpha-protocol)
- [Robot parameter-isolated direction-support pilot](#robot-separate-trunks-protocol)
- [Task1 300k executed-variance-matched discovery protocol](#robot-variance-matched-task1-300k-protocol)

<a id="robot-awr-balanced-protocol"></a>
## Robot replay-supported actor and stage-balanced replay pilot

原文位置：`docs/experiments/robotics/robot_awr_balanced_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

<a id="robot-awr-balanced-protocol--purpose"></a>
### Purpose

The controlled off-policy shell experiment showed that direct actor gradients
through a learned critic can leave replay support, while replay-supported AWR
recovers learning. This pilot asks whether that interface and reward-stage
balanced replay improve OGBench task2 before any fresh-seed confirmation.

<a id="robot-awr-balanced-protocol--fixed-pilot"></a>
### Fixed pilot

- Environment: `cube-triple-play-singletask-task2-v0`, H=1, dense relabeled
  reward, RLPD-style 50/50 prior and online batches.
- Budget: 100k online steps; evaluation every 25k with 20 episodes and fixed
  evaluation seed 20260925.
- Tuning seeds: 40 and 41. They may never enter confirmation.
- Cartesian uses the ordinary TanhNormal actor. DS uses integrated spherical,
  sphere radius, semantic-product regularization, env-action critic, and target
  entropy -5.3187063194210715.
- Replay AWR scores actually executed batch actions with the same critic used
  by the baseline. Weights are `exp((Q_data-Q_policy)/temperature)`, clipped at
  20 and normalized to mean one. The actor maximizes their replay log
  likelihood; it has no direct gradient through Q. Entropy scale is 0.01.
- Stage-balanced sampling divides each prior and online half-batch equally
  across reward stages currently present in that source. It oversamples only
  observed transitions and never creates counterfactual labels.

Conditions include Cartesian SAC/uniform, Cartesian AWR/balanced, DS
SAC/uniform, DS AWR/uniform, DS SAC/balanced, and DS AWR/balanced. For the full
DS intervention, AWR temperatures 0.1, 0.3, and 1.0 are screened.

<a id="robot-awr-balanced-protocol--pilot-selection-rule"></a>
### Pilot selection rule

First reject any candidate with NaN/Inf, critic divergence, or normalized AWR
weight mean outside [0.99, 1.01]. Among full DS AWR+balanced candidates, select
the temperature with the largest mean normalized return AUC; break ties within
2% relative AUC by choosing the larger temperature. The selected full method
must then satisfy at least one progression gate relative to DS SAC/uniform:

1. at least +2 percentage points success at 100k; or
2. at least +0.05 normalized-return AUC and both paired seeds improve; or
3. at least +5 percentage points in the fraction of online states beyond stage
   zero, with both seeds improving.

If no gate passes, do not launch confirmation. If one passes, freeze the
temperature and compare the six mechanism conditions on fresh seeds, 500k
steps, 100-episode checkpoint reevaluation. Pilot seeds and smoke seed 30 are
excluded from confirmation.

Normalized return is defined before analysis as
`clip((episode_return + 3000) / 3000, 0, 1)` for the 1,000-step cube-triple
episodes; its AUC includes an explicit zero value at online step zero. Stage
progression is the post-warmup fraction of online environment transitions with
the current dense reward greater than -3.

<a id="robot-awr-balanced-protocol--completed-pilot-outcome"></a>
### Completed pilot outcome

All 16 scheduled runs completed on tuning seeds 40 and 41. Every replay-AWR
condition was finite and its normalized weight mean remained one, so the
result is not a numerical-divergence failure. Nevertheless, Cartesian AWR, DS
AWR with uniform replay, and all three DS AWR + balanced temperatures had zero
normalized-return AUC and zero post-warmup stage progression. Their mean
translation norms were also lower than their SAC controls.

The deterministic selection rule chose temperature 1.0 only because all three
full candidates tied at zero return AUC and the rule prefers the larger
temperature in a tie. Relative to DS SAC with uniform replay, its paired
normalized-return AUC changes were -0.00565 and -0.00427, while stage
progression changed by -0.02174 and -0.01259. Success, return-AUC, and stage
progression gates all failed; `launch_confirmation` is therefore false.

Stage-balanced SAC was a useful mechanism control: it increased mean online
stage progression from 1.72% to 2.67% and increased mean translation norm from
0.714 to 0.825, but did not produce final success at 100k. Ordinary Cartesian
SAC had the strongest 100k discovery signal (normalized-return AUC 0.0164 and
12.44% stage progression). These are two-seed discovery observations, not
confirmatory comparisons.

<a id="robot-direction-support-protocol"></a>
## Robot direction-only replay-support pilot

原文位置：`docs/experiments/robotics/robot_direction_support_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

<a id="robot-direction-support-protocol--motivation-and-hypothesis"></a>
### Motivation and hypothesis

The preceding frozen AWR pilot showed a diagnostic failure rather than
numerical divergence: pure replay AWR reduced the mean translation norm while
DS SAC retained enough movement to reach the second task stage.  This follow-up
tests the factorized hypothesis suggested by direction--speed coordinates:
replay support may be useful for the *direction* decision without replacing
the SAC update that maintains speed and exploration.

The actor therefore keeps its ordinary SAC objective.  For each replay action,
the same environment-action critic computes
`Q(s, a_data) - Q(s, a_policy)`.  Mean-one clipped exponential weights are
formed at a fixed temperature of 1.0, but they multiply only the spherical
direction component of the replay log density.  Speed and the two scalar robot
controls receive no behavior-cloning term.  The added loss is
`-lambda * E[w log pi_direction(d_data | s)]`.

<a id="robot-direction-support-protocol--frozen-discovery-design"></a>
### Frozen discovery design

- Environment and budget: `cube-triple-play-singletask-task2-v0`, 100k online
  steps, evaluation every 25k with 20 episodes and evaluation seed 20260925.
- Fixed representation: integrated spherical DS, sphere bound,
  semantic-product SAC regularization, environment-action critic.
- Fixed sampling: reward-stage balanced prior and online half-batches.
- Tuning seeds: 40 and 41, reused only for discovery and excluded from any
  confirmation claim.
- Direction-support coefficients: 0.1, 1.0, and 10.0.  Temperature is fixed at
  1.0 so this screen changes one hyperparameter only.
- Paired control: the already frozen and completed `ds_sac_balanced` condition
  from `robot_awr_balanced_pilot_100k_20260925` on the same seeds.

<a id="robot-direction-support-protocol--selection-and-launch-rule"></a>
### Selection and launch rule

Reject a coefficient if either run has a non-finite training metric, an AWR
weight mean outside [0.99, 1.01], or a mean translation norm below 90% of its
paired DS-SAC-balanced control.  Among valid candidates, choose the largest
mean normalized-return AUC; within a 2% relative tie choose the smaller
coefficient.

Launch a fresh-seed 500k confirmation only if the selected candidate satisfies
at least one of the following paired gates relative to DS-SAC-balanced:

1. normalized-return AUC improves by at least 0.02 on average and both seeds
   improve;
2. final success improves by at least 2 percentage points on average; or
3. post-warmup second-stage visitation improves by at least 1 percentage point
   on average and both seeds improve, while the translation-norm constraint
   above holds.

The third gate is a mechanism gate, not proof of final task performance.  If it
alone passes, confirmation must retain final success and return AUC as primary
outcomes and stage visitation as secondary.  If no gate passes, the
direction-only replay-support hypothesis is rejected for this task and no
confirmation is launched.

<a id="robot-direction-support-protocol--completed-outcome"></a>
### Completed outcome

All six runs completed and all training metrics were finite with mean-one AWR
weights.  No coefficient satisfied the frozen translation-norm constraint, so
the analyzer selected no candidate and all confirmation gates were false.

The weakest coefficient (0.1) was closest to the control.  Its paired
translation norms were 0.736/0.805 versus 0.826/0.825 for DS SAC balanced, and
its stage-progression fractions were 2.423%/2.947% versus 2.162%/3.185%.
Thus stage visitation moved in opposite paired directions and averaged almost
exactly the same.  Its normalized-return AUCs were 0.00295/0.00018 versus
0/0.00381, again one seed up and one down.  Coefficients 1 and 10 reduced mean
translation norms to roughly 0.44--0.54 and produced little or no stage
progression.

The intended loss-level factorization was therefore not a parameter-level
factorization: direction loss gradients still flowed through the shared actor
trunk and altered the speed head.  A future attempt must use separate
direction/speed trunks, explicit gradient isolation, or a constrained gradient
method.  Merely sweeping this coefficient further is not justified by the
frozen pilot.

<a id="robot-manifold-support-protocol"></a>
## Robot manifold-support discovery protocol

原文位置：`docs/experiments/robotics/robot_manifold_support_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

> Status: complete.  All 12 training runs and all 12 fixed-seed,
> 100-episode checkpoint reevaluations finished successfully.  Neither
> manifold candidate passed the frozen 500k-expansion gate.

<a id="robot-manifold-support-protocol--purpose"></a>
### Purpose

Test whether the controlled matched-support result transfers to neural,
state-conditioned SAC with real OGBench replay and environment-action critics.
This is a discovery pilot, not robot confirmation evidence.

<a id="robot-manifold-support-protocol--conditions"></a>
### Conditions

1. Cartesian TanhNormal plus XYZ unit-L2 execution projection.
2. `exp_tangent`: learned direction center, Gaussian local tangent whose norm
   is mapped to geodesic angle by the sphere exponential map, plus radial speed.
3. `projected_tangent`: the identical state network and parameter heads, but
   direction is `normalize(center + Bv)` with exact
   `-3/2 log(1 + ||v||^2)` direction Jacobian.

Both manifold policies use a unit-sphere speed bound, action-volume reference,
single SAC temperature, target entropy -2.5, and an environment-action critic.
The two manifold conditions differ only in exp-map versus normalized-tangent
retraction.  Cartesian projection controls the unit-norm constraint without
curved policy support.

<a id="robot-manifold-support-protocol--frozen-pilot"></a>
### Frozen pilot

- OGBench cube-triple task1 and task2, H=1;
- fresh seeds 50 and 51;
- 100k online interactions, training starts at 5k;
- evaluation at 25k intervals with 20 fixed-seed episodes;
- ordinary SAC, uniform replay, no reward-stage labels or auxiliary losses;
- final checkpoints saved for a common 100-episode reevaluation if any method
  has nonzero success or a return-AUC screening signal.

Primary screening metric is normalized-return AUC using the exact cube-triple
reward range: OGBench defines each step's reward as completed cubes minus
three, so the 1,000-step return is normalized as
`clip((return + 3000) / 3000, 0, 1)` for both task IDs.  Secondary metrics are
success AUC/final success, occupancy after at least one (`reward > -3`) and at
least two (`reward > -2`) completed cubes, executed XYZ norm, entropy, and
numerical failure rate.  This common-scale correction was made during the
analysis-code audit, before loading the complete 12-run result table.

Expansion to fresh 500k confirmation is allowed only if one manifold method:

1. improves mean normalized-return AUC over Cartesian projection by >=0.02;
2. is positive in at least 3/4 task-by-seed pairs;
3. retains at least 90% of Cartesian mean executed XYZ norm; and
4. has no non-finite run.

The exp-map versus projected-tangent comparison is considered practically
similar in this pilot if their mean normalized-return AUC differs by <=0.02 and
neither has more than one task-by-seed loss exceeding 0.05.  These gates were
frozen before launching the 12 runs.

<a id="robot-manifold-support-protocol--results"></a>
### Results

All runs were finite.  Mean normalized-return AUC, averaged over the two seeds
within each task, was:

| Task | Cartesian projection | Exp tangent | Projected tangent |
|---|---:|---:|---:|
| task1 | 54.719% | 53.745% | 53.927% |
| task2 | 0.440% | 1.769% | 1.011% |

Across the four task-by-seed pairs, exp tangent minus Cartesian was
`[-3.548, +1.600, +0.649, +2.008]` percentage points (mean `+0.177`): three
pairs were positive and the mean executed-translation-norm ratio was 0.945,
but the mean improvement missed the frozen `+2`-point gate.  Projected tangent
minus Cartesian was `[+0.316, -1.900, +1.342, -0.201]` points (mean `-0.111`):
only two pairs were positive, with norm ratio 0.943.  Neither method therefore
launches a 500k confirmation.

Projected tangent minus exp tangent was `[+3.864, -3.500, +0.693, -2.209]`
points.  Its signed mean difference was `-0.288` points and no pair lost more
than 5 points, so the two sphere retractions passed the frozen practical-
similarity rule.  This supports the controlled finding that the coordinate map
is not the principal effect, but it does not establish a robot return gain.

The common 100-episode reevaluation gave per-checkpoint success rates
Cartesian `[1, 0, 0, 0]%`, exp tangent `[2, 0, 0, 0]%`, and projected tangent
`[0, 0, 0, 0]%` in task1-seed50, task1-seed51, task2-seed50, task2-seed51
order.  A zero count in 100 episodes still has a 95% Wilson upper bound of
3.70%; the isolated 1% and 2% estimates have intervals `[0.18, 5.45]%` and
`[0.55, 7.00]%`.  These heavily overlapping intervals are discovery-level
noise, not evidence of success-rate separation.

A post-hoc conditional-geometry diagnostic then sampled 32 actions at each of
512 shared prior-data states for every checkpoint and applied the actual
execution postprocessor.  Mean angular deviation in task1 was 0.161/0.414/0.548
rad for Cartesian/exp/projected, and 0.239/0.465/0.495 rad in task2.  Mean XYZ
radius was 0.981/0.895/0.889 in task1 and 0.979/0.885/0.870 in task2; mean
within-state radial standard deviation was 0.027/0.101/0.108 and
0.043/0.111/0.124, respectively.  Thus both manifold actors materially changed
the conditional action distribution and broadened angular exploration.  The
negative return result is not explained by an inactive intervention; broader
angular/radial exploration was simply insufficient under this task and budget.

Canonical artifacts are
`/data/chainpray/qc/reports/analysis/robot_manifold_support_pilot_per_run.csv`,
`/data/chainpray/qc/reports/analysis/robot_manifold_support_pilot_summary.csv`,
`/data/chainpray/qc/reports/analysis/robot_manifold_support_pilot_decision.json`, and
`/data/chainpray/qc/reports/analysis/robot_manifold_support_pilot.png`.  The fixed-state diagnostics are
`/data/chainpray/qc/reports/analysis/robot_manifold_policy_geometry.csv`, its `_summary.csv`, and its
`.png` figure.

<a id="robot-manifold-task1-300k-protocol"></a>
## Task1 300k manifold-support discovery protocol

原文位置：`docs/experiments/robotics/robot_manifold_task1_300k_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

> Status: completed on 2026-09-26; the frozen expansion gate did not pass.
> This is a budget-adequacy discovery study, not a continuation or confirmation
> on seeds 50/51.

<a id="robot-manifold-task1-300k-protocol--motivation"></a>
### Motivation

The completed task1/task2 100k pilot did not pass its expansion gate.  Its
fixed-state diagnostic nevertheless verified that both manifold actors
materially broadened conditional angular exploration.  Independently existing
ten-seed historical curves show that task1, unlike task2--4, has a usable
learning signal within this range: the historical Cartesian H=1 success mean
was 3.7%, 18.8%, and 50.3% at 100k, 200k, and 300k.  We therefore test whether
the 100k null result was a budget floor on the only pre-existing task with a
clear 300k signal.  Task and budget were chosen from those historical curves,
not from candidate performance in the new seeds.

<a id="robot-manifold-task1-300k-protocol--frozen-design"></a>
### Frozen design

- Environment: `cube-triple-play-singletask-task1-v0`, H=1.
- Fresh seeds: 52, 53, 54, 55.
- Budget: 300k online interactions; training starts at 5k.
- Evaluation: 20 fixed-seed episodes every 50k steps.
- Final checkpoint: independent 100-episode fixed-seed reevaluation.
- Trainer: ordinary SAC actor update, uniform replay, prior/online data mixed
  as in the repository, target entropy -2.5.
- All critics consume the five-dimensional executed environment action.

Conditions:

1. Cartesian TanhNormal plus executed XYZ unit-L2 projection.
2. Exponential-map tangent direction plus radial speed.
3. Normalized projected-tangent direction plus radial speed.

The two manifold actors share network architecture, parameter heads, unit-ball
support, action-volume reference, entropy target, and critic interface.  They
differ only in the tangent-to-sphere retraction.

<a id="robot-manifold-task1-300k-protocol--metrics-and-gates"></a>
### Metrics and gates

Primary metric is normalized-return AUC over 0--300k, with each 1,000-step
return normalized as `clip((return + 3000) / 3000, 0, 1)`.  Success AUC,
fixed-reevaluation success, stage occupancy, executed XYZ norm, conditional
policy geometry, entropy, alpha, and critic loss are secondary diagnostics.

A manifold candidate qualifies for a later cross-task confirmation only if it:

1. improves paired mean normalized-return AUC over Cartesian by at least 0.02;
2. is positive in at least 3/4 paired seeds;
3. retains at least 90% of Cartesian mean executed XYZ norm;
4. has no non-finite run; and
5. does not reduce the mean 100-episode final success by more than 0.02.

The support-level hypothesis requires both manifold candidates to pass gates
1--5.  If only one passes, the result is retraction-specific discovery rather
than evidence for manifold support in general.

Exp and projected tangent are practically similar when the absolute signed
mean paired AUC difference is at most 0.02 and no more than one pair loses over
0.05 in either direction.  These rules and all seeds were frozen before launch.

<a id="robot-manifold-task1-300k-protocol--completed-results"></a>
### Completed results

All 12 runs reached 300k steps with finite training metrics and a final
checkpoint.  Every checkpoint was then reevaluated for 100 episodes with the
same independent evaluation seed (`20260925`).

| Condition | Normalized-return AUC | 300k online success (20 ep.) | Fixed success (100 ep.) | Executed XYZ norm |
|---|---:|---:|---:|---:|
| Cartesian projection | 61.88% | 28.75% | 21.75% | 0.772 |
| Exponential tangent | 59.21% | 38.75% | 39.50% | 0.774 |
| Projected tangent | 58.78% | 17.50% | 20.50% | 0.779 |

Exponential tangent minus Cartesian had paired normalized-return AUC changes
of `+3.289/-8.782/-0.302/-4.875` percentage points for seeds 52--55.  The mean
was -2.668 points, its paired bootstrap interval was [-6.829,+1.494], and only
1/4 pairs was positive.  Projected tangent had
`+2.838/-8.801/-2.677/-3.771` points, mean -3.103
[-7.270,+1.186], again with only 1/4 positive pairs.  Both candidates retained
the executed-action norm and had finite metrics, but both failed the primary
+2-point AUC gate and the 3/4 sign gate.  Neither qualifies for cross-task
confirmation.

The fixed 100-episode paired success changes were
`+39/+40/+14/-22` points for exponential tangent and
`+46/-22/-3/-26` for projected tangent.  Thus exponential tangent had a higher
mean final success (+17.75 points) despite lower learning-curve AUC, but this
late, seed-sensitive improvement is not evidence of better sample efficiency.
Projected tangent was within the frozen -2-point noninferiority margin on
average (-1.25 points).  Both final-success gates passed, but cannot override
the failed primary gates.

Projected-minus-exponential AUC differences were
`-0.451/-0.019/-2.375/+1.104` points, with mean -0.435
[-1.786,+0.715].  No pair differed by more than five points, so the two
retractions passed the frozen practical-similarity rule.  The support-level
hypothesis nevertheless failed because neither candidate passed the Cartesian
comparison gates.

<a id="robot-manifold-task1-300k-protocol--conditional-action-geometry-diagnostic"></a>
### Conditional action-geometry diagnostic

For each final policy, 32 stochastic actions were sampled at the same 512
states drawn uniformly through the prior dataset, followed by the actual
environment-action postprocessor.

| Condition | Mean XYZ radius | Within-state radial std | Angular deviation |
|---|---:|---:|---:|
| Cartesian projection | 0.989 | 0.020 | 0.159 rad |
| Exponential tangent | 0.907 | 0.092 | 0.380 rad |
| Projected tangent | 0.902 | 0.096 | 0.579 rad |

The intervention therefore remained behaviorally active at 300k: both
manifold actors produced much wider angular and radial conditional action
distributions.  The lack of AUC improvement cannot be attributed to a dormant
implementation.  In this task and with the shared entropy target, the broader
conditional support instead appears to delay exploitation; projected tangent
also shows substantial seed sensitivity in angular spread (0.426--0.800 rad).
This is a mechanism-consistent interpretation, not a causal proof of the
remaining performance variance.

Machine-readable outputs are in
`/data/chainpray/qc/reports/analysis/robot_manifold_task1_300k_{per_run,summary,decision}` and
`/data/chainpray/qc/reports/analysis/robot_manifold_task1_300k_policy_geometry{,_summary}`.

<a id="robot-moment-control-task1-pilot-protocol"></a>
## Task1 executed-moment-control pilot protocol

原文位置：`docs/experiments/robotics/robot_moment_control_task1_pilot_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

> Status: revision 2 frozen before the controlled 300k pilot restarts.  The
> first controlled processes were stopped before their first 50k evaluation
> because geometry-only logs showed that an additive speed bias was still
> cancelled by the actor.  Revision 2 replaces that bias with an absolute,
> stopped raw-speed mean.  Cartesian/default-tangent baselines had already
> started, but no return or success statistic was inspected while making this
> repair.  Aborted directories are excluded from analysis.

<a id="robot-moment-control-task1-pilot-protocol--causal-question"></a>
### Causal question

The preceding static-offset experiment failed its manipulation check because
the trainable actor and automatic entropy temperature compensated for fixed
log-scale and speed-mean offsets.  This pilot asks whether tangent
direction--speed exploration is competitive with Cartesian projection when
its *executed* conditional action moments are kept close to the Cartesian
reference throughout learning.

The controller is deliberately not an RL component.  It receives a frozen
panel of 512 uniformly spaced prior-data observations and 32 stochastic actor
samples per observation every 5k environment steps.  It never receives
reward, success, Q values, advantages, replay priorities, or evaluation
results.

<a id="robot-moment-control-task1-pilot-protocol--controller"></a>
### Controller

The actor learns direction centers, scalar controls, and all critic parameters
normally.  For this strict causal ablation its ordinary raw-speed-center head
is disconnected, because leaving an additive learned center enabled reproduced
the compensating drift seen in the static-offset experiment.  Three shared
sampler values are stored in the actor parameter tree but wrapped in
`stop_gradient`:

- angular tangent log standard deviation;
- raw speed log standard deviation; and
- an absolute raw-speed mean.

Let `a`, `s`, and `r` denote conditional mean angular deviation, within-state
radius standard deviation, and mean translation radius.  Every 5k steps the
controller applies

```text
log_sigma_angle <- clip(log_sigma_angle + 0.35 log(0.161 / a), -5, 0.5)
log_sigma_speed <- clip(log_sigma_speed + 0.35 log(0.022 / s), -5, 0.5)
speed_mean <- clip(speed_mean + 4.0 (0.989 - r), -6, 6)
```

Initial values are `-1.7`, `-2.1`, and `+0.4`.  The targets
`radius/angular/radial-SD = 0.989/0.161/0.022` are the four-seed Cartesian
means from the completed seeds 56--59 study.  Controller leaves and update
count are checkpointed with the agent.  Each update logs both pre-update and
post-update geometry, target errors, controller values, and update count.

The proportional gains were calibrated only on action geometry.  A separate
seed999, 25k revision-2 smoke run produced post-update
`radius/radial-SD/angle = 0.876/0.0117/0.201` at 5k and
`0.954/0.0126/0.168` at 25k.  Its radius error decreased monotonically while
the other two moments stayed within tolerance; extrapolation is not counted as
a passed checkpoint, so the first formal manipulation decision remains 50k.

<a id="robot-moment-control-task1-pilot-protocol--frozen-experiment"></a>
### Frozen experiment

- Environment: `cube-triple-play-singletask-task1-v0`, horizon 1.
- Pilot seeds: 60 and 61.
- Budget: 300k online interactions; optimization starts at 5k.
- Evaluation: 20 episodes with seed 20260925 every 50k.
- Checkpoint: every 50k, including all six evaluation points.
- Shared learner: SAC, uniform prior/online replay mixture, target entropy
  -2.5, and otherwise identical repository defaults.

Conditions are Cartesian plus L2 projection, default exponential tangent,
and moment-controlled exponential tangent.  The controller condition differs
from default tangent only in the stopped sampler leaves and reward-blind
geometry controller.

<a id="robot-moment-control-task1-pilot-protocol--frozen-gates"></a>
### Frozen gates

The manipulation check passes at a checkpoint when all hold:

- angular absolute error <= 0.05 rad;
- mean-radius absolute error <= 0.03; and
- within-state radial-SD absolute error <= 0.02.

The pilot manipulation gate requires at least five of the six checkpoints
(50k--300k) to pass for each controlled run.  Performance is summarized by
normalized-return AUC over 0--300k, using
`clip((return + 3000) / 3000, 0, 1)`.  Formal seeds 62--65 are launched only
if both pilot seeds have nonnegative paired AUC differences relative to their
Cartesian counterparts and the manipulation gate passes.  The default
tangent condition is diagnostic and is not used to weaken this gate.

No pilot hyperparameter may be changed after return or success curves are
opened.  If manipulation fails, only geometry-only controller repair is
allowed and all repaired runs use new seeds.

<a id="robot-moment-control-task1-pilot-protocol--completed-revision-2-results"></a>
### Completed revision-2 results

All four baseline runs and both valid revision-2 controlled runs reached 300k
with finite training metrics and checkpoints every 50k.  The two aborted
additive-bias directories never reached the first evaluation and are excluded.

| condition | seed 60 return AUC | seed 61 return AUC | mean return AUC | mean success AUC | final success |
|---|---:|---:|---:|---:|---:|
| Cartesian projection | 66.18% | 60.14% | 63.16% | 15.21% | 37.5% |
| Default tangent | 57.49% | 63.69% | 60.59% | 8.33% | 15.0% |
| Moment-controlled tangent | 60.57% | 60.30% | 60.44% | 0.00% | 0.0% |

The manipulation succeeded at all six checkpoints for both controlled seeds.
At 300k their mean executed geometry was radius `0.9859`, within-state radial
SD `0.02196`, and angular deviation `0.16110` rad, versus targets
`0.989/0.022/0.161`.  Thus this is the first robot comparison in the project
where the intended executed-space intervention survived learning.

The paired controlled-minus-Cartesian return-AUC differences were -5.605 and
+0.157 percentage points.  Because seed 60 was negative, the frozen
both-nonnegative pilot gate failed and formal seeds 62--65 must not be
launched.  Relative to default tangent the paired differences were +3.079 and
-3.393 points, mean -0.157 points, so strict moment control did not provide a
consistent repair either.  It produced intermediate task progress (normalized
return near two-thirds) but no complete successes.

This result rejects the narrow claim that excess executed angular/radial
variance alone explains the tangent policy's robot gap.  It does not cleanly
rank representational families because revision 2 disconnects the learned
state-dependent speed center.  The next causal ablation should retain a
bounded state-dependent speed residual around the externally controlled global
mean, use fresh pilot seeds, and again require the manipulation check before
interpreting returns.

<a id="robot-moment-fixed-alpha-task1-confirmation-protocol"></a>
## Fixed-alpha perturbation-control confirmation

原文位置：`docs/experiments/robotics/robot_moment_fixed_alpha_task1_confirmation_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

> Status: frozen before any seed 68--71 run.  Pilot seeds 66/67 passed the
> predeclared expansion rule but are not part of the confirmation decision.

<a id="robot-moment-fixed-alpha-task1-confirmation-protocol--frozen-setup"></a>
### Frozen setup

- Environment `cube-triple-play-singletask-task1-v0`, horizon 1.
- Fresh confirmation seeds 68, 69, 70, 71.
- Conditions: Cartesian L2 projection, default exponential tangent, and
  perturbation-only controlled tangent.
- All conditions use fixed SAC `alpha=0.1`, 300k interactions, training after
  5k, 20 fixed-seed evaluations and checkpoints every 50k.
- Controlled targets: within-state radial SD 0.025 and angular deviation 0.169
  rad on the fixed 512-state x 32-sample reward-blind panel.  Mean radius is
  measured but neither controlled nor gated.

<a id="robot-moment-fixed-alpha-task1-confirmation-protocol--confirmation-gates"></a>
### Confirmation gates

Manipulation/capacity gates:

1. Every controlled seed must pass the angular <=0.05 and radial-SD <=0.02
   tolerance at at least five of six 50k--300k checkpoints.
2. At 300k every controlled seed must retain direction-center SD mean >0.05,
   raw-speed-center SD >0.01, and both scalar-center SDs >0.01 on the shared
   512-state panel.
3. Alpha must be 0.1, alpha loss zero, and all logged values finite.

Primary performance gate, evaluated on confirmation seeds only:

- controlled-minus-Cartesian normalized-return AUC mean >0;
- paired AUC median >0;
- at least three of four paired differences >0.

Success AUC, 300k success, and default tangent comparisons are reported but do
not override the primary gate.  Pilot and confirmation may be pooled only as a
secondary estimate clearly labeled as six seeds.  No hyperparameter or gate
may change after reading confirmation returns.

<a id="robot-moment-fixed-alpha-task1-confirmation-protocol--evidence-motivating-expansion-not-part-of-the-test"></a>
### Evidence motivating expansion, not part of the test

Pilot seeds 66/67 both passed 6/6 manipulation checkpoints and retained
state-dependent centers through 300k.  Controlled-minus-Cartesian AUC was
`+0.1311/+0.0408`; mean controlled AUC was `63.21%` versus Cartesian `54.62%`.
This satisfies the pilot expansion rule but is insufficient for a paper claim
without the independent cohort above.

<a id="robot-moment-fixed-alpha-task1-protocol"></a>
## Task1 perturbation-only control with fixed SAC temperature

原文位置：`docs/experiments/robotics/robot_moment_fixed_alpha_task1_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

> Status: seed-996 engineering gate passed; the fresh seeds 66/67 pilot is
> frozen before launch.  This revision is a response to a diagnosed controller
> conflict, not a post-hoc claim from the failed revision-4 return curves.

<a id="robot-moment-fixed-alpha-task1-protocol--diagnosed-failure-in-revision-4"></a>
### Diagnosed failure in revision 4

Revision 4 correctly disabled mean-radius control and hit both perturbation
targets at all 12 evaluation checkpoints.  Nevertheless, controlled-minus-
Cartesian normalized-return AUC was `-0.02492/-0.09603` on seeds 64/65 and both
controlled runs had zero success AUC.  The full learned action centers again
became constant by 50k.

The upstream cause was automatic-temperature divergence.  Conditional action-
volume entropy stayed near `-5.4` while the common target was `-2.5`; because
angular and radial scales were externally locked, the actor could not satisfy
that target through those scales.  Alpha grew from 1 to about 2,620 by 25k,
`5.95e7` by 50k, and approximately `6e18` by 115k.  This makes revision 4 an
incompatible dual-controller experiment, not a clean rejection of manifold
noise.

<a id="robot-moment-fixed-alpha-task1-protocol--revision-5-intervention"></a>
### Revision-5 intervention

The external reward-blind controller remains responsible for translation
angular/radial perturbation.  SAC temperature adaptation is disabled for every
condition and the entropy coefficient is fixed to `alpha=0.1`.  This value is
the conventional scale and is close to the final alpha of the two fresh
default-tangent controls (`0.1217/0.0967`).  If a performance pilot is launched,
Cartesian, default tangent, and controlled tangent must all use the same fixed
coefficient.

<a id="robot-moment-fixed-alpha-task1-protocol--engineering-gate"></a>
### Engineering gate

- Seed 996; 50k interactions; no evaluation episodes or evaluation interval.
- Save at 25k and 50k; controller panel and targets remain 512 x 32,
  radial SD `0.025`, and angular deviation `0.169` rad.
- `alpha` must equal 0.1 and `alpha_loss` must equal zero at every log point.
- All training and controller metrics must be finite.
- At 50k, angular error <=0.05 and radial-SD error <=0.02.
- On the fixed 512-state panel: mean XYZ direction-center SD >0.05,
  raw-speed-center SD >0.01, and each scalar-center SD >0.01.

Only if every engineering gate passes may fresh seeds 66/67 be used for a
three-condition 300k pilot.  Its manipulation and paired-AUC decision rules
remain identical to revision 4.  No threshold may be relaxed after seeing the
engineering checkpoint.

Seed 996 passed every gate.  Alpha was exactly 0.1 and alpha loss zero at all
ten log points.  At 50k, angular/radial-SD absolute errors were
`0.00121/0.00198`.  Direction-center SD mean was `0.549`, speed-center SD
`0.461`, and scalar-center SDs `0.295/0.738`; corresponding 25k values were
also far from collapse.  The engineering run produced no evaluation file.

The 300k pilot therefore uses fresh seeds 66/67 and three conditions:
Cartesian L2 projection, default exponential tangent, and perturbation-only
controlled tangent.  All three use fixed `alpha=0.1`; all other evaluation,
checkpoint, manipulation, and paired-AUC gates are unchanged from revision 4.

<a id="robot-moment-matched-fixed-alpha-protocol"></a>
## Fixed-alpha full-moment-matched counterfactual

原文位置：`docs/experiments/robotics/robot_moment_matched_fixed_alpha_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

> Status: failed at the frozen seed-995 engineering gate; no performance
> evaluation was run.  This protocol was created after the
> reward-blind geometry audit showed that the earlier fixed-alpha controlled
> condition used substantially less perturbation than its Cartesian control.

<a id="robot-moment-matched-fixed-alpha-protocol--why-another-revision-is-required"></a>
### Why another revision is required

At 300k on confirmation seeds 68--71, fixed-alpha Cartesian policies had mean
radius `0.969005`, within-state radial SD `0.067227`, and angular deviation
`0.345378` rad on the shared 512-state x 32-sample panel.  The earlier
controlled condition reached about `0.933`, `0.025`, and `0.169`, respectively.
Its positive return result is therefore a valid lower-noise ablation, not a
same-moment counterfactual.

The new target is the Cartesian four-seed mean above.  Target construction
uses no additional return statistic.  The controller now controls mean radius
and retains the actor's learned speed center through a bounded residual with
scale 1.0.  All conditions keep fixed SAC `alpha=0.1`.

<a id="robot-moment-matched-fixed-alpha-protocol--seed-995-engineering-gate"></a>
### Seed-995 engineering gate

- 50k interactions, no evaluation episodes, checkpoints at 25k and 50k.
- Exponential-map tangent support with sphere speed bound.
- Controller targets: radius mean `0.969005`, radial SD `0.067227`, angular
  deviation `0.345378` rad on the fixed reward-blind 512 x 32 panel.
- At 50k, absolute errors must be <=`0.02`, <=`0.02`, and <=`0.05`, in that
  order.  At least four of the final five controller updates must meet all
  three tolerances.
- Alpha must be exactly 0.1, alpha loss zero, and all logs finite.
- At 50k, direction-center SD mean must exceed 0.05, raw-speed-center SD must
  exceed 0.01, and both scalar-center SDs must exceed 0.01.

Only if every engineering gate passes may a fresh two-seed pilot be launched.
That pilot will use new seeds 72/73 and compare Cartesian projection, default
tangent, and full-moment-matched tangent for 300k interactions.  Its expansion
rule is fixed in advance: both matched seeds pass at least five of six moment
checkpoints, retain center capacity, and matched-minus-Cartesian return AUC is
nonnegative in both seeds.  Any later confirmation must use new seeds and may
not revise targets or gates after viewing pilot returns.

<a id="robot-moment-matched-fixed-alpha-protocol--frozen-gate-outcome"></a>
### Frozen-gate outcome

At 50k the radius-mean, radial-SD, and angular absolute errors were `0.02120`,
`0.01101`, and `0.00038`.  The first value narrowly missed its fixed `0.02`
tolerance.  More importantly, raw speed-center SD across the panel fell to
`2.38e-7`, far below the `0.01` capacity threshold, while direction and scalar
centers remained non-collapsed.  Thus the actor learned an equal-and-opposite
speed residual against the global mean controller.  This is a failed
engineering design, and seeds 72/73 are not launched under this protocol.

<a id="robot-moment-noise-only-task1-pilot-protocol"></a>
## Task1 perturbation-only moment-control experiment

原文位置：`docs/experiments/robotics/robot_moment_noise_only_task1_pilot_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

> Status: completed; manipulation passed, performance gate failed, and no
> confirmation cohort was launched.  No performance result from this revision
> was inspected when choosing the targets or decision rules.

<a id="robot-moment-noise-only-task1-pilot-protocol--question-and-intervention"></a>
### Question and intervention

Revision 3 controlled a global raw-speed mean as well as angular and radial
noise.  Training then mapped distinct hidden state features to nearly constant
direction, speed, and scalar centers.  Revision 4 removes that pathway.  The
actor learns the complete state-dependent action center:

```text
direction center = normalize(actor_xyz(s))
raw speed center = actor_speed(s)
scalar centers   = actor_scalar(s)
```

The reward-blind controller can change only the shared angular log standard
deviation and shared raw-speed log standard deviation.  The controller still
measures mean translation radius for diagnosis, but neither its value nor a
reward, success, critic value, or replay priority can affect an update.

Targets are frozen from the fresh Cartesian-L2 checkpoints used in the prior
audit: within-state radial SD `0.025` and mean angular deviation `0.169` rad.
The old radius-mean value `0.989` remains in the task schema for backward
compatibility and logging only; `ds_moment_control_speed_mean=false` guarantees
that it is ignored.

<a id="robot-moment-noise-only-task1-pilot-protocol--engineering-run"></a>
### Engineering run

- Environment: `cube-triple-play-singletask-task1-v0`, horizon 1.
- Seed: 997 (engineering only).
- Budget: 25k interactions; training starts at 5k.
- Controller: 512 fixed prior-data states, 32 samples/state, update every 5k.
- Checkpoint: 25k.  Evaluation interval is zero and evaluation episodes are
  zero, so this run supplies no performance evidence.

The revision is mechanically valid only if:

1. the 25k checkpoint and controller log exist and all training diagnostics
   are finite;
2. the final angular absolute error is at most 0.05 rad and radial-SD absolute
   error is at most 0.02;
3. the saved actor has nontrivial across-state variation in direction, raw
   speed, and scalar centers, rather than the near-zero variation observed in
   revision 3.

No radius-mean error threshold is used.

The seed-997 run completed successfully in 25k steps and produced no evaluation
file.  At 25k, angular absolute error was `0.00591` rad and radial-SD absolute
error was `0.000545`; all logged training metrics were finite.  On the frozen
512-state panel, direction-coordinate standard deviations were
`0.598/0.447/0.425`, raw-speed-center SD was `0.00355`, and the two scalar-center
SDs were `0.000900/0.000992`.  These are clearly above the numerical collapse
seen in revision 3 (approximately `1e-6` for XYZ and `1e-8` for scalars), so the
mechanical gate passed.  These diagnostics are feasibility evidence only.

<a id="robot-moment-noise-only-task1-pilot-protocol--frozen-two-seed-pilot-after-engineering-validation"></a>
### Frozen two-seed pilot after engineering validation

If the engineering run is mechanically valid, launch fresh seeds 64 and 65
for three conditions: Cartesian L2 projection, default exponential tangent,
and perturbation-only controlled tangent.  Each run uses 300k interactions,
20 fixed-seed evaluation episodes every 50k, and checkpoints every 50k.  All
learner settings are shared: SAC, target entropy -2.5, horizon 1, and the same
offline/online data pipeline.

For each controlled seed, at least five of the six 50k--300k checkpoints must
simultaneously meet the angular and radial-SD tolerances above.  The
manipulation gate deliberately ignores mean radius.  The small pilot expands
only if controlled-minus-Cartesian normalized-return AUC is nonnegative for
both paired seeds.  Default tangent is diagnostic and cannot relax the gate.
No setting may be changed after inspecting pilot return or success.

<a id="robot-moment-noise-only-task1-pilot-protocol--pilot-result-and-failure-audit"></a>
### Pilot result and failure audit

All six runs completed.  Both controlled seeds passed all six 50k--300k
manipulation checks.  Mean normalized-return AUC was `64.61%` for Cartesian,
`59.26%` for default tangent, and `58.56%` for perturbation-only control;
success AUC was `18.75%/7.71%/0%`, and 300k success was `65.0%/17.5%/0%`.
Controlled-minus-Cartesian paired AUC differences were `-2.492/-9.603`
percentage points, so the expansion gate failed.

The complete learned centers were present in the graph but training still made
them constant by 50k.  At 300k, direction-center state SD was approximately
`1e-6`, speed-center SD `0--1e-7`, and scalar-center SD near machine precision;
default tangent retained order `0.3--0.8` variation.  Automatic alpha provides
the upstream explanation: controlled entropy stayed near `-5.4` against target
`-2.5`, while locked translation scales could not close that error.  Alpha rose
to about 2,620 at 25k, `5.95e7` at 50k, and `6e18` after 115k.  Thus revision 4
tests an incompatible external-moment/automatic-entropy dual loop, not a clean
capacity-matched manifold-noise hypothesis.  The next audit must replace that
automatic loop rather than retune moment targets.

<a id="robot-moment-residual-task1-pilot-protocol"></a>
## Task1 bounded residual moment-control pilot

原文位置：`docs/experiments/robotics/robot_moment_residual_task1_pilot_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

> Status: completed; manipulation passed, performance gate failed, and no
> confirmation cohort was launched.  This revision used fresh pilot seeds
> after the strict revision-2 performance gate failed.  It is not the blocked
> seeds62--65 confirmation cohort from revision 2.

<a id="robot-moment-residual-task1-pilot-protocol--motivation-and-intervention"></a>
### Motivation and intervention

Revision 2 matched all three executed moments at every checkpoint but removed
the learned state-dependent translation-speed center.  It therefore could not
separate a direction--speed representation effect from lost radial policy
capacity.  Revision 3 restores a bounded learned residual:

```text
effective_raw_speed_mean(s)
  = external_global_mean + 0.5 * tanh(actor_speed_residual(s)).
```

The actor receives gradients through the residual, but its global
counteraction is bounded to ±0.5 raw units.  The external global mean and the
angular/radial log standard deviations remain stopped controller leaves.  The
reward-blind 512-state by 32-sample panel, 5k update interval, targets,
controller gains, and clipping bounds are identical to revision 2.

A seed998 engineering smoke run used no evaluation episodes.  At 5/10/15/20/
25k its post-update mean-radius errors were 0.112/0.095/0.065/0.049/0.039;
angular and radial-SD errors were already inside their tolerances.  These data
justify feasibility only and are not performance evidence.

<a id="robot-moment-residual-task1-pilot-protocol--frozen-experiment-and-gates"></a>
### Frozen experiment and gates

- Environment: `cube-triple-play-singletask-task1-v0`, horizon 1.
- Fresh pilot seeds: 62 and 63.
- Conditions: Cartesian L2 projection, default exponential tangent, and
  bounded-residual moment-controlled tangent.
- Budget: 300k interactions; train after 5k; evaluate 20 fixed-seed episodes
  and checkpoint every 50k.
- Shared learner: SAC, uniform prior/online mixture, target entropy -2.5.

For each controlled seed, at least 5/6 checkpoints must simultaneously satisfy
angular error <=0.05 rad, mean-radius error <=0.03, and radial-SD error <=0.02.
Performance passes only if controlled-minus-Cartesian normalized-return AUC is
nonnegative for both seeds.  If both gates pass, a new confirmation cohort
will use seeds64--67; otherwise no confirmation is launched.  Default tangent
is diagnostic and cannot relax either gate.  No return or success result may
be used to modify revision-3 settings.

<a id="robot-moment-residual-task1-pilot-protocol--results"></a>
### Results

All six runs completed normally.  Both controlled seeds passed the geometry
criterion at all six 50k--300k checkpoints.  At 300k, the two-seed mean
`radius/radial-SD/angle` was `0.9860/0.02159/0.16096`, compared with targets
`0.989/0.022/0.161`.

| Condition | Mean normalized-return AUC | Mean success AUC | Mean 300k success |
|---|---:|---:|---:|
| Cartesian L2 projection | 66.89% | 26.46% | 77.5% |
| Default exponential tangent | 59.74% | 7.71% | 32.5% |
| Bounded-residual moment tangent | 58.62% | 0.00% | 0.0% |

The controlled-minus-Cartesian paired AUC differences were -9.179 and -7.359
percentage points for seeds 62 and 63.  The predeclared requirement that both
differences be nonnegative therefore failed, so seeds64--67 were not launched.

<a id="robot-moment-residual-task1-pilot-protocol--capacity-manipulation-audit"></a>
### Capacity manipulation audit

The bounded residual existed in the graph and has a nonzero actor gradient in
the unit test, but the trained actor did not retain state-dependent radial
capacity.  On the shared 512-state panel, the final raw-speed-location standard
deviation was 0 (seed 62) and `4.77e-7` (seed 63).  The external controller
means were 5.59996 and 5.59818, while the effective locations were 5.09996 and
5.09818: in both cases the difference was the exact residual lower bound
`-0.5`.  Thus the actor saturated `tanh(residual)` at -1 and the reward-blind
controller compensated by raising its global mean.

The collapse was not confined to the residual.  At 300k, the mean
state-wise standard deviation of the executed XYZ center was approximately
`1.4e-6/1.8e-6` for the two controlled seeds, versus `0.484/0.484` for default
tangent.  The resultant length of the 512 direction centers was approximately
1.0, versus `0.450/0.428`; scalar raw-location standard deviations were about
`1e-8`, versus order `0.3--0.8`.  The same near-constant centers were already
present at 50k and persisted at every saved checkpoint.  Parameter-tree tests
verify that controller replacement preserves unrelated actor and critic
leaves, and the trained MLP weights differ materially from initialization, so
the current evidence points to closed-loop optimization collapse rather than a
simple checkpoint or parameter-overwrite error.  A layer-wise audit further
shows that the last hidden representation did not become constant: its mean
across-unit state standard deviation was approximately 0.022 at initialization
and 0.031 at both 50k and 300k.  The collapse therefore occurs in the learned
location output mapping, which projects distinct state features onto almost
identical action centers.

This is a reproducible actor--controller interaction rather than evidence that
state-dependent radial capacity is unimportant.  The experiment cleanly shows
that matching three global executed moments is insufficient for good robot
learning, but it does **not** constitute a capacity-matched rejection of a
direction--speed policy.  A fourth controller revision should not be launched
without changing the estimand: the next causal audit should preserve the full
learned action center and control only perturbations around that center, or
study the reference-measure/temperature mechanism directly.

<a id="robot-perturbation-matched-fixed-alpha-protocol"></a>
## Fixed-alpha perturbation-matched counterfactual

原文位置：`docs/experiments/robotics/robot_perturbation_matched_fixed_alpha_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

> Status: seed-994 engineering, seed-72/73 pilot, and fresh seed-74--77
> confirmation gates all passed.  No return from seed 994 was evaluated.

<a id="robot-perturbation-matched-fixed-alpha-protocol--estimand"></a>
### Estimand

This experiment asks whether tangent direction-speed exploration changes
learning when its two conditional *perturbation* magnitudes match fixed-alpha
Cartesian projection: within-state radius SD and within-state angular
deviation.  Mean radius is measured but deliberately uncontrolled because the
full-moment engineering attempt caused the actor's state-dependent speed
center to collapse through controller compensation.

Targets are the Cartesian confirmation-panel means measured without rewards:
radial SD `0.067227` and angular deviation `0.345378` rad.  Fixed `alpha=0.1`
is shared.  The direction, speed, and scalar centers remain actor-controlled.

<a id="robot-perturbation-matched-fixed-alpha-protocol--seed-994-engineering-gate"></a>
### Seed-994 engineering gate

- 50k interactions, no evaluation, checkpoints at 25k and 50k.
- The final five controller updates must each have radial-SD error <=`0.02`
  and angular error <=`0.05`; the 50k point must also pass both.
- Alpha is exactly 0.1, alpha loss is zero, and all logs are finite.
- At 50k, direction-center SD mean >0.05, raw-speed-center SD >0.01, and both
  scalar-center SDs >0.01 on the shared 512-state panel.

If all gates pass, launch the frozen 300k pilot with new seeds 72/73 and three
conditions: Cartesian projection, default tangent, and perturbation-matched
tangent.  Expansion requires at least 5/6 manipulation checkpoints per seed,
all capacity gates, and nonnegative matched-minus-Cartesian normalized-return
AUC in both seeds.  Mean radius remains a reported limitation rather than a
hidden matching claim.

<a id="robot-perturbation-matched-fixed-alpha-protocol--engineering-outcome"></a>
### Engineering outcome

All frozen gates passed.  The final five updates (30k--50k) were within both
tolerances.  At 50k, radial SD was `0.061245` (absolute error `0.005982`) and
angular deviation was `0.344058` rad (absolute error `0.001320`).  Alpha was
0.1, alpha loss was zero, and all values were finite.  Direction, speed, and
the two scalar center SDs were `0.5619`, `0.4610`, `0.3030`, and `0.5413`.
The pilot may therefore proceed exactly as specified above.

<a id="robot-perturbation-matched-fixed-alpha-protocol--pilot-outcome-and-confirmation-rule"></a>
### Pilot outcome and confirmation rule

Both pilot seeds passed all six 50k--300k manipulation checkpoints and the
final center-capacity gate.  Matched-minus-Cartesian normalized-return AUC was
`+0.021528/+0.028401`; mean AUC was `0.631789` versus `0.606824`.  Default
tangent reached `0.582955`.  Final success was `42.5%/37.5%` for matched and
Cartesian, although the two seed-level final-success comparisons were mixed.

The confirmation therefore uses fresh seeds 74--77 with unchanged targets,
controller, fixed alpha, training budget, and evaluation protocol.  It passes
only if every matched seed passes at least 5/6 manipulation checkpoints and
the final capacity gate, while matched-minus-Cartesian normalized-return AUC
has positive mean, positive median, and at least 3/4 positive pairs.

The first scheduler invocation exited before training because the original
run-group string was 66 characters while W&B tags are limited to 64.  The
identifier was shortened to `robot_pert_match_fa_task1_confirm_20260926`; no
algorithmic setting, seed, gate, or budget changed, and the failed invocation
created no experiment data.

<a id="robot-perturbation-matched-fixed-alpha-protocol--confirmation-outcome"></a>
### Confirmation outcome

All four matched seeds passed all six manipulation checkpoints and the final
center-capacity gate.  Matched-minus-Cartesian normalized-return AUC was
`+0.026931`, `+0.064522`, `+0.024190`, and `-0.032885`; the mean was
`+0.020690`, median `+0.025560`, and 3/4 pairs were positive.  Thus every
predeclared primary confirmation criterion passed.  Confirmation mean AUC was
`0.640755` for matched, `0.620066` for Cartesian, and `0.587792` for default
tangent.

The secondary success metrics did not improve: success AUC was `0.18125`
versus Cartesian `0.21146`, and final success was `52.5%` versus `78.75%`.
Accordingly, this experiment supports a task1 normalized-return AUC claim,
not a general success-rate or asymptotic-performance claim.  Pooling pilot and
confirmation only as a secondary estimate gives mean paired AUC `+0.022115`,
median `+0.025560`, and 5/6 positive pairs.

<a id="robot-separate-trunks-protocol"></a>
## Robot parameter-isolated direction-support pilot

原文位置：`docs/experiments/robotics/robot_separate_trunks_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

<a id="robot-separate-trunks-protocol--question"></a>
### Question

The shared-trunk direction-support pilot failed because a loss applied only to
the direction density still changed shared features used by the speed head.
This pilot asks the narrower counterfactual: when direction, speed, and scalar
actors have independent parameters, does replay-supported direction learning
improve exploration without shrinking speed?

An automated gradient test is part of the intervention definition.  For the
direction replay loss, the direction trunk/head gradient norm must be nonzero
and the combined speed/scalar trunk/head norm must be zero to tolerance 1e-10.

<a id="robot-separate-trunks-protocol--frozen-design"></a>
### Frozen design

- Same cube-triple task, 100k budget, 25k evaluations, 20 evaluation episodes,
  tuning seeds 40/41, stage-balanced replay, integrated spherical DS, sphere
  bound, semantic-product regularization, and environment-action critic as the
  preceding discovery pilots.
- All conditions use three independent 4x512 actor trunks: direction, speed,
  and scalar.  This capacity-matched control prevents a parameter-count
  explanation.
- Conditions: separate-trunk SAC; separate-trunk SAC plus direction replay
  support with coefficient 0.1; and the same with coefficient 1.0.
- Direction AWR temperature is fixed at 1.0 and weights are clipped before
  mean-one normalization.  Speed and scalar branches receive only SAC loss.

<a id="robot-separate-trunks-protocol--frozen-decision-rule"></a>
### Frozen decision rule

Reject a treatment if either seed has a non-finite training metric, AWR weight
mean outside [0.99, 1.01], or translation norm below 90% of its paired
separate-trunk SAC control.  Select the valid treatment with largest mean
normalized-return AUC, breaking a relative 2% tie toward coefficient 0.1.

Launch a fresh-seed 500k confirmation only if the selected treatment passes at
least one paired gate versus separate-trunk SAC:

1. mean final success is at least +2 percentage points;
2. normalized-return AUC is at least +0.02 on average and both seeds improve;
3. second-stage visitation is at least +1 percentage point on average and both
   seeds improve, with the translation-norm constraint satisfied.

Pilot seeds cannot be used for confirmation.  If no gate passes, stop tuning
direction replay coefficients on this task and prioritize state-conditional
covariance learning or a different exploration mechanism.

<a id="robot-separate-trunks-protocol--completed-outcome"></a>
### Completed outcome

All six runs completed, all metrics were finite, and the gradient-isolation
test passed.  Nevertheless, neither treatment satisfied the paired
translation-norm constraint, so no candidate was selected and every
confirmation gate was false.

For coefficient 0.1, paired normalized-return AUCs were 0.00154/0.00005 versus
0.00001/0 for capacity-matched SAC, but translation norms were 0.720/0.762
versus 0.815/0.831; seed 40 retained only 88.3%.  Stage progression moved from
3.736%/0.908% to 1.921%/2.352%, again one seed down and one up.

Coefficient 1.0 produced a small paired return-AUC increase
(0.00291/0.00313) but was far below the frozen +0.02 gate.  More importantly,
translation norms fell to 0.503/0.532 and mean stage progression fell from
2.322% to 0.539%.  Direct direction-loss gradients were isolated, so this
behavioral coupling must arise indirectly through changed actions, critic/SAC
gradients, and the visited-state distribution.  No 500k confirmation is
launched, and further coefficient sweeps are not justified.

<a id="robot-variance-matched-task1-300k-protocol"></a>
## Task1 300k executed-variance-matched discovery protocol

原文位置：`docs/experiments/robotics/robot_variance_matched_task1_300k_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

> Status: frozen before launch and completed on 2026-09-26.  No return or
> success result from seeds 56--59 was inspected when the rules below were
> written.  The frozen performance and manipulation gates did not pass.

<a id="robot-variance-matched-task1-300k-protocol--question-and-calibration-boundary"></a>
### Question and calibration boundary

The preceding fresh-seed task1 study found that default exponential tangent
exploration had lower normalized-return AUC than Cartesian projection while
producing substantially wider conditional angular and radial action spreads.
This study asks whether that loss is caused by excessive executed-space
variance rather than by curved manifold support itself.

Calibration used only conditional action geometry from the already completed
seeds 52--55, never their reward to select scale values.  On the same prior
states, post-hoc sampling showed that an angular log-scale offset of `-1.85`
approximately matches exponential tangent angular deviation to Cartesian.
A radial raw-mean offset of `+1.5`, radial log-scale offset of `-0.5`, and cap
of `0.0` were then selected by interpolation to bring mean radius and
within-state radial standard deviation close to Cartesian.  All values below
are frozen for fresh seeds; they will not be retuned from seeds 56--59.

<a id="robot-variance-matched-task1-300k-protocol--frozen-design"></a>
### Frozen design

- Environment: `cube-triple-play-singletask-task1-v0`, H=1.
- Fresh seeds: 56, 57, 58, 59.
- Budget: 300k online interactions; training begins at 5k.
- Evaluation: 20 fixed-seed episodes every 50k; final independent 100-episode
  reevaluation with evaluation seed `20260925`.
- Shared trainer: SAC actor update, uniform replay, the repository's fixed
  prior/online mixture, five-dimensional environment-action critic, and
  target entropy -2.5.

Conditions:

1. Cartesian TanhNormal with executed XYZ unit-L2 projection.
2. Default exponential tangent (`angular offset=-1`, `angular max=0`, no
   radial offsets).
3. Executed-variance-matched exponential tangent (`angular offset=-1.85`,
   `angular max=-1.3`, `speed offset=-0.5`, `speed max=0`, `speed mean
   offset=+1.5`).

The matched and default tangent conditions have identical networks, support,
critic inputs, replay, reference measure, and optimization.  Only explicit
raw sampling-scale/mean transformations differ.

<a id="robot-variance-matched-task1-300k-protocol--metrics-and-frozen-decisions"></a>
### Metrics and frozen decisions

Primary metric is normalized-return AUC over 0--300k, using
`clip((return + 3000) / 3000, 0, 1)`.  Fixed success, online success AUC,
stage occupancy, action norm, entropy/alpha, critic loss, and 512-state by
32-sample conditional action geometry are secondary.

Evidence that over-wide variance caused the default tangent loss requires the
matched condition, relative to default tangent, to satisfy all of:

1. paired mean normalized-return AUC improvement at least +0.02;
2. at least 3/4 paired seeds positive;
3. all training metrics finite;
4. mean fixed-reevaluation success no worse by more than 0.02; and
5. final conditional geometry closer to Cartesian in both angular deviation
   and within-state radial standard deviation.

Evidence that variance-matched curved support improves robot sample efficiency
additionally requires, relative to Cartesian, mean AUC at least +0.02 and at
least 3/4 positive pairs.  If the first set passes but this second comparison
does not, the conclusion is only that calibration repairs over-exploration,
not that manifold support beats Cartesian.

The geometry intervention counts as successfully matched when the absolute
mean angular-deviation error from Cartesian is at most 0.05 rad, mean-radius
error at most 0.03, and radial-standard-deviation error at most 0.02.  Failure
of these manipulation checks makes the return comparison descriptive rather
than a clean variance-matched causal test.

<a id="robot-variance-matched-task1-300k-protocol--completed-results"></a>
### Completed results

All 12 fresh-seed runs reached 300k interactions with finite metrics and a
300k checkpoint.  The final reevaluation used the frozen seed and 100 episodes
per checkpoint.  Conditional geometry used the same 512 prior-data states and
32 stochastic actions per state for every checkpoint.

| condition | normalized-return AUC | online success at 300k | fixed success | radius / within-state radial SD / angular deviation | tail alpha |
|---|---:|---:|---:|---:|---:|
| Cartesian projection | 63.14% | 41.25% | 43.50% | 0.989 / 0.022 / 0.161 rad | 0.025 |
| Default exponential tangent | 58.51% | 17.50% | 24.75% | 0.904 / 0.094 / 0.389 rad | 0.117 |
| Variance-matched exponential tangent | 59.63% | 21.25% | 19.50% | 0.787 / 0.147 / 0.333 rad | 1.313 |

Relative to default tangent, the matched condition's paired normalized-return
AUC differences were `-1.802/+4.140/+2.451/-0.301` percentage points for
seeds 56--59.  The mean was +1.122 points with paired-bootstrap 95% interval
[-1.051,+3.296], and only 2/4 pairs were positive.  Fixed-success differences
were `-27/+53/-40/-7` points, mean -5.25 points.  Thus the +2-point AUC,
3/4-positive, and fixed-success noninferiority gates all failed.

Relative to Cartesian, the matched condition's paired AUC differences were
`-3.682/+0.453/-10.772/-0.026` points: mean -3.507 points, 95% interval
[-8.085,+0.213], and 1/4 positive.  Its fixed-success mean was 24 points lower.
The proposed robot sample-efficiency claim therefore also failed.

Most importantly, the manipulation check failed.  Compared with Cartesian,
the matched policy's angular-deviation, mean-radius, and radial-SD errors were
0.172 rad, 0.202, and 0.125, all far outside the frozen 0.05/0.03/0.02 bounds.
The static offsets slightly reduced angular deviation relative to default
tangent, but made mean radius and radial variation less Cartesian-like.
Consequently these returns cannot be interpreted as a clean causal test of
what would happen under genuinely matched executed-space variance.

The learned-distribution diagnostic explains why the intended match did not
survive training.  On the shared states, default versus matched tangent had
mean effective raw speed location 2.793 versus 1.566, angular raw standard
deviation 0.319 versus 0.273, and speed raw standard deviation 1.159 versus
1.000.  Although the matched configuration added `+1.5` to speed location,
the actor learned a compensating lower location; its entropy temperature rose
to 1.313 while tail entropy remained near the common -2.5 target.  This is
mechanism-consistent evidence that actor adaptation and the single automatic
temperature counteracted static calibration, not proof that either component
alone caused the return loss.

The next causal study must enforce executed-space angular and radial moments
through a non-learnable sampler or an explicit constraint, and keep those
moments outside the shared automatic-entropy feedback loop.  Another sweep of
trainable log-scale offsets would repeat the failed manipulation rather than
test the hypothesis.
