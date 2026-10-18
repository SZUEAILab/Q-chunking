# 合成机制：历史协议

[历史归档索引](index.md) · [文档入口](../index.md)

> 历史归档：各节保留原文的日期、实验身份和结论范围。旧文中的“当前”“推荐”“待做”不代表今天的执行状态；日常规则以 [实验](../current/experiments.md)、[运行与存储](../current/runbook.md) 为准。

## 内容

- [Contextual rotation generalization protocol](#contextual-rotation-protocol)
- [Joint contextual mean--covariance generalization protocol](#joint-contextual-policy-protocol)
- [Cartesian mean-frame counterfactual protocol](#joint-mean-frame-counterfactual-protocol)
- [Axis-aligned learned-shell counterexample](#learned-shell-axis-protocol)
- [Learned sparse two-stage shell MDP](#learned-shell-rl-protocol)
- [Off-policy projected-tangent equivalence protocol](#offpolicy-projected-tangent-protocol)
- [Sparse-shell off-policy critic bridge](#offpolicy-shell-protocol)
- [Projected-tangent Cartesian flow protocol](#projected-tangent-flow-protocol)
- [Controlled staged direction--speed diagnostic](#synthetic-stage-protocol)

<a id="contextual-rotation-protocol"></a>
## Contextual rotation generalization protocol

原文位置：`docs/experiments/synthetic/contextual_rotation_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

<a id="contextual-rotation-protocol--question"></a>
### Question

Does a direction--speed exploration covariance transfer to unseen rotations,
or are the fixed-angle shell results only a memorized coordinate effect?

<a id="contextual-rotation-protocol--task-and-controls"></a>
### Task and controls

For context angle theta, the policy mean is fixed at radius 0.75 and direction
theta. The sparse target is at the same radius and direction theta + 30
degrees. Only the exploration distribution is learned. A hit requires angular
error below 7.5 degrees and radial error below 0.025.

Training contexts are 0, 30, ..., 330 degrees. Held-out contexts are the
midpoints 15, 45, ..., 345 degrees. This isolates rotation transfer of the
exploration distribution; it does not claim neural state or mean-policy
generalization.

Conditions are global Cartesian diagonal, constant Cartesian full covariance,
a 9-parameter full covariance whose Cholesky entries use second-order Fourier
features of context, a Cartesian oracle whose covariance basis rotates along
the target chord, two-parameter direction--speed, and a one-scale coupled DS
ablation. Every condition uses the same REINFORCE estimator, per-context
baseline, samples, optimizer family, train angles, and held-out angles.

<a id="contextual-rotation-protocol--pilot-and-formal-boundary"></a>
### Pilot and formal boundary

Pilot seeds 500--507 may tune learning rate and update count. The pilot is
valid only if the rotating Cartesian oracle and DS both learn above their
initial performance. After freezing settings, formal paired seeds 600--631
must not be used for tuning.

The primary endpoint is held-out-angle success, DS minus Cartesian Fourier
full covariance. Secondary contrasts are DS versus the rotating oracle,
constant full covariance, and coupled DS. Report train/test success,
generalization gap, every held-out angle, paired bootstrap intervals, and the
number of positive pairs. A positive result supports compact rotation transfer
of exploration covariance only; it does not establish robot-task performance.

<a id="contextual-rotation-protocol--frozen-confirmation-configuration-and-decisions"></a>
### Frozen confirmation configuration and decisions

Pilot seeds showed that 600 updates stopped while the Fourier baseline was
still improving. At 1,200 updates, DS and the rotating oracle had plateaued;
the Fourier baseline reached 7.87% on train angles and 6.67% on held-out
angles. The frozen formal configuration is therefore 1,200 updates, batch 512
per training context, Adam learning rate 3e-3, and 20,000 evaluation samples
per angle. Formal paired seeds are 600--631.

Before reading formal seeds, the confirmation gates are fixed as follows:

- DS minus Fourier held-out success >=3 percentage points, paired-bootstrap
  lower bound >0, and at least 28/32 positive seed pairs;
- absolute DS train--test gap <=0.5 percentage points;
- DS is non-inferior to the rotating Cartesian oracle with a -1 percentage
  point margin, both for the mean difference and bootstrap lower bound;
- Fourier improves on constant full covariance by >=3 points, demonstrating a
  learned rather than deliberately broken Cartesian control;
- DS improves on coupled DS by >=8 points;
- mean DS success exceeds mean Fourier success on at least 10/12 held-out
  angles.

No configuration may change after reading seeds 600--631.

<a id="contextual-rotation-protocol--confirmation-outcome"></a>
### Confirmation outcome

All frozen gates passed. On seeds 600--631, held-out success was 12.10% for DS,
11.36% for the rotating Cartesian oracle, 6.62% for Fourier full covariance,
0.78% for constant full covariance, and 0.76% for coupled DS. DS minus Fourier
was +5.48 percentage points with a paired-bootstrap 95% interval of
[+5.38,+5.58] and 32/32 positive pairs. DS minus the rotating oracle was +0.74
points, also 32/32. The DS train--test gap was -0.0005 points, whereas the
Fourier gap was +1.07 points. DS exceeded Fourier at all 12 held-out angles.

The Fourier control improved over constant full covariance by 5.84 points, so
the comparison is not against a control that failed to learn. Its periodic
held-out valleys show that a low-order context-to-Cholesky map does not enforce
rotation equivariance. A richer equivariant Cartesian network could close this
gap; the result establishes compact covariance transfer, not universal
superiority over every possible Cartesian architecture.

<a id="joint-contextual-policy-protocol"></a>
## Joint contextual mean--covariance generalization protocol

原文位置：`docs/experiments/synthetic/joint_contextual_policy_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

<a id="joint-contextual-policy-protocol--question"></a>
### Question

Can direction--speed exploration retain its rotation-aligned sample-efficiency
advantage when the policy mean is learned jointly from a high-dimensional
state, rather than fixed by the experimenter?

<a id="joint-contextual-policy-protocol--task"></a>
### Task

Each context angle is encoded as a 32-dimensional vector containing sine and
cosine harmonics 1 through 16.  The policy begins at a radius-0.75 anchor and
must learn a shared 32--64--2 residual-mean network that reaches a sparse shell
target rotated by 30 degrees.  Simultaneously, it learns its exploration
covariance.  A hit requires angular error below 7.5 degrees and radial error
below 0.025.

Training contexts are every 10 degrees; held-out contexts are the 5-degree
midpoints.  Thus both splits contain 36 rotations.  The initial policy mean,
mean-network architecture, reward, REINFORCE estimator, per-context baseline,
optimizer, samples, and update budget are shared across methods.

<a id="joint-contextual-policy-protocol--conditions"></a>
### Conditions

- Cartesian diagonal state: a 32--64--2 state covariance network.
- Cartesian full-cov state: a larger 32--64--3 Cholesky network.
- Cartesian rotating oracle: two global scales in a target-chord-aligned basis.
- DS global: two global angular/radial scales in the learned mean's local frame.
- DS state: a 32--64--2 state-dependent angular/radial scale network.
- DS coupled: one global local scale shared by angular and radial perturbations.

All methods learn the same Cartesian residual mean.  DS global has 2,244 total
parameters, while Cartesian full-cov state has 4,549 and DS state has 4,484;
the primary comparison therefore does not favor DS by capacity.

<a id="joint-contextual-policy-protocol--pilotformal-boundary"></a>
### Pilot/formal boundary

Pilot seeds 700--727 were used to diagnose the underdetermined 12-context
version and select context density, learning rate, and convergence budget.  In
the 36-context pilot, DS global and the rotating oracle plateaued by roughly
1,600--2,400 updates; Cartesian full covariance changed by less than one point
from 2,000 to 2,400.  The frozen formal configuration is therefore:

- seeds 800--831, excluded from all tuning;
- 2,400 updates, batch 512 per training context, Adam learning rate 2e-3;
- checkpoints 0, 600, 1,200, 1,600, 2,000, and 2,400;
- 10,000 fresh evaluation samples per context and checkpoint.

The 12-context failure is retained as a negative capacity/generalization
diagnostic and is not pooled with formal evidence.

<a id="joint-contextual-policy-protocol--frozen-confirmation-gates"></a>
### Frozen confirmation gates

Primary endpoint: final held-out success of DS global minus Cartesian full-cov
state.  All of the following must pass:

1. mean difference >=10 percentage points, paired-bootstrap 95% lower bound
   >8 points, and at least 28/32 paired seeds positive;
2. held-out learning-AUC difference >=10 points with bootstrap lower bound
   >8 points;
3. DS global is non-inferior to the rotating Cartesian oracle with a -2 point
   margin for both mean and bootstrap lower bound;
4. DS global held-out mean error <=0.03 and train--test success gap <=20 points;
5. Cartesian full covariance beats Cartesian diagonal by >=5 points;
6. DS global beats DS state by >=10 points and DS coupled by >=30 points;
7. DS global beats Cartesian full covariance on at least 30/36 held-out angles.

The claim supported by passing is joint high-dimensional state-to-mean and
covariance generalization in this controlled contextual bandit.  It is not a
robot-performance claim and does not replace fresh multi-task robot evidence.

<a id="joint-contextual-policy-protocol--formal-outcome"></a>
### Formal outcome

The frozen seeds 800--831 completed without missing conditions or checkpoints.
DS global reached 63.03% held-out success, compared with 46.13% for Cartesian
full-cov state, 34.25% for Cartesian diagonal state, 62.54% for the rotating
Cartesian oracle, 45.63% for DS state, and 11.92% for DS coupled.  Its paired
advantage over full-cov state was 16.90 percentage points (95% paired-bootstrap
CI [14.46, 19.33]); all 32/32 paired seeds were positive.  Held-out AUC was
52.54% versus 38.11%, a 14.43-point difference (95% CI [12.42, 16.45]), again
32/32 positive.

DS global was non-inferior to the rotating oracle and differed by only +0.49
points (95% CI [0.06, 0.93]).  Its held-out mean error was 0.0262 and its
train--test success gap was 19.33 points.  Full covariance beat diagonal
Cartesian by 11.88 points; DS global beat state-dependent DS by 17.40 points
and coupled DS by 51.11 points.  Averaged over seeds, DS global exceeded
full-cov Cartesian on all 36/36 held-out angles.  Therefore all 13 frozen
decision criteria passed.

The result supports the stated controlled-bandit claim.  It should not be
described as evidence of robot-task return improvement: the target family has
exact rotational structure, contexts densely cover the circle, and the
experiment has no temporal credit assignment or learned critic.

<a id="joint-mean-frame-counterfactual-protocol"></a>
## Cartesian mean-frame counterfactual protocol

原文位置：`docs/experiments/synthetic/joint_mean_frame_counterfactual_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

<a id="joint-mean-frame-counterfactual-protocol--question"></a>
### Question

Does the positive joint contextual result require sampling in direction--speed
coordinates, or is a Cartesian Gaussian whose covariance rotates with the
learned mean sufficient?

<a id="joint-mean-frame-counterfactual-protocol--intervention"></a>
### Intervention

`Cartesian mean-frame` keeps actions and log densities in Cartesian space.  At
each context it builds a radial unit vector from the current learned mean and a
perpendicular tangent vector, then learns two global standard deviations in
that rotating basis.  Its configured tangential/radial deviations are 0.25 and
0.008192; the shared log-scale floor makes the effective radial deviation
`exp(-4)=0.018315` for both methods, so their first-order local covariances still match.  It has the
same 2,244 parameters, the same residual-mean network, initialization, random
noise, optimizer, samples, and updates as DS global.  The only difference is
straight Cartesian perturbations versus angular/radial perturbations on the
circle.

<a id="joint-mean-frame-counterfactual-protocol--frozen-pilot"></a>
### Frozen pilot

- conditions: Cartesian mean-frame and DS global;
- seeds 900--911;
- 36 training contexts every 10 degrees and 36 held-out midpoint contexts;
- 2,400 updates, batch 512 per context, Adam learning rate 2e-3;
- checkpoints 0, 600, 1,200, 1,600, 2,000, and 2,400;
- 10,000 fresh evaluation samples per context and checkpoint.

This is a new mechanism pilot, not pooled with seeds 800--831.  Interpret the
paired final held-out difference before any follow-up tuning:

1. absolute mean difference <=2 points and its 95% paired-bootstrap interval
   lies within [-3,+3] points: local rotating covariance is sufficient;
2. DS minus mean-frame >=5 points with lower confidence bound >2 points: the
   nonlinear polar sampler contributes beyond first-order covariance;
3. otherwise: inconclusive and no strong coordinate-level claim.

The same comparison is also reported for held-out AUC, mean error, and all 36
angles.  Any later 32-seed confirmation must use fresh seeds and a separately
frozen decision rule.

<a id="joint-mean-frame-counterfactual-protocol--pilot-outcome-and-frozen-confirmation"></a>
### Pilot outcome and frozen confirmation

The untouched 12-seed pilot produced 63.35% held-out success for DS global and
40.12% for Cartesian mean-frame.  The paired difference was +23.24 points
(95% bootstrap CI [16.46, 29.75]), with 11/12 seeds positive.  Held-out AUC was
52.79% versus 32.90%, a +19.90-point difference [14.20, 25.40].  This passes
pilot interpretation rule 2 and rejects first-order local covariance as a
sufficient explanation.

Before running fresh seeds, the formal confirmation is frozen as seeds
1000--1031 with the identical task and optimization settings.  It passes only
if all criteria hold:

1. final DS-minus-mean-frame difference >=10 points, paired-bootstrap lower
   bound >8 points, and at least 28/32 paired seeds positive;
2. held-out AUC difference >=10 points with lower bound >8 points;
3. DS held-out mean error <=0.03;
4. DS exceeds mean-frame on at least 30/36 held-out angles;
5. mean-frame itself reaches at least 25% final held-out success, guarding
   against a non-learning control.

The intended interpretation is finite-rotation manifold support: a straight
tangent perturbation matches DS only to first order, but for a 30-degree move
it increases radius unless paired with an inward radial displacement.  DS
angular sampling follows the radius-preserving arc directly.  The confirmation
does not establish that polar sampling beats an unrestricted expressive flow.

<a id="joint-mean-frame-counterfactual-protocol--formal-outcome"></a>
### Formal outcome

Fresh seeds 1000--1031 confirmed the pilot.  DS global reached 63.24% final
held-out success and Cartesian mean-frame reached 34.47%, a paired difference
of +28.77 points (95% bootstrap CI [24.69, 32.63]); 31/32 seed pairs were
positive.  Held-out AUC was 53.03% versus 28.67%, a +24.36-point difference
[20.89, 27.64], also positive in 31/32 pairs.  DS mean error was 0.0253 and it
won 36/36 angle averages.  The control exceeded its 25% learning floor.  All
eight frozen confirmation criteria passed.

The single negative pair was seed 1013 (66.57% DS versus 68.56% mean-frame),
so the result is strong but not deterministic seed-wise dominance.  Together
with the expressive full-covariance comparison, this supports a specific
claim: rotating a thin Cartesian tangent ellipse is not enough for finite
angular moves on a narrow radius shell; curved angular support matters.  It
does not show that DS dominates a flow or mixture capable of representing the
same curved support.

<a id="learned-shell-axis-protocol"></a>
## Axis-aligned learned-shell counterexample

原文位置：`docs/experiments/synthetic/learned_shell_axis_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

> Status: complete.  The formal equivalence protocol was frozen before
> untouched paired seeds 200--231, and all four checks passed.

<a id="learned-shell-axis-protocol--purpose"></a>
### Purpose

The rotated sparse-shell MDP favors a policy whose radial/tangential covariance
automatically rotates with its mean direction.  A credible scope claim needs a
counterexample.  Here the stage-0 target angle is zero and the initial mean is
already on that Cartesian axis.  The Cartesian full-covariance oracle and DS
stage policy therefore begin with the same local radial/tangential covariance
orientation and area.

All rewards, radii, tolerances, REINFORCE updates, batch size, evaluation size,
learning rate, training length, and policy families are identical to
`learned_shell_rl_protocol.md`; only the target angle changes from 30 degrees
to zero.  Formal estimates use paired seeds 200--231.  Pilot seeds 0--7 are
excluded.

<a id="learned-shell-axis-protocol--pilot"></a>
### Pilot

Pilot AUC was 79.201% for Cartesian full-cov oracle and 79.215% for DS stage,
a difference of only +0.014 points.  Both reached 50% success at 122.88k
stage-0 interactions.  Cartesian stage AUC was 55.981%, showing that the
structured covariance still matters even when DS and full covariance tie.

<a id="learned-shell-axis-protocol--frozen-decision-rules"></a>
### Frozen decision rules

The axis-aligned task is considered an informative equivalence/counterexample
only if all of the following hold:

- The absolute paired mean AUC difference between DS stage and Cartesian
  full-cov oracle is at most 0.5 percentage points.
- The paired bootstrap 95% interval lies entirely inside the equivalence band
  [-1,+1] AUC points.
- Their mean interactions to 50% differ by no more than one evaluation
  interval (20,480 stage-0 interactions).
- Both structured methods exceed Cartesian stage AUC by at least 15 points.

No hyperparameter or equivalence-bound change is allowed after reading formal
seeds 200--231.  Passing supports a bounded claim: DS is a compact,
rotation-aligned inductive bias, not a universally superior policy family.

<a id="learned-shell-axis-protocol--formal-results"></a>
### Formal results

Cartesian full-cov oracle and DS stage obtained AUC 79.215% and 79.260%,
respectively.  Their paired difference was +0.045 AUC points with bootstrap
interval [-0.0046,+0.0919] points, fully inside the preregistered [-1,+1]
equivalence band.  The methods required 123.52k and 122.88k interactions to
reach 50%, a difference of only 640 interactions.  Both exceeded Cartesian
stage by at least 23.24 AUC points.

This counterexample narrows the positive claim.  When the covariance axes are
already aligned with the task, a sufficiently expressive Cartesian policy with
the same structured initialization matches DS.  The rotated-task advantage is
therefore evidence for compact rotation-aligned structure and sample-efficient
optimization, not universal superiority or extra asymptotic capacity.

Artifacts:

- `/data/chainpray/qc/reports/analysis/learned_shell_axis_formal.png`
- `/data/chainpray/qc/reports/analysis/learned_shell_axis_formal_paired.png`
- `/data/chainpray/qc/reports/analysis/learned_shell_axis_formal_paired.csv`
- `/data/chainpray/qc/reports/analysis/learned_shell_axis_formal_decision.csv`

<a id="learned-shell-rl-protocol"></a>
## Learned sparse two-stage shell MDP

原文位置：`docs/experiments/synthetic/learned_shell_rl_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

> Status: complete.  Pilot seeds 0--7 were used only to freeze the protocol;
> all formal estimates use untouched paired seeds 100--131.  All ten machine
> decision checks passed.

<a id="learned-shell-rl-protocol--purpose"></a>
### Purpose

The fixed-sampler shell diagnostic showed that direction--speed coordinates can
express wide angular search, a narrow radial shell, and a later precision
schedule.  This experiment tests the missing learning interface: can a policy
learn from sparse rewards when its mean initially points 30 degrees away from
the shell target?

This is a two-step episodic MDP.  At stage 0, an action receives a binary
discovery reward only if its direction is within 7.5 degrees of the target and
its radius lies in `0.75 +/- 0.025`.  Only discovered episodes reach stage 1.
At stage 1, success requires the same angular tolerance and radius
`0.15 +/- 0.015`.  The stage-0 mean starts at angle zero; the target is at 30
degrees.  The stage-1 mean is initialized on target so that the experiment
isolates exploration-scale learning rather than a second mean-search problem.

<a id="learned-shell-rl-protocol--learning-algorithm"></a>
### Learning algorithm

All conditions use the same on-policy REINFORCE implementation:

- stagewise mean baselines;
- reward-to-go of discovery plus terminal success for stage 0 and terminal
  success for stage 1;
- Adam learning rate 0.01 with global gradient clipping at 10;
- 300 updates, batch size 2048, or 614,400 stage-0 interactions;
- evaluation every 10 updates using 20,000 fixed random samples per seed;
- 32 paired seeds 100--131 with common training and evaluation random numbers.

The primary outcome is normalized one-shot success AUC over stage-0
interactions.  Secondary outcomes are interactions to 50% success, final
success, discovery probability, and conditional precision probability.  Final
success is expected to saturate and is not the primary ranking metric.

<a id="learned-shell-rl-protocol--policy-families"></a>
### Policy families

1. **Cartesian global** (6 parameters): diagonal Gaussian with one shared scale
   across stages; initial sigma 0.27.
2. **Cartesian stage** (8): separate diagonal scales; approach sigma 0.27 and
   precision sigma 0.01.
3. **Cartesian full-cov** (10): state-specific learned Cholesky covariance,
   initialized isotropically as above.
4. **Cartesian full-cov oracle** (10): the stronger baseline.  Its approach
   covariance is initialized with radial/tangential standard deviations
   0.0256/0.25, exactly matching the 2D local covariance area
   `0.0256 * 0.25 = 0.08**2`; precision is isotropic 0.01.
5. **DS global** (6): shared angular/radial scales across stages.
6. **DS stage** (8): approach angular/radial scales
   `(0.25/0.75, 0.0256)` and precision scales `(0.01/0.15, 0.01)`.
7. **DS stage, coupled radius** (8): identical to DS stage except the approach
   radial standard deviation is 0.08.

Every method learns its mean and scale parameters.  Full-covariance Cartesian
also learns the off-diagonal Cholesky term.  The oracle full-covariance baseline
has more parameters than DS and receives the desired initial covariance axes;
it is included to distinguish structured rotational alignment from raw
Gaussian expressivity.

<a id="learned-shell-rl-protocol--pilot-and-frozen-decision-rules"></a>
### Pilot and frozen decision rules

Pilot seeds 0--7 gave AUC 73.78% for DS stage and 67.40% for Cartesian
full-cov oracle.  Their mean interactions to 50% success were 163.8k and
220.2k.  DS global and coupled-radius AUCs were 71.39% and 69.01%.

The learned mechanism is considered confirmed only if all of the following
hold on untouched seeds 100--131:

- DS stage minus Cartesian full-cov oracle AUC is at least +3 percentage
  points, the paired bootstrap lower bound is positive, and at least 28/32
  paired seeds favor DS.
- DS stage reduces mean interactions to 50% success by at least 15%, and at
  least 28/32 paired seeds reach the threshold no later than full-cov oracle.
- DS stage exceeds Cartesian stage AUC by at least 10 points.
- DS stage exceeds DS global AUC by at least 1 point with at least 24/32
  positive pairs.
- DS stage exceeds the coupled-radius ablation by at least 3 AUC points with
  at least 24/32 positive pairs.

No learning-rate, iteration-count, initialization, or threshold change is
allowed after reading formal seeds.  A failed rule remains a failed ablation;
it must not trigger tuning on seeds 100--131.

<a id="learned-shell-rl-protocol--evidence-boundary"></a>
### Evidence boundary

Passing establishes sample-efficient policy-gradient learning in a controlled
sparse MDP.  It still does not establish robot-task improvement, neural-network
state generalization, off-policy critic stability, or automatic phase
discovery from raw observations.  Those remain separate gates before a
performance paper claim.

<a id="learned-shell-rl-protocol--formal-results"></a>
### Formal results

All methods eventually saturated near 99% success, validating the decision to
use learning-curve AUC rather than the endpoint.  Formal means were:

| Condition | Parameters | AUC | Steps to 50% | Final success |
|---|---:|---:|---:|---:|
| Cartesian global | 6 | 36.98% | 390.40k | 98.60% |
| Cartesian stage | 8 | 54.29% | 286.72k | 99.39% |
| Cartesian full-cov | 10 | 59.48% | 266.24k | 99.67% |
| Cartesian full-cov oracle | 10 | 67.46% | 223.36k | 99.57% |
| DS global | 6 | 71.55% | 184.32k | 99.59% |
| DS stage | 8 | **73.89%** | **163.84k** | 99.68% |
| DS stage, coupled radius | 8 | 69.06% | 204.80k | 99.72% |

DS stage exceeded the stronger full-covariance oracle by 6.43 AUC points.  The
paired bootstrap interval was [6.32, 6.53] points and all 32 paired seeds were
positive.  It reached 50% success 59.52k interactions earlier on average, a
26.65% reduction; all 32 pairs were no later than the oracle.  Relative to
ordinary Cartesian stage, DS gained 19.60 AUC points.

Both preregistered mechanism ablations replicated.  Sharing angular/radial
scales across stages reduced AUC by 2.34 points, while coupling the approach
radius to Cartesian-scale noise reduced it by 4.83 points; all 32 pairs favored
the full DS-stage condition in both comparisons.

Final policy diagnostics support the intended mechanism.  DS stage learned a
stage-0 tangential/radial local action standard deviation of 0.0323/0.00627 and
mean direction error 0.18 degrees.  The full-covariance oracle learned
0.0184/0.00832 with 2.62 degrees mean error.  DS therefore did not win through
larger radial actions: it rotated its mean to the target more accurately while
retaining greater tangential coverage.

Artifacts:

- `/data/chainpray/qc/reports/analysis/learned_shell_formal.png`
- `/data/chainpray/qc/reports/analysis/learned_shell_formal_paired.png`
- `/data/chainpray/qc/reports/analysis/learned_shell_formal_per_run.csv`
- `/data/chainpray/qc/reports/analysis/learned_shell_formal_summary.csv`
- `/data/chainpray/qc/reports/analysis/learned_shell_formal_paired.csv`
- `/data/chainpray/qc/reports/analysis/learned_shell_formal_decision.csv`
- `/data/chainpray/qc/reports/analysis/learned_shell_formal_parameter_summary.csv`

<a id="offpolicy-projected-tangent-protocol"></a>
## Off-policy projected-tangent equivalence protocol

原文位置：`docs/experiments/synthetic/offpolicy_projected_tangent_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

<a id="offpolicy-projected-tangent-protocol--question"></a>
### Question

Does the contextual-bandit equivalence between DS and a Cartesian
projected-tangent flow survive replay, twin environment-action critics, sparse
two-stage credit assignment, and AWR actor fitting?

<a id="offpolicy-projected-tangent-protocol--matched-conditions"></a>
### Matched conditions

- `DS stage`: Gaussian angular/radial perturbations with stage-specific scales.
- `Cartesian projected-tangent stage`: Gaussian Cartesian tangent/radial
  perturbations, with the tangent proposal normalized back to the radius shell
  and its exact Jacobian included in replay-action log density.

Both actors have eight parameters, identical stage means, matched first-order
local scales, common random numbers, identical environment-action critics,
replay, AWR weights, and optimization.  Unlike the contextual bandit, the
actor is fitted only to executed replay actions ranked by learned twin Qs.

<a id="offpolicy-projected-tangent-protocol--frozen-pilot"></a>
### Frozen pilot

- seeds 1300--1311;
- 1,000 iterations x 512 collections = 512k stage-0 interactions;
- evaluation every 25 iterations with 20,000 fixed samples;
- replay 65,536, 50% discovery-balanced batches, batch 512, four critic
  updates per iteration;
- twin 128x128 critics, BCE targets, output bias -2, critic/actor learning rate
  1e-3, discount 0.99, Polyak 0.01;
- critic learning starts at 10, actor at 100, policy delay 2;
- AWR temperature 0.10 and entropy coefficient 0.002.

The pilot interpretation is frozen before inspection:

1. practical AUC equivalence if the absolute paired mean difference is <=2
   points and its 95% bootstrap interval lies in [-3,+3] points;
2. DS advantage if DS-minus-flow AUC >=3 points with lower bound >1 point;
3. flow advantage under the symmetric criterion;
4. otherwise inconclusive.

Both methods must exceed 40% final success for any equivalence interpretation.
We additionally report final success, discovery, conditional precision,
steps-to-50%, action norms, and paired signs.  A fresh 32-seed confirmation
requires a new decision rule frozen after this pilot.

<a id="offpolicy-projected-tangent-protocol--pilot-outcome-and-frozen-confirmation"></a>
### Pilot outcome and frozen confirmation

The pilot supports practical AUC equivalence.  DS/flow success AUC was
38.757%/38.840%; DS minus flow was -0.083 points with 95% paired-bootstrap CI
[-0.604,+0.390].  Final success was 70.03%/68.60%, a +1.44-point difference
[+0.37,+2.62].  Both reached 50% in every seed, at 308.3k/310.4k interactions;
conditional precision was 98.92%/98.87%.

Before running fresh data, confirmation is frozen to seeds 1400--1431 and the
identical 512k configuration.  Equivalence passes only if all hold:

1. absolute paired AUC mean difference <=1 point and its 95% interval is
   contained in [-1.5,+1.5] points;
2. absolute final-success difference <=3 points and its interval is contained
   in [-4,+4] points;
3. both methods average >=60% final success and all 32 seeds reach 50%;
4. absolute mean steps-to-50 difference <=15k and its interval is contained in
   [-20k,+20k];
5. absolute conditional-precision difference <=1 point;
6. neither method has non-finite metrics or missing checkpoints.

The wider equivalence bands than the contextual experiment reflect learned-Q
and replay variance and were fixed from scientific relevance, not from formal
results.  Passing would show that the manifold-support equivalence survives an
off-policy critic interface.

<a id="offpolicy-projected-tangent-protocol--formal-outcome"></a>
### Formal outcome

Fresh seeds 1400--1431 passed all 13 scalar checks implementing the grouped
criteria above.  DS/flow AUC was 39.03%/39.08%; DS minus flow was -0.051 points
with 95% paired-bootstrap CI [-0.271,+0.173].  Final success was 69.11%/68.15%,
a +0.96-point difference [-0.56,+2.48].  Every one of the 64 condition-runs
reached 50%.  DS reached it 8.8k interactions earlier on average (302.4k
versus 311.2k; paired CI for DS-minus-flow [-11.2k,-6.0k]), which is detectable
but lies inside the predeclared +/-20k practical-equivalence interval.
Conditional precision differed by only +0.078 points.

Together with contextual equivalence, this shows that the useful mechanism is
preserved through replay, learned twin environment-action critics, and AWR
actor fitting.  DS retains a small threshold-time advantage in this
implementation, but not enough under the frozen rule to support a distinct
sample-efficiency claim over the matched-support flow.

<a id="offpolicy-shell-protocol"></a>
## Sparse-shell off-policy critic bridge

原文位置：`docs/experiments/synthetic/offpolicy_shell_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

<a id="offpolicy-shell-protocol--purpose"></a>
### Purpose

The learned shell result currently uses on-policy REINFORCE, while the robot
experiments use replay and a learned critic. This bridge tests whether the
direction--speed advantage survives an off-policy value-learning interface
before spending more compute on OGBench.

<a id="offpolicy-shell-protocol--algorithm-and-causal-controls"></a>
### Algorithm and causal controls

- Two-stage sparse shell MDP, target offset 30 degrees, identical tolerances and
  actor initializations to the completed REINFORCE experiment.
- Rolling replay buffer containing executed Cartesian environment actions.
- Two stage-specific twin Q networks, target critics, Polyak updates, and a
  stochastic actor. The initial reparameterized gradient through
  `Q(s, a_env)` was retained as a negative pilot ablation: it exploited
  critic peaks outside replay support and collapsed all methods toward zero.
- The confirmatory actor update is AWR/CRR-style weighted maximum likelihood
  over executed replay actions. Weights come from the conservative minimum of
  the twin critics. Thus the critic still ranks environment actions, while the
  actor cannot follow a gradient into an action region absent from replay.
- Both critics predict bounded normalized returns. Stage 0 bootstraps from the
  target stage-1 critic only after an observed discovery transition.
- The stage-1 critic loss is masked by observed discovery. Actions sampled for
  an unreached stage may not contribute a reward label; this prevents
  counterfactual reward leakage.
- Every condition has identical critic architecture and optimization. Only the
  actor family differs.

Conditions are Cartesian stage, Cartesian full-covariance oracle, DS global,
DS stage, and DS stage with coupled radius. The primary comparison is DS stage
versus the full-covariance oracle; DS global and coupled radius are mechanism
ablations.

<a id="offpolicy-shell-protocol--pilot-and-confirmation-boundary"></a>
### Pilot and confirmation boundary

Pilot seeds 300--307 may tune optimizer rates, update ratio, replay capacity,
and total interactions. The pilot must first show that both the full-covariance
oracle and DS stage learn above chance; otherwise it diagnoses the critic rather
than ranks policy families. After freezing settings, confirmation must use new
paired seeds 400--431. No settings may be changed after reading those seeds.

Primary metric is normalized success AUC over stage-0 interactions. Report
final discovery, conditional precision, steps to 50%, all paired differences,
and bootstrap intervals. Passing this experiment establishes only a controlled
off-policy critic bridge—not robot-task improvement or neural state
generalization.

<a id="offpolicy-shell-protocol--pilot-diagnosis-and-frozen-confirmation-configuration"></a>
### Pilot diagnosis and frozen confirmation configuration

The original direct-Q actor ended near zero for every condition. Action
diagnostics showed the DS stage-0 norm drifting from the correct 0.750 shell to
0.853 even while its direction error improved. The stabilized critic uses a
pessimistic output bias, BCE on bounded targets, 50% discovery-stratified
replay, a critic-only warm-up, and delayed actor updates. AWR temperatures
0.01, 0.05, 0.10, 0.20, and 0.50 were screened on pilot seeds; 0.10 gave the
best stable learning. Actor learning rates 3e-4 and 1e-3 were screened; 1e-3
allowed both strong controls to exceed 50% within budget.

The frozen confirmation configuration is:

- paired seeds 400--431; 1,000 iterations x 512 collections = 512,000 stage-0
  interactions per seed and condition;
- evaluation every 25 iterations with 20,000 fixed evaluation samples;
- replay capacity 65,536, update batch 512, four critic updates per iteration,
  and a 50% discovery stratum;
- two hidden layers of 128, twin stage critics, BCE loss, output bias -2,
  critic learning rate 1e-3, discount 0.99, Polyak coefficient 0.01;
- learning begins at iteration 10; actor begins at 100 and updates every second
  critic update; AWR temperature 0.10, actor learning rate 1e-3, entropy
  coefficient 0.002;
- critic and replay RNG seed 20260926. Only the paired environment seed changes.

The frozen primary endpoint is DS-stage minus full-covariance-oracle normalized
success AUC. Confirmation requires all of: mean AUC benefit >=1 percentage
point, paired-bootstrap lower bound >0, at least 24/32 positive pairs, at least
5% mean reduction in interactions to 50%, and at least 24/32 pairs no later.
Both strong methods must average >=50% final success. DS final success must be
within a -15 percentage-point non-inferiority margin. Mechanism checks require
AUC benefits of >=15 points over Cartesian-stage, >=3 over DS-global, and >=5
over coupled-radius DS. These thresholds were fixed before reading seeds
400--431.

The exact single-GPU reproduction command is:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m research.shell_offpolicy \
  --seed-start=400 --num-seeds=32 \
  --output-prefix=offpolicy_shell_formal_single_gpu
```

The script defaults now equal the frozen AWR temperature and actor learning
rate. The archived direct-Q negative control is reproduced with
`--actor-update=q_gradient --actor-learning-rate=3e-4`.

<a id="projected-tangent-flow-protocol"></a>
## Projected-tangent Cartesian flow protocol

原文位置：`docs/experiments/synthetic/projected_tangent_flow_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

<a id="projected-tangent-flow-protocol--question"></a>
### Question

Does DS global still help after the Cartesian control is given curved,
radius-preserving support rather than only a rotating tangent ellipse?

<a id="projected-tangent-flow-protocol--matched-flow-control"></a>
### Matched flow control

For learned mean (m=r e_r), the control samples a Cartesian tangent offset
(t\sim\mathcal N(0,\sigma_t^2)) and radial offset
(d\sim\mathcal N(0,\sigma_r^2)), then executes

\[
a=(r+d)\frac{m+t e_t}{\lVert m+t e_t\rVert_2}.
\]

Thus the proposal is Cartesian but is projected back onto the radius shell.
The implementation uses the exact change-of-variables density.  With
(\delta=\arctan(t/r)), the Jacobian magnitude is

\[
|J|=|r+d|\frac{r}{r^2+t^2}.
\]

The flow and DS global have the same 2,244 parameters, shared residual-mean
network, effective local tangential/radial deviations 0.25/0.018315 after the
shared `log_scale >= -4` constraint, optimizer,
samples, random noise, and checkpoints.  Their difference is Gaussian tangent
offset followed by normalization versus Gaussian angular offset.

<a id="projected-tangent-flow-protocol--frozen-pilot"></a>
### Frozen pilot

- conditions: Cartesian projected-tangent flow and DS global;
- seeds 1100--1111, never used by the preceding experiments;
- 36 training contexts every 10 degrees and 36 held-out midpoint contexts;
- 2,400 updates, batch 512 per training context, Adam learning rate 2e-3;
- checkpoints 0, 600, 1,200, 1,600, 2,000, 2,400;
- 10,000 evaluation samples per context and checkpoint.

Interpretation is frozen before reading the pilot:

1. equivalence support: absolute final difference <=2 points and the 95%
   paired-bootstrap interval lies inside [-3,+3] points;
2. DS-specific advantage: DS minus flow >=5 points with lower bound >2 points;
3. flow advantage: flow minus DS >=5 points with lower bound >2 points;
4. otherwise the pilot is inconclusive.

We also report held-out AUC, mean error, paired signs, and wins across 36 test
angles.  This pilot does not support a formal claim until a fresh-seed
confirmation rule is frozen after inspecting it.

<a id="projected-tangent-flow-protocol--pilot-outcome-and-frozen-equivalence-confirmation"></a>
### Pilot outcome and frozen equivalence confirmation

The untouched pilot supports equivalence: projected-tangent flow and DS global
reached 62.97% and 62.94% final held-out success.  DS minus flow was -0.03
points with a 95% paired-bootstrap interval [-0.58,+0.45].  AUC was
52.709%/52.702%, a -0.008-point difference [-0.50,+0.44].  Mean errors were
0.02514/0.02528.  The largest absolute difference among the 36 angle averages
was 2.91 points and the mean absolute angle difference was 0.90 points.

Before any fresh run, the formal confirmation is frozen to seeds 1200--1231
and the identical optimization protocol.  Equivalence is accepted only if all
criteria pass:

1. absolute paired mean final difference <=0.5 points and its 95% bootstrap
   interval lies entirely inside [-1,+1] points;
2. absolute paired mean AUC difference <=0.5 points and its interval lies
   entirely inside [-1,+1] points;
3. both methods reach at least 55% final held-out success;
4. their absolute mean-error difference is <=0.003;
5. mean absolute per-angle difference <=1.5 points and maximum absolute
   per-angle difference <=4 points.

These are equivalence rather than superiority gates.  Passing would identify
the useful mechanism as curved, radius-preserving support; the choice between
Gaussian angle and normalized Gaussian tangent coordinates would then be an
implementation detail for this task.

<a id="projected-tangent-flow-protocol--formal-outcome"></a>
### Formal outcome

Fresh seeds 1200--1231 passed all 11 frozen equivalence criteria.  Final
held-out success was 61.41% for projected-tangent flow and 61.73% for DS; the
paired DS-minus-flow difference was +0.318 points with 95% interval
[+0.020,+0.606].  The interval excludes zero but is fully contained in the
predeclared [-1,+1]-point equivalence region.  Held-out AUC was 51.43% versus
51.75%, a +0.324-point difference [+0.085,+0.557], also equivalent.  Mean
errors were 0.02635/0.02622.  Across held-out angles, the mean and maximum
absolute differences were only 0.612 and 2.257 points.

This confirmation overturns a coordinate-specific reading of the preceding
positive results.  The straight mean-frame Gaussian fails because its support
leaves the narrow shell at finite angular displacement; once a Cartesian flow
is normalized back onto the shell, it matches DS.  The supported mechanism is
therefore manifold-matched, radius-preserving exploration support, not a
unique benefit of naming the latent variables direction and speed.

<a id="synthetic-stage-protocol"></a>
## Controlled staged direction--speed diagnostic

原文位置：`docs/experiments/synthetic/synthetic_stage_protocol.md`。正文按原记录保留，原文件名用于历史追溯。

> Status: complete.  The protocol was frozen after a floor/ceiling pilot and
> before formal Monte Carlo seeds 100--115.  All four decision rules passed.
> This is an action-sampling mechanism experiment, not a learned-RL
> performance result.

<a id="synthetic-stage-protocol--question"></a>
### Question

Can a direction--speed policy allocate a fixed local stochasticity volume
toward angular search while preserving a narrow radial shell, and can an
explicit phase change then recover precision?  This isolates the mechanism
that the reward-stage OGBench experiment attempted to induce indirectly.

<a id="synthetic-stage-protocol--two-stage-task"></a>
### Two-stage task

The action is a 3D vector in the unit ball.  In the approach stage, the policy
mean is radius 0.75 along `e1`.  A hidden target direction is uniformly
oriented at a fixed angular offset from `e1`.  One sample discovers the target
only if it lies within 7.5 degrees of that direction and within a narrow shell
around radius 0.75.  The policy has 100 independent samples.

After discovery, the mean is recentered on the target direction and radius
0.15.  The precision stage succeeds if a sample is within 7.5 degrees and
within 0.015 radius, with a budget of 20 samples.  Recentring represents the
idealized learning update after an informative transition; it deliberately
removes critic and network optimization from the geometric test.

The preregistered main setting uses a 30-degree target offset and shell
tolerance 0.025.  Robustness cells cross offsets 15/30/45 degrees with shell
tolerances 0.015/0.025/0.05.

<a id="synthetic-stage-protocol--noise-matching-and-conditions"></a>
### Noise matching and conditions

The fixed Cartesian comparator uses isotropic standard deviation 0.08.  The
DS approach distribution uses local tangential action standard deviation 0.25
and radial standard deviation

`0.08**3 / 0.25**2 = 0.008192`.

Thus both have the same square root of the local covariance determinant:
`sigma_tangent**2 * sigma_radius = 0.08**3 = 0.000512`.  This is a local
small-noise volume match, not a claim of exact finite-noise density equality.
The DS allocation was selected in the pilot and is frozen across all formal
cells.

Conditions are:

1. Cartesian global: standard deviation 0.08 in both stages.
2. Cartesian stage: 0.08 in approach and 0.01 in precision.
3. Cartesian stage + L2 projection: the same schedule with unit-ball projection.
4. Cartesian oracle stage: its approach sigma is separately calibrated on an
   independent sample for every robustness cell; precision sigma is 0.01.
   This intentionally favors the Cartesian comparator.
5. DS global: the matched approach direction/radius noise is retained in the
   precision stage.
6. DS stage: matched approach noise, then local tangential/radial standard
   deviations 0.01/0.01 in precision.
7. DS stage, coupled radius: DS angular allocation but Cartesian radial
   standard deviation 0.08 in approach, followed by the same precision schedule.

Formal estimates use 100,000 action samples for each of seeds 100--115.  All
conditions within a seed share target orientations and standard-normal draws.
The Cartesian oracle sigma is calibrated using 500,000 separate samples and
never sees formal seeds.

<a id="synthetic-stage-protocol--frozen-decision-rules"></a>
### Frozen decision rules

The geometric hypothesis is supported only if all of the following hold:

- In the main setting, DS stage exceeds Cartesian oracle stage by at least 20
  percentage points in end-to-end success and all 16 paired seed estimates are
  positive.
- DS stage exceeds DS global by at least 15 points, identifying a need for the
  phase change rather than direction--speed allocation alone.
- DS stage exceeds the coupled-radius ablation by at least 20 points,
  identifying independent radial control.
- DS stage beats Cartesian oracle stage in every one of the nine robustness
  cells.

Passing this diagnostic supports a causal geometry claim only.  It does not
establish that a learned robot policy can infer the correct phase or allocation.

<a id="synthetic-stage-protocol--results"></a>
### Results

The local stochasticity-volume check is exact by construction:
`0.08**3 = 0.25**2 * 0.008192 = 0.000512`.  In the main 30-degree,
plus/minus-0.025-shell setting, the within-budget results were:

| Condition | Discovery | Precision | End-to-end |
|---|---:|---:|---:|
| Cartesian global | 0.06% | 8.54% | 0.01% |
| Cartesian stage | 0.06% | 100.00% | 0.06% |
| Cartesian stage + L2 | 0.06% | 100.00% | 0.06% |
| Cartesian oracle stage | 17.54% | 100.00% | 17.54% |
| DS global | 93.06% | 76.65% | 71.34% |
| DS stage | 93.06% | 100.00% | 93.06% |
| DS stage, coupled radius | 47.93% | 100.00% | 47.93% |

DS stage exceeded the separately tuned Cartesian upper comparator by 75.52
percentage points.  All 16 paired formal seeds were positive and the paired
bootstrap interval was [75.01, 76.07] points.  The explicit phase change added
21.72 points relative to DS global, while independent radial control added
45.13 points relative to the coupled-radius ablation.

DS stage also beat Cartesian oracle stage in every robustness cell.  The nine
advantages, ordered by shell tolerance rows 0.015/0.025/0.05 and offset columns
15/30/45 degrees, were:

`[[+36.07, +81.13, +22.46],
  [+18.45, +75.52, +21.45],
  [ +3.65, +60.83, +16.69]]` percentage points.

The result identifies two separate causal ingredients.  First, reallocating a
fixed local covariance volume from radial to tangential axes makes sparse
direction discovery compatible with a narrow speed shell.  Second, retaining
that exploratory angular allocation in the precision stage loses 21.72 points,
so a phase-dependent reduction is required.  L2 projection has no effect here
because the target is an interior shell rather than the unit-ball boundary.

<a id="synthetic-stage-protocol--evidence-boundary"></a>
### Evidence boundary

This experiment validates the proposed geometry under controlled sampling,
including stronger tuned Cartesian and mechanism ablations.  It does not
resolve how a learned actor should infer phases, choose the allocation, or
propagate sparse rewards.  The OGBench reward-stage heuristic failed precisely
at that learning/interface layer.  The next method must therefore parameterize
state-dependent angular and radial scales directly, rather than treating a
scalar multiplier on the actor regularizer as equivalent to those scales.

Artifacts:

- `/data/chainpray/qc/reports/analysis/synthetic_stage_task.png`
- `/data/chainpray/qc/reports/analysis/synthetic_stage_task_per_seed.csv`
- `/data/chainpray/qc/reports/analysis/synthetic_stage_task_summary.csv`
- `/data/chainpray/qc/reports/analysis/synthetic_stage_task_paired.csv`
- `/data/chainpray/qc/reports/analysis/synthetic_stage_task_decision.csv`
- `/data/chainpray/qc/reports/analysis/synthetic_stage_allocation_curve.csv`
