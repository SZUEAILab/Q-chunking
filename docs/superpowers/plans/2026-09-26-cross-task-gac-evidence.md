# Cross-task and GAC Evidence Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement reward-blind cross-task geometry calibration, three auditable GAC variants, preregistered task1/task2/task3 experiment grids, and analysis artifacts sufficient to update the paper without selecting favorable tasks, seeds, or metrics.

**Architecture:** Keep the current ACRLPD critic, replay, encoder, and action-execution pipeline. Add a non-density geometric actor interface whose sampled action carries an entropy surrogate, add task-agnostic geometry/provenance logging, and put all statistical decisions in pure analysis modules driven only by raw run directories. Execute the approved stages in order: implementation verification, task2 calibration/formal, task1 GAC engineering/formal, task3 calibration/formal, then paper freeze.

**Tech Stack:** Python 3.12, JAX 0.9.x, Flax, Optax, TensorFlow Probability JAX substrate, NumPy 2.x, Matplotlib, SciPy, pytest/unittest, OGBench, existing `schedule.py` GPU scheduler.

**Spec:** `docs/superpowers/specs/2026-09-26-cross-task-gac-evidence-design.md`

## Global Constraints

- Preserve the approved method labels exactly: `cart_projection`, `default_tangent`, `matched_tangent`, `gac_code`, `gac_paper`, and `gac_ds`.
- Pin upstream GAC to commit `374ae2c9ef284a6ef715a67c1fdce864ebad6f44`; archive the source, MIT license, hashes, and every porting deviation.
- `GAC-code` uses the released unnormalized mixture, `sigmoid(0.5*kappa)`, fixed `alpha=1.0`, radius `1.0`, released kappa transform, and released kappa regularizer.
- `GAC-paper` changes only the two paper/code discrepancies: `sigmoid(kappa)` and final L2 normalization; all other GAC-code choices remain fixed.
- `GAC-DS` uses the paper-normalized XYZ mixture with `sigmoid(kappa)`, sphere-bounded learned speed, tanh-Gaussian yaw/gripper, the approved mixed entropy surrogate, and fixed `alpha=0.1`.
- Restrict every new GAC experiment to `horizon_length=1`, 5D robot actions, `actor_update_mode=sac`, `bc_alpha=0`, and fixed positive alpha; fail fast otherwise.
- Geometry control consumes only actor samples from a fixed 512-state offline panel with 32 samples per state. It must reject reward, success, Q, or replay-priority fields.
- Match only within-state radius SD and average angular deviation. Radius mean and second moment are diagnostics, never controller targets.
- Engineering tolerances are absolute radius-SD error `<=0.02` and angular error `<=0.05 rad`; formal manipulation requires at least 8 of 10 nonzero checkpoints plus the approved final capacity gates.
- Record actual evaluation at step 0; do not synthesize a zero origin. Use 50 evaluation episodes at every formal checkpoint.
- Use normalized-return AUC as primary, success AUC/final success/final normalized return/time-to-50%-success as key secondary endpoints.
- Use all formal seeds. Infrastructure reruns retain an incident record; no seed, task, or metric may be removed because of its result.
- Run groups stay below 64 characters. Formal GPU concurrency is at most 6 processes per RTX 6000 Ada after the engineering run validates memory.
- The current worktree is heavily dirty and contains the existing DS implementation. Never stage an unrelated path. Before every commit run `git diff --cached --name-status` and `git diff --cached --check`; if an already-dirty shared file contains unrelated user work, keep the task change uncommitted and report the overlap rather than absorbing it.
- Do not edit generated `.compiled.json` status fields by hand. Compile source JSON, inspect commands, then let `schedule.py` own runtime status.

## File Structure

**Create:**

- `third_party/gac/GAC.py` — byte-for-byte upstream snapshot.
- `third_party/gac/LICENSE` — upstream MIT license.
- `third_party/gac/PORTING.md` — source hashes and explicit paper/code/local differences.
- `direction_speed/gac_objective.py` — pure kappa transform, regularizer, entropy-surrogate, actor, and target equations.
- `direction_speed/distributions/gac.py` — geometric action container and Flax actor heads for all three variants.
- `direction_speed/diagnostics.py` — reward-blind action-geometry and center-capacity measurements.
- `experiment_analysis/__init__.py` — analysis package marker.
- `experiment_analysis/run_io.py` — strict run discovery and grid validation.
- `experiment_analysis/statistics.py` — AUC, paired/stratified bootstrap, improvement probability, and Holm correction.
- `experiment_analysis/geometry.py` — Cartesian target extraction and manipulation/capacity gates.
- `experiment_analysis/reporting.py` — CSV/JSON/plot writers with provenance.
- `analyze_cross_task_gac.py` — one CLI for calibration, per-task, GAC, and cross-task reports.
- `build_evidence_tasks.py` — deterministic generator for target-dependent engineering/formal task JSON.
- `tests/test_gac_objective.py`, `tests/test_gac_distribution.py`, `tests/test_gac_agent.py` — method and learner tests.
- `tests/test_schedule_gac.py` — GAC CLI/compiler preservation tests.
- `tests/test_robot_diagnostics.py`, `tests/test_evidence_analysis.py`, `tests/test_evidence_task_builder.py` — logging, statistics, gates, and config-grid tests.
- `experiment_configs/task2_cartesian_calibration.json`, `experiment_configs/task1_gac_engineering.json`, `experiment_configs/task1_gac_formal.json`, `experiment_configs/task3_cartesian_calibration.json` — static preregistered grids.
- Generated after calibration: `experiment_configs/task2_matched_engineering.json`, `experiment_configs/task2_formal.json`, `experiment_configs/task3_matched_engineering.json`, `experiment_configs/task3_formal.json`.
- Generated evidence under `docs/images/cross_task_gac/` and protocol updates under `docs/`.

**Modify:**

- `direction_speed/specs.py` — add the GAC mode/variant enums.
- `direction_speed/config.py` — validate GAC combinations and serialize the selected recipe.
- `direction_speed/distributions/factory.py` and `direction_speed/distributions/__init__.py` — assemble GAC recipes.
- `agents/acrlpd.py` — consume the geometric entropy surrogate in actor and target-Q losses and apply actor-only clipping for GAC.
- `main_online.py` — flags, step-0 evaluation, geometry/provenance/runtime logging.
- `main.py` and `eval_video.py` — accept saved GAC configurations for reconstruction.
- `schedule.py` and `tasks.example.json` — compile new fields without silently dropping them.
- `docs/gac_baseline_protocol.md`, `docs/robot_perturbation_matched_fixed_alpha_protocol.md`, and `docs/paper_draft_direction_speed_cn.md` — final protocol/results/claim updates.

## Review Focus

- **Degenerate directions:** zero or nearly zero `mu_raw`/noise must remain finite and yield legal actions; Task 2 pins safe normalization and gradients.
- **Singleton kappa batch:** the released unbiased variance is undefined at batch size one; Task 1 pins the documented finite local fallback without changing normal-batch semantics.
- **Invalid learner combinations:** GAC with action chunks, automatic alpha, BC, AWR, non-5D action, or GAC-DS cube speed must fail before training; Task 3 pins every rejection.
- **Incomplete/duplicate formal grid:** analysis must abort rather than average partial seeds or repeated runs; Task 6 pins missing and duplicate run failures.
- **Interrupted or resumed training:** runtime metadata must preserve incident/resume history and analysis must not mistake a truncated run for a completed seed; Tasks 5 and 6 pin terminal-step and incident validation.

---

### Task 1: Archive GAC and Implement the Pure Objective Algebra

**Files:**

- Create: `third_party/gac/GAC.py`
- Create: `third_party/gac/LICENSE`
- Create: `third_party/gac/PORTING.md`
- Create: `direction_speed/gac_objective.py`
- Test: `tests/test_gac_objective.py`

**Interfaces:**

- Consumes: upstream commit `374ae2c9ef284a6ef715a67c1fdce864ebad6f44`.
- Produces: `soft_clamp_kappa(raw, init) -> Array`, `gac_mix_weight(kappa, variant) -> Array`, `kappa_regularization(kappa, minimum, maximum) -> scalar Array`, `actor_entropy_surrogate(kappa, rest_log_prob) -> Array`, `gac_actor_objective(q, surrogate, alpha, kappa_reg, reg_weight) -> scalar`, and `gac_target_value(next_q, surrogate, alpha) -> Array`.

- [ ] **Step 1: Add failing equation and edge-case tests**

```python
def test_code_and_paper_weights_match_frozen_equations():
    kappa = jnp.array([2.0])
    np.testing.assert_allclose(gac_mix_weight(kappa, "code"), jax.nn.sigmoid(1.0))
    np.testing.assert_allclose(gac_mix_weight(kappa, "paper"), jax.nn.sigmoid(2.0))

def test_mixed_objective_has_approved_signs():
    q = jnp.array([3.0, 4.0])
    surrogate = jnp.array([1.5, 2.0])
    assert float(gac_target_value(q, surrogate, 0.1)[0]) == pytest.approx(2.85)
    loss = gac_actor_objective(q, surrogate, 0.1, jnp.array(0.25), 1.0)
    assert float(loss) == pytest.approx(np.mean([0.15 - 3.0, 0.2 - 4.0]) + 0.25)

def test_singleton_kappa_regularizer_is_finite():
    value = kappa_regularization(jnp.array([[2.0]]), 1.0, 20.0)
    assert bool(jnp.isfinite(value))
```

- [ ] **Step 2: Run the new tests and verify the import failure**

Run: `PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_gac_objective.py`

Expected: FAIL because `direction_speed.gac_objective` does not exist.

- [ ] **Step 3: Implement the frozen algebra**

```python
def soft_clamp_kappa(raw, init=2.0):
    safe = -10.0 + 14.0 * jax.nn.sigmoid(raw / 5.0)
    return jax.nn.softplus(safe + jnp.log(init)) + 1e-6

def gac_mix_weight(kappa, variant):
    if variant == "code":
        return jax.nn.sigmoid(jnp.minimum(kappa, 20.0) * 0.5)
    if variant in ("paper", "ds"):
        return jax.nn.sigmoid(kappa)
    raise ValueError(f"unknown GAC variant: {variant}")

def kappa_regularization(kappa, minimum=1.0, maximum=20.0):
    values = jnp.squeeze(kappa, axis=-1)
    lower = jax.nn.relu(minimum - values) ** 2
    upper = jax.nn.relu(values - maximum) ** 2
    variance = jnp.where(values.size > 1, jnp.var(values, ddof=1), 0.0)
    protection = jax.nn.relu(values.mean() - 3.0) ** 2 + jax.nn.relu(0.1 - variance) ** 2
    entropy_reg = -jnp.log(jnp.maximum(values, 0.1) + 1e-8).mean()
    return (lower + upper).mean() + 0.3 * protection + 0.05 * entropy_reg

def actor_entropy_surrogate(kappa, rest_log_prob=None):
    result = jnp.squeeze(kappa, axis=-1)
    return result if rest_log_prob is None else result + rest_log_prob

def gac_target_value(next_q, surrogate, alpha):
    return next_q - alpha * surrogate

def gac_actor_objective(q, surrogate, alpha, kappa_reg, reg_weight):
    return (alpha * surrogate - q).mean() + reg_weight * kappa_reg
```

- [ ] **Step 4: Archive and verify the upstream bytes**

Use `apply_patch` to add the exact upstream `GAC.py` and `LICENSE` contents. Record these fixed checksums in `PORTING.md`:

```text
GAC.py  13182 bytes  sha256 6d37b8bcc9388b4de54036940985a3ee3fab5e89c6da2dd3391176653a627bfd
LICENSE 1067 bytes   sha256 7508c9a9b37c2190a7a5d62669587df38728db4fbfda0f5f873c0e6df22adda9
```

Also record the local singleton-variance guard, shared ACRLPD critic/replay, common 512x4 actor backbone, and 5D radius-1 robot action domain as explicit deviations.

- [ ] **Step 5: Run focused tests and hash verification**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_gac_objective.py
sha256sum third_party/gac/GAC.py third_party/gac/LICENSE
```

Expected: tests PASS and hashes equal the two fixed values above.

- [ ] **Step 6: Commit only Task 1 paths**

```bash
git add third_party/gac direction_speed/gac_objective.py tests/test_gac_objective.py
git diff --cached --check
git diff --cached --name-status
git commit -m "feat: freeze GAC objective semantics"
```

### Task 2: Implement GAC-code, GAC-paper, and GAC-DS Action Distributions

**Files:**

- Create: `direction_speed/distributions/gac.py`
- Modify: `direction_speed/distributions/__init__.py`
- Test: `tests/test_gac_distribution.py`

**Interfaces:**

- Consumes: Task 1 objective helpers.
- Produces: `GACActionDistribution.sample_with_surrogate(seed, sample_shape=()) -> tuple[Array, Array]`, `.sample(...)`, `.mode()`, `.kappa_regularizer()`, and `GACActor(base_cls, action_dim, variant, radius, ...)`.

- [ ] **Step 1: Write failing tests for support, formula discrepancy, limits, isotropy, and gradients**

```python
def test_code_keeps_radial_mixture_while_paper_normalizes():
    mu = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    noise = jnp.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
    kappa = jnp.array([[2.0]])
    code = mix_direction(mu, noise, kappa, "code", radius=1.0)
    paper = mix_direction(mu, noise, kappa, "paper", radius=1.0)
    assert float(jnp.linalg.norm(code, axis=-1)[0]) < 1.0
    np.testing.assert_allclose(jnp.linalg.norm(paper, axis=-1), 1.0, atol=1e-6)

def test_gac_ds_bounds_and_surrogate():
    dist = fixed_gac_ds_distribution()
    action, surrogate = dist.sample_with_surrogate(jax.random.key(4), sample_shape=4096)
    assert action.shape == (4096, 1, 5)
    assert bool(jnp.all(jnp.linalg.norm(action[..., :3], axis=-1) <= 1.0 + 1e-6))
    assert bool(jnp.all(jnp.abs(action[..., 3:]) <= 1.0))
    assert bool(jnp.all(jnp.isfinite(surrogate)))

def test_zero_center_and_extreme_kappa_are_finite_and_differentiable():
    value, grad = jax.value_and_grad(sample_sum_from_raw_center)(jnp.zeros((2, 5)))
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(grad)))
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_gac_distribution.py`

Expected: FAIL because the GAC distribution is absent.

- [ ] **Step 3: Add safe normalization and the exact action mixtures**

```python
def safe_normalize(value, eps=1e-8):
    fallback = jnp.zeros_like(value).at[..., 0].set(1.0)
    squared_norm = jnp.sum(jnp.square(value), axis=-1, keepdims=True)
    normalized = value * jax.lax.rsqrt(jnp.maximum(squared_norm, eps**2))
    return jnp.where(squared_norm > eps**2, normalized, fallback)

def mix_direction(mu, noise, kappa, variant, radius=1.0):
    mu = safe_normalize(mu)
    noise = safe_normalize(noise)
    weight = gac_mix_weight(kappa, variant)
    mixture = weight * mu + (1.0 - weight) * noise
    if variant in ("paper", "ds"):
        mixture = safe_normalize(mixture)
    return radius * mixture
```

- [ ] **Step 4: Implement the PyTree action container**

Use this concrete PyTree surface; `sample_with_surrogate` splits the RNG, normalizes integer/tuple `sample_shape`, computes the speed-sigmoid and scalar-tanh Jacobians from the sampled raw rest coordinates, and returns `kappa + log_p_rest` only for `gac_ds`:

```python
@flax.struct.dataclass
class GACActionDistribution:
    mu: jax.Array
    kappa: jax.Array
    rest_loc: jax.Array | None = None
    rest_log_scale: jax.Array | None = None
    variant: str = flax.struct.field(pytree_node=False, default="code")
    radius: float = flax.struct.field(pytree_node=False, default=1.0)
    max_speed: float = flax.struct.field(pytree_node=False, default=1.0)
    kappa_min: float = flax.struct.field(pytree_node=False, default=1.0)
    kappa_max: float = flax.struct.field(pytree_node=False, default=20.0)

    def sample_with_surrogate(self, seed, sample_shape=()):
        sample_shape = (sample_shape,) if isinstance(sample_shape, int) else tuple(sample_shape)
        noise_key, rest_key = jax.random.split(seed)
        noise = jax.random.normal(noise_key, sample_shape + self.mu.shape)
        direction = mix_direction(self.mu, noise, self.kappa, self.variant, self.radius)
        if self.variant != "ds":
            return direction, jnp.broadcast_to(jnp.squeeze(self.kappa, -1), sample_shape + self.kappa.shape[:-1])
        raw = self.rest_loc + jnp.exp(self.rest_log_scale) * jax.random.normal(
            rest_key, sample_shape + self.rest_loc.shape
        )
        return compose_gac_ds(direction, raw, self.rest_loc, self.rest_log_scale, self.kappa, self.max_speed)

    def log_prob(self, action):
        del action
        raise NotImplementedError("GAC uses an entropy surrogate, not an invertible density")

    def kappa_regularizer(self):
        return kappa_regularization(self.kappa, self.kappa_min, self.kappa_max)
```

Define the rest-coordinate composition in the same file, with no TFP inverse through the direction mixture:

```python
def compose_gac_ds(direction, raw, rest_loc, rest_log_scale, kappa, max_speed):
    speed_fraction = jax.nn.sigmoid(raw[..., :1])
    scalar = jnp.tanh(raw[..., 1:])
    action = jnp.concatenate([max_speed * speed_fraction * direction, scalar], axis=-1)
    standardized = (raw - rest_loc) * jnp.exp(-rest_log_scale)
    raw_log_prob = -0.5 * (
        standardized**2 + 2.0 * rest_log_scale + jnp.log(2.0 * jnp.pi)
    )
    speed_log_det = (
        jnp.log(max_speed)
        + jax.nn.log_sigmoid(raw[..., 0])
        + jax.nn.log_sigmoid(-raw[..., 0])
    )
    scalar_log_det = jnp.log(jnp.maximum(1.0 - scalar**2, 1e-8)).sum(axis=-1)
    rest_log_prob = raw_log_prob.sum(axis=-1) - speed_log_det - scalar_log_det
    return action, actor_entropy_surrogate(kappa, rest_log_prob)
```

Add `.sample()` as the first element of `.sample_with_surrogate()`, `.mode()` using `mu`, `sigmoid(rest_loc[...,0])`, and `tanh(rest_loc[...,1:])`, and `.kappa_regularizer()` delegating to Task 1. Tests call all three methods.

- [ ] **Step 5: Implement the Flax actor heads**

```python
class GACActor(nn.Module):
    base_cls: Type[nn.Module]
    action_dim: int
    variant: str
    radius: float = 1.0
    kappa_init: float = 2.0
    kappa_min: float = 1.0
    kappa_max: float = 20.0

    @nn.compact
    def __call__(self, inputs):
        if self.action_dim != 5:
            raise ValueError("GAC robot experiments require action_dim=5")
        features = self.base_cls()(inputs)
        mu_dim = 3 if self.variant == "ds" else self.action_dim
        mu_raw = nn.Dense(
            mu_dim,
            kernel_init=nn.initializers.orthogonal(0.1),
            bias_init=nn.initializers.zeros,
            name="MuHead",
        )(features)
        kappa_hidden = nn.relu(nn.Dense(64, name="KappaHidden")(features))
        kappa_raw = nn.Dense(
            1,
            bias_init=nn.initializers.constant(jnp.log(self.kappa_init)),
            name="KappaOutput",
        )(kappa_hidden)
        kappa = soft_clamp_kappa(kappa_raw, self.kappa_init)
        if self.variant == "ds":
            rest_loc, rest_log_scale = DistributionParameterHeads(
                location_size=3,
                log_scale_size=3,
                log_scale_min=-20.0,
                log_scale_max=2.0,
                name="RestParameterHeads",
            )(features)
        else:
            rest_loc = rest_log_scale = None
        return GACActionDistribution(
            mu=safe_normalize(mu_raw),
            kappa=kappa,
            rest_loc=rest_loc,
            rest_log_scale=rest_log_scale,
            variant=self.variant,
            radius=self.radius,
            max_speed=1.0,
            kappa_min=self.kappa_min,
            kappa_max=self.kappa_max,
        )
```

- [ ] **Step 6: Pin deterministic mode, isotropy, sample shape, and checkpoint serialization**

Add tests using 50,000 normalized Gaussian draws; require each coordinate mean absolute value `<0.015` and second moment within `0.02` of `1/d`. Initialize an actor, serialize with `flax.serialization.to_bytes`, restore it, and require identical `mu`, `kappa`, mode, and seeded sample.

- [ ] **Step 7: Run focused tests**

Run: `PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_gac_objective.py tests/test_gac_distribution.py`

Expected: PASS.

- [ ] **Step 8: Commit only Task 2 paths**

```bash
git add direction_speed/distributions/gac.py direction_speed/distributions/__init__.py tests/test_gac_distribution.py
git diff --cached --check
git diff --cached --name-status
git commit -m "feat: add auditable GAC action distributions"
```

### Task 3: Wire GAC Through Configuration and Action-Pipeline Assembly

**Files:**

- Modify: `direction_speed/specs.py`
- Modify: `direction_speed/config.py`
- Modify: `direction_speed/distributions/factory.py`
- Modify: `main_online.py`
- Modify: `main.py`
- Modify: `eval_video.py`
- Modify: `schedule.py`
- Modify: `tasks.example.json`
- Test: `tests/test_gac_distribution.py`
- Create: `tests/test_schedule_gac.py`

**Interfaces:**

- Consumes: `GACActor` from Task 2.
- Produces: `--ds_mode=gac`, `--gac_variant={code,paper,ds}`, serialized `config["gac_variant"]`, `validate_gac_training_config(config, action_dim) -> None`, and `GACPipelineRecipe`.

- [ ] **Step 1: Write failing recipe and invalid-combination tests**

```python
@pytest.mark.parametrize("variant", ["code", "paper", "ds"])
def test_gac_recipe_builds_named_actor(variant):
    config = base_config(ds_mode="gac", gac_variant=variant, fixed_alpha=1.0)
    resolved = configure_direction_speed(config=config, agent_name="acrlpd", **cli(config))
    assert resolved.mode.value == "gac"
    assert resolved.recipe.components.raw_sampler == f"GAC[{variant}]"

@pytest.mark.parametrize(
    "override, message",
    [
        ({"horizon_length": 2}, "horizon_length=1"),
        ({"fixed_alpha": -1.0}, "fixed positive alpha"),
        ({"bc_alpha": 0.1}, "bc_alpha=0"),
        ({"actor_update_mode": "replay_awr"}, "actor_update_mode=sac"),
        ({"action_dim": 7}, "action_dim=5"),
        ({"gac_variant": "ds", "ds_speed_bound": "cube"}, "sphere speed"),
    ],
)
def test_invalid_gac_combinations_fail_before_training(override, message):
    with pytest.raises(ValueError, match=message):
        validate_gac_training_config({**valid_gac_config(), **override})
```

- [ ] **Step 2: Verify the recipe tests fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_gac_distribution.py -k 'recipe or invalid'`

Expected: FAIL because the enum, recipe, and validator are absent.

- [ ] **Step 3: Add enums and recipe selection**

```python
class DirectionSpeedMode(StrEnum):
    NONE = "none"
    PROJECTION = "projection"
    ADAPTER = "adapter"
    INTEGRATED = "integrated"
    GAC = "gac"

class GACVariant(StrEnum):
    CODE = "code"
    PAPER = "paper"
    DS = "ds"
```

Add `GACPipelineRecipe(variant, radius, kappa_init)` to `factory.py`. Resolve it before integrated/projection recipes when `config["gac_variant"] != "none"`; its postprocessor is identity and its actor is `GACActor`.

- [ ] **Step 4: Add CLI and scheduler fields without implicit defaults in formal configs**

Add `gac_variant` to `main.py`, `main_online.py`, and `eval_video.py`, pass it to `configure_direction_speed`, and add these compiler mappings:

```python
"gac_variant": "--gac_variant",
"gac_radius": "--agent.gac_radius",
"gac_kappa_init": "--agent.gac_kappa_init",
"gac_kappa_min": "--agent.gac_kappa_min",
"gac_kappa_max": "--agent.gac_kappa_max",
"gac_kappa_reg_weight": "--agent.gac_kappa_reg_weight",
```

Every formal JSON must state `gac_variant`, `fixed_alpha`, and `gac_radius` explicitly.

- [ ] **Step 5: Run config/factory/scheduler tests**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_direction_speed_adapter.py tests/test_gac_distribution.py tests/test_schedule_gac.py
PYTHONPATH=. .venv/bin/python schedule.py --tasks_config=tasks.example.json --compile
```

Expected: tests PASS and the compiled example retains every GAC field.

- [ ] **Step 6: Commit only Task 3 paths after auditing dirty shared files**

```bash
git add direction_speed/specs.py direction_speed/config.py direction_speed/distributions/factory.py main_online.py main.py eval_video.py schedule.py tasks.example.json tests/test_gac_distribution.py tests/test_schedule_gac.py
git diff --cached --check
git diff --cached --name-status
git commit -m "feat: configure GAC policy pipelines"
```

If the staged diff includes unrelated pre-existing hunks, unstage the affected shared file and leave it for the final reviewed integration commit.

### Task 4: Integrate the Geometric Entropy Surrogate Into ACRLPD

**Files:**

- Modify: `agents/acrlpd.py`
- Create: `tests/test_gac_agent.py`

**Interfaces:**

- Consumes: `sample_with_surrogate`, `kappa_regularizer`, `gac_actor_objective`, and `gac_target_value`.
- Produces: GAC target-Q backup, actor objective, logs `kappa_mean`, `kappa_std`, `kappa_reg_loss`, `entropy_surrogate`, `action_norm`, and actor-only global-norm clipping at `1.0`.

- [ ] **Step 1: Write failing algebra-to-agent tests**

```python
def test_gac_target_backup_subtracts_entropy_surrogate():
    value = gac_target_value(jnp.array([5.0]), jnp.array([2.0]), 1.0)
    np.testing.assert_allclose(value, [3.0])

def test_gac_ds_actor_uses_kappa_plus_rest_log_prob():
    agent, batch = make_gac_agent("ds", fixed_alpha=0.1)
    _, info = agent.actor_loss(batch, agent.network.params, jax.random.key(7))
    expected = info["kappa_mean"] + info["rest_log_prob_mean"]
    np.testing.assert_allclose(info["entropy_surrogate"], expected, rtol=1e-5)
    assert float(info["alpha_loss"]) == 0.0

def test_ordinary_sac_loss_is_unchanged():
    before = frozen_standard_loss_fixture()
    after = ordinary_agent_loss_from_same_fixture()
    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-6)
```

- [ ] **Step 2: Run tests and verify GAC loss failure**

Run: `PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_gac_agent.py`

Expected: FAIL because ACRLPD still calls ordinary `.sample()`/`.log_prob()` and omits the target entropy surrogate.

- [ ] **Step 3: Add one explicit geometric branch to critic and actor loss**

```python
geometric = self.config.get("gac_variant", "none") != "none"
if geometric:
    next_action_pi, next_surrogate = next_dist.sample_with_surrogate(sample_rng)
else:
    next_action_pi = next_dist.sample(seed=sample_rng)
# ... compute next_q ...
if geometric:
    next_q = gac_target_value(next_q, next_surrogate, self.config["fixed_alpha"])
```

In `actor_loss`, branch immediately after obtaining `dist`: sample `(action, surrogate)`, evaluate Q, compute the released kappa regularizer, and call `gac_actor_objective`. Return before the ordinary density/AWR/BC branch. Keep the standard code path byte-for-byte equivalent apart from helper extraction.

- [ ] **Step 4: Add actor-only clipping without changing standard optimizers**

For GAC configurations, label only `modules_actor` as `gac_actor` and use:

```python
optax.multi_transform(
    {
        "gac_actor": optax.chain(optax.clip_by_global_norm(1.0), optax.adam(config["lr"])),
        "other": optax.adam(config["lr"]),
    },
    parameter_labels,
)
```

For every non-GAC configuration retain the current single `optax.adam` transform. Add a test with a constructed large actor gradient showing its applied update norm is bounded while the ordinary agent update fixture is unchanged.

- [ ] **Step 5: Test finite gradients at kappa bounds and checkpoint round-trip**

Create agents with kappa-output bias forcing both soft-clamp extremes; run `jax.value_and_grad(agent.total_loss)` and require all leaves finite. Save and restore with existing `save_agent`/`restore_agent_with_file`, then compare a seeded sample and all GAC logs.

- [ ] **Step 6: Run focused and regression tests**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_gac_objective.py tests/test_gac_distribution.py tests/test_gac_agent.py tests/test_moment_controller.py tests/test_tangent_distribution.py
```

Expected: PASS.

- [ ] **Step 7: Commit the learner integration**

```bash
git add agents/acrlpd.py tests/test_gac_agent.py
git diff --cached --check
git diff --cached --name-status
git commit -m "feat: train GAC variants with frozen objectives"
```

### Task 5: Add Task-Agnostic Geometry, Step-0 Evaluation, and Run Provenance

**Files:**

- Create: `direction_speed/diagnostics.py`
- Modify: `direction_speed/moment_control.py`
- Modify: `main_online.py`
- Modify: `schedule.py`
- Create: `tests/test_robot_diagnostics.py`
- Modify: `tests/test_evaluation_seeding.py`

**Interfaces:**

- Consumes: actor distribution, offline observation panel, execution postprocessor, and evaluation function.
- Produces: `policy_geometry.csv`, `run_manifest.json`, `runtime_metrics.json`, `incidents.jsonl`, and real `eval.csv` row at step 0.

- [ ] **Step 1: Write failing diagnostics tests**

```python
def test_geometry_is_conditional_and_reports_declared_diagnostics():
    actions = synthetic_actions_with_two_state_centers()
    metrics = measure_action_geometry(actions, max_speed=1.0)
    assert metrics["radius_std_within_state"] == pytest.approx(0.1)
    assert metrics["radius_second_moment"] == pytest.approx(1.01)
    assert 0.0 <= metrics["direction_bin_occupancy"] <= 1.0
    assert 0.0 <= metrics["speed_upper_saturation"] <= 1.0

def test_reward_fields_are_rejected():
    with pytest.raises(ValueError, match="reward-blind"):
        validate_diagnostic_inputs({"observations": np.zeros((2, 4)), "reward": np.ones(2)})

def test_step_zero_and_periodic_schedule():
    assert evaluation_steps(300_000, 50_000, include_initial=True) == tuple(range(0, 300_001, 50_000))
```

- [ ] **Step 2: Verify diagnostics tests fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_robot_diagnostics.py tests/test_evaluation_seeding.py`

Expected: FAIL because the diagnostics module and step-0 schedule are absent.

- [ ] **Step 3: Implement fixed-panel diagnostics**

```python
def measure_action_geometry(actions, max_speed=1.0, azimuth_bins=8, elevation_bins=4):
    actions = np.asarray(actions)
    if actions.ndim != 3 or actions.shape[-1] != 5:
        raise ValueError("actions must have shape [samples, states, 5]")
    xyz = actions[..., :3]
    radii = np.linalg.norm(xyz, axis=-1)
    directions = xyz / np.maximum(radii[..., None], 1e-8)
    resultant = directions.mean(axis=0)
    centers = resultant / np.maximum(np.linalg.norm(resultant, axis=-1, keepdims=True), 1e-8)
    cosines = np.sum(directions * centers[None, ...], axis=-1)
    angle = float(np.arccos(np.clip(cosines, -1.0, 1.0)).mean())
    azimuth = np.mod(np.arctan2(directions[..., 1], directions[..., 0]), 2.0 * np.pi)
    azimuth_id = np.minimum((azimuth * azimuth_bins / (2.0 * np.pi)).astype(int), azimuth_bins - 1)
    elevation_id = np.minimum(
        (((np.clip(directions[..., 2], -1.0, 1.0) + 1.0) / 2.0) * elevation_bins).astype(int),
        elevation_bins - 1,
    )
    occupied = np.unique(azimuth_id * elevation_bins + elevation_id).size
    return {
        "radius_mean": float(radii.mean()),
        "radius_second_moment": float(np.square(radii).mean()),
        "radius_std_within_state": float(radii.std(axis=0).mean()),
        "angular_deviation_rad": angle,
        "direction_bin_occupancy": float(occupied / (azimuth_bins * elevation_bins)),
        "speed_mean": float(radii.mean()),
        "speed_std": float(radii.std()),
        "speed_lower_saturation": float(np.mean(radii <= 0.01 * max_speed)),
        "speed_upper_saturation": float(np.mean(radii >= 0.99 * max_speed)),
        "yaw_saturation": float(np.mean(np.abs(actions[..., 3]) >= 0.99)),
        "gripper_saturation": float(np.mean(np.abs(actions[..., 4]) >= 0.99)),
    }
```

Add `measure_center_capacity(mode_actions)` using actor modes after execution: direction-coordinate SD mean, speed SD, yaw SD, and gripper SD across the 512 states.

- [ ] **Step 4: Generalize panel creation and logging in `main_online.py`**

Add flags `geometry_diagnostics`, `geometry_interval`, `geometry_panel_size`, `geometry_samples_per_state`, and `eval_at_start`; add matching scheduler fields. Select the panel before online replay, store its integer indices and SHA256 in `run_manifest.json`, and use a dedicated RNG stream. At each geometry interval log all diagnostics for every enabled method. Pass only the three existing control metrics into `update_moment_controller`, keeping its strict reward-blind key check intact.

- [ ] **Step 5: Add actual step-0 evaluation**

After creating the agent/logger but before the online loop, call the existing deterministic evaluator when `eval_at_start=True`, and log at `step=0`. Do not insert a zero in analysis. Add an integration test with `_OneStepEnv` requiring the step-0 row and the same episode seeds as later evaluations.

- [ ] **Step 6: Add provenance and runtime metadata**

`run_manifest.json` must include expanded flags, git commit, `git status --porcelain` hash, Python/JAX/Flax/OGBench versions, device model, actor/critic parameter counts, panel indices/hash, start timestamp, and optional `resume_from`. At shutdown write wall seconds, environment steps/second, and JAX `peak_bytes_in_use` when `device.memory_stats()` exposes it; otherwise write `peak_vram_available=false` rather than a fabricated zero.

- [ ] **Step 7: Run diagnostics and existing controller tests**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_robot_diagnostics.py tests/test_evaluation_seeding.py tests/test_moment_controller.py
```

Expected: PASS and the old controller still rejects extra keys.

- [ ] **Step 8: Commit Task 5 paths after staged-diff audit**

```bash
git add direction_speed/diagnostics.py direction_speed/moment_control.py main_online.py schedule.py tests/test_robot_diagnostics.py tests/test_evaluation_seeding.py
git diff --cached --check
git diff --cached --name-status
git commit -m "feat: log reward-blind robot policy geometry"
```

### Task 6: Build Strict, Reusable Evidence Analysis

**Files:**

- Create: `experiment_analysis/__init__.py`
- Create: `experiment_analysis/run_io.py`
- Create: `experiment_analysis/statistics.py`
- Create: `experiment_analysis/geometry.py`
- Create: `experiment_analysis/reporting.py`
- Create: `analyze_cross_task_gac.py`
- Create: `tests/test_evidence_analysis.py`

**Interfaces:**

- Consumes: raw run directories containing flags, eval, training, geometry, manifest, runtime, and incident files.
- Produces: target JSON, per-seed CSV, summary CSV, paired CSV, Holm CSV, cross-task CSV, decision JSON, curves, paired plots, and geometry plots.

- [ ] **Step 1: Write failing grid, AUC, bootstrap, Holm, and gate tests**

```python
def test_auc_uses_real_step_zero():
    steps = np.array([0, 50_000, 100_000])
    values = np.array([0.2, 0.4, 0.8])
    assert normalized_auc(steps, values, 100_000) == pytest.approx(0.45)

def test_grid_rejects_missing_and_duplicate_seed():
    with pytest.raises(IncompleteGridError):
        validate_grid(missing_seed_rows(), ("a", "b"), (1, 2), (0, 50_000))
    with pytest.raises(DuplicateRunError):
        validate_grid(duplicate_seed_rows(), ("a", "b"), (1, 2), (0, 50_000))

def test_truncated_or_unresolved_incident_is_not_complete():
    with pytest.raises(IncompleteGridError):
        validate_grid(truncated_rows(last_step=40_000), ("a",), (1,), (0, 50_000))
    with pytest.raises(UnresolvedIncidentError):
        load_run(run_with_incident(status="open"))

def test_formal_manipulation_requires_eight_of_ten_and_capacity():
    assert manipulation_gate([True] * 8 + [False] * 2, valid_capacity())
    assert not manipulation_gate([True] * 7 + [False] * 3, valid_capacity())
    assert not manipulation_gate([True] * 10, {**valid_capacity(), "speed_sd": 0.0})

def test_holm_is_monotone_in_sorted_p_values():
    adjusted = holm_adjust({"a": 0.001, "b": 0.02, "c": 0.5})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]

def test_exact_paired_sign_flip_uses_all_sign_assignments():
    assert paired_sign_flip_pvalue(np.ones(6)) == pytest.approx(2.0 / 64.0)
```

- [ ] **Step 2: Verify analysis tests fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_evidence_analysis.py`

Expected: FAIL because the analysis package does not exist.

- [ ] **Step 3: Implement strict run loading**

`discover_runs(root, run_group)` must require exactly one terminal run for each `(condition, seed)`, exact expected eval steps including zero, finite training metrics, matching manifest commit/config, and either no incident or an explicitly resolved incident chain. Never choose “latest” among duplicates.

- [ ] **Step 4: Implement fixed statistics**

```python
def normalized_auc(steps, values, max_steps):
    steps = np.asarray(steps)
    if steps[0] != 0 or steps[-1] != max_steps or np.any(np.diff(steps) <= 0):
        raise ValueError("AUC requires a complete, strictly increasing grid with step 0")
    return float(np.trapezoid(values, steps / max_steps))

def improvement_probability(differences, samples=100_000, seed=20260926):
    rng = np.random.default_rng(seed)
    values = np.asarray(differences)
    draws = values[rng.integers(0, len(values), (samples, len(values)))].mean(axis=1)
    return float(np.mean(draws > 0.0))
```

Implement paired percentile bootstrap, task-stratified equal-weight bootstrap, all 10 task1 pairwise method contrasts with Holm correction, and time-to-50%-success as a censored `(reached, step)` pair.

Compute each raw two-sided p-value by exhaustive paired sign flipping (`2**6=64` assignments), using the absolute mean difference as the statistic. Apply Holm to exactly the ten unordered pairs of the five task1 methods; do not select contrasts after seeing results.

For all cube-triple tasks, convert `episode.return` to `clip((return + 3000) / 3000, 0, 1)` before AUC. Within task2/task3, report all three paired method contrasts; the preregistered cross-task primary contrast is `matched_tangent - cart_projection`, with tasks weighted equally and seeds resampled within task. A method reaches the success threshold at the first checkpoint whose 50-episode success rate is at least `0.50`; otherwise store `reached=false, step=null`.

- [ ] **Step 5: Implement calibration and gate analysis**

`extract_cartesian_target` must accept exactly seeds `984,985` at task2 steps `400k,500k` or seeds `986,987` at task3 steps `800k,1M`, average only radius SD and angle into controller targets, carry radius mean/second moment as diagnostics, and write source paths, panel hashes, commit, and extraction timestamp. Engineering gate reads the last five controller updates. Formal gate reads the ten preregistered nonzero checkpoints and final center capacities.

- [ ] **Step 6: Implement CLI/report writers**

Support these explicit modes:

```text
analyze_cross_task_gac.py calibration --task task2|task3
analyze_cross_task_gac.py engineering --task task2|task3|task1-gac
analyze_cross_task_gac.py formal --task task1-gac|task2|task3
analyze_cross_task_gac.py cross-task --task2-dir ... --task3-dir ...
analyze_cross_task_gac.py verify --root ... --output-dir ...
```

Every mode writes machine-readable CSV/JSON before plots and includes uncertainty-band semantics in plot metadata/titles.

- [ ] **Step 7: Run analysis tests**

Run: `PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_evidence_analysis.py`

Expected: PASS.

- [ ] **Step 8: Commit the reusable analyzer**

```bash
git add experiment_analysis analyze_cross_task_gac.py tests/test_evidence_analysis.py
git diff --cached --check
git diff --cached --name-status
git commit -m "feat: add preregistered evidence analysis"
```

### Task 7: Generate and Validate Every Experiment Grid

**Files:**

- Create: `build_evidence_tasks.py`
- Create: `experiment_configs/task2_cartesian_calibration.json`
- Create: `experiment_configs/task1_gac_engineering.json`
- Create: `experiment_configs/task1_gac_formal.json`
- Create: `experiment_configs/task3_cartesian_calibration.json`
- Test: `tests/test_evidence_task_builder.py`

**Interfaces:**

- Consumes: approved spec and generated `task2_geometry_target.json` / `task3_geometry_target.json`.
- Produces: deterministic task JSONs with exact grids and no placeholder target values.

- [ ] **Step 1: Write failing grid cardinality and seed tests**

```python
def test_static_grids_match_preregistration():
    assert grid("task2_cartesian_calibration").pairs == {("cart_projection", 984), ("cart_projection", 985)}
    assert grid("task1_gac_formal").conditions == (
        "cart_projection", "matched_tangent", "gac_code", "gac_paper", "gac_ds"
    )
    assert grid("task1_gac_formal").seeds == tuple(range(92, 98))

def test_target_dependent_grid_embeds_hash_and_values(tmp_path):
    target = write_target(tmp_path, radius_sd=0.061, angle=0.34, sha="abc")
    built = build_task2_formal(target)
    matched = next(task for task in built["tasks"] if task["name"] == "matched_tangent")
    assert matched["ds_moment_target_radius_std"] == 0.061
    assert built["meta"]["target_sha256"] == "abc"
```

- [ ] **Step 2: Verify builder tests fail**

Run: `PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_evidence_task_builder.py`

Expected: FAIL because the builder/configs are absent.

- [ ] **Step 3: Add static calibration and GAC configurations**

Use exactly these grids and budgets:

```text
task2 calibration: cart_projection x seeds 984,985 x 500k; eval=0; geometry=50k
task1 GAC engineering: gac_code,gac_paper,gac_ds x seed 992 x 50k; eval=0; geometry=50k
task1 GAC formal: cart_projection,matched_tangent,gac_code,gac_paper,gac_ds x seeds 92..97 x 300k; eval=0,50k..300k; 50 episodes
task3 calibration: cart_projection x seeds 986,987 x 1M; eval=0; geometry=100k
```

Use run groups `xev_t2_cart_cal_20260926`, `xev_t1_gac_eng_20260926`, `xev_t1_gac_formal_20260926`, and `xev_t3_cart_cal_20260926`.

Every JSON explicitly freezes `horizon_length=1`, `start_training=5000`, `utd_ratio=1`, `buffer_size=1000000`, `discount=0.99`, `log_interval=5000`, `sparse=false`, `video_episodes=0`, `eval_seed=20260925`, `stage_balanced_sampling=false`, and `actor_update_mode=sac`. Calibration/engineering sets `eval_interval=0`, `eval_episodes=0`, and `eval_at_start=false`; formal runs set `eval_at_start=true`, `eval_episodes=50`, and save/geometry intervals equal to their eval interval. task1, task2, and task3 use `cube-triple-play-singletask-task1-v0`, `...-task2-v0`, and `...-task3-v0`, respectively.

- [ ] **Step 4: Implement target-dependent generation**

```python
TASKS = {
    "task2": {"engineering_seed": [993], "engineering_steps": 100_000, "formal_seeds": list(range(80, 86)), "steps": 500_000, "eval_interval": 50_000},
    "task3": {"engineering_seed": [994], "engineering_steps": 200_000, "formal_seeds": list(range(86, 92)), "steps": 1_000_000, "eval_interval": 100_000},
}
```

The builder reads the target JSON, validates task/seed/checkpoint/panel provenance, inserts exact radius-SD and angle values, disables radius-mean control, fixes `alpha=0.1`, and writes engineering plus three-condition formal JSON. It must refuse a task2 target when building task3.

The static task1 matched condition reuses the already frozen task1 target exactly: `radius_mean=0.969005` (diagnostic only), `radius_std=0.067227`, and `angle=0.345378`; radius-mean control remains disabled.

- [ ] **Step 5: Compile and audit static commands**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_evidence_task_builder.py
PYTHONPATH=. .venv/bin/python schedule.py --tasks_config=experiment_configs/task2_cartesian_calibration.json --compile
PYTHONPATH=. .venv/bin/python schedule.py --tasks_config=experiment_configs/task1_gac_engineering.json --compile
PYTHONPATH=. .venv/bin/python schedule.py --tasks_config=experiment_configs/task1_gac_formal.json --compile
PYTHONPATH=. .venv/bin/python schedule.py --tasks_config=experiment_configs/task3_cartesian_calibration.json --compile
```

Inspect each compiled command for environment, seed, steps, eval count, fixed alpha, geometry flags, and run-group length. Expected command counts are `2`, `3`, `30`, and `2`.

- [ ] **Step 6: Run the full CPU test suite before GPU work**

Run: `PYTHONPATH=. .venv/bin/python -m pytest -q`

Expected: all existing and new tests PASS; no xfail is accepted for a GAC or evidence test.

- [ ] **Step 7: Commit configs and builder**

```bash
git add build_evidence_tasks.py experiment_configs tests/test_evidence_task_builder.py
git diff --cached --check
git diff --cached --name-status
git commit -m "exp: preregister cross-task and GAC grids"
```

### Task 8: Execute task2 Calibration, Engineering Gate, and Formal Grid

**Files:**

- Generate: `docs/images/cross_task_gac/task2_geometry_target.json`
- Generate: `experiment_configs/task2_matched_engineering.json`
- Generate: `experiment_configs/task2_formal.json`
- Generate: `docs/images/cross_task_gac/task2_*`
- Create: `docs/task2_perturbation_matched_protocol.md`

**Interfaces:**

- Consumes: Task 7 calibration config and Task 6 analyzer.
- Produces: frozen task2 target, engineering decision, complete 18-run formal grid, and report artifacts.

- [ ] **Step 1: Dry-run then launch the two Cartesian calibration seeds**

```bash
PYTHONPATH=. .venv/bin/python schedule.py --run=experiment_configs/task2_cartesian_calibration.compiled.json --dry_run
PYTHONPATH=. .venv/bin/python schedule.py --run=experiment_configs/task2_cartesian_calibration.compiled.json --gpus=0,1 --gpu_tasks=1
```

Expected: both commands finish `done`, with geometry at every 50k and no eval rows.

- [ ] **Step 2: Extract and freeze the reward-blind target**

```bash
PYTHONPATH=. .venv/bin/python analyze_cross_task_gac.py calibration \
  --root=exp --run-group=xev_t2_cart_cal_20260926 --task=task2 \
  --output-dir=docs/images/cross_task_gac
PYTHONPATH=. .venv/bin/python build_evidence_tasks.py \
  --task=task2 \
  --target=docs/images/cross_task_gac/task2_geometry_target.json \
  --output-dir=experiment_configs
```

Expected: target provenance names seeds 984/985 and steps 400k/500k; generated JSON contains no `null`, sentinel, or template token.

- [ ] **Step 3: Compile, inspect, and run seed-993 engineering**

```bash
PYTHONPATH=. .venv/bin/python schedule.py --tasks_config=experiment_configs/task2_matched_engineering.json --compile
PYTHONPATH=. .venv/bin/python schedule.py --run=experiment_configs/task2_matched_engineering.compiled.json --dry_run
PYTHONPATH=. .venv/bin/python schedule.py --run=experiment_configs/task2_matched_engineering.compiled.json --gpus=0 --gpu_tasks=1
PYTHONPATH=. .venv/bin/python analyze_cross_task_gac.py engineering \
  --root=exp --run-group=xev_t2_match_eng_20260926 --task=task2 \
  --output-dir=docs/images/cross_task_gac
```

Expected gate: last five updates each meet both tolerances, final capacity passes, fixed alpha is 0.1, evaluation is absent, and all values are finite. If it fails, follow the spec's capacity-failure branch; do not inspect reward.

- [ ] **Step 4: Compile and audit the 18 formal commands**

```bash
PYTHONPATH=. .venv/bin/python schedule.py --tasks_config=experiment_configs/task2_formal.json --compile
PYTHONPATH=. .venv/bin/python schedule.py --run=experiment_configs/task2_formal.compiled.json --dry_run
```

Expected: 3 conditions x 6 seeds, seeds 80–85, 500k steps, step-0 plus 50k evaluation grid, 50 episodes, identical target hash for all matched runs.

- [ ] **Step 5: Launch task2 formal runs**

Run: `PYTHONPATH=. .venv/bin/python schedule.py --run=experiment_configs/task2_formal.compiled.json --gpus=0,1 --gpu_tasks=6`

Expected: all 18 commands reach `done`. Poll scheduler/session output rather than starting a duplicate scheduler. Record any infrastructure failure in the affected run's `incidents.jsonl` before exact-seed resume.

- [ ] **Step 6: Analyze task2 without selection**

```bash
PYTHONPATH=. .venv/bin/python analyze_cross_task_gac.py formal \
  --root=exp --run-group=xev_t2_formal_20260926 --task=task2 \
  --output-dir=docs/images/cross_task_gac
```

Expected: complete-grid validation, six paired differences for every endpoint, bootstrap CI/improvement probability, 8/10 manipulation gates, capacity gates, curves, and runtime table.

- [ ] **Step 7: Write protocol outcome and commit only compact evidence**

Update `docs/task2_perturbation_matched_protocol.md` with target, gate, exact commands, incidents, all metrics, and allowed claim. Commit source configs, protocol, CSV/JSON/PNG summaries, but not raw checkpoints.

### Task 9: Execute GAC Engineering and the Fresh task1 Competition

**Files:**

- Generate: `docs/images/cross_task_gac/task1_gac_*`
- Update: `docs/gac_baseline_protocol.md`
- Update: `third_party/gac/PORTING.md`

**Interfaces:**

- Consumes: verified GAC implementation and static grids.
- Produces: engineering safety decision and complete 30-run task1 competition.

- [ ] **Step 1: Run the three seed-992 engineering jobs one at a time**

```bash
PYTHONPATH=. .venv/bin/python schedule.py --run=experiment_configs/task1_gac_engineering.compiled.json --dry_run
PYTHONPATH=. .venv/bin/python schedule.py --run=experiment_configs/task1_gac_engineering.compiled.json --gpus=0 --gpu_tasks=1
PYTHONPATH=. .venv/bin/python analyze_cross_task_gac.py engineering \
  --root=exp --run-group=xev_t1_gac_eng_20260926 --task=task1-gac \
  --output-dir=docs/images/cross_task_gac
```

Expected: no NaN/Inf, no action-bound violation, checkpoint round-trip passes, manifest hashes match, and measured peak VRAM permits the planned concurrency. No return evaluation exists.

- [ ] **Step 2: Freeze the engineering decision and re-run the full tests**

Run: `PYTHONPATH=. .venv/bin/python -m pytest -q`

Expected: full suite PASS after the actual checkpoint path has been exercised.

- [ ] **Step 3: Audit and launch all 30 fresh task1 formal runs**

```bash
PYTHONPATH=. .venv/bin/python schedule.py --run=experiment_configs/task1_gac_formal.compiled.json --dry_run
PYTHONPATH=. .venv/bin/python schedule.py --run=experiment_configs/task1_gac_formal.compiled.json --gpus=0,1 --gpu_tasks=6
```

Expected: 5 methods x seeds 92–97, 300k steps, real step-0 plus 50k evaluations, 50 episodes. Do not pool old task1 seeds into inference.

- [ ] **Step 4: Produce the GAC competition report**

```bash
PYTHONPATH=. .venv/bin/python analyze_cross_task_gac.py formal \
  --root=exp --run-group=xev_t1_gac_formal_20260926 --task=task1-gac \
  --output-dir=docs/images/cross_task_gac
```

Expected: all ten pairwise normalized-return-AUC contrasts with Holm-adjusted p-values, six paired seed differences per contrast, success endpoints, radial/action-norm diagnostics, parameter counts, throughput, wall time, and peak VRAM availability.

- [ ] **Step 5: Update and commit the GAC protocol outcome**

State separately: algorithm competition (`GAC-code`, `GAC-paper`), common robot skeleton (`GAC-DS`), paper/code discrepancy, and any local port deviation. Commit compact evidence and protocol; never label the better of code/paper post hoc as unqualified “GAC.”

### Task 10: Execute task3 Calibration, Engineering, and the 1M Stress Test

**Files:**

- Generate: `docs/images/cross_task_gac/task3_geometry_target.json`
- Generate: `experiment_configs/task3_matched_engineering.json`
- Generate: `experiment_configs/task3_formal.json`
- Generate: `docs/images/cross_task_gac/task3_*`
- Create: `docs/task3_perturbation_matched_protocol.md`

**Interfaces:**

- Consumes: task3 calibration grid and the same frozen code used for task2.
- Produces: independent target, engineering gate, complete 18-run stress test, and floor-aware report.

- [ ] **Step 1: Run task3 Cartesian calibration**

```bash
PYTHONPATH=. .venv/bin/python schedule.py --run=experiment_configs/task3_cartesian_calibration.compiled.json --dry_run
PYTHONPATH=. .venv/bin/python schedule.py --run=experiment_configs/task3_cartesian_calibration.compiled.json --gpus=0,1 --gpu_tasks=1
PYTHONPATH=. .venv/bin/python analyze_cross_task_gac.py calibration \
  --root=exp --run-group=xev_t3_cart_cal_20260926 --task=task3 \
  --output-dir=docs/images/cross_task_gac
```

Expected: target uses only seeds 986/987 at 800k/1M and is independent of task2.

- [ ] **Step 2: Generate, compile, and run seed-994 engineering**

```bash
PYTHONPATH=. .venv/bin/python build_evidence_tasks.py \
  --task=task3 \
  --target=docs/images/cross_task_gac/task3_geometry_target.json \
  --output-dir=experiment_configs
PYTHONPATH=. .venv/bin/python schedule.py --tasks_config=experiment_configs/task3_matched_engineering.json --compile
PYTHONPATH=. .venv/bin/python schedule.py --run=experiment_configs/task3_matched_engineering.compiled.json --gpus=0 --gpu_tasks=1
PYTHONPATH=. .venv/bin/python analyze_cross_task_gac.py engineering \
  --root=exp --run-group=xev_t3_match_eng_20260926 --task=task3 \
  --output-dir=docs/images/cross_task_gac
```

Expected: same geometry/capacity/finite/fixed-alpha gate as task2, with no reward evaluation.

- [ ] **Step 3: Compile, audit, and launch 18 formal 1M runs**

```bash
PYTHONPATH=. .venv/bin/python schedule.py --tasks_config=experiment_configs/task3_formal.json --compile
PYTHONPATH=. .venv/bin/python schedule.py --run=experiment_configs/task3_formal.compiled.json --dry_run
PYTHONPATH=. .venv/bin/python schedule.py --run=experiment_configs/task3_formal.compiled.json --gpus=0,1 --gpu_tasks=6
```

Expected: three conditions x seeds 86–91, 1M steps, real step 0 plus every 100k, 50 episodes. Finish the grid even if early success is zero.

- [ ] **Step 4: Analyze and commit the floor-aware task3 report**

```bash
PYTHONPATH=. .venv/bin/python analyze_cross_task_gac.py formal \
  --root=exp --run-group=xev_t3_formal_20260926 --task=task3 \
  --output-dir=docs/images/cross_task_gac
```

Write `docs/task3_perturbation_matched_protocol.md`; if all methods floor, state exactly that and do not append steps or replace the task.

### Task 11: Freeze Cross-task Statistics and Update the Paper

**Files:**

- Generate: `docs/images/cross_task_gac/cross_task_*`
- Modify: `docs/paper_draft_direction_speed_cn.md`
- Modify: `docs/experiments.md`
- Modify: `docs/ds_exp_checklist.md`

**Interfaces:**

- Consumes: complete task1 GAC, task2, and task3 artifacts.
- Produces: final tables/figures, bounded claims, reproducibility checklist, and revised paper draft.

- [ ] **Step 1: Run the equal-task-weight cross-task analysis**

```bash
PYTHONPATH=. .venv/bin/python analyze_cross_task_gac.py cross-task \
  --task2-dir=docs/images/cross_task_gac \
  --task3-dir=docs/images/cross_task_gac \
  --output-dir=docs/images/cross_task_gac
```

Expected: task-stratified bootstrap gives task2 and task3 equal weight, reports normalized-return AUC and success AUC together, and retains task3 floor status.

- [ ] **Step 2: Audit every preregistered artifact**

Run a verifier that exits nonzero unless it finds: 2+1+18 task2 runs, 3+30 task1 GAC runs, 2+1+18 task3 runs; all expected checkpoints; all six formal seeds per condition; target hashes; manipulation/capacity decisions; incident resolutions; and every CSV/JSON/PNG named by the report manifest.

- [ ] **Step 3: Update paper methods and related work**

Describe GAC-code/paper/DS equations, upstream commit, common and differing components, reward-blind calibration, real step-0 AUC, fixed seeds/budgets, bootstrap/Holm procedures, and task4 scope limitation. Do not claim invention of geometric directional policies.

- [ ] **Step 4: Update results and claims from generated tables only**

Insert task1 GAC, task2, task3, and cross-task tables by reading the generated CSVs. If return and success disagree, state both in abstract, results, limitations, and conclusion. Use only the claim language allowed by Section 6.3 of the spec.

- [ ] **Step 5: Run final verification**

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q
PYTHONPATH=. .venv/bin/python analyze_cross_task_gac.py verify \
  --root=exp --output-dir=docs/images/cross_task_gac
git diff --check
```

Expected: full suite PASS, verifier PASS, no missing formal cell, and no whitespace errors.

- [ ] **Step 6: Commit the final evidence and paper revision**

```bash
git add docs/paper_draft_direction_speed_cn.md docs/experiments.md docs/ds_exp_checklist.md docs/task2_perturbation_matched_protocol.md docs/task3_perturbation_matched_protocol.md docs/gac_baseline_protocol.md docs/images/cross_task_gac experiment_configs
git diff --cached --check
git diff --cached --name-status
git commit -m "paper: add cross-task and GAC evidence"
```

Do not mark the overarching publication goal complete at this commit. Completion still requires a final requirement-by-requirement audit of the paper, raw evidence, tests, and submission-ready formatting.
