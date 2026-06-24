import { onLanguageChange, tx } from "../i18n.js";

export function initPipeline() {
  const modeTabs = Array.from(document.querySelectorAll(".mode-tab"));
  const modePanels = Array.from(document.querySelectorAll(".mode-panel"));

  modeTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const mode = tab.dataset.mode;
      modeTabs.forEach((item) => {
        const active = item === tab;
        item.classList.toggle("active", active);
        item.setAttribute("aria-selected", String(active));
      });
      modePanels.forEach((panel) => {
        panel.classList.toggle("active", panel.id === `panel-${mode}`);
      });
    });
  });

  const flowButtons = Array.from(document.querySelectorAll(".flow-button"));
  const flowNarration = document.getElementById("flowNarration");
  const flowStage = document.getElementById("flowStage");
  const flowPlayButton = document.getElementById("flowPlayButton");
  const flowOrder = ["overview", "data", "actor", "env", "density"];
  let flowTimer = null;
  let currentFlowStep = "overview";
  const flowCopy = {
    overview: "总览两个层级：Integrated DS 把 latent ξ 变成 policy action aπ；Action-adapter DS 决定 aπ 是否还要 compose 成 a_env。",
    data: "Dataset 阶段：Action-adapter DS 先 decompose 成 policy-action batch；Integrated DS 通常保持 env action 维度。",
    actor: "Actor / Critic 阶段：SAC 内部始终对齐 Q(s,aπ) 与 log π(aπ|s)；Integrated DS 的 forward 输出就是这个 aπ。",
    env: "Execution 阶段：a_env = g(aπ)。Integrated DS 里 g 是 identity；Action-adapter DS 或组合方案里 g 可以是 compose。",
    density: "log_prob 阶段：Integrated DS 的 bijector 有 inverse 和 log-det，因此能给 aπ 严格密度；只有 aπ=a_env 时才等价于 env-action density。"
  };
  const flowStageLabel = {
    overview: "Overview（总览）",
    data: "1 / Dataset（数据集）",
    actor: "2 / Actor + Critic（策略 + 价值网络）",
    env: "3 / Execution（环境执行）",
    density: "4 / log_prob（对数概率）"
  };

  function setFlowStep(step) {
    currentFlowStep = step;
    flowButtons.forEach((button) => {
      button.classList.toggle("active", button.dataset.flowStep === step);
    });
    if (flowNarration) {
      flowNarration.textContent = tx(flowCopy[step] || flowCopy.overview);
    }
    if (flowStage) {
      flowStage.textContent = tx(flowStageLabel[step] || flowStageLabel.overview);
    }

    document.querySelectorAll("#unifiedPipeline [data-flow]").forEach((element) => {
      const flow = element.dataset.flow;
      const active = step === "overview"
        || flow === step
        || (step === "density" && element.id === "pipelineTransformModule");
      element.classList.toggle("is-hot", active);
    });
  }

  flowButtons.forEach((button) => {
    button.addEventListener("click", () => {
      stopFlowPlayback();
      setFlowStep(button.dataset.flowStep);
    });
  });

  function stopFlowPlayback() {
    window.clearInterval(flowTimer);
    flowTimer = null;
    if (flowPlayButton) flowPlayButton.textContent = tx("播放流程");
  }

  function toggleFlowPlayback() {
    if (flowTimer) {
      stopFlowPlayback();
      return;
    }
    if (flowPlayButton) flowPlayButton.textContent = tx("暂停流程");
    flowTimer = window.setInterval(() => {
      const currentIndex = flowOrder.indexOf(currentFlowStep);
      const nextIndex = (currentIndex + 1) % flowOrder.length;
      setFlowStep(flowOrder[nextIndex]);
    }, 2200);
  }

  if (flowPlayButton) {
    flowPlayButton.addEventListener("click", toggleFlowPlayback);
  }
  setFlowStep("overview");

  const pipelineShell = document.getElementById("unifiedPipeline");
  const pipelineModeButtons = Array.from(document.querySelectorAll(".pipeline-mode-button"));
  const pipelineEntityToggle = document.getElementById("pipelineEntityToggle");
  const pipelineEntityReplay = document.getElementById("pipelineEntityReplay");
  const pipelineFields = {
    summary: document.getElementById("pipelineSummary"),
    keyIdea: document.getElementById("pipelineKeyIdea"),
    dataset: document.getElementById("pipelineDatasetText"),
    datasetOperation: document.getElementById("pipelineDatasetOperation"),
    heads: document.getElementById("pipelineHeadsText"),
    sampler: document.getElementById("pipelineSamplerText"),
    transformName: document.getElementById("pipelineTransformName"),
    transformText: document.getElementById("pipelineTransformText"),
    postName: document.getElementById("pipelinePostName"),
    postText: document.getElementById("pipelinePostText"),
    env: document.getElementById("pipelineEnvText"),
    rawShape: document.getElementById("pipelineRawShape"),
    piShape: document.getElementById("pipelinePiShape"),
    envShape: document.getElementById("pipelineEnvShape"),
    detail: document.getElementById("pipelineDetail")
  };
  const pipelineModes = {
    integrated: {
      summary: "Integrated（分布内集成）Recipe（组装方案）选择 Parameter Heads（参数输出头）、Gaussian Raw Sampler（高斯原始采样器）、DS Bijector（方向速度可逆变换器）和 Identity Postprocessor（恒等后处理器）。",
      keyIdea: "Integrated（分布内集成）改变“如何采样策略动作”；action_pi（策略动作）已经是环境可执行的 XYZ（笛卡尔坐标）动作。",
      dataset: "环境动作直接作为 action_pi（策略动作）进入 Critic（价值网络），不执行 decompose（分解）。",
      datasetOperation: "Identity（恒等映射）",
      heads: "输出 location（位置参数）与 log-scale（对数尺度参数）；Tangent（切空间）方法允许两个输出头宽度不同。",
      sampler: "MultivariateNormalDiag（对角多元高斯）产生 action_raw（原始动作）。",
      transformName: "DS Bijector（方向速度可逆变换器）",
      transformText: "选择 Spherical / Stereographic / Tangent（球坐标 / 立体投影 / 切空间）几何，并提供 inverse（逆变换）与 log-det（对数行列式）。",
      postName: "Identity（恒等映射）",
      postText: "action_pi（策略动作）不再改变。",
      env: "action_env = action_pi；环境收到 Bijector（可逆变换器）的输出。",
      rawShape: "Gaussian latent（高斯潜在坐标）",
      piShape: "5D XYZ + yaw + gripper",
      envShape: "5D，与 action_pi 相同",
      detail: "Integrated（分布内集成）中，action_raw（原始动作）经过几何变换成为 action_pi（策略动作）；Critic（价值网络）和 log_prob（对数概率）都读取 action_pi，Identity（恒等映射）再把同一个实体交给环境。"
    },
    adapter: {
      summary: "Adapter（动作适配器）Recipe（组装方案）选择 Parameter Heads（参数输出头）、Gaussian Raw Sampler（高斯原始采样器）、Mixed Squash（混合压缩器）和 DirectionSpeedAdapter（方向速度适配器）。",
      keyIdea: "Adapter（动作适配器）让 SAC（软演员评论家算法）在 direction + speed（方向 + 速度）坐标中训练，执行前再把 action_pi（策略动作）翻译为 XYZ（笛卡尔坐标）。",
      dataset: "环境动作先经过 decompose（分解）得到 policy-space action_pi（策略空间动作），Replay Buffer（回放缓冲区）与 Critic（价值网络）存取这个坐标。",
      datasetOperation: "decompose（分解）",
      heads: "输出逐策略坐标的 μ（均值）与 log σ（对数标准差）。",
      sampler: "MultivariateNormalDiag（对角多元高斯）产生 action_raw（原始动作）。",
      transformName: "MixedSquashBijector（混合压缩变换器）",
      transformText: "方向坐标选择 tanh / sigmoid / identity（双曲正切 / 逻辑函数 / 恒等），速度压缩与速度解释独立配置。",
      postName: "DirectionSpeedAdapter（方向速度适配器）",
      postText: "compose（合成）把 action_pi（策略动作）解释成 XYZ（笛卡尔坐标）与其它环境维度。",
      env: "action_env = compose(action_pi)；Cartesian（笛卡尔）表示可发生 D+1 → D。",
      rawShape: "6D Gaussian sample（六维高斯样本）",
      piShape: "6D direction + speed + scalars",
      envShape: "5D XYZ + yaw + gripper",
      detail: "Adapter（动作适配器）中，action_pi（策略动作）仍是 Critic（价值网络）和 log_prob（对数概率）的变量；只有执行边界把它 compose（合成）为 action_env（环境动作），因此不要把 action_pi 的密度误称为环境动作密度。"
    }
  };

  function setPipelineMode(mode) {
    const copy = pipelineModes[mode] || pipelineModes.integrated;
    pipelineShell.dataset.pipelineMode = mode;
    pipelineModeButtons.forEach((button) => {
      const active = button.dataset.pipelineMode === mode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    Object.entries(pipelineFields).forEach(([key, element]) => {
      if (element && copy[key]) element.textContent = tx(copy[key]);
    });
    replayPipelineEntities();
  }

  function replayPipelineEntities() {
    document.querySelectorAll(".action-entity").forEach((entity) => {
      entity.style.animation = "none";
      void entity.offsetWidth;
      entity.style.animation = "";
    });
  }

  pipelineModeButtons.forEach((button) => {
    button.addEventListener("click", () => setPipelineMode(button.dataset.pipelineMode));
  });

  if (pipelineEntityToggle) {
    pipelineEntityToggle.addEventListener("click", () => {
      const paused = pipelineShell.classList.toggle("entities-paused");
      pipelineEntityToggle.textContent = tx(paused ? "继续实体动画" : "暂停实体动画");
    });
  }
  if (pipelineEntityReplay) pipelineEntityReplay.addEventListener("click", replayPipelineEntities);
  setPipelineMode("integrated");

  onLanguageChange(() => {
    setFlowStep(currentFlowStep);
    setPipelineMode(pipelineShell?.dataset.pipelineMode || "integrated");
    if (flowTimer && flowPlayButton) flowPlayButton.textContent = tx("暂停流程");
  });
}
