const LANGUAGE_KEY = "ds-visualizer-language";
const THEME_KEY = "ds-visualizer-theme";

let language = "zh";
const languageListeners = new Set();

const normalize = (value) => String(value || "").replace(/\s+/g, " ").trim();

const EN_TEXT = new Map(Object.entries({
  "Direction-Speed Architecture（方向速度架构）": "Direction-Speed Architecture",
  "Direction-Speed Policy Parameterization（方向速度策略参数化）": "Direction-Speed Policy Parameterization",
  "A modular action-space study for SAC / RLPD（面向 SAC / RLPD 的模块化动作空间研究）": "A modular action-space study for SAC / RLPD",
  "Interactive Research Note（交互式研究笔记）": "Interactive Research Note",
  "OGBench 5D Action（OGBench 五维动作）": "OGBench 5D Action",
  "Cube Radial Bound（立方体径向边界）": "Cube Radial Bound",
  "Method（方法）": "Method",
  "Parameterization（参数化）": "Parameterization",
  "Pipeline（流水线）": "Pipeline",
  "Abstract（摘要）": "Abstract",
  "本页按“问题动机 → 方法定义 → 概率参数化 → 执行流水线”的顺序组织：先交代 action space（动作空间）、 SAC actor/critic（策略网络 / 价值网络）和 explore/exploit（探索 / 利用），再说明 Cartesian Gaussian（笛卡尔高斯） 为什么会诱导不稳定的 direction cone（方向视锥），随后引出 direction × speed（方向 × 速度）分层， 最后对比 Integrated DS（分布内集成）与 Adapter DS（动作适配器）的训练变量、执行变量和 log_prob（对数概率）边界。": "This page is organized as motivation, method definition, probabilistic parameterization, and execution pipeline. It first defines the action space, SAC actor/critic, and the explore/exploit role of policy variance, then explains why Cartesian Gaussian exploration induces unstable direction cones. It then introduces a direction-speed factorization and compares the training variable, execution variable, and log-probability boundary of Integrated DS and Adapter DS.",
  "Full Cube Support（完整立方体覆盖）": "Full Cube Support",
  "使用 R(u)=1/max|uᵢ| 覆盖 normalized cube（归一化立方体）而不是只覆盖 unit ball（单位球）。": "Use R(u)=1/max|uᵢ| to cover the normalized cube, rather than only the unit ball.",
  "Separated Semantics（语义解耦）": "Separated Semantics",
  "direction（方向）控制“往哪里走”，speed ratio（速度比例）控制“沿边界比例走多远”。": "Direction controls where to move; speed ratio controls how far to move along the radial boundary.",
  "Correct Training Variable（正确训练变量）": "Correct Training Variable",
  "Critic（价值网络）和 log_prob（对数概率）始终对齐 aπ ，不是误用 a_env 。": "The critic and log_prob are always aligned with aπ, not incorrectly with a_env.",
  "1. 背景：SAC 随机策略": "1. Background: SAC Stochastic Policy",
  "Actor（策略网络）输出 μ、σ；Critic（价值网络）学习 Q(s,aπ)；σ 承担探索语义。": "The actor outputs μ and σ; the critic learns Q(s,aπ); σ carries the exploration semantics.",
  "2. 动机：XYZ 高斯耦合": "2. Motivation: Coupled XYZ Gaussian",
  "固定 σ 的 95% 动作置信椭球会随 ||μ|| 诱导不同方向视锥。": "With fixed σ, a 95% action confidence ellipsoid induces different direction cones as ||μ|| changes.",
  "3. 方法：方向 × 速度": "3. Method: Direction × Speed",
  "把 3D 位移拆成 unit direction（单位方向）和 speed ratio（速度比例），让随机性沿任务语义流动。": "Factorize 3D displacement into unit direction and speed ratio so stochasticity follows task semantics.",
  "4. 流水线：训练动作与执行动作": "4. Pipeline: Training Action vs. Execution Action",
  "Integrated DS（分布内集成）改变概率分布内部；Adapter DS（动作适配器）改变 policy/env action 边界。": "Integrated DS changes the policy distribution internally; Adapter DS changes the policy/env action boundary.",
  "Contents（目录）": "Contents",
  "按方法论顺序阅读：问题 → 方法 → 实现 → 总结": "Read in method order: problem → method → implementation → summary",
  "摘要": "Abstract",
  "背景与动机": "Background & Motivation",
  "方向视锥": "Direction Cone",
  "方法定义": "Method Definition",
  "概率参数化": "Probabilistic Parameterization",
  "速度边界": "Speed Bound",
  "模块流水线": "Modular Pipeline",
  "总结边界": "Boundary Summary",
  "当前章节": "Current Section",
  "1. 背景：Action Space（动作空间）与 SAC": "1. Background: Action Space and SAC",
  "先看随机策略在什么坐标里探索": "Start with the coordinates used by the stochastic policy",
  "Action Space（动作空间）先决定方向语义": "Action Space First Determines Direction Semantics",
  "OGBench 5D 动作里，前三维是末端位置增量 dx, dy, dz ，通常是 expressed in world frame 的 EE position delta；后两维是 yaw 和 gripper。SAC 在这个动作空间上学习随机策略。": "In OGBench 5D actions, the first three dimensions are end-effector position deltas dx, dy, dz, usually expressed in the world frame; the last two are yaw and gripper. SAC learns a stochastic policy in this action space.",
  "Delta World 的方向是全局坐标方向；Delta EE 的方向是末端局部方向。本文讨论的是前三维位移向量上的方向与速度。": "Delta World uses global-frame directions; Delta EE uses end-effector local directions. This page focuses on direction and speed in the first three displacement dimensions.",
  "Critic（价值网络）拟合 TD Target（时序差分目标）": "The Critic Fits the TD Target",
  "Q(s,aπ) 评估训练用 policy action（策略动作），训练目标是 Bellman backup（贝尔曼备份）形成的一步 TD target；采样的下一步动作来自当前 actor（策略网络）。": "Q(s,aπ) evaluates the policy action used for training. The target is the one-step TD target induced by Bellman backup; the next sampled action comes from the current actor.",
  "Actor（策略网络）输出 μ 和 σ": "The Actor Outputs μ and σ",
  "μ 表示动作分布中心，偏 exploit（利用）； σ 表示采样尺度，偏 explore（探索）。问题在于 σ 的坐标语义是否可解释。": "μ represents the action-distribution center and leans toward exploitation; σ represents sampling scale and leans toward exploration. The question is whether σ has interpretable coordinate semantics.",
  "Explore / Exploit（探索 / 利用）要能被解释": "Explore / Exploit Should Be Interpretable",
  "如果 σ 只是在 XYZ 坐标轴上变大，就很难说它增加的是方向探索、速度探索，还是二者的混合。": "If σ only grows along XYZ axes, it is unclear whether it increases direction exploration, speed exploration, or a mixture of both.",
  "2. 动机：XYZ 表示不等于任务语义": "2. Motivation: XYZ Coordinates Are Not Task Semantics",
  "问题不是维度，而是坐标系": "The issue is the coordinate system, not dimensionality",
  "直接用 TanhNormal 在 dx,dy,dz 上采样可以运行，但三维 Cartesian 坐标不是任务真正关心的语义坐标。位移动作同时包含方向和速度。": "Sampling dx, dy, dz directly with TanhNormal can run, but 3D Cartesian coordinates are not the task-semantic coordinates. Displacement actions contain both direction and speed.",
  "1. 方向和速度纠缠": "1. Direction and Speed Are Entangled",
  "(dx,dy,dz) 同时决定方向 v/||v|| 和速度 ||v|| ，一次噪声会同时改变两个语义。": "(dx,dy,dz) determines both direction v/||v|| and speed ||v||, so one noise sample changes both semantics.",
  "2. σ 先形成动作置信椭球": "2. σ First Forms an Action Confidence Ellipsoid",
  "Cartesian 高斯的 σx,σy,σz 定义的是固定置信水平下的动作采样椭球，而不是直接定义方向探索。": "The σx, σy, σz of a Cartesian Gaussian define an action-sampling ellipsoid at a fixed confidence level, not direction exploration directly.",
  "3. μ 的长度会改变方向探索": "3. The Length of μ Changes Direction Exploration",
  "同样置信水平、同样体积的蓝色椭球，离原点越近，从原点看过去张开的方向视锥越大。": "For the same confidence level and ellipsoid volume, moving the blue ellipsoid closer to the origin opens a larger direction cone.",
  "4. chunk 会重复这个问题": "4. Action Chunks Repeat the Same Issue",
  "H=5 时每一步都有一组三维位移；方向、速度和标量动作的边界更需要清楚。": "With H=5, each step has its own 3D displacement, so the boundary between direction, speed, and scalar actions must be explicit.",
  "3. 可视化：固定 σ 如何变成不同方向视锥": "3. Visualization: How Fixed σ Becomes Different Direction Cones",
  "Three.js（3D 可视化）交互视锥": "Interactive Three.js Direction Cone",
  "联合置信区域": "joint confidence region",
  "蓝色 = 固定置信水平的采样椭球；红色 = 方向探索视锥": "Blue = fixed-confidence sampling ellipsoid; red = direction exploration cone",
  "当蓝色 95% 置信椭球靠近原点时，同样的体积会张开更大的方向视锥，所以小动作更容易出现方向抖动。": "When the blue 95% confidence ellipsoid moves closer to the origin, the same volume opens a larger direction cone, so small actions are more prone to directional jitter.",
  "蓝色球/椭球表示 Cartesian 高斯在 [dx,dy,dz] 空间里的固定置信水平采样区域，例如 95% 概率质量对应的一层等概率置信椭球；红色锥体表示从原点看过去，这个局部采样区域诱导出的方向探索范围。 保持 σ 不变时，只有 l=||μ|| 变小，方向角也会被放大。": "The blue sphere/ellipsoid is a fixed-confidence sampling region of the Cartesian Gaussian in [dx,dy,dz]. The red cone shows the induced direction-exploration range from the origin. With fixed σ, decreasing l=||μ|| enlarges the direction angle.",
  "这里的\"95% 置信椭球\"指 联合置信区域 （Mahalanobis 距离 ≤ χ²_D(p)，保证总概率质量 = 95%），不是\"每维 95% 区间拼成的长方体\"——后者只包含 0.95³ ≈ 85.7% 的联合质量。两者形状、含义都不同，不能混用。": "Here, the “95% confidence ellipsoid” means a joint confidence region, not a box formed by per-dimension 95% intervals. The latter contains only 0.95³ ≈ 85.7% joint mass, so the two should not be mixed.",
  "重置视角": "Reset View",
  "自动旋转": "Auto Rotate",
  "手动视角": "Manual View",
  "视锥参数": "Cone Parameters",
  "这里把均值方向固定在 x 轴，只改变均值长度和探索半径，用来隔离说明方向噪声为何会随动作幅度变化。": "The mean direction is fixed on the x-axis; only mean length and exploration radius are changed to isolate why directional noise varies with action magnitude.",
  "均值长度 l = ||μ||": "Mean length l = ||μ||",
  "l 越小，蓝色置信椭球越接近原点，红色方向视锥越容易张开。": "Smaller l brings the blue confidence ellipsoid closer to the origin, opening the red direction cone.",
  "固定探索半径 σ": "Fixed exploration radius σ",
  "σ 表示 Cartesian 三维高斯的局部采样尺度；在 DS 中，方向 σ 和速度 σ 会被分开解释。": "σ is the local sampling scale of the Cartesian 3D Gaussian; in DS, direction σ and speed σ are interpreted separately.",
  "方向半角 θ。越大表示同样采样噪声会造成更剧烈的方向变化。": "Direction half-angle θ. Larger values mean the same sampling noise causes stronger direction changes.",
  "σ / l。这个比值越高，Cartesian 探索越不像“沿原方向微调”。": "σ / l. The larger this ratio, the less Cartesian exploration behaves like a small adjustment along the original direction.",
  "这正是 direction-speed 分层的动机：让方向维度的 σ 直接控制方向探索视锥，让速度维度的 σ 只影响速度探索， 不再让 ||μ|| 隐式改变方向探索强度。": "This motivates direction-speed factorization: directional σ directly controls the direction cone, while speed σ only affects speed exploration, instead of letting ||μ|| implicitly change direction exploration.",
  "把上一节的问题拆开：方向由 u 控制，速度由 m 控制": "Factorize the previous problem: direction is controlled by u, speed by m",
  "红色锥形空间表示方向探索 σu ，蓝色锥台壳段表示 normalized speed ratio 的探索区间。": "The red cone shows direction exploration σu; the blue frustum shell shows the exploration interval of the normalized speed ratio.",
  "4. 方法定义：把 XYZ 探索改写成 Direction × Speed（方向 × 速度）": "4. Method: Rewrite XYZ Exploration as Direction × Speed",
  "环境仍执行 v=(dx,dy,dz) ；策略内部把它拆成方向 u 和 normalized speed ratio ρ ，再 compose 回 3D 位移。": "The environment still executes v=(dx,dy,dz). Internally, the policy factorizes it into direction u and normalized speed ratio ρ, then composes it back into a 3D displacement.",
  "拖 yaw/pitch 看方向变；拖 m 看半径变；拖 σu 看红色方向锥变宽，拖 σm 看蓝色球壳锥台展开。": "Drag yaw/pitch to change direction; drag m to change radius; drag σu to widen the red direction cone; drag σm to expand the blue speed shell.",
  "方向 yaw": "Direction yaw",
  "方向 pitch": "Direction pitch",
  "normalized speed ρ": "Normalized speed ρ",
  "方向探索 σu": "Direction exploration σu",
  "速度 raw 探索 σm": "Raw speed exploration σm",
  "改变单位方向 u 。": "Change unit direction u.",
  "和 yaw 一起定义 u 。": "Defines u together with yaw.",
  "表示沿当前方向走到 cube 边界的比例。": "Ratio of the distance to the cube boundary along the current direction.",
  "控制红色方向锥角。": "Controls the red direction cone angle.",
  "经 sigmoid 后变成蓝色球壳锥台。": "After sigmoid, it becomes the blue spherical-shell frustum.",
  "方向 u ，始终在单位球面上。": "Direction u, always on the unit sphere.",
  "环境动作 v=m u 。": "Environment action v=m u.",
  "5. 概率参数化：Action（动作）层级与方向分布": "5. Probabilistic Parameterization: Action Levels and Direction Distributions",
  "5.1 三种 Action（动作）变量": "5.1 Three Action Variables",
  "从 Actor（策略网络）采样到 env.step": "From actor sampling to env.step",
  "latent value（潜在变量）": "latent value",
  "policy action（策略动作）": "policy action",
  "env action（环境动作）": "env action",
  "5.2 方法卡片：Baseline（基线）、Adapter（动作适配器）与 Integrated（分布内集成）": "5.2 Method Cards: Baseline, Adapter, and Integrated",
  "选择 tab（标签页）查看对应实现": "Select a tab to inspect the implementation",
  "Baseline（基线）/ XYZ": "Baseline / XYZ",
  "Adapter（动作适配器）": "Adapter",
  "Spherical（球坐标）": "Spherical",
  "Stereographic（立体投影）": "Stereographic",
  "Tangent（切空间）": "Tangent",
  "Baseline（基线）实现": "Baseline Implementation",
  "Baseline（基线）数据流": "Baseline Data Flow",
  "Adapter DS（动作适配器）实现": "Adapter DS Implementation",
  "Adapter（动作适配器）核心映射": "Adapter Core Mapping",
  "Spherical（球坐标）实现": "Spherical Implementation",
  "Spherical（球坐标）核心映射": "Spherical Core Mapping",
  "Stereographic（立体投影）实现": "Stereographic Implementation",
  "Stereographic（立体投影）核心映射": "Stereographic Core Mapping",
  "Tangent（切空间）实现": "Tangent Implementation",
  "Tangent（切空间）核心映射": "Tangent Core Mapping",
  "严格 log_prob": "exact log_prob",
  "XYZ 轴向探索": "XYZ axis exploration",
  "D+1 表示": "D+1 representation",
  "非可逆": "non-invertible",
  "FQL 可用": "FQL-compatible",
  "actor 输出 D 维 TanhNormal；OGBench 是 [dx,dy,dz,dyaw,grip] 。": "The actor outputs a D-dimensional TanhNormal; OGBench uses [dx,dy,dz,dyaw,grip].",
  "critic、dataset、env 都使用同一个 raw 5D action。": "The critic, dataset, and environment all use the same raw 5D action.",
  "每个维度独立建模，三维位移的方向和速度没有显式结构。": "Each dimension is modeled independently; 3D displacement direction and speed have no explicit structure.",
  "decompose : xyz -> unit_direction + speed_coord ，其余维度原样保留。": "decompose: xyz -> unit_direction + speed_coord, while other dimensions are preserved.",
  "speed_coord = 2 * (||xyz|| / R(u)) - 1 ，其中 R(u)=1/max(|u_i|) 。": "speed_coord = 2 * (||xyz|| / R(u)) - 1, where R(u)=1/max(|u_i|).",
  "compose : 先 normalize direction，再用 ρ=(speed_coord+1)/2 还原 xyz=ρR(u)u 。": "compose: normalize direction, then recover xyz=ρR(u)u with ρ=(speed_coord+1)/2.",
  "可逆 bijector": "invertible bijector",
  "极点附近更敏感": "sensitive near poles",
  "推荐主线": "recommended main line",
  "方差更稳": "more stable variance",
  "旋转不变探索": "rotation-invariant exploration",
  "对跖点 cut locus": "antipodal cut locus",
  "5.3 Direction Distribution Lab（方向分布实验台）": "5.3 Direction Distribution Lab",
  "相同均值 · 相同 σ · 三种坐标": "Same mean · same σ · three coordinates",
  "彩色点是 240 个单位方向样本。拖动均值方向和方向 σ，切换方法后比较 sphere angular distance（球面角距离）统计；speed（速度）不参与本实验。": "Colored points are 240 unit-direction samples. Adjust the mean direction and direction σ, then switch methods to compare sphere angular distance statistics. Speed is excluded from this lab.",
  "均值方向预设": "Mean Direction Presets",
  "北极": "North Pole",
  "赤道": "Equator",
  "南极": "South Pole",
  "斜方向": "Oblique",
  "角距离 mean": "angular distance mean",
  "角距离 std": "angular distance std",
  "角距离 P95": "angular distance P95",
  "6. Speed Bound（速度边界）：Sphere（球）与 Cube（立方体）": "6. Speed Bound: Sphere vs. Cube",
  "相同方向 u · 相同速度比例 ρ": "Same direction u · same speed ratio ρ",
  "速度比例 ρ 独立于方向分布。Sphere 使用固定半径 R(u)=1 ；Cube 使用 R(u)=1/max|uᵢ| ，因此沿斜方向能够覆盖单位球之外、cube 之内的合法动作。": "The speed ratio ρ is independent of the direction distribution. Sphere uses the fixed radius R(u)=1; Cube uses R(u)=1/max|uᵢ|, so oblique directions can cover legal cube actions outside the unit ball.",
  "黄色 = sphere action · 蓝色 = cube action": "Yellow = sphere action · blue = cube action",
  "虚线表示同一单位方向；拖动方向可观察 cube 径向边界随 u 改变。": "The dashed line indicates the same unit direction; drag the direction to observe how the cube radial boundary changes with u.",
  "Sphere bound：只能覆盖单位球。": "Sphere bound: covers only the unit ball.",
  "Cube bound：沿方向缩放到 cube 表面。": "Cube bound: scales along the direction to the cube surface.",
  "Sphere 环境动作。": "Sphere environment action.",
  "Cube 环境动作；每一维仍在 [-1,1]。": "Cube environment action; every dimension remains in [-1,1].",
  "7. Modular Pipeline（模块化流水线）：从 action_raw 到 action_env": "7. Modular Pipeline: From action_raw to action_env",
  "先看数据流，再看实现层级": "Read data flow first, then implementation layers",
  "Dataset（数据集）": "Dataset",
  "Actor / Critic（策略 / 价值网络）": "Actor / Critic",
  "Execution（环境执行）": "Execution",
  "log_prob（对数概率）": "log_prob",
  "播放流程": "Play Flow",
  "暂停流程": "Pause Flow",
  "Modular Action Pipeline（模块化动作流水线）": "Modular Action Pipeline",
  "同一条 Pipeline（流水线），不同方法只替换模块": "One pipeline; each method only swaps modules",
  "一句话理解": "In One Sentence",
  "Integrated（分布内集成）改变“如何采样策略动作”，但环境与 SAC（软演员评论家算法）看到的是同一个动作。": "Integrated changes how the policy action is sampled, while SAC and the environment see the same action.",
  "Offline path（离线数据路径）": "Offline Path",
  "Policy Distribution（策略分布内部）": "Inside the Policy Distribution",
  "从 observation（观测）产生可训练的策略动作": "Produce the trainable policy action from an observation",
  "Observation（观测）": "Observation",
  "把状态编码为 feature（特征）h(s)。": "Encode the state into feature h(s).",
  "Parameter Heads（参数输出头）": "Parameter Heads",
  "Gaussian Sampler（高斯采样器）": "Gaussian Sampler",
  "内部样本": "Internal Sample",
  "SAC boundary（SAC 训练边界）": "SAC Boundary",
  "Execution output（执行输出）": "Execution Output",
  "SAC Training Boundary（SAC 训练边界）": "SAC Training Boundary",
  "两个训练目标都读取 action_pi（策略动作）": "Both training objectives read action_pi",
  "唯一训练动作": "The Only Training Action",
  "Execution Boundary（环境执行边界）": "Execution Boundary",
  "这里只负责把训练动作交给环境": "This stage only delivers the training action to the environment",
  "Integrated（分布内集成）": "Integrated",
  "Actor Backbone（策略主干）": "Actor Backbone",
  "把 observation（观测）编码为状态特征 h(s)。": "Encode observation into state feature h(s).",
  "μ / log σ 参数头": "μ / log σ parameter heads",
  "Raw Sampler（原始采样器）": "Raw Sampler",
  "DS Bijector（方向速度可逆变换器）": "DS Bijector",
  "Identity（恒等映射）": "Identity",
  "Critic（价值网络）": "Critic",
  "Density（概率密度）": "Density",
  "暂停实体动画": "Pause Entities",
  "继续实体动画": "Resume Entities",
  "重新流动": "Replay Flow",
  "页面章节导航": "Page section navigation",
  "返回顶部": "Back to top",
  "8. 总结：三类实现边界": "8. Summary: Three Implementation Boundaries",
  "建议实验汇报中按此口径描述": "Use this wording in experiment reports",
  "问题": "Question",
  "Action-adapter DS": "Action-adapter DS",
  "Integrated DS": "Integrated DS",
  "数据集动作如何进入模型？": "How does a dataset action enter the model?",
  "5D 先 decompose 成 6D。": "5D is first decomposed into 6D.",
  "Actor / critic 看到几维？": "How many dimensions do actor / critic see?",
  "6D decomposed action。": "6D decomposed action.",
  "Critic 里的 Q(s,aπ) ， aπ 是什么？": "In critic Q(s,aπ), what is aπ?",
  "Actor 输出能否直接给 env？": "Can actor output be sent directly to env?",
  "不能，必须先 compose 成 5D。": "No. It must first be composed into 5D.",
  "是否是严格 bijection？": "Is it a strict bijection?",
  "不是。 D+1 → D 多对一。": "No. D+1 → D is many-to-one.",
  "适合什么实验定位？": "What experiment role does it fit?",
  "FQL 或 D+1 表示消融；RLPD 中只能算近似 ablation。": "FQL or D+1 representation ablation; in RLPD it is only an approximate ablation.",
  "说明：本页优先使用 Three.js 做 3D 动作语义演示；如果 CDN 不可用，会自动切换到内置 Canvas 动画。 当前页面由 Vite（前端构建工具）独立管理，源码位于 docs/ds-visualizer 。": "Note: this page prefers Three.js for 3D action-semantics demos; if the CDN is unavailable, it automatically falls back to built-in Canvas animation. The page is managed as an independent Vite project under docs/ds-visualizer.",
  "总览两个层级：Integrated DS 把 latent ξ 变成 policy action aπ；Action-adapter DS 决定 aπ 是否还要 compose 成 a_env。": "Overview of two layers: Integrated DS maps latent ξ into policy action aπ; Action-adapter DS decides whether aπ must still be composed into a_env.",
  "Dataset 阶段：Action-adapter DS 先 decompose 成 policy-action batch；Integrated DS 通常保持 env action 维度。": "Dataset stage: Action-adapter DS first decomposes into policy-action batches; Integrated DS usually keeps the env-action dimensionality.",
  "Actor / Critic 阶段：SAC 内部始终对齐 Q(s,aπ) 与 log π(aπ|s)；Integrated DS 的 forward 输出就是这个 aπ。": "Actor / Critic stage: SAC always aligns Q(s,aπ) with log π(aπ|s); the forward output of Integrated DS is this aπ.",
  "Execution 阶段：a_env = g(aπ)。Integrated DS 里 g 是 identity；Action-adapter DS 或组合方案里 g 可以是 compose。": "Execution stage: a_env = g(aπ). In Integrated DS, g is identity; in Action-adapter DS or combined variants, g may be compose.",
  "log_prob 阶段：Integrated DS 的 bijector 有 inverse 和 log-det，因此能给 aπ 严格密度；只有 aπ=a_env 时才等价于 env-action density。": "log_prob stage: the Integrated DS bijector has inverse and log-det, so it gives an exact density for aπ; it equals env-action density only when aπ=a_env.",
  "Overview（总览）": "Overview",
  "1 / Dataset（数据集）": "1 / Dataset",
  "2 / Actor + Critic（策略 + 价值网络）": "2 / Actor + Critic",
  "3 / Execution（环境执行）": "3 / Execution",
  "4 / log_prob（对数概率）": "4 / log_prob",
  "Integrated（分布内集成）Recipe（组装方案）选择 Parameter Heads（参数输出头）、Gaussian Raw Sampler（高斯原始采样器）、DS Bijector（方向速度可逆变换器）和 Identity Postprocessor（恒等后处理器）。": "The Integrated recipe selects parameter heads, Gaussian raw sampler, DS bijector, and identity postprocessor.",
  "Integrated（分布内集成）改变“如何采样策略动作”；action_pi（策略动作）已经是环境可执行的 XYZ（笛卡尔坐标）动作。": "Integrated changes how the policy action is sampled; action_pi is already an executable XYZ environment action.",
  "环境动作直接作为 action_pi（策略动作）进入 Critic（价值网络），不执行 decompose（分解）。": "The environment action directly enters the critic as action_pi; no decomposition is applied.",
  "输出 location（位置参数）与 log-scale（对数尺度参数）；Tangent（切空间）方法允许两个输出头宽度不同。": "Outputs location and log-scale; Tangent allows the two heads to have different widths.",
  "MultivariateNormalDiag（对角多元高斯）产生 action_raw（原始动作）。": "MultivariateNormalDiag produces action_raw.",
  "选择 Spherical / Stereographic / Tangent（球坐标 / 立体投影 / 切空间）几何，并提供 inverse（逆变换）与 log-det（对数行列式）。": "Selects Spherical / Stereographic / Tangent geometry and provides inverse plus log-det.",
  "action_pi（策略动作）不再改变。": "action_pi is unchanged.",
  "action_env = action_pi；环境收到 Bijector（可逆变换器）的输出。": "action_env = action_pi; the environment receives the bijector output.",
  "Gaussian latent（高斯潜在坐标）": "Gaussian latent",
  "Environment shape（环境动作维度）": "Environment shape",
  "与 action_pi 相同": "Same as action_pi",
  "5D，与 action_pi 相同": "5D, same as action_pi",
  "Integrated（分布内集成）中，action_raw（原始动作）经过几何变换成为 action_pi（策略动作）；Critic（价值网络）和 log_prob（对数概率）都读取 action_pi，Identity（恒等映射）再把同一个实体交给环境。": "In Integrated DS, action_raw becomes action_pi through geometric transformation; both critic and log_prob read action_pi, and identity passes the same entity to the environment.",
  "Adapter（动作适配器）Recipe（组装方案）选择 Parameter Heads（参数输出头）、Gaussian Raw Sampler（高斯原始采样器）、Mixed Squash（混合压缩器）和 DirectionSpeedAdapter（方向速度适配器）。": "The Adapter recipe selects parameter heads, Gaussian raw sampler, mixed squash, and DirectionSpeedAdapter.",
  "Adapter（动作适配器）让 SAC（软演员评论家算法）在 direction + speed（方向 + 速度）坐标中训练，执行前再把 action_pi（策略动作）翻译为 XYZ（笛卡尔坐标）。": "Adapter trains SAC in direction-plus-speed coordinates, then translates action_pi into XYZ before execution.",
  "环境动作先经过 decompose（分解）得到 policy-space action_pi（策略空间动作），Replay Buffer（回放缓冲区）与 Critic（价值网络）存取这个坐标。": "The environment action is first decomposed into policy-space action_pi; replay buffer and critic store this coordinate.",
  "decompose（分解）": "decompose",
  "输出逐策略坐标的 μ（均值）与 log σ（对数标准差）。": "Outputs μ and log σ for each policy coordinate.",
  "方向坐标选择 tanh / sigmoid / identity（双曲正切 / 逻辑函数 / 恒等），速度压缩与速度解释独立配置。": "Direction coordinates choose tanh / sigmoid / identity; speed squashing and speed interpretation are configured independently.",
  "DirectionSpeedAdapter（方向速度适配器）": "DirectionSpeedAdapter",
  "compose（合成）把 action_pi（策略动作）解释成 XYZ（笛卡尔坐标）与其它环境维度。": "compose interprets action_pi as XYZ plus the remaining environment dimensions.",
  "action_env = compose(action_pi)；Cartesian（笛卡尔）表示可发生 D+1 → D。": "action_env = compose(action_pi); Cartesian representation can involve D+1 → D.",
  "6D Gaussian sample（六维高斯样本）": "6D Gaussian sample",
  "6D direction + speed + scalars": "6D direction + speed + scalars",
  "5D XYZ + yaw + gripper": "5D XYZ + yaw + gripper",
  "Adapter（动作适配器）中，action_pi（策略动作）仍是 Critic（价值网络）和 log_prob（对数概率）的变量；只有执行边界把它 compose（合成）为 action_env（环境动作），因此不要把 action_pi 的密度误称为环境动作密度。": "In Adapter DS, action_pi remains the variable for critic and log_prob; only the execution boundary composes it into action_env, so do not call action_pi density an env-action density.",
  "spherical · fixed global angles": "spherical · fixed global angles",
  "θ、φ 的局部尺度随极点位置和 sigmoid 饱和程度变化。": "The local scale of θ and φ changes with pole position and sigmoid saturation.",
  "stereographic · unbounded plane chart": "stereographic · unbounded plane chart",
  "平面方差固定；球面角尺度按 2/(1+||p||²) 缩放。": "Plane variance is fixed; spherical angular scale is multiplied by 2/(1+||p||²).",
  "tangent · mean-centered exponential map": "tangent · mean-centered exponential map",
  "切平面各向同性采样；角距离分布不依赖平均方向。": "Isotropic tangent-plane sampling; angular-distance distribution is independent of the mean direction."
}));

export function currentLanguage() {
  return language;
}

export function tx(zhText) {
  if (language !== "en") return zhText;
  return EN_TEXT.get(normalize(zhText)) || zhText;
}

export function onLanguageChange(listener) {
  languageListeners.add(listener);
  return () => languageListeners.delete(listener);
}

function applyStoredTheme() {
  const stored = localStorage.getItem(THEME_KEY);
  const preferredDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches;
  const theme = stored || (preferredDark ? "dark" : "light");
  document.documentElement.classList.toggle("dark", theme === "dark");
  document.documentElement.classList.toggle("light", theme !== "dark");
}

function setTheme(theme) {
  localStorage.setItem(THEME_KEY, theme);
  document.documentElement.classList.toggle("dark", theme === "dark");
  document.documentElement.classList.toggle("light", theme !== "dark");
  updateControlState();
}

function setLanguage(nextLanguage) {
  language = nextLanguage === "en" ? "en" : "zh";
  localStorage.setItem(LANGUAGE_KEY, language);
  document.documentElement.lang = language === "en" ? "en" : "zh-CN";
  document.title = language === "en"
    ? "Direction-Speed Method and Action Pipeline"
    : "Direction-Speed 方法与动作流水线";
  applyStaticI18n();
  updateControlState();
  languageListeners.forEach((listener) => listener(language));
}

function updateControlState() {
  const isDark = document.documentElement.classList.contains("dark");
  document.querySelectorAll("[data-lang-option]").forEach((button) => {
    const active = button.dataset.langOption === language;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  const themeToggle = document.getElementById("themeToggle");
  if (themeToggle) {
    themeToggle.textContent = isDark
      ? (language === "en" ? "Light" : "日间")
      : (language === "en" ? "Dark" : "夜间");
    themeToggle.setAttribute("aria-label", isDark
      ? (language === "en" ? "Switch to light mode" : "切换到日间模式")
      : (language === "en" ? "Switch to dark mode" : "切换到夜间模式"));
  }
}

function applyStaticI18n() {
  const candidates = document.querySelectorAll("h1,h2,h3,h4,p,li,span,strong,a,button,small,b,div.eyebrow,div.cell,div.shadcn-nav-title");
  candidates.forEach((element) => {
    if (element.closest("pre, code, canvas, svg")) return;
    if (!element.dataset.i18nOriginalHtml) {
      element.dataset.i18nOriginalHtml = element.innerHTML;
      element.dataset.i18nOriginalText = normalize(element.textContent);
    }
    if (language === "zh") {
      element.innerHTML = element.dataset.i18nOriginalHtml;
      return;
    }
    const translated = EN_TEXT.get(element.dataset.i18nOriginalText);
    if (!translated) return;
    element.textContent = translated;
  });
}

export function initI18n() {
  applyStoredTheme();
  const storedLanguage = localStorage.getItem(LANGUAGE_KEY);
  const initialLanguage = storedLanguage === "en" ? "en" : "zh";

  document.querySelectorAll("[data-lang-option]").forEach((button) => {
    button.addEventListener("click", () => setLanguage(button.dataset.langOption));
  });

  const themeToggle = document.getElementById("themeToggle");
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      setTheme(document.documentElement.classList.contains("dark") ? "light" : "dark");
    });
  }

  setLanguage(initialLanguage);
}
