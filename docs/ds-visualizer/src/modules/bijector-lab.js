import { currentLanguage, onLanguageChange, tx } from "../i18n.js";

export function initBijectorLab() {
  const canvas = document.getElementById("bijectorCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const methodButtons = Array.from(document.querySelectorAll(".bijector-method"));
    const presetButtons = Array.from(document.querySelectorAll(".bijector-preset"));
    const ui = {
      yaw: document.getElementById("bijectorYaw"),
      pitch: document.getElementById("bijectorPitch"),
      sigma: document.getElementById("bijectorDirSigma"),
      yawValue: document.getElementById("bijectorYawValue"),
      pitchValue: document.getElementById("bijectorPitchValue"),
      sigmaValue: document.getElementById("bijectorDirSigmaValue"),
      angleMean: document.getElementById("bijectorAngleMean"),
      angleStd: document.getElementById("bijectorAngleStd"),
      angleP95: document.getElementById("bijectorAngleP95"),
      title: document.getElementById("bijectorPlotTitle"),
      subtitle: document.getElementById("bijectorPlotSubtitle"),
      formula: document.getElementById("bijectorFormulaLive")
    };
    const methodMeta = {
      spherical: {
        color: "#ffc857",
        title: "spherical · fixed global angles",
        subtitle: "θ、φ 的局部尺度随极点位置和 sigmoid 饱和程度变化。"
      },
      stereographic: {
        color: "#6ee7f9",
        title: "stereographic · unbounded plane chart",
        subtitle: "平面方差固定；球面角尺度按 2/(1+||p||²) 缩放。"
      },
      tangent: {
        color: "#c9a3ff",
        title: "tangent · mean-centered exponential map",
        subtitle: "切平面各向同性采样；角距离分布不依赖平均方向。"
      }
    };
  let method = "spherical";

  const isDarkTheme = () => document.documentElement.classList.contains("dark");

    const clampLab = (value, min, max) => Math.min(max, Math.max(min, value));
    const sigmoidLab = (value) => 1 / (1 + Math.exp(-value));
    const logitLab = (value) => {
      const bounded = clampLab(value, 1e-6, 1 - 1e-6);
      return Math.log(bounded / (1 - bounded));
    };
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const cross = (a, b) => [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0]
    ];
    const norm = (v) => Math.hypot(...v);
    const normalize = (v) => {
      const length = Math.max(norm(v), 1e-12);
      return v.map((value) => value / length);
    };
    const addScaled = (a, scaleA, b, scaleB) => [
      a[0] * scaleA + b[0] * scaleB,
      a[1] * scaleA + b[1] * scaleB,
      a[2] * scaleA + b[2] * scaleB
    ];

    function mulberry32(seed) {
      return () => {
        let value = seed += 0x6D2B79F5;
        value = Math.imul(value ^ value >>> 15, value | 1);
        value ^= value + Math.imul(value ^ value >>> 7, value | 61);
        return ((value ^ value >>> 14) >>> 0) / 4294967296;
      };
    }

    function gaussianGenerator(seed) {
      const random = mulberry32(seed);
      let spare = null;
      return () => {
        if (spare !== null) {
          const value = spare;
          spare = null;
          return value;
        }
        const u = Math.max(random(), 1e-12);
        const v = random();
        const radius = Math.sqrt(-2 * Math.log(u));
        spare = radius * Math.sin(2 * Math.PI * v);
        return radius * Math.cos(2 * Math.PI * v);
      };
    }

    function meanDirection() {
      const yaw = Number(ui.yaw.value) * Math.PI / 180;
      const pitch = Number(ui.pitch.value) * Math.PI / 180;
      return normalize([
        Math.cos(pitch) * Math.cos(yaw),
        Math.sin(pitch),
        Math.cos(pitch) * Math.sin(yaw)
      ]);
    }

    function sphericalSample(mean, sigma, normal) {
      const theta = Math.acos(clampLab(mean[2], -1, 1));
      let phi = Math.atan2(mean[1], mean[0]);
      if (phi < 0) phi += 2 * Math.PI;
      const thetaRaw = logitLab(theta / Math.PI);
      const phiRaw = logitLab(phi / (2 * Math.PI));
      const sampledTheta = Math.PI * sigmoidLab(thetaRaw + sigma * normal());
      const sampledPhi = 2 * Math.PI * sigmoidLab(phiRaw + sigma * normal());
      return [
        Math.sin(sampledTheta) * Math.cos(sampledPhi),
        Math.sin(sampledTheta) * Math.sin(sampledPhi),
        Math.cos(sampledTheta)
      ];
    }

    function stereographicCenter(mean) {
      const denominator = 1 + mean[2];
      if (denominator < 1e-5) return [1e4, 0];
      return [mean[0] / denominator, mean[1] / denominator];
    }

    function stereographicSample(mean, sigma, normal) {
      const center = stereographicCenter(mean);
      const px = center[0] + sigma * normal();
      const py = center[1] + sigma * normal();
      const radiusSquared = px * px + py * py;
      const denominator = 1 + radiusSquared;
      return [2 * px / denominator, 2 * py / denominator, (1 - radiusSquared) / denominator];
    }

    function tangentSample(mean, sigma, normal) {
      const reference = Math.abs(mean[2]) < 0.9 ? [0, 0, 1] : [0, 1, 0];
      const basisX = normalize(cross(reference, mean));
      const basisY = cross(mean, basisX);
      const rawX = sigma * normal();
      const rawY = sigma * normal();
      const rawRadius = Math.hypot(rawX, rawY);
      const maxAngle = Math.PI - 1e-4;
      const angle = maxAngle * Math.tanh(rawRadius / maxAngle);
      if (rawRadius < 1e-10) return mean.slice();
      const tangentUnit = normalize([
        basisX[0] * rawX + basisY[0] * rawY,
        basisX[1] * rawX + basisY[1] * rawY,
        basisX[2] * rawX + basisY[2] * rawY
      ]);
      return normalize(addScaled(mean, Math.cos(angle), tangentUnit, Math.sin(angle)));
    }

    function generateSamples() {
      const normal = gaussianGenerator(20260620);
      const mean = meanDirection();
      const sigma = Number(ui.sigma.value);
      const samples = [];
      for (let index = 0; index < 240; index += 1) {
        let direction;
        if (method === "spherical") direction = sphericalSample(mean, sigma, normal);
        if (method === "stereographic") direction = stereographicSample(mean, sigma, normal);
        if (method === "tangent") direction = tangentSample(mean, sigma, normal);
        const angle = Math.acos(clampLab(dot(mean, direction), -1, 1));
        samples.push({ direction, angle });
      }
      return { mean, samples };
    }

    function cameraProject(vector, radius, centerX, centerY) {
      const yaw = -0.62;
      const pitch = 0.28;
      const cosYaw = Math.cos(yaw);
      const sinYaw = Math.sin(yaw);
      const x1 = cosYaw * vector[0] + sinYaw * vector[2];
      const z1 = -sinYaw * vector[0] + cosYaw * vector[2];
      const cosPitch = Math.cos(pitch);
      const sinPitch = Math.sin(pitch);
      const y2 = cosPitch * vector[1] - sinPitch * z1;
      const z2 = sinPitch * vector[1] + cosPitch * z1;
      return { x: centerX + radius * x1, y: centerY - radius * y2, depth: z2 };
    }

    function drawCurve(points, projection, color, width, alpha) {
      ctx.beginPath();
      points.forEach((point, index) => {
        const projected = projection(point);
        if (index === 0) ctx.moveTo(projected.x, projected.y);
        else ctx.lineTo(projected.x, projected.y);
      });
      ctx.strokeStyle = color;
      ctx.globalAlpha = alpha;
      ctx.lineWidth = width;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

  function drawCoordinateAxes(projection, extent = 1.28) {
      const axes = [
        { unit: [1, 0, 0], color: "#ff6b6b", label: "X" },
        { unit: [0, 1, 0], color: "#61e6a0", label: "Y" },
        { unit: [0, 0, 1], color: "#6ea8ff", label: "Z" }
      ];
      const origin = projection([0, 0, 0]);
      axes.forEach(({ unit, color, label }) => {
        const negative = projection(unit.map((value) => -extent * value));
        const positive = projection(unit.map((value) => extent * value));
        ctx.beginPath();
        ctx.moveTo(negative.x, negative.y);
        ctx.lineTo(origin.x, origin.y);
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.38;
        ctx.lineWidth = 1.2;
        ctx.setLineDash([5, 5]);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(origin.x, origin.y);
        ctx.lineTo(positive.x, positive.y);
        ctx.globalAlpha = 0.9;
        ctx.lineWidth = 1.8;
        ctx.setLineDash([]);
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.font = "700 12px SFMono-Regular, Consolas, monospace";
        ctx.fillText(label, positive.x + 6, positive.y - 5);
      });
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.arc(origin.x, origin.y, 3, 0, 2 * Math.PI);
    ctx.fillStyle = isDarkTheme() ? "#edf7ff" : "#0f1729";
    ctx.fill();
      ctx.font = "700 10px SFMono-Regular, Consolas, monospace";
      ctx.fillText("O", origin.x + 6, origin.y + 13);
    }

  function drawSphere(width, height, mean, samples) {
    const dark = isDarkTheme();
    ctx.clearRect(0, 0, width, height);
      const centerX = width * 0.5;
      const centerY = height * 0.53;
      const radius = Math.min(width * 0.36, height * 0.39);
      const projection = (point) => cameraProject(point, radius, centerX, centerY);
      const gradient = ctx.createRadialGradient(
        centerX - radius * 0.32, centerY - radius * 0.38, radius * 0.08,
        centerX, centerY, radius
      );
    gradient.addColorStop(0, dark ? "rgba(110,231,249,0.16)" : "rgba(8,145,178,0.14)");
    gradient.addColorStop(0.72, dark ? "rgba(24,45,65,0.11)" : "rgba(15,23,42,0.045)");
    gradient.addColorStop(1, dark ? "rgba(3,8,14,0.28)" : "rgba(15,23,42,0.08)");
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
      ctx.fillStyle = gradient;
      ctx.fill();
    ctx.strokeStyle = dark ? "rgba(110,231,249,0.42)" : "rgba(8,145,178,0.5)";
    ctx.lineWidth = dark ? 1.4 : 1.6;
      ctx.stroke();

      for (let latitude = -60; latitude <= 60; latitude += 30) {
        const lat = latitude * Math.PI / 180;
        const points = [];
        for (let step = 0; step <= 96; step += 1) {
          const lon = 2 * Math.PI * step / 96;
          points.push([Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon)]);
        }
      drawCurve(points, projection, dark ? "#8ea0b5" : "#64748b", dark ? 0.7 : 0.9, dark ? 0.22 : 0.34);
      }
      for (let longitude = 0; longitude < 180; longitude += 30) {
        const lon = longitude * Math.PI / 180;
        const points = [];
        for (let step = 0; step <= 96; step += 1) {
          const lat = -Math.PI / 2 + Math.PI * step / 96;
          points.push([Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon)]);
        }
      drawCurve(points, projection, dark ? "#8ea0b5" : "#64748b", dark ? 0.7 : 0.9, dark ? 0.18 : 0.3);
      }

      drawCoordinateAxes(projection);

      const color = methodMeta[method].color;
      samples
        .map((sample) => ({ ...sample, projected: projection(sample.direction) }))
        .sort((a, b) => a.projected.depth - b.projected.depth)
        .forEach((sample) => {
          ctx.beginPath();
          ctx.arc(sample.projected.x, sample.projected.y, 3.2, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.globalAlpha = sample.projected.depth < 0 ? (dark ? 0.28 : 0.38) : 0.88;
          ctx.fill();
        });
      ctx.globalAlpha = 1;

      const projectedMean = projection(mean);
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(projectedMean.x, projectedMean.y);
      ctx.strokeStyle = dark ? "rgba(97,230,160,0.76)" : "rgba(5,150,105,0.9)";
      ctx.lineWidth = 2.4;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(projectedMean.x, projectedMean.y, 8, 0, 2 * Math.PI);
      ctx.fillStyle = "#61e6a0";
      ctx.fill();
      ctx.strokeStyle = dark ? "#ecfff5" : "#0f1729";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    function percentile(sorted, fraction) {
      const index = Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * fraction));
      return sorted[index];
    }

    function updateFormula(mean) {
      if (method === "spherical") {
        const theta = Math.acos(clampLab(mean[2], -1, 1));
        ui.formula.textContent = currentLanguage() === "en"
          ? `θ=${(theta * 180 / Math.PI).toFixed(1)}°, sin(θ)=${Math.sin(theta).toFixed(3)}; the actual angular scale of φ noise is multiplied by sin(θ).`
          : `θ=${(theta * 180 / Math.PI).toFixed(1)}°, sin(θ)=${Math.sin(theta).toFixed(3)}；φ 噪声的实际角尺度会乘 sin(θ)。`;
      } else if (method === "stereographic") {
        const center = stereographicCenter(mean);
        const length = Math.hypot(...center);
        const scale = 2 / (1 + length * length);
        ui.formula.textContent = currentLanguage() === "en"
          ? `||pμ||=${length > 999 ? "∞" : length.toFixed(3)}, local angular scale≈2σ/(1+||pμ||²)=${(scale * Number(ui.sigma.value)).toFixed(3)} rad.`
          : `||pμ||=${length > 999 ? "∞" : length.toFixed(3)}，局部角尺度≈2σ/(1+||pμ||²)=${(scale * Number(ui.sigma.value)).toFixed(3)} rad。`;
      } else {
        ui.formula.textContent = currentLanguage() === "en"
          ? "angle(μ,u)=r=(π-δ)tanh(||ε||/(π-δ)); the same σ yields an angular-distance distribution independent of μ."
          : "angle(μ,u)=r=(π-δ)tanh(||ε||/(π-δ))；相同 σ 产生与 μ 无关的角距离分布。";
      }
    }

    function renderLab() {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(320, rect.width);
      const height = Math.max(360, rect.height);
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const { mean, samples } = generateSamples();
      drawSphere(width, height, mean, samples);

      const angles = samples.map((sample) => sample.angle).sort((a, b) => a - b);
      const angleMean = angles.reduce((sum, value) => sum + value, 0) / angles.length;
      const variance = angles.reduce((sum, value) => sum + (value - angleMean) ** 2, 0) / angles.length;
      const degrees = 180 / Math.PI;
      ui.angleMean.textContent = `${(angleMean * degrees).toFixed(2)}°`;
      ui.angleStd.textContent = `${(Math.sqrt(variance) * degrees).toFixed(2)}°`;
      ui.angleP95.textContent = `${(percentile(angles, 0.95) * degrees).toFixed(2)}°`;
      ui.yawValue.value = `${Number(ui.yaw.value).toFixed(0)}°`;
      ui.pitchValue.value = `${Number(ui.pitch.value).toFixed(0)}°`;
      ui.sigmaValue.value = Number(ui.sigma.value).toFixed(2);
      ui.title.textContent = tx(methodMeta[method].title);
      ui.subtitle.textContent = tx(methodMeta[method].subtitle);
      updateFormula(mean);
      window.__bijectorLabState = {
        method,
        angleMeanDegrees: angleMean * degrees,
        angleStdDegrees: Math.sqrt(variance) * degrees,
        angleP95Degrees: percentile(angles, 0.95) * degrees
      };
    }

    function setMethod(nextMethod, syncTab = true) {
      method = nextMethod;
      methodButtons.forEach((button) => button.classList.toggle("active", button.dataset.bijectorMethod === method));
      if (syncTab) {
        const tabId = method === "stereographic" ? "tab-stereo" : `tab-${method}`;
        const tab = document.getElementById(tabId);
        if (tab && !tab.classList.contains("active")) tab.click();
      }
      renderLab();
    }

    methodButtons.forEach((button) => button.addEventListener("click", () => setMethod(button.dataset.bijectorMethod)));
    presetButtons.forEach((button) => button.addEventListener("click", () => {
      ui.yaw.value = button.dataset.yaw;
      ui.pitch.value = button.dataset.pitch;
      presetButtons.forEach((item) => item.classList.toggle("active", item === button));
      renderLab();
    }));
    [ui.yaw, ui.pitch, ui.sigma].forEach((input) => input.addEventListener("input", () => {
      if (input === ui.yaw || input === ui.pitch) presetButtons.forEach((item) => item.classList.remove("active"));
      renderLab();
    }));
    Array.from(document.querySelectorAll(".mode-tab")).forEach((tab) => tab.addEventListener("click", () => {
      const mapped = tab.dataset.mode === "stereo" ? "stereographic" : tab.dataset.mode;
      if (methodMeta[mapped] && mapped !== method) setMethod(mapped, false);
    }));
    window.addEventListener("resize", renderLab);
    onLanguageChange(renderLab);
    renderLab();
}
