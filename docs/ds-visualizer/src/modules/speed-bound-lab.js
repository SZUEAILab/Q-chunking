import { currentLanguage, onLanguageChange } from "../i18n.js";

export function initSpeedBoundLab() {
  const canvas = document.getElementById("speedBoundCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const isDarkTheme = () => document.documentElement.classList.contains("dark");
    const ui = {
      yaw: document.getElementById("speedBoundYaw"),
      pitch: document.getElementById("speedBoundPitch"),
      rho: document.getElementById("speedBoundRho"),
      yawValue: document.getElementById("speedBoundYawValue"),
      pitchValue: document.getElementById("speedBoundPitchValue"),
      rhoValue: document.getElementById("speedBoundRhoValue"),
      sphere: document.getElementById("speedSphereReadout"),
      cube: document.getElementById("speedCubeReadout"),
      sphereVector: document.getElementById("speedSphereVector"),
      cubeVector: document.getElementById("speedCubeVector"),
      formula: document.getElementById("speedBoundFormula")
    };

    function direction() {
      const yaw = Number(ui.yaw.value) * Math.PI / 180;
      const pitch = Number(ui.pitch.value) * Math.PI / 180;
      return [
        Math.cos(pitch) * Math.cos(yaw),
        Math.sin(pitch),
        Math.cos(pitch) * Math.sin(yaw)
      ];
    }

    function project(vector, scale, centerX, centerY) {
      const cameraYaw = -0.62;
      const cameraPitch = 0.28;
      const cy = Math.cos(cameraYaw);
      const sy = Math.sin(cameraYaw);
      const x1 = cy * vector[0] + sy * vector[2];
      const z1 = -sy * vector[0] + cy * vector[2];
      const cp = Math.cos(cameraPitch);
      const sp = Math.sin(cameraPitch);
      return {
        x: centerX + scale * x1,
        y: centerY - scale * (cp * vector[1] - sp * z1),
        depth: sp * vector[1] + cp * z1
      };
    }

    function drawLine(from, to, projection, color, width = 1, dash = []) {
      const a = projection(from);
      const b = projection(to);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    function drawPoint(vector, projection, color, radius) {
      const point = projection(vector);
      ctx.beginPath();
      ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = isDarkTheme() ? "rgba(255,255,255,0.9)" : "rgba(15,23,42,0.72)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    function drawSpeedCoordinateAxes(projection, extent = 1.42) {
      const axes = [
        { unit: [1, 0, 0], color: "#ff6b6b", label: "X" },
        { unit: [0, 1, 0], color: "#61e6a0", label: "Y" },
        { unit: [0, 0, 1], color: "#6ea8ff", label: "Z" }
      ];
      axes.forEach(({ unit, color, label }) => {
        const negative = unit.map((value) => -extent * value);
        const positive = unit.map((value) => extent * value);
        drawLine(negative, [0, 0, 0], projection, color, 1.2, [5, 5]);
        drawLine([0, 0, 0], positive, projection, color, 1.8);
        const endpoint = projection(positive);
        ctx.fillStyle = color;
        ctx.font = "700 12px SFMono-Regular, Consolas, monospace";
        ctx.fillText(label, endpoint.x + 6, endpoint.y - 5);
      });
      const origin = projection([0, 0, 0]);
      ctx.beginPath();
      ctx.arc(origin.x, origin.y, 3, 0, 2 * Math.PI);
      ctx.fillStyle = isDarkTheme() ? "#edf7ff" : "#0f1729";
      ctx.fill();
      ctx.font = "700 10px SFMono-Regular, Consolas, monospace";
      ctx.fillText("O", origin.x + 6, origin.y + 13);
    }

    function renderSpeedBound() {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(320, rect.width);
      const height = Math.max(360, rect.height);
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      const dark = isDarkTheme();

      const centerX = width * 0.5;
      const centerY = height * 0.54;
      const scale = Math.min(width * 0.24, height * 0.28);
      const projection = (point) => project(point, scale, centerX, centerY);
      const u = direction();
      const rho = Number(ui.rho.value);
      const maxComponent = Math.max(...u.map(Math.abs));
      const cubeRadius = 1 / maxComponent;
      const sphereAction = u.map((value) => rho * value);
      const cubeAction = u.map((value) => rho * cubeRadius * value);
      const cubeBoundary = u.map((value) => cubeRadius * value);

      const vertices = [];
      for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) vertices.push([x, y, z]);
      const edges = [];
      vertices.forEach((vertex, i) => {
        vertices.forEach((other, j) => {
          if (j <= i) return;
          const changed = vertex.reduce((count, value, axis) => count + (value !== other[axis] ? 1 : 0), 0);
          if (changed === 1) edges.push([vertex, other]);
        });
      });
    edges.forEach(([a, b]) => drawLine(a, b, projection, dark ? "rgba(110,231,249,0.35)" : "rgba(15,23,42,0.42)", dark ? 1.2 : 1.35));

      for (let latitude = -60; latitude <= 60; latitude += 30) {
        const lat = latitude * Math.PI / 180;
        let previous = null;
        for (let step = 0; step <= 72; step += 1) {
          const lon = 2 * Math.PI * step / 72;
          const point = [Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon)];
        if (previous) drawLine(previous, point, projection, dark ? "rgba(255,200,87,0.18)" : "rgba(180,83,9,0.22)", dark ? 0.7 : 0.9);
          previous = point;
        }
      }
      for (let longitude = 0; longitude < 180; longitude += 30) {
        const lon = longitude * Math.PI / 180;
        let previous = null;
        for (let step = 0; step <= 72; step += 1) {
          const lat = -Math.PI / 2 + Math.PI * step / 72;
          const point = [Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon)];
        if (previous) drawLine(previous, point, projection, dark ? "rgba(255,200,87,0.15)" : "rgba(180,83,9,0.18)", dark ? 0.7 : 0.9);
          previous = point;
        }
      }

      drawSpeedCoordinateAxes(projection);

    drawLine([0, 0, 0], cubeBoundary, projection, dark ? "rgba(255,255,255,0.5)" : "rgba(15,23,42,0.45)", 1.4, [6, 5]);
      drawLine([0, 0, 0], sphereAction, projection, "#ffc857", 3);
      drawLine([0, 0, 0], cubeAction, projection, "#6ee7f9", 2.4);
      drawPoint(sphereAction, projection, "#ffc857", 6);
      drawPoint(cubeAction, projection, "#6ee7f9", 7);

      const formatVector = (vector) => `(${vector.map((value) => value.toFixed(3)).join(", ")})`;
      ui.yawValue.value = `${Number(ui.yaw.value).toFixed(0)}°`;
      ui.pitchValue.value = `${Number(ui.pitch.value).toFixed(0)}°`;
      ui.rhoValue.value = rho.toFixed(2);
      ui.sphere.textContent = `R=1.000 · ‖v‖=${rho.toFixed(3)}`;
      ui.cube.textContent = `R=${cubeRadius.toFixed(3)} · ‖v‖=${(rho * cubeRadius).toFixed(3)}`;
      ui.sphereVector.textContent = `v=${formatVector(sphereAction)}`;
      ui.cubeVector.textContent = `v=${formatVector(cubeAction)}`;
      ui.formula.textContent = currentLanguage() === "en"
        ? `max|uᵢ|=${maxComponent.toFixed(3)} → Rcube=${cubeRadius.toFixed(3)}; at the same ρ=${rho.toFixed(2)}, cube radial length is ${cubeRadius.toFixed(3)}× the sphere length.`
        : `max|uᵢ|=${maxComponent.toFixed(3)} → Rcube=${cubeRadius.toFixed(3)}；相同 ρ=${rho.toFixed(2)} 时，cube 的径向长度是 sphere 的 ${cubeRadius.toFixed(3)} 倍。`;
      window.__speedBoundState = { direction: u, rho, cubeRadius, sphereAction, cubeAction };
    }

    [ui.yaw, ui.pitch, ui.rho].forEach((input) => input.addEventListener("input", renderSpeedBound));
    window.addEventListener("resize", renderSpeedBound);
    onLanguageChange(renderSpeedBound);
    renderSpeedBound();
}
