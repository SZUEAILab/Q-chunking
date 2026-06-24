import { tx } from "../i18n.js";

export async function initDsScenes() {
  const host = document.getElementById("threeScene");
  const fallback = document.getElementById("sceneFallback");
  const coneHost = document.getElementById("coneScene");
  const coneFallback = document.getElementById("coneFallback");
  const dsResetButton = document.getElementById("dsReset");
  const dsAutoButton = document.getElementById("dsAuto");
  const coneResetButton = document.getElementById("coneReset");
  const coneAutoButton = document.getElementById("coneAuto");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const isDarkTheme = () => document.documentElement.classList.contains("dark");
  const controls = {
    yaw: document.getElementById("dsYaw"),
    pitch: document.getElementById("dsPitch"),
    speed: document.getElementById("dsSpeed"),
    dirSigma: document.getElementById("dsDirSigma"),
    speedSigma: document.getElementById("dsSpeedSigma"),
    yawValue: document.getElementById("dsYawValue"),
    pitchValue: document.getElementById("dsPitchValue"),
    speedValue: document.getElementById("dsSpeedValue"),
    dirSigmaValue: document.getElementById("dsDirSigmaValue"),
    speedSigmaValue: document.getElementById("dsSpeedSigmaValue"),
    uValue: document.getElementById("dsUValue"),
    vValue: document.getElementById("dsVValue")
  };
  const viewDefaults = { x: -0.18, y: -0.28, zoom: 1 };
  const coneDefaults = { x: -0.12, y: -0.34, zoom: 1 };
  const viewRotation = { x: viewDefaults.x, y: viewDefaults.y };
  let dragging = false;
  let lastPointer = { x: 0, y: 0 };
  let viewZoom = viewDefaults.zoom;
  let viewAutoRotate = true;
  const coneRotation = { x: coneDefaults.x, y: coneDefaults.y };
  let coneDragging = false;
  let lastConePointer = { x: 0, y: 0 };
  let coneZoom = coneDefaults.zoom;
  let coneAutoRotate = true;

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function setAutoButton(button, active) {
    if (!button) return;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
    button.textContent = tx(active ? "自动旋转" : "手动视角");
  }

  function updateSceneToolState() {
    setAutoButton(dsAutoButton, viewAutoRotate);
    setAutoButton(coneAutoButton, coneAutoRotate);
  }

  host.addEventListener("pointerdown", (event) => {
    dragging = true;
    viewAutoRotate = false;
    updateSceneToolState();
    lastPointer = { x: event.clientX, y: event.clientY };
    host.setPointerCapture(event.pointerId);
  });

  host.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const dx = event.clientX - lastPointer.x;
    const dy = event.clientY - lastPointer.y;
    lastPointer = { x: event.clientX, y: event.clientY };
    viewRotation.y += dx * 0.006;
    viewRotation.x = clamp(viewRotation.x + dy * 0.004, -1.05, 0.8);
  });

  host.addEventListener("pointerup", (event) => {
    dragging = false;
    host.releasePointerCapture(event.pointerId);
  });

  host.addEventListener("pointercancel", () => {
    dragging = false;
  });

  host.addEventListener("wheel", (event) => {
    event.preventDefault();
    viewZoom = clamp(viewZoom + (event.deltaY > 0 ? -0.08 : 0.08), 0.72, 1.35);
  }, { passive: false });

  if (dsResetButton) {
    dsResetButton.addEventListener("click", () => {
      viewRotation.x = viewDefaults.x;
      viewRotation.y = viewDefaults.y;
      viewZoom = viewDefaults.zoom;
      viewAutoRotate = true;
      updateSceneToolState();
    });
  }

  if (dsAutoButton) {
    dsAutoButton.addEventListener("click", () => {
      viewAutoRotate = !viewAutoRotate;
      updateSceneToolState();
    });
  }

  if (coneHost) {
    coneHost.addEventListener("pointerdown", (event) => {
      coneDragging = true;
      coneAutoRotate = false;
      updateSceneToolState();
      lastConePointer = { x: event.clientX, y: event.clientY };
      coneHost.setPointerCapture(event.pointerId);
    });

    coneHost.addEventListener("pointermove", (event) => {
      if (!coneDragging) return;
      const dx = event.clientX - lastConePointer.x;
      const dy = event.clientY - lastConePointer.y;
      lastConePointer = { x: event.clientX, y: event.clientY };
      coneRotation.y += dx * 0.006;
      coneRotation.x = clamp(coneRotation.x + dy * 0.004, -0.95, 0.72);
    });

    coneHost.addEventListener("pointerup", (event) => {
      coneDragging = false;
      coneHost.releasePointerCapture(event.pointerId);
    });

    coneHost.addEventListener("pointercancel", () => {
      coneDragging = false;
    });

    coneHost.addEventListener("wheel", (event) => {
      event.preventDefault();
      coneZoom = clamp(coneZoom + (event.deltaY > 0 ? -0.08 : 0.08), 0.72, 1.35);
    }, { passive: false });
  }

  if (coneResetButton) {
    coneResetButton.addEventListener("click", () => {
      coneRotation.x = coneDefaults.x;
      coneRotation.y = coneDefaults.y;
      coneZoom = coneDefaults.zoom;
      coneAutoRotate = true;
      updateSceneToolState();
    });
  }

  if (coneAutoButton) {
    coneAutoButton.addEventListener("click", () => {
      coneAutoRotate = !coneAutoRotate;
      updateSceneToolState();
    });
  }

  updateSceneToolState();

  async function loadThree() {
    const sources = [
      "https://unpkg.com/three@0.164.1/build/three.module.js",
      "https://cdn.jsdelivr.net/npm/three@0.164.1/build/three.module.js",
      "https://esm.sh/three@0.164.1"
    ];
    let lastError;
    for (const source of sources) {
      try {
        return await import(/* @vite-ignore */ source);
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError;
  }

  function startCanvasFallback(error) {
    console.warn("Three.js scene failed to load; using Canvas fallback:", error);
    fallback.style.display = "none";
    host.style.display = "block";
    host.innerHTML = "";

    const canvas = document.createElement("canvas");
    canvas.setAttribute("aria-label", "Canvas 3D direction-speed fallback scene");
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.minHeight = "430px";
    host.appendChild(canvas);

    const ctx = canvas.getContext("2d");
    const rawNoisePositions = [
      [-1.05, -0.24, 0.18], [-0.72, 0.38, -0.42], [-0.42, -0.66, 0.54],
      [-0.12, 0.76, -0.24], [0.22, -0.38, 0.82], [0.48, 0.18, -0.78],
      [0.76, -0.72, 0.12], [1.04, 0.52, -0.08], [-0.88, 0.02, 0.72],
      [0.08, -0.98, -0.52], [0.92, 0.82, 0.34], [-0.28, 0.24, -1.0]
    ];
    const mode = 3;

    function resize() {
      const rect = host.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function rotate(point, t) {
      const yAngle = reduceMotion ? 0.12 : Math.sin(t * 0.2) * 0.18;
      const xAngle = reduceMotion ? 0.04 : Math.sin(t * 0.16) * 0.08;
      const cosY = Math.cos(yAngle);
      const sinY = Math.sin(yAngle);
      const cosX = Math.cos(xAngle);
      const sinX = Math.sin(xAngle);
      const x1 = point[0] * cosY - point[2] * sinY;
      const z1 = point[0] * sinY + point[2] * cosY;
      const y2 = point[1] * cosX - z1 * sinX;
      const z2 = point[1] * sinX + z1 * cosX;
      return [x1, y2, z2];
    }

    function project(point, t) {
      const width = host.clientWidth || 500;
      const height = host.clientHeight || 430;
      const rotated = rotate(point, t);
      const depth = rotated[2] + 4.3;
      const scale = Math.min(width, height) * 0.72 / depth;
      return {
        x: width * 0.5 + rotated[0] * scale,
        y: height * 0.47 - rotated[1] * scale,
        depth,
        scale
      };
    }

    function line(a, b, color, alpha, width, t) {
      const pa = project(a, t);
      const pb = project(b, t);
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    function point(p, radius, color, alpha, t) {
      const projected = project(p, t);
      ctx.globalAlpha = alpha;
      ctx.fillStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = 16;
      ctx.beginPath();
      ctx.arc(projected.x, projected.y, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.globalAlpha = 1;
    }

    function drawSphere(t, actorFocus, xyzFocus) {
      const alpha = actorFocus || xyzFocus ? 0.44 : 0.24;
      for (let ring = -4; ring <= 4; ring += 1) {
        const y = ring / 4;
        const r = Math.sqrt(Math.max(0, 1 - y * y));
        let previous = null;
        for (let i = 0; i <= 96; i += 1) {
          const angle = (i / 96) * Math.PI * 2;
          const current = [Math.cos(angle) * r, y, Math.sin(angle) * r];
          if (previous) line(previous, current, "#61e6a0", alpha, 0.75, t);
          previous = current;
        }
      }
      for (let meridian = 0; meridian < 8; meridian += 1) {
        const offset = (meridian / 8) * Math.PI;
        let previous = null;
        for (let i = -48; i <= 48; i += 1) {
          const angle = (i / 48) * Math.PI;
          const current = [Math.sin(angle) * Math.cos(offset), Math.cos(angle), Math.sin(angle) * Math.sin(offset)];
          if (previous) line(previous, current, "#61e6a0", alpha, 0.7, t);
          previous = current;
        }
      }
    }

    function drawGrid(t) {
      for (let i = -5; i <= 5; i += 1) {
        const v = i * 0.42;
        line([-2.1, -1.05, v], [2.1, -1.05, v], "#2e4a5d", 0.55, 0.8, t);
        line([v, -1.05, -2.1], [v, -1.05, 2.1], "#1b2d3a", 0.65, 0.8, t);
      }
      line([-2.1, 0, 0], [2.1, 0, 0], "#6ee7f9", 0.42, 1, t);
      line([0, -1.45, 0], [0, 1.45, 0], "#6ee7f9", 0.42, 1, t);
      line([0, 0, -2.1], [0, 0, 2.1], "#6ee7f9", 0.42, 1, t);
    }

    function animate(timeMs) {
      const t = timeMs * 0.001;
      const width = host.clientWidth || 500;
      const height = host.clientHeight || 430;
      ctx.clearRect(0, 0, width, height);

      const gradient = ctx.createRadialGradient(width * 0.52, height * 0.38, 30, width * 0.52, height * 0.38, Math.min(width, height) * 0.54);
      gradient.addColorStop(0, "rgba(97, 230, 160, 0.18)");
      gradient.addColorStop(1, "rgba(0, 0, 0, 0)");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);

      const phase = mode * 0.75;
      const radius = mode === 2 ? 0.72 : mode === 4 ? 1.22 : 1.0 + Math.sin(t * 1.7) * 0.08;
      const theta = t * 0.8 + phase;
      const phi = Math.sin(t * 0.62 + phase) * 0.8 + 1.15;
      const direction = [
        Math.cos(theta) * Math.sin(phi),
        Math.cos(phi),
        Math.sin(theta) * Math.sin(phi)
      ];
      const outputPosition = direction.map((value) => value * radius);
      const latentPosition = [
        Math.sin(t * 1.3 + phase) * 1.35,
        Math.cos(t * 0.9 + phase) * 0.72,
        Math.cos(t * 1.1 + phase) * 1.2
      ];

      const xyzFocus = mode === 1;
      const actorFocus = mode === 3 || mode === 5;
      const envFocus = mode === 4;

      drawGrid(t);
      drawSphere(t, actorFocus, xyzFocus);

      rawNoisePositions.forEach((p, index) => {
        const pulse = xyzFocus ? 1 + Math.sin(t * 3.2 + index) * 0.28 : 0.72;
        point(p, 3.7 * pulse, "#ff6b6b", xyzFocus ? 0.82 : 0.18, t);
      });

      line([0, 0, 0], outputPosition, "#ffc857", envFocus || actorFocus ? 0.98 : 0.72, 2.2, t);
      line(latentPosition, outputPosition, "#c9a3ff", actorFocus ? 0.95 : 0.34, 1.8, t);
      point(latentPosition, actorFocus ? 8 : 6, "#55b7ff", 0.95, t);
      point(outputPosition, envFocus ? 9 : 7, "#ffc857", 0.98, t);

      if (!reduceMotion) {
        requestAnimationFrame(animate);
      }
    }

    window.addEventListener("resize", resize);
    resize();
    requestAnimationFrame(animate);
  }

  function startConeFallback(error) {
    console.warn("Three.js cone scene failed to load:", error);
    if (coneHost) coneHost.style.display = "none";
    if (coneFallback) coneFallback.style.display = "grid";
  }

  function initConeExplorer(THREE) {
    if (!coneHost) return null;

    const muSlider = document.getElementById("coneMu");
    const sigmaSlider = document.getElementById("coneSigma");
    const muValue = document.getElementById("coneMuValue");
    const sigmaValue = document.getElementById("coneSigmaValue");
    const thetaValue = document.getElementById("coneThetaValue");
    const ratioValue = document.getElementById("coneRatioValue");

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
    const cameraBase = new THREE.Vector3(3.55, 2.25, 4.35);
    const cameraTarget = new THREE.Vector3(0, 0.02, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    coneHost.appendChild(renderer.domElement);

    const group = new THREE.Group();
    scene.add(group);
    const content = new THREE.Group();
    content.position.set(-1.16, -0.02, 0);
    group.add(content);

    const grid = new THREE.GridHelper(4.8, 16, 0x2e4a5d, 0x1b2d3a);
    grid.position.y = -0.72;
    grid.position.x = 1.15;
    content.add(grid);

    const axisMaterial = new THREE.LineBasicMaterial({ color: 0x6ee7f9, transparent: true, opacity: 0.36 });
    [
      [new THREE.Vector3(-0.35, 0, 0), new THREE.Vector3(3.15, 0, 0)],
      [new THREE.Vector3(0, -1.25, 0), new THREE.Vector3(0, 1.25, 0)],
      [new THREE.Vector3(0, 0, -1.25), new THREE.Vector3(0, 0, 1.25)]
    ].forEach((points) => {
      content.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), axisMaterial));
    });

    const origin = new THREE.Mesh(
      new THREE.SphereGeometry(0.045, 16, 8),
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.4 })
    );
    content.add(origin);

    const meanMaterial = new THREE.LineBasicMaterial({ color: 0xffc857, transparent: true, opacity: 0.95 });
    const meanLine = new THREE.Line(new THREE.BufferGeometry(), meanMaterial);
    content.add(meanLine);

    const sphereGeometry = new THREE.SphereGeometry(1, 48, 24);
    const sphere = new THREE.Mesh(
      sphereGeometry,
      new THREE.MeshStandardMaterial({
        color: 0x55b7ff,
        transparent: true,
        opacity: 0.18,
        roughness: 0.32,
        metalness: 0.08,
        depthWrite: false
      })
    );
    content.add(sphere);

    const sphereWire = new THREE.Mesh(
      sphereGeometry,
      new THREE.MeshBasicMaterial({ color: 0x55b7ff, wireframe: true, transparent: true, opacity: 0.38 })
    );
    content.add(sphereWire);

    const coneMaterial = new THREE.MeshBasicMaterial({
      color: 0xff6b6b,
      transparent: true,
      opacity: 0.18,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    const ringMaterial = new THREE.LineBasicMaterial({ color: 0xff6b6b, transparent: true, opacity: 0.78 });
    let coneMesh = new THREE.Mesh(new THREE.BufferGeometry(), coneMaterial);
    let coneRing = new THREE.LineLoop(new THREE.BufferGeometry(), ringMaterial);
    const coneEdges = new THREE.Group();
    content.add(coneMesh, coneRing, coneEdges);

    const sampleGeometry = new THREE.SphereGeometry(0.025, 10, 8);
    const sampleMaterial = new THREE.MeshBasicMaterial({ color: 0x9fd7ff, transparent: true, opacity: 0.75 });
    const sampleGroup = new THREE.Group();
    const sampleOffsets = [
      [0.15, 0.45, 0.12], [-0.34, -0.22, 0.38], [0.48, -0.36, -0.24], [-0.18, 0.58, -0.18],
      [0.04, -0.08, 0.62], [0.35, 0.16, -0.48], [-0.55, 0.2, 0.08], [0.08, -0.58, 0.18],
      [0.42, 0.38, 0.32], [-0.28, -0.42, -0.38], [0.6, 0.05, 0.02], [-0.08, 0.18, -0.62]
    ];
    sampleOffsets.forEach(() => {
      sampleGroup.add(new THREE.Mesh(sampleGeometry, sampleMaterial));
    });
    content.add(sampleGroup);

    scene.add(new THREE.AmbientLight(0x9db8ff, 1.0));
    const light = new THREE.DirectionalLight(0xffffff, 1.8);
    light.position.set(3, 4, 4);
    scene.add(light);

    function coneGeometry(length, radius, segments = 96) {
      const vertices = [0, 0, 0];
      const indices = [];
      for (let i = 0; i <= segments; i += 1) {
        const a = (i / segments) * Math.PI * 2;
        vertices.push(length, Math.cos(a) * radius, Math.sin(a) * radius);
      }
      for (let i = 1; i <= segments; i += 1) {
        indices.push(0, i, i + 1);
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
      geometry.setIndex(indices);
      geometry.computeVertexNormals();
      return geometry;
    }

    function ringGeometry(length, radius, segments = 128) {
      const points = [];
      for (let i = 0; i < segments; i += 1) {
        const a = (i / segments) * Math.PI * 2;
        points.push(new THREE.Vector3(length, Math.cos(a) * radius, Math.sin(a) * radius));
      }
      return new THREE.BufferGeometry().setFromPoints(points);
    }

    function edgeLine(length, y, z) {
      return new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(length, y, z)]),
        ringMaterial
      );
    }

    function updateCamera() {
      camera.position.copy(cameraBase).multiplyScalar(1 / coneZoom);
      camera.lookAt(cameraTarget);
    }

    function resize() {
      const width = coneHost.clientWidth || 520;
      const height = coneHost.clientHeight || 520;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      updateCamera();
    }

    function update() {
      const dark = isDarkTheme();
      const l = Number(muSlider.value);
      const sigma = Number(sigmaSlider.value);
      const ratio = sigma / Math.max(l, 0.001);
      const theta = Math.asin(Math.min(ratio, 0.99));
      const thetaDeg = theta * 180 / Math.PI;
      const coneLength = Math.min(3.0, Math.max(l + sigma * 1.6, 1.0));
      const coneRadius = Math.tan(theta) * coneLength;

      muValue.value = l.toFixed(2);
      sigmaValue.value = sigma.toFixed(2);
      thetaValue.textContent = `${thetaDeg.toFixed(1)}°`;
      ratioValue.textContent = ratio.toFixed(2);

      sphere.position.set(l, 0, 0);
      sphere.scale.setScalar(sigma);
      sphereWire.position.copy(sphere.position);
      sphereWire.scale.copy(sphere.scale);

      meanLine.geometry.dispose();
      meanLine.geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(l, 0, 0)
      ]);

      coneMesh.geometry.dispose();
      coneMesh.geometry = coneGeometry(coneLength, coneRadius);
      coneRing.geometry.dispose();
      coneRing.geometry = ringGeometry(coneLength, coneRadius);

      while (coneEdges.children.length) {
        const child = coneEdges.children.pop();
        child.geometry.dispose();
      }
      coneEdges.add(
        edgeLine(coneLength, coneRadius, 0),
        edgeLine(coneLength, -coneRadius, 0),
        edgeLine(coneLength, 0, coneRadius),
        edgeLine(coneLength, 0, -coneRadius)
      );

      sampleGroup.children.forEach((dot, index) => {
        const offset = sampleOffsets[index];
        dot.position.set(
          l + offset[0] * sigma,
          offset[1] * sigma,
          offset[2] * sigma
        );
      });

      sphere.material.color.set(dark ? 0x55b7ff : 0x0284c7);
      sphere.material.opacity = dark ? 0.18 : 0.24;
      sphereWire.material.color.set(dark ? 0x55b7ff : 0x0369a1);
      sphereWire.material.opacity = dark ? 0.38 : 0.52;
      axisMaterial.color.set(dark ? 0x6ee7f9 : 0x0891b2);
      axisMaterial.opacity = dark ? 0.36 : 0.54;
      coneMaterial.opacity = Math.min(dark ? 0.34 : 0.4, (dark ? 0.12 : 0.16) + ratio * 0.18);
      ringMaterial.opacity = Math.min(0.95, 0.55 + ratio * 0.28);
      renderer.render(scene, camera);
    }

    muSlider.addEventListener("input", update);
    sigmaSlider.addEventListener("input", update);
    window.addEventListener("resize", () => {
      resize();
      update();
    });

    function animate(timeMs) {
      const t = timeMs * 0.001;
      updateCamera();
      if (!reduceMotion) {
        const idle = coneAutoRotate ? Math.sin(t * 0.18) * 0.16 : 0;
        group.rotation.y = coneRotation.y + idle;
        group.rotation.x = coneRotation.x + (coneAutoRotate ? Math.sin(t * 0.14) * 0.04 : 0);
        sampleGroup.children.forEach((dot, index) => {
          const pulse = 0.9 + Math.sin(t * 2.6 + index * 0.7) * 0.16;
          dot.scale.setScalar(pulse);
        });
      } else {
        group.rotation.y = coneRotation.y;
        group.rotation.x = coneRotation.x;
      }
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }

    resize();
    update();
    requestAnimationFrame(animate);
    return { update, resize };
  }

  try {
    const THREE = await loadThree();
    try {
      initConeExplorer(THREE);
    } catch (coneError) {
      startConeFallback(coneError);
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    const cameraBase = new THREE.Vector3(3.35, 2.55, 5.25);
    const cameraTarget = new THREE.Vector3(0, 0.06, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    host.appendChild(renderer.domElement);

    const group = new THREE.Group();
    group.position.set(0, -0.08, 0);
    scene.add(group);

    const grid = new THREE.GridHelper(4.2, 16, 0x2e4a5d, 0x1b2d3a);
    grid.position.y = -1.05;
    group.add(grid);

    const axisMaterial = new THREE.LineBasicMaterial({ color: 0x6ee7f9, transparent: true, opacity: 0.42 });
    const axisPoints = [
      [new THREE.Vector3(-2.1, 0, 0), new THREE.Vector3(2.1, 0, 0)],
      [new THREE.Vector3(0, -1.45, 0), new THREE.Vector3(0, 1.45, 0)],
      [new THREE.Vector3(0, 0, -2.1), new THREE.Vector3(0, 0, 2.1)]
    ];
    axisPoints.forEach((points) => {
      group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), axisMaterial));
    });

    const unitSphereMaterial = new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.24 });
    const unitSphere = unitSphereGrid(1.0, 12, 7, 96);
    group.add(unitSphere);

    const rawMaterial = new THREE.LineBasicMaterial({ color: 0xffc857, transparent: true, opacity: 0.95 });
    const rawLine = new THREE.Line(new THREE.BufferGeometry(), rawMaterial);
    group.add(rawLine);

    const directionMaterial = new THREE.LineBasicMaterial({ color: 0x61e6a0, transparent: true, opacity: 0.95 });
    const directionLine = new THREE.Line(new THREE.BufferGeometry(), directionMaterial);
    group.add(directionLine);

    const speedMaterial = new THREE.LineBasicMaterial({ color: 0x55b7ff, transparent: true, opacity: 0.9 });
    const speedLine = new THREE.Line(new THREE.BufferGeometry(), speedMaterial);
    group.add(speedLine);

    const composeMaterial = new THREE.LineBasicMaterial({ color: 0xc9a3ff, transparent: true, opacity: 0.75 });
    const composeLine = new THREE.Line(new THREE.BufferGeometry(), composeMaterial);
    group.add(composeLine);

    const directionConeGroup = new THREE.Group();
    const directionConeMaterial = new THREE.MeshBasicMaterial({
      color: 0xff6b6b,
      transparent: true,
      opacity: 0.18,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    const directionConeRingMaterial = new THREE.LineBasicMaterial({ color: 0xff6b6b, transparent: true, opacity: 0.82 });
    let directionConeMesh = new THREE.Mesh(new THREE.BufferGeometry(), directionConeMaterial);
    let directionConeRing = new THREE.LineLoop(new THREE.BufferGeometry(), directionConeRingMaterial);
    const directionConeEdges = new THREE.Group();
    directionConeGroup.add(directionConeMesh, directionConeRing, directionConeEdges);
    group.add(directionConeGroup);

    const speedExploreGroup = new THREE.Group();
    const speedBandMaterial = new THREE.MeshBasicMaterial({
      color: 0x55b7ff,
      transparent: true,
      opacity: 0.2,
      side: THREE.DoubleSide,
      depthWrite: false
    });
    const speedBand = new THREE.Mesh(new THREE.BufferGeometry(), speedBandMaterial);
    const speedEdgeMaterial = new THREE.LineBasicMaterial({ color: 0x0b4fd8, transparent: true, opacity: 1, linewidth: 2 });
    const speedGuideMaterial = new THREE.LineBasicMaterial({
      color: 0x083fba,
      transparent: true,
      opacity: 0.82,
      depthWrite: false
    });
    const speedInnerEdge = new THREE.LineLoop(new THREE.BufferGeometry(), speedEdgeMaterial);
    const speedOuterEdge = new THREE.LineLoop(new THREE.BufferGeometry(), speedEdgeMaterial);
    const speedRadialEdges = new THREE.Group();
    const speedGuideLines = new THREE.Group();
    speedExploreGroup.add(speedBand, speedGuideLines, speedInnerEdge, speedOuterEdge, speedRadialEdges);
    group.add(speedExploreGroup);

    const light = new THREE.DirectionalLight(0xffffff, 2.1);
    light.position.set(3, 4, 5);
    scene.add(light);
    scene.add(new THREE.AmbientLight(0x9db8ff, 0.95));

    function updateCamera() {
      camera.position.copy(cameraBase).multiplyScalar(1 / viewZoom);
      camera.lookAt(cameraTarget);
    }

    function resize() {
      const width = host.clientWidth || 500;
      const height = host.clientHeight || 430;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      updateCamera();
    }

    window.addEventListener("resize", resize);
    resize();

    function unitSphereGrid(radius = 1, meridians = 8, parallels = 5, segments = 96) {
      const sphere = new THREE.Group();

      for (let i = 0; i < meridians; i += 1) {
        const longitude = (i / meridians) * Math.PI * 2;
        const points = [];
        for (let j = 0; j < segments; j += 1) {
          const latitude = -Math.PI * 0.5 + (j / segments) * Math.PI;
          const cosLat = Math.cos(latitude);
          points.push(new THREE.Vector3(
            Math.cos(longitude) * cosLat * radius,
            Math.sin(latitude) * radius,
            Math.sin(longitude) * cosLat * radius
          ));
        }
        sphere.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points), unitSphereMaterial));
      }

      for (let i = 1; i <= parallels; i += 1) {
        const latitude = -Math.PI * 0.5 + (i / (parallels + 1)) * Math.PI;
        const ringRadius = Math.cos(latitude) * radius;
        const y = Math.sin(latitude) * radius;
        const points = [];
        for (let j = 0; j < segments; j += 1) {
          const angle = (j / segments) * Math.PI * 2;
          points.push(new THREE.Vector3(
            Math.cos(angle) * ringRadius,
            y,
            Math.sin(angle) * ringRadius
          ));
        }
        sphere.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points), unitSphereMaterial));
      }

      return sphere;
    }

    function directionConeGeometry(length, radius, segments = 80) {
      const vertices = [0, 0, 0];
      const indices = [];
      for (let i = 0; i <= segments; i += 1) {
        const a = (i / segments) * Math.PI * 2;
        vertices.push(length, Math.cos(a) * radius, Math.sin(a) * radius);
      }
      for (let i = 1; i <= segments; i += 1) {
        indices.push(0, i, i + 1);
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
      geometry.setIndex(indices);
      geometry.computeVertexNormals();
      return geometry;
    }

    function directionConeRingGeometry(length, radius, segments = 96) {
      const points = [];
      for (let i = 0; i < segments; i += 1) {
        const a = (i / segments) * Math.PI * 2;
        points.push(new THREE.Vector3(length, Math.cos(a) * radius, Math.sin(a) * radius));
      }
      return new THREE.BufferGeometry().setFromPoints(points);
    }

    function directionConeEdge(length, y, z) {
      return new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(length, y, z)]),
        directionConeRingMaterial
      );
    }

    function speedFrustumShellGeometry(innerRadius, outerRadius, coneAngle, radialSegments = 8, angularSegments = 72) {
      const vertices = [];
      const indices = [];

      function addQuadGrid(base, rows, columns) {
        const row = columns + 1;
        for (let i = 0; i < rows; i += 1) {
          for (let j = 0; j < columns; j += 1) {
            const p = base + i * row + j;
            indices.push(p, p + row, p + 1, p + 1, p + row, p + row + 1);
          }
        }
      }

      function pushSpherePatch(radius, angleStart, angleEnd) {
        const base = vertices.length / 3;
        for (let i = 0; i <= radialSegments; i += 1) {
          const theta = angleStart + (angleEnd - angleStart) * (i / radialSegments);
          for (let j = 0; j <= angularSegments; j += 1) {
            const a = (j / angularSegments) * Math.PI * 2;
            vertices.push(
              Math.cos(theta) * radius,
              Math.sin(theta) * Math.cos(a) * radius,
              Math.sin(theta) * Math.sin(a) * radius
            );
          }
        }
        addQuadGrid(base, radialSegments, angularSegments);
      }

      function pushConeSide(radiusStart, radiusEnd, theta) {
        const base = vertices.length / 3;
        for (let i = 0; i <= radialSegments; i += 1) {
          const radius = radiusStart + (radiusEnd - radiusStart) * (i / radialSegments);
          for (let j = 0; j <= angularSegments; j += 1) {
            const a = (j / angularSegments) * Math.PI * 2;
            vertices.push(
              Math.cos(theta) * radius,
              Math.sin(theta) * Math.cos(a) * radius,
              Math.sin(theta) * Math.sin(a) * radius
            );
          }
        }
        addQuadGrid(base, radialSegments, angularSegments);
      }

      // Spherical caps at the bounded speed range, plus the conical side surface.
      pushSpherePatch(outerRadius, 0, coneAngle);
      pushSpherePatch(innerRadius, coneAngle, 0);
      pushConeSide(innerRadius, outerRadius, coneAngle);

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
      geometry.setIndex(indices);
      geometry.computeVertexNormals();
      return geometry;
    }

    function speedShellEdgeGeometry(radius, coneAngle, segments = 96) {
      const points = [];
      for (let i = 0; i < segments; i += 1) {
        const a = (i / segments) * Math.PI * 2;
        points.push(new THREE.Vector3(
          Math.cos(coneAngle) * radius,
          Math.sin(coneAngle) * Math.cos(a) * radius,
          Math.sin(coneAngle) * Math.sin(a) * radius
        ));
      }
      return new THREE.BufferGeometry().setFromPoints(points);
    }

    function speedShellMeridianGeometry(radius, coneAngle, azimuth, segments = 28) {
      const points = [];
      for (let i = 0; i <= segments; i += 1) {
        const theta = (i / segments) * coneAngle;
        points.push(new THREE.Vector3(
          Math.cos(theta) * radius,
          Math.sin(theta) * Math.cos(azimuth) * radius,
          Math.sin(theta) * Math.sin(azimuth) * radius
        ));
      }
      return new THREE.BufferGeometry().setFromPoints(points);
    }

    function speedShellRadialEdge(innerRadius, outerRadius, coneAngle, azimuth) {
      return new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(
            Math.cos(coneAngle) * innerRadius,
            Math.sin(coneAngle) * Math.cos(azimuth) * innerRadius,
            Math.sin(coneAngle) * Math.sin(azimuth) * innerRadius
          ),
          new THREE.Vector3(
            Math.cos(coneAngle) * outerRadius,
            Math.sin(coneAngle) * Math.cos(azimuth) * outerRadius,
            Math.sin(coneAngle) * Math.sin(azimuth) * outerRadius
          )
        ]),
        speedEdgeMaterial
      );
    }

    function sigmoid(value) {
      return 1 / (1 + Math.exp(-value));
    }

    function logit(value) {
      const clamped = clamp(value, 0.001, 0.999);
      return Math.log(clamped / (1 - clamped));
    }

    function dsDirection() {
      const yaw = Number(controls.yaw.value) * Math.PI / 180;
      const pitch = Number(controls.pitch.value) * Math.PI / 180;
      return new THREE.Vector3(
        Math.cos(pitch) * Math.cos(yaw),
        Math.sin(pitch),
        Math.cos(pitch) * Math.sin(yaw)
      ).normalize();
    }

    function updateReadouts() {
      const direction = dsDirection();
      const speed = Number(controls.speed.value);
      const v = direction.clone().multiplyScalar(speed);
      controls.yawValue.value = `${Number(controls.yaw.value).toFixed(0)}°`;
      controls.pitchValue.value = `${Number(controls.pitch.value).toFixed(0)}°`;
      controls.speedValue.value = speed.toFixed(2);
      controls.dirSigmaValue.value = Number(controls.dirSigma.value).toFixed(2);
      controls.speedSigmaValue.value = Number(controls.speedSigma.value).toFixed(2);
      controls.uValue.textContent = `(${direction.x.toFixed(2)}, ${direction.y.toFixed(2)}, ${direction.z.toFixed(2)})`;
      controls.vValue.textContent = `(${v.x.toFixed(2)}, ${v.y.toFixed(2)}, ${v.z.toFixed(2)})`;
    }

    Object.values(controls).forEach((element) => {
      if (element && element.tagName === "INPUT") {
        element.addEventListener("input", updateReadouts);
      }
    });

    function animate(timeMs) {
      try {
      const t = timeMs * 0.001;
      updateCamera();
      const baseDirection = dsDirection();
      const speed = Number(controls.speed.value);
      const dirSigma = Number(controls.dirSigma.value);
      const speedSigma = Number(controls.speedSigma.value);
      const unitPosition = baseDirection.clone();
      const outputPosition = baseDirection.clone().multiplyScalar(speed);

      rawLine.geometry.setFromPoints([new THREE.Vector3(0, 0, 0), outputPosition]);
      directionLine.geometry.setFromPoints([new THREE.Vector3(0, 0, 0), unitPosition]);
      speedLine.geometry.setFromPoints([unitPosition, outputPosition]);
      composeLine.geometry.setFromPoints([unitPosition, outputPosition]);

      const coneAngle = Math.min(0.72, Math.max(0.05, dirSigma * 1.55));
      const coneLength = Math.cos(coneAngle);
      const coneRadius = Math.sin(coneAngle);
      directionConeMesh.geometry.dispose();
      directionConeMesh.geometry = directionConeGeometry(coneLength, coneRadius);
      directionConeRing.geometry.dispose();
      directionConeRing.geometry = directionConeRingGeometry(coneLength, coneRadius);
      while (directionConeEdges.children.length) {
        const child = directionConeEdges.children.pop();
        child.geometry.dispose();
      }
      directionConeEdges.add(
        directionConeEdge(coneLength, coneRadius, 0),
        directionConeEdge(coneLength, -coneRadius, 0),
        directionConeEdge(coneLength, 0, coneRadius),
        directionConeEdge(coneLength, 0, -coneRadius)
      );
      directionConeGroup.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), baseDirection);
      directionConeMaterial.opacity = Math.min(0.32, 0.12 + dirSigma * 0.45);
      directionConeRingMaterial.opacity = Math.min(0.95, 0.58 + dirSigma * 0.58);

      const speedRaw = logit(speed);
      const rawSpan = speedSigma * 3.2;
      const speedMin = sigmoid(speedRaw - rawSpan);
      const speedMax = sigmoid(speedRaw + rawSpan);
      speedBand.geometry.dispose();
      speedBand.geometry = speedFrustumShellGeometry(speedMin, speedMax, coneAngle, 8, 72);
      speedInnerEdge.geometry.dispose();
      speedInnerEdge.geometry = speedShellEdgeGeometry(speedMin, coneAngle);
      speedOuterEdge.geometry.dispose();
      speedOuterEdge.geometry = speedShellEdgeGeometry(speedMax, coneAngle);
      while (speedGuideLines.children.length) {
        const child = speedGuideLines.children.pop();
        child.geometry.dispose();
      }
      const speedMid = (speedMin + speedMax) * 0.5;
      speedGuideLines.add(
        new THREE.LineLoop(speedShellEdgeGeometry(speedMin, coneAngle * 0.35), speedGuideMaterial),
        new THREE.LineLoop(speedShellEdgeGeometry(speedMin, coneAngle * 0.7), speedGuideMaterial),
        new THREE.LineLoop(speedShellEdgeGeometry(speedMid, coneAngle * 0.35), speedGuideMaterial),
        new THREE.LineLoop(speedShellEdgeGeometry(speedMid, coneAngle * 0.7), speedGuideMaterial),
        new THREE.LineLoop(speedShellEdgeGeometry(speedMid, coneAngle), speedGuideMaterial),
        new THREE.LineLoop(speedShellEdgeGeometry(speedMax, coneAngle * 0.35), speedGuideMaterial),
        new THREE.LineLoop(speedShellEdgeGeometry(speedMax, coneAngle * 0.7), speedGuideMaterial),
        new THREE.Line(speedShellMeridianGeometry(speedMin, coneAngle, 0), speedGuideMaterial),
        new THREE.Line(speedShellMeridianGeometry(speedMin, coneAngle, Math.PI * 0.5), speedGuideMaterial),
        new THREE.Line(speedShellMeridianGeometry(speedMin, coneAngle, Math.PI), speedGuideMaterial),
        new THREE.Line(speedShellMeridianGeometry(speedMin, coneAngle, Math.PI * 1.5), speedGuideMaterial),
        new THREE.Line(speedShellMeridianGeometry(speedMid, coneAngle, Math.PI * 0.25), speedGuideMaterial),
        new THREE.Line(speedShellMeridianGeometry(speedMid, coneAngle, Math.PI * 0.75), speedGuideMaterial),
        new THREE.Line(speedShellMeridianGeometry(speedMid, coneAngle, Math.PI * 1.25), speedGuideMaterial),
        new THREE.Line(speedShellMeridianGeometry(speedMid, coneAngle, Math.PI * 1.75), speedGuideMaterial),
        new THREE.Line(speedShellMeridianGeometry(speedMax, coneAngle, 0), speedGuideMaterial),
        new THREE.Line(speedShellMeridianGeometry(speedMax, coneAngle, Math.PI * 0.5), speedGuideMaterial),
        new THREE.Line(speedShellMeridianGeometry(speedMax, coneAngle, Math.PI), speedGuideMaterial),
        new THREE.Line(speedShellMeridianGeometry(speedMax, coneAngle, Math.PI * 1.5), speedGuideMaterial)
      );
      while (speedRadialEdges.children.length) {
        const child = speedRadialEdges.children.pop();
        child.geometry.dispose();
      }
      speedRadialEdges.add(
        speedShellRadialEdge(speedMin, speedMax, coneAngle, 0),
        speedShellRadialEdge(speedMin, speedMax, coneAngle, Math.PI * 0.5),
        speedShellRadialEdge(speedMin, speedMax, coneAngle, Math.PI),
        speedShellRadialEdge(speedMin, speedMax, coneAngle, Math.PI * 1.5)
      );
      speedBand.position.set(0, 0, 0);
      speedBand.scale.set(1, 1, 1);
      speedBand.rotation.set(0, 0, 0);
      speedExploreGroup.position.set(0, 0, 0);
      speedExploreGroup.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), baseDirection);
      speedBandMaterial.opacity = Math.min(0.32, 0.12 + speedSigma * 0.42);
      speedEdgeMaterial.opacity = Math.min(1, 0.7 + speedSigma * 0.55);
      speedGuideMaterial.opacity = Math.min(0.92, 0.68 + speedSigma * 0.38);

      const dark = isDarkTheme();
      unitSphere.visible = true;
      unitSphereMaterial.color.set(dark ? 0x7dffb8 : 0x059669);
      unitSphereMaterial.opacity = dark ? 0.22 : 0.36;
      axisMaterial.color.set(dark ? 0x6ee7f9 : 0x0891b2);
      axisMaterial.opacity = dark ? 0.42 : 0.58;
      speedEdgeMaterial.color.set(dark ? 0x0b4fd8 : 0x075985);
      speedGuideMaterial.color.set(dark ? 0x083fba : 0x0369a1);
      rawMaterial.opacity = 0.98;
      directionMaterial.opacity = 0.94;
      speedMaterial.opacity = 0.82;
      composeMaterial.opacity = 0.5;

      directionConeGroup.visible = true;
      speedExploreGroup.visible = true;

      if (!reduceMotion) {
        group.rotation.y = viewRotation.y + (viewAutoRotate ? Math.sin(t * 0.18) * 0.08 : 0);
        group.rotation.x = viewRotation.x + (viewAutoRotate ? Math.sin(t * 0.13) * 0.035 : 0);
      } else {
        group.rotation.y = viewRotation.y;
        group.rotation.x = viewRotation.x;
      }

      renderer.render(scene, camera);
      window.__dsSceneStatus = "rendered";
      } catch (error) {
        window.__dsSceneStatus = "error";
        window.__dsSceneError = error && (error.stack || error.message || String(error));
      }
      requestAnimationFrame(animate);
    }

    updateReadouts();
    requestAnimationFrame(animate);
  } catch (error) {
    window.__dsSceneStatus = "init-error";
    window.__dsSceneError = error && (error.stack || error.message || String(error));
    startCanvasFallback(error);
    startConeFallback(error);
  }
}
