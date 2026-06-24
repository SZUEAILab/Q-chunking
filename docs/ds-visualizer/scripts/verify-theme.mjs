import { spawn } from "node:child_process";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { setTimeout as sleep } from "node:timers/promises";

const PORT = Number(process.env.DS_VISUALIZER_VERIFY_PORT || 8786);
const BASE_URL = `http://127.0.0.1:${PORT}/`;

const TEXT_CHECKS = [
  { selector: "body", label: "body（页面正文）", role: "strong" },
  { selector: "h1", label: "hero title（首屏标题）", role: "strong" },
  { selector: ".shadcn-nav a.is-active", label: "active nav（当前导航）", role: "strong" },
  { selector: ".reason-item strong", label: "reason card title（原因卡片标题）", role: "strong" },
  { selector: ".optimize-item strong", label: "optimize card title（优化卡片标题）", role: "strong" },
  { selector: ".action-token strong", label: "action token title（动作变量标题）", role: "strong" },
  { selector: ".action-path b", label: "action path label（动作路径标签）", role: "strong" },
  { selector: ".bijector-lab-header h3", label: "bijector lab title（可逆变换实验台标题）", role: "strong" },
  { selector: ".bijector-plot-label strong", label: "plot label（图表标签）", role: "strong" },
  { selector: ".bijector-control-group > strong", label: "control group title（控制组标题）", role: "strong" },
  { selector: ".formula-box h4", label: "formula title（公式标题）", role: "strong" },
  { selector: ".code-box pre", label: "code block（代码块）", role: "soft" },
  { selector: ".formula-list li", label: "formula list（公式列表）", role: "soft" },
  { selector: ".dataset-strip strong", label: "dataset label（数据集标签）", role: "strong" },
  { selector: ".pipeline-module strong", label: "pipeline module title（流水线模块标题）", role: "strong" },
  { selector: ".policy-consumer", label: "policy consumer（策略消费者）", role: "soft" },
  { selector: ".cell", label: "summary matrix cell（总结矩阵单元格）", role: "soft" },
  { selector: "code", label: "inline code（行内代码）", role: "soft" },
];

const THEME_RULES = {
  light: {
    strong: { max: 0.35 },
    soft: { max: 0.52 },
  },
  dark: {
    strong: { min: 0.72 },
    soft: { min: 0.48 },
  },
};

function findChrome() {
  const candidates = [
    process.env.CHROME_BIN,
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
  ].filter(Boolean);

  return new Promise((resolve) => {
    const tryCandidate = (index) => {
      if (index >= candidates.length) {
        resolve(null);
        return;
      }
      const child = spawn("bash", ["-lc", `command -v ${candidates[index]}`], { stdio: ["ignore", "pipe", "ignore"] });
      let output = "";
      child.stdout.on("data", (chunk) => {
        output += chunk.toString();
      });
      child.on("close", (code) => {
        if (code === 0 && output.trim()) resolve(output.trim().split("\n")[0]);
        else tryCandidate(index + 1);
      });
    };
    tryCandidate(0);
  });
}

function spawnServer() {
  const child = spawn(
    "npm",
    ["run", "dev", "--", "--port", String(PORT), "--strictPort"],
    {
      cwd: new URL("..", import.meta.url),
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
      detached: process.platform !== "win32",
    },
  );

  child.stdout.on("data", (chunk) => process.stdout.write(`[vite] ${chunk}`));
  child.stderr.on("data", (chunk) => process.stderr.write(`[vite] ${chunk}`));
  return child;
}

function terminateProcessTree(child) {
  if (!child?.pid || child.exitCode !== null) return;
  try {
    if (process.platform === "win32") child.kill("SIGTERM");
    else process.kill(-child.pid, "SIGTERM");
  } catch {
    child.kill("SIGTERM");
  }
}

async function waitForServer() {
  const started = Date.now();
  while (Date.now() - started < 15_000) {
    try {
      const response = await fetch(BASE_URL);
      if (response.ok) return;
    } catch {
      // Retry until Vite is ready.
    }
    await sleep(250);
  }
  throw new Error(`Vite（前端开发服务器） did not become ready at ${BASE_URL}`);
}

async function connectToPage(debugPort) {
  const tabs = await fetch(`http://127.0.0.1:${debugPort}/json/list`).then((response) => response.json());
  const page = tabs.find((tab) => tab.type === "page") || tabs[0];
  if (!page?.webSocketDebuggerUrl) throw new Error("No Chrome page target（浏览器页面目标） found");

  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let id = 0;

  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const messageId = ++id;
    const timeout = setTimeout(() => reject(new Error(`CDP（浏览器调试协议） timeout: ${method}`)), 8_000);
    const onMessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.id !== messageId) return;
      clearTimeout(timeout);
      ws.removeEventListener("message", onMessage);
      if (message.error) reject(new Error(JSON.stringify(message.error)));
      else resolve(message.result);
    };
    ws.addEventListener("message", onMessage);
    ws.send(JSON.stringify({ id: messageId, method, params }));
  });

  await new Promise((resolve) => ws.addEventListener("open", resolve, { once: true }));
  return { ws, send };
}

async function waitForChrome(debugPort) {
  const started = Date.now();
  while (Date.now() - started < 10_000) {
    try {
      const response = await fetch(`http://127.0.0.1:${debugPort}/json/list`);
      if (response.ok) return;
    } catch {
      // Retry until Chrome exposes its debugging endpoint.
    }
    await sleep(200);
  }
  throw new Error(`Chrome CDP（浏览器调试接口） did not become ready on port ${debugPort}`);
}

function parseColor(value) {
  const text = String(value || "").trim();
  const rgbMatch = text.match(/rgba?\(([^)]+)\)/);
  if (rgbMatch) {
    const [r, g, b] = rgbMatch[1]
      .split(/[, ]+/)
      .filter(Boolean)
      .slice(0, 3)
      .map(Number);
    return [r / 255, g / 255, b / 255];
  }

  const srgbMatch = text.match(/color\(srgb\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)/);
  if (srgbMatch) {
    return srgbMatch.slice(1, 4).map(Number);
  }

  throw new Error(`Unsupported color（不支持的颜色格式）: ${value}`);
}

function channelToLinear(value) {
  return value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
}

function luminance(color) {
  const [r, g, b] = parseColor(color).map(channelToLinear);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function formatLuminance(value) {
  return value.toFixed(3);
}

async function verifyTheme(send, theme) {
  await send("Runtime.evaluate", {
    expression: `
      localStorage.setItem("ds-visualizer-theme", "${theme}");
      document.documentElement.classList.toggle("dark", "${theme}" === "dark");
      window.scrollTo(0, 0);
    `,
  });
  await sleep(500);

  const result = await send("Runtime.evaluate", {
    expression: `(() => {
      const checks = ${JSON.stringify(TEXT_CHECKS)};
      return JSON.stringify({
        theme: document.documentElement.classList.contains("dark") ? "dark" : "light",
        overlay: !!document.querySelector(".vite-error-overlay,#webpack-dev-server-client-overlay,[data-nextjs-dialog]"),
        textLength: document.body.innerText.trim().length,
        checks: checks.map((check) => {
          const element = document.querySelector(check.selector);
          if (!element) return { ...check, missing: true };
          const style = getComputedStyle(element);
          return {
            ...check,
            color: style.color,
            text: element.textContent.trim().slice(0, 80),
          };
        }),
      });
    })()`,
    returnByValue: true,
  });

  const payload = JSON.parse(result.result.value);
  if (payload.theme !== theme) throw new Error(`Expected theme（主题） ${theme}, got ${payload.theme}`);
  if (payload.overlay) throw new Error("Framework error overlay（框架错误覆盖层） is visible");
  if (payload.textLength < 1000) throw new Error("Page content（页面内容） looks too short");

  const failures = [];
  for (const check of payload.checks) {
    if (check.missing) {
      failures.push(`${check.label}: missing selector ${check.selector}`);
      continue;
    }
    const lum = luminance(check.color);
    const rule = THEME_RULES[theme][check.role];
    if (typeof rule.max === "number" && lum > rule.max) {
      failures.push(`${check.label}: luminance ${formatLuminance(lum)} > ${rule.max} (${check.color})`);
    }
    if (typeof rule.min === "number" && lum < rule.min) {
      failures.push(`${check.label}: luminance ${formatLuminance(lum)} < ${rule.min} (${check.color})`);
    }
  }

  if (failures.length) {
    throw new Error(`${theme} theme（${theme === "light" ? "白天" : "黑夜"}主题） failed:\n- ${failures.join("\n- ")}`);
  }

  console.log(`✓ ${theme} theme（${theme === "light" ? "白天" : "黑夜"}主题） text luminance（文字亮度） passed`);
}

async function capturePipeline(send, outputDirectory, mode, width, height) {
  await send("Emulation.setDeviceMetricsOverride", {
    width,
    height,
    deviceScaleFactor: 1,
    mobile: width < 600,
  });
  await sleep(150);
  await send("Runtime.evaluate", {
    expression: `(() => {
      const button = document.querySelector('[data-pipeline-mode="${mode}"]');
      button?.click();
      document.documentElement.style.scrollBehavior = "auto";
      const pipeline = document.getElementById("pipeline");
      if (pipeline) window.scrollTo(0, window.scrollY + pipeline.getBoundingClientRect().top - 18);
    })()`,
  });
  await sleep(450);
  const screenshot = await send("Page.captureScreenshot", {
    format: "png",
    fromSurface: true,
  });
  const path = `${outputDirectory}/pipeline-${mode}-${width}.png`;
  await writeFile(path, Buffer.from(screenshot.data, "base64"));
  console.log(`✓ Pipeline screenshot（流水线截图） written to ${path}`);

  if (width < 600) {
    await send("Runtime.evaluate", {
      expression: `(() => {
        const target = document.querySelector(".pipeline-policy-zone");
        if (target) window.scrollTo(0, window.scrollY + target.getBoundingClientRect().top - 12);
      })()`,
    });
    await sleep(250);
    const detailScreenshot = await send("Page.captureScreenshot", {
      format: "png",
      fromSurface: true,
    });
    const detailPath = `${outputDirectory}/pipeline-${mode}-${width}-detail.png`;
    await writeFile(detailPath, Buffer.from(detailScreenshot.data, "base64"));
    console.log(`✓ Pipeline detail screenshot（流水线细节截图） written to ${detailPath}`);
  }
}

async function main() {
  const chrome = await findChrome();
  if (!chrome) {
    throw new Error("Chrome/Chromium（浏览器） not found. Set CHROME_BIN to run theme verification.");
  }

  const debugPort = Number(process.env.DS_VISUALIZER_CHROME_PORT || 9237);
  const profilePath = `/tmp/ds-visualizer-theme-verify-${process.pid}`;
  const server = spawnServer();
  let browser;

  try {
    await waitForServer();
    browser = spawn(chrome, [
      "--headless=new",
      `--remote-debugging-port=${debugPort}`,
      "--disable-gpu",
      "--no-sandbox",
      `--user-data-dir=${profilePath}`,
      BASE_URL,
    ], {
      stdio: ["ignore", "ignore", "pipe"],
      detached: process.platform !== "win32",
    });

    browser.stderr.on("data", (chunk) => {
      const line = chunk.toString();
      if (!line.includes("DevTools listening")) return;
      process.stderr.write(`[chrome] ${line}`);
    });

    await waitForChrome(debugPort);
    const { ws, send } = await connectToPage(debugPort);
    await send("Page.enable");
    await send("Runtime.enable");
    await send("Emulation.setDeviceMetricsOverride", {
      width: 1440,
      height: 1200,
      deviceScaleFactor: 1,
      mobile: false,
    });
    await send("Page.navigate", { url: BASE_URL });
    await sleep(900);

    await verifyTheme(send, "light");
    await verifyTheme(send, "dark");

    const screenshotDirectory = process.env.DS_VISUALIZER_SCREENSHOT_DIR;
    if (screenshotDirectory) {
      await mkdir(screenshotDirectory, { recursive: true });
      await verifyTheme(send, "light");
      await capturePipeline(send, screenshotDirectory, "integrated", 1440, 1200);
      await capturePipeline(send, screenshotDirectory, "adapter", 1440, 1200);
      await capturePipeline(send, screenshotDirectory, "adapter", 430, 932);
    }

    ws.close();
    console.log("✓ Theme verification（主题验证） completed");
  } finally {
    terminateProcessTree(browser);
    terminateProcessTree(server);
    await sleep(250);
    await rm(profilePath, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(`\nTheme verification failed（主题验证失败）:\n${error.stack || error.message}`);
  process.exit(1);
});
