# Direction-Speed Visualizer（方向速度可视化）

This is the independent frontend（独立前端） for the Direction-Speed（方向速度） architecture explanation.

## Commands（命令）

```bash
npm install
npm run dev
npm run build
npm run preview
npm run verify:theme
```

`verify:theme`（主题验证）会启动临时 Vite / Chrome（开发服务器 / 浏览器）进程，检查白天和黑夜主题的文字亮度，并在结束时回收整个进程树。

需要输出 Pipeline（流水线）的桌面端与移动端截图时：

```bash
DS_VISUALIZER_SCREENSHOT_DIR=/tmp/ds-pipeline-shots npm run verify:theme
```

## Git Management（版本管理）

- Track（跟踪）source files（源码文件）: `index.html`, `src/`, `package.json`, Tailwind（原子化样式）/ Vite（前端构建工具）/ shadcn（组件风格） config.
- Ignore（忽略）generated files（生成文件）: `node_modules/`, `dist/`, `.vite/`, logs.
- The legacy page（旧页面） at `../ds_arch_comparison.html` is a small migration note.  It no longer contains the full visualization source, so this directory is the single frontend source of truth（唯一前端源码）。

## Structure（结构）

- `index.html`: static page shell（静态页面结构）.
- `src/main.js`: application bootstrap（应用启动入口） only.
- `src/i18n.js`: i18n（国际化） and theme（主题） state.
- `src/modules/navigation.js`: section nav（章节导航）, scroll progress（阅读进度）, and back-to-top（返回顶部） behavior.
- `src/modules/pipeline.js`: tabs（标签页）, action flow（动作流） and pipeline animation（流水线动画） state.
- `src/modules/bijector-lab.js`: direction distribution lab（方向分布实验台）.
- `src/modules/speed-bound-lab.js`: sphere/cube speed bound lab（速度边界实验台）.
- `src/modules/ds-scenes.js`: Three.js（3D 可视化） and Canvas fallback（画布降级） scenes.
- `src/styles.css`: Tailwind entry（Tailwind 入口） plus shadcn tokens（设计变量） and custom visualization styles.

Keep new interactions inside `src/modules/` unless they are true app bootstrap（应用启动） concerns.
