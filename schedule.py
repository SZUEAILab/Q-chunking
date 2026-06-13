#!/usr/bin/env python3
"""GPU-aware 实验调度器：编译 tasks.json → 命令列表 → 按 GPU slot 空闲事件驱动执行。

用法:
  # 第一步：编译（生成命令列表，人工审查）
  python schedule.py --tasks_config=tasks.json --compile

  # 第二步：执行（从编译文件读取，中断后可重跑——自动跳过已完成）
  python schedule.py --run=tasks.compiled.json --gpus=0,1 --gpu_tasks=4

  # 预览
  python schedule.py --run=tasks.compiled.json --dry_run
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from absl import app, flags

FLAGS = flags.FLAGS

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------

flags.DEFINE_string('tasks_config', 'tasks.json',
                    '任务配置文件 (JSON)。默认 tasks.json。')
flags.DEFINE_string('run', None,
                    '编译后的命令文件路径。未指定时尝试 ./tasks.compiled.json。')
flags.DEFINE_bool('compile', False,
                  '仅编译 tasks.json → tasks.compiled.json，不执行。')
flags.DEFINE_list('gpus', [0],
                  '使用的 GPU ID 列表，逗号分隔。默认只用 GPU 0。')
flags.DEFINE_integer('gpu_tasks', 1,
                     '每张 GPU 同时跑的进程数。A6000 建议 6，4090 建议 4。')
flags.DEFINE_integer('stagger', 30,
                     '同一 GPU 上两次启动之间的冷却秒数，等 JIT 编译完成。')
flags.DEFINE_bool('dry_run', False, '只打印命令，不实际启动。')


# ---------------------------------------------------------------------------
# GPU 检测
# ---------------------------------------------------------------------------

def query_gpus():
    """返回 {index: {total_mb, used_mb, free_mb}}."""
    out = subprocess.check_output(
        ['nvidia-smi',
         '--query-gpu=index,memory.total,memory.used,memory.free',
         '--format=csv,noheader,nounits'],
        text=True,
    )
    gpus = {}
    for line in out.strip().split('\n'):
        if not line:
            continue
        idx, total, used, free = line.strip().split(',')
        gpus[int(idx.strip())] = {
            'total_mb': int(total.strip()),
            'used_mb': int(used.strip()),
            'free_mb': int(free.strip()),
        }
    return gpus


def calc_concurrency(free_mb, horizon_length):
    """按显存公式估算单卡并发数。"""
    if horizon_length == 1:
        h_factor = 1.0
    elif horizon_length == 5:
        h_factor = 1.7
    else:
        h_factor = 1.0 + (horizon_length - 1) * (0.7 / 4)
    per_proc_mb = 1500 * h_factor
    return max(1, int(free_mb / per_proc_mb))


# ---------------------------------------------------------------------------
# 任务配置 → CLI flag 映射
# ---------------------------------------------------------------------------

FIELD_TO_FLAG = {
    'entry':                 None,
    'env_name':              '--env_name',
    'horizon_length':        '--horizon_length',
    'ds_mode':               '--ds_mode',
    'sparse':                '--sparse',
    'online_steps':          '--online_steps',
    'offline_steps':         '--offline_steps',
    'save_interval':         '--save_interval',
    'eval_episodes':         '--eval_episodes',
    'video_episodes':        '--video_episodes',
    'run_group':             '--run_group',
    'action_chunking':       '--agent.action_chunking',
    'agent':                 '--agent',
    'discount':              '--discount',
    'utd_ratio':             '--utd_ratio',
    'buffer_size':           '--buffer_size',
    'log_interval':          '--log_interval',
    'eval_interval':         '--eval_interval',
    'start_training':        '--start_training',
    'save_dir':              '--save_dir',
    'dataset_proportion':    '--dataset_proportion',
    'dataset_replace_interval':          '--dataset_replace_interval',
    'ogbench_dataset_dir':                '--ogbench_dataset_dir',
    'allow_posthoc_direction_speed_rlpd': '--allow_posthoc_direction_speed_rlpd',
}


def _bool_str(v):
    """absl bool flag 格式."""
    return 'True' if v else 'False'


# ---------------------------------------------------------------------------
# 编译阶段：tasks.json → 命令列表
# ---------------------------------------------------------------------------

def load_tasks_config(path):
    """读取并校验任务配置文件。"""
    with open(path) as f:
        config = json.load(f)
    if 'tasks' not in config:
        sys.exit(f'{path}: missing "tasks" list.')
    if 'seeds' not in config:
        sys.exit(f'{path}: missing "seeds" list.')
    return config


def merge_task(common, task):
    """合并 common + task 配置，task 优先。"""
    merged = dict(common)
    merged.update(task)
    return merged


def build_command(seed, task):
    """task dict → 完整 CLI 命令字符串。"""
    entry = task.pop('entry', 'main_online.py')
    name = task.pop('name', 'unnamed')
    parts = [sys.executable, entry, f'--seed={seed}']
    for field, flag in FIELD_TO_FLAG.items():
        if field == 'entry' or field not in task:
            continue
        value = task[field]
        if flag is None:
            continue
        if isinstance(value, bool):
            parts.append(f'{flag}={_bool_str(value)}')
        else:
            parts.append(f'{flag}={value}')
    return ' '.join(parts), name


def compile_tasks(config_path):
    """编译 tasks.json → 命令列表，返回 compiled dict。"""
    config = load_tasks_config(config_path)
    common = config.get('common', {})
    seeds = config['seeds']

    commands = []
    for task_def in config['tasks']:
        task = merge_task(common, task_def)
        for seed in seeds:
            cmd, name = build_command(seed, dict(task))
            commands.append({
                'name': name,
                'seed': seed,
                'cmd': cmd,
                'status': 'pending',
                'pid': None,
                'log': None,
                'gpu': None,
            })

    compiled = {
        'meta': {
            'source': os.path.abspath(config_path),
            'compiled_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        },
        'commands': commands,
    }
    return compiled


def write_compiled(compiled, config_path):
    """写入编译文件，默认路径为 {config_path}.compiled.json。"""
    out_path = config_path.replace('.json', '.compiled.json')
    if out_path == config_path:
        out_path = config_path + '.compiled.json'
    with open(out_path, 'w') as f:
        json.dump(compiled, f, indent=2)
    return out_path


def load_compiled(path):
    """读取编译后的命令文件。"""
    with open(path) as f:
        return json.load(f)


def save_compiled(compiled, path):
    """更新编译文件（原地修改状态）。"""
    compiled['meta']['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(path, 'w') as f:
        json.dump(compiled, f, indent=2)


# ---------------------------------------------------------------------------
# 执行阶段：调度器主循环
# ---------------------------------------------------------------------------

def _pid_alive(pid):
    """检查进程是否存活。"""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def run_scheduler(compiled, compiled_path, gpus, gpu_tasks, stagger):
    """从编译文件执行调度。编译文件会被原地更新以记录进度。"""
    commands = compiled['commands']
    total_commands = len(commands)

    # ---- 日志目录 ----
    log_dir = Path('exp_logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- 环境变量 ----
    env = os.environ.copy()
    env.setdefault('MUJOCO_GL', 'egl')
    env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    env.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.05')

    # ---- 恢复状态：分类命令 ----
    queue = []
    running = {g: [] for g in gpus}
    launched = 0
    done_ok = 0
    done_fail = 0
    log_index = 0

    for c in commands:
        if c['status'] == 'done':
            done_ok += 1
            launched += 1
        elif c['status'] == 'running':
            if _pid_alive(c.get('pid')):
                log_path = c.get('log')
                fh = open(log_path, 'a') if log_path else None
                if fh is None:
                    log_path = str(log_dir / f'{c["name"]}_seed{c["seed"]}_{log_index}.log')
                    fh = open(log_path, 'w')
                    c['log'] = log_path
                log_index += 1
                launched += 1
                gpu = c.get('gpu', gpus[0])
                if gpu not in running:
                    gpu = gpus[0]
                running[gpu].append((c, None, fh))
                print(f'[scheduler] 重连: GPU={gpu} {c["name"]} '
                      f'seed={c["seed"]} pid={c["pid"]}')
            else:
                c['status'] = 'pending'
                c['pid'] = None
                c['log'] = None
                c['gpu'] = None
                queue.append(c)
        else:
            queue.append(c)

    if not any(c['status'] in ('done', 'running') for c in commands):
        # 首次运行
        queue = list(commands)
        launched = 0
        done_ok = 0
        done_fail = 0

    if FLAGS.dry_run:
        for i, c in enumerate(commands):
            marker = '✓' if c['status'] == 'done' else ('▶' if c['status'] == 'running' else '·')
            print(f'[{i}] [{marker}] {c["name"]} seed={c["seed"]}')
            print(f'    {c["cmd"]}')
        return

    # ---- 调度辅助函数 ----
    cooldown = {g: 0.0 for g in gpus}
    start_time = time.time()
    last_heartbeat = 0.0

    def _gpu_available_slots(gpu_id):
        return gpu_tasks - len(running[gpu_id])

    def _gpu_bar(gpu_id):
        n = len(running[gpu_id])
        return '[' + '█' * n + '░' * (gpu_tasks - n) + ']'

    def _print_status(now, force=False):
        nonlocal last_heartbeat
        if not force and now - last_heartbeat < 60:
            return
        last_heartbeat = now
        done = done_ok + done_fail
        elapsed = now - start_time
        bars = '  '.join(f'GPU{g}:{_gpu_bar(g)}' for g in gpus)
        eta = ''
        if done > 0:
            eta_sec = (elapsed / done) * (total_commands - done)
            if eta_sec < 120:
                eta = f'  ETA: {eta_sec:.0f}s'
            elif eta_sec < 7200:
                eta = f'  ETA: {eta_sec/60:.1f}m'
            else:
                eta = f'  ETA: {eta_sec/3600:.1f}h'
        print(f'[{time.strftime("%T")}] {bars}  '
              f'queue={len(queue)}  done={done}/{total_commands}'
              f'{eta}  elapsed={elapsed/60:.1f}m')

    def _try_launch(gpu_id, now):
        nonlocal launched, log_index
        if not queue:
            return False
        if _gpu_available_slots(gpu_id) <= 0:
            return False
        if now < cooldown[gpu_id]:
            return False

        c = queue.pop(0)
        log_path = log_dir / f'{c["name"]}_seed{c["seed"]}_{log_index}.log'
        log_index += 1
        fh = open(log_path, 'w')

        proc_env = env.copy()
        proc_env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        proc = subprocess.Popen(
            c['cmd'], shell=True, env=proc_env,
            stdout=fh, stderr=subprocess.STDOUT,
        )
        c['status'] = 'running'
        c['pid'] = proc.pid
        c['log'] = str(log_path)
        c['gpu'] = gpu_id
        running[gpu_id].append((c, proc, fh))
        cooldown[gpu_id] = now + stagger
        launched += 1
        save_compiled(compiled, compiled_path)

        remaining = _gpu_available_slots(gpu_id)
        print(f'[{time.strftime("%T")}] ▶ GPU={gpu_id}  {c["name"]}  seed={c["seed"]}  '
              f'pid={proc.pid}  [{launched}/{total_commands}]  '
              f'GPU{gpu_id}{_gpu_bar(gpu_id)} slot_free={remaining}')
        _print_status(now, force=True)
        return True

    # ---- 信号处理：SIGTERM → 等同于 Ctrl+C ----
    def _on_terminate(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _on_terminate)

    # ---- 首次填充 ----
    print(f'\n{"─"*60}')
    for i, c in enumerate(commands):
        marker = '✓' if c['status'] == 'done' else '·'
        print(f'  [{i}] [{marker}] {c["name"]}  seed={c["seed"]}')
    print(f'{"─"*60}\n')

    print(f'[scheduler] 命令总数={total_commands}  '
          f'已完成={done_ok + done_fail}  待执行={len(queue)}  '
          f'gpus={gpus}  gpu_tasks={gpu_tasks}  stagger={stagger}s')

    # ---- 主循环 ----
    try:
        while queue or any(running[g] for g in gpus):
            now = time.time()

            # 1. 回收
            for gpu_id in gpus:
                still = []
                for c, proc, fh in running[gpu_id]:
                    if proc is None:
                        # 重连的进程（非本调度器启动），检查 PID
                        if not _pid_alive(c.get('pid')):
                            fh.close()
                            c['status'] = 'done'
                            done_ok += 1
                            save_compiled(compiled, compiled_path)
                            print(f'[{time.strftime("%T")}] ✓ GPU={gpu_id}  {c["name"]}  '
                                  f'seed={c["seed"]}  pid={c["pid"]}  DETACHED-OK  '
                                  f'[{done_ok + done_fail}/{total_commands}]')
                        else:
                            still.append((c, proc, fh))
                        continue

                    ret = proc.poll()
                    if ret is not None:
                        fh.close()
                        if ret == 0:
                            done_ok += 1
                            status = 'OK'
                        else:
                            done_fail += 1
                            status = f'FAIL({ret})'
                        c['status'] = 'done'
                        save_compiled(compiled, compiled_path)
                        print(f'[{time.strftime("%T")}] ✓ GPU={gpu_id}  {c["name"]}  '
                              f'seed={c["seed"]}  pid={proc.pid}  {status}  '
                              f'[{done_ok + done_fail}/{total_commands}]')
                    else:
                        still.append((c, proc, fh))
                running[gpu_id] = still

            # 2. 填空位
            for gpu_id in sorted(gpus, key=lambda g: _gpu_available_slots(g), reverse=True):
                while _try_launch(gpu_id, now):
                    pass

            # 3. 心跳
            _print_status(now)

            if queue or any(running[g] for g in gpus):
                time.sleep(1)

    except KeyboardInterrupt:
        print(f'\n[scheduler] 正在终止所有子进程...')
        for gpu_id in gpus:
            for c, proc, fh in running[gpu_id]:
                fh.close()
                if proc is not None:
                    proc.terminate()
                elif c.get('pid'):
                    # 重连的进程：用 PID 杀
                    try:
                        os.kill(c['pid'], 15)  # SIGTERM
                    except OSError:
                        pass
        # 等子进程退出
        time.sleep(2)
        for gpu_id in gpus:
            for c, proc, fh in running[gpu_id]:
                if proc is not None:
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                elif c.get('pid'):
                    try:
                        os.kill(c['pid'], 9)  # SIGKILL
                    except OSError:
                        pass
        print(f'[scheduler] 已终止。进度已保存至 {compiled_path}')
        print(f'[scheduler] 恢复时运行: python schedule.py --run={compiled_path} '
              f'--gpus={",".join(str(g) for g in gpus)} --gpu_tasks={gpu_tasks}')
        sys.exit(1)

    # ---- 汇总 ----
    elapsed = time.time() - start_time
    print(f'\n{"="*60}')
    print(f'  结束时间: {time.strftime("%Y-%m-%d %T")}')
    print(f'  总耗时:   {elapsed/60:.1f}m ({elapsed/3600:.2f}h)')
    print(f'  完成:     {done_ok + done_fail}/{total_commands}')
    print(f'  成功:     {done_ok}')
    if done_fail > 0:
        print(f'  失败:     {done_fail}')
    print(f'{"="*60}')


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main(_):
    # ---- 确定运行路径 ----
    run_path = FLAGS.run
    if not run_path and FLAGS.tasks_config:
        # 根据 tasks_config 推导 compiled 路径
        candidate = FLAGS.tasks_config.replace('.json', '.compiled.json')
        if candidate == FLAGS.tasks_config:
            candidate = FLAGS.tasks_config + '.compiled.json'
        if os.path.exists(candidate) and not FLAGS.compile:
            # 已有编译文件，直接走执行（除非显式要求 --compile）
            run_path = candidate
    if not run_path and not FLAGS.tasks_config:
        if os.path.exists('tasks.compiled.json'):
            run_path = 'tasks.compiled.json'
        else:
            sys.exit('未找到 tasks.compiled.json，请先 --tasks_config=X --compile 或指定 --run。')

    # ---- 模式：--run（或自动检测到 compiled 文件）直接执行 ----
    if run_path and not FLAGS.compile:
        print(f'[schedule] 执行: {run_path}')
        compiled = load_compiled(run_path)
        gpus = [int(g) for g in FLAGS.gpus]
        run_scheduler(compiled, run_path, gpus, FLAGS.gpu_tasks, FLAGS.stagger)
        return

    # ---- 模式：--compile 编译（或编译 + 预览） ----
    if not FLAGS.tasks_config:
        sys.exit('编译需要 --tasks_config。')

    compiled = compile_tasks(FLAGS.tasks_config)
    out_path = write_compiled(compiled, FLAGS.tasks_config)

    total = len(compiled['commands'])
    print(f'[compile] {FLAGS.tasks_config} → {out_path}')
    print(f'[compile] 共 {total} 条命令')

    if FLAGS.dry_run:
        for i, c in enumerate(compiled['commands']):
            print(f'  [{i}] {c["name"]}  seed={c["seed"]}')
            print(f'      {c["cmd"]}')
    print(f'\n执行: python schedule.py --gpus={",".join(str(g) for g in FLAGS.gpus)} '
          f'--gpu_tasks={FLAGS.gpu_tasks}')


if __name__ == '__main__':
    app.run(main)
