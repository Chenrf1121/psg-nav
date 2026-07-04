#!/usr/bin/env python3
"""
并行运行PSG-Nav - 使用多个GPU同时处理不同场景
每个GPU会启动独立的LLM server和PSG_Nav进程

使用示例:
    # HSSD数据集 (自动检测场景数)
    python run_parallel_with_server.py --gpus 5 --dataset hssd

    # 自动检测所有可用GPU
    python run_parallel_with_server.py --auto-detect --dataset hssd

    # 从第6个场景开始运行 (跳过0-5)
    python run_parallel_with_server.py --gpus 0,1,2,3,4,5 --scenes-per-gpu 1,1,1,1,1,1 --dataset hssd --start-scene 6
"""

import os
import subprocess
import time
import argparse
import signal
import sys
import gzip
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

from configs.dataset_registry import (
    SUPPORTED_DATASETS,
    get_dataset_config_path,
    get_default_dataset_split,
    get_fallback_scene_count,
)

# ============================================
# 用户配置 - 根据你的环境修改这里
# ============================================
PSGNAV_WORK_DIR = Path(__file__).resolve().parent

# Conda 安装路径
CONDA_BASE = os.environ.get("CONDA_BASE", "/home/RufengChen/miniconda3")
# CONDA_BASE = "/home/YueChang/miniconda3"  # YueChang 的路径

# Conda 环境名称
MLLM_ENV = os.environ.get("MLLM_ENV", "mllm")      # LLM server 使用的环境
PSGNAV_ENV = os.environ.get("PSGNAV_ENV", "psg_nav")   # PSG_Nav 使用的环境
# ============================================


def infer_scene_count_from_dataset(dataset: str, dataset_split: Optional[str] = None) -> Optional[int]:
    """Infer scene count from downloaded dataset files when possible."""
    dataset_split = dataset_split or get_default_dataset_split(dataset)
    config_path = Path(get_dataset_config_path(dataset))
    if not config_path.exists():
        return None

    data_path_template = None
    for line in config_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("DATA_PATH:"):
            data_path_template = stripped.split("DATA_PATH:", 1)[1].strip().strip('"').strip("'")
            break

    if not data_path_template:
        return None

    dataset_file = Path(data_path_template.format(split=dataset_split))
    content_dir = dataset_file.parent / "content"

    if content_dir.exists():
        content_files = sorted(content_dir.glob("*.json.gz"))
        if content_files:
            return len(content_files)

    if dataset_file.exists():
        try:
            with gzip.open(dataset_file, "rt") as fin:
                data = json.load(fin)
            episodes = data.get("episodes", [])
            if episodes:
                return len({episode["scene_id"] for episode in episodes if "scene_id" in episode})
        except Exception:
            return None

    return None


def get_default_scene_count(dataset: str, dataset_split: Optional[str] = None) -> int:
    """
    根据数据集返回默认的场景数量

    Args:
        dataset: 数据集名称
        dataset_split: 数据集 split，用于从已下载文件自动推断场景数

    Returns:
        该数据集的场景总数
    """
    inferred_count = infer_scene_count_from_dataset(dataset, dataset_split=dataset_split)
    if inferred_count is not None:
        return inferred_count

    fallback_count = get_fallback_scene_count(dataset)
    if fallback_count is None:
        raise ValueError(
            f"无法自动确定数据集 {dataset} 的场景数量。"
            f"请先下载数据集，或使用 --total-scenes 手动指定。"
        )

    return fallback_count


class ParallelRunner:
    def __init__(self, gpus: List[int], dataset: str = "hssd", dataset_split: Optional[str] = None, total_scenes: Optional[int] = None, start_scene: int = 0):
        self.gpus = gpus
        self.dataset = dataset
        self.dataset_split = dataset_split or get_default_dataset_split(dataset)
        # 如果未指定场景数，根据数据集自动设置
        self.total_scenes = total_scenes if total_scenes is not None else get_default_scene_count(dataset, self.dataset_split)
        self.start_scene = start_scene  # 起始场景索引
        self.processes = []  # (gpu_id, server_pid, psgnav_proc, log_file, split_l, split_r)
        self.log_dir = None

        # Generate unified timestamp for all parallel runs (format: MMDD_HHMM)
        self.timestamp = datetime.now().strftime("%m%d_%H%M")

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """处理中断信号，清理所有进程"""
        _ = (signum, frame)  # 未使用但需要符合信号处理器签名
        print("\n\n[中断] 收到终止信号，正在清理所有进程...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """清理所有启动的进程"""
        print("\n[清理] 开始清理所有进程...")

        for gpu_id, server_pid, psgnav_proc, _, split_l, split_r in self.processes:
            print(f"[清理] GPU {gpu_id}: 场景[{split_l}:{split_r}]")

            # 1. 终止PSG_Nav进程
            if psgnav_proc and psgnav_proc.poll() is None:
                try:
                    print(f"  - 终止PSG_Nav进程 (PID: {psgnav_proc.pid})")
                    psgnav_proc.terminate()
                    psgnav_proc.wait(timeout=5)
                except:
                    print(f"  - 强制杀死PSG_Nav进程")
                    psgnav_proc.kill()

            # 2. 终止LLM Server进程
            if server_pid:
                try:
                    print(f"  - 终止LLM Server (PID: {server_pid})")
                    os.kill(server_pid, signal.SIGTERM)
                    time.sleep(1)
                except ProcessLookupError:
                    pass
                except Exception as e:
                    print(f"  - 警告: {e}")

            # 3. 清理可能残留的server进程（通过端口查找）
            server_port = 5001 + gpu_id
            try:
                subprocess.run(
                    f"lsof -ti:{server_port} | xargs kill -9 2>/dev/null || true",
                    shell=True, timeout=2
                )
            except:
                pass

        print("[清理] 完成！")

    def calculate_scene_splits(self, scenes_per_gpu: List[int] = None) -> List[Tuple[int, int]]:
        """
        计算每个GPU处理的场景范围

        Args:
            scenes_per_gpu: 每个GPU处理的场景数量。如果为None，则均匀分配

        Returns:
            List of (split_l, split_r) tuples
        """
        # 计算从 start_scene 开始的剩余场景数
        remaining_scenes = self.total_scenes - self.start_scene

        if scenes_per_gpu:
            requested_scenes = sum(scenes_per_gpu)
            if requested_scenes > remaining_scenes:
                raise ValueError(f"请求的场景数({requested_scenes})超过从场景{self.start_scene}开始的剩余场景数({remaining_scenes})")
            if len(scenes_per_gpu) != len(self.gpus):
                raise ValueError(f"GPU数量与分配方案不匹配: {len(self.gpus)} != {len(scenes_per_gpu)}")

            # 如果只跑部分场景
            if requested_scenes < remaining_scenes:
                print(f"注意: 只运行 {requested_scenes} 个场景（从场景{self.start_scene}开始，共剩余 {remaining_scenes} 个场景）")
        else:
            # 均匀分配剩余场景
            base_count = remaining_scenes // len(self.gpus)
            remainder = remaining_scenes % len(self.gpus)
            scenes_per_gpu = [base_count + (1 if i < remainder else 0) for i in range(len(self.gpus))]

        splits = []
        current = self.start_scene  # 从 start_scene 开始
        for count in scenes_per_gpu:
            splits.append((current, current + count))
            current += count

        return splits

    def start_job(self, gpu_id: int, split_l: int, split_r: int) -> Tuple:
        """
        启动单个GPU上的任务（包括server和PSG_Nav）

        Returns:
            (gpu_id, server_pid, psgnav_proc, log_file, split_l, split_r)
        """
        server_port = 5001 + gpu_id

        print(f"\n{'='*60}")
        print(f"[启动] GPU {gpu_id}: 场景 [{split_l}:{split_r}]")
        print(f"       Server端口: {server_port}")
        print(f"{'='*60}")

        # 创建日志文件
        log_file = self.log_dir / f"gpu{gpu_id}_scenes[{split_l}-{split_r}].log"

        # 1. 检查并清理端口
        print(f"[1/4] 检查端口 {server_port}...")
        subprocess.run(
            f"lsof -ti:{server_port} | xargs kill -9 2>/dev/null || true",
            shell=True
        )
        time.sleep(1)

        # 2. 启动LLM Server
        print(f"[2/4] 启动LLM Server...")

        # 构建server启动命令 - 使用Popen直接启动
        server_cmd = [
            '/bin/bash', '-c',
            f"""
            source {CONDA_BASE}/etc/profile.d/conda.sh && \
            conda activate {MLLM_ENV} && \
            export CUDA_VISIBLE_DEVICES={gpu_id} && \
            python server.py --port {server_port}
            """
        ]

        # 打开server日志文件
        server_log_file = self.log_dir / f"server_gpu{gpu_id}.log"
        with open(server_log_file, 'w') as log_f:
            server_proc = subprocess.Popen(
                server_cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=PSGNAV_WORK_DIR
            )

        server_pid = server_proc.pid
        print(f"     Server PID: {server_pid}")

        # 3. 等待Server启动
        print(f"[3/4] 等待Server启动...")
        max_wait = 60
        for i in range(max_wait):
            try:
                result = subprocess.run(
                    f"lsof -Pi :{server_port} -sTCP:LISTEN -t",
                    shell=True,
                    capture_output=True,
                    timeout=1
                )
                if result.returncode == 0:
                    print(f"     Server已启动 (等待{i+1}秒)")
                    break
            except:
                pass

            if i < max_wait - 1:
                time.sleep(1)
        else:
            print(f"     警告: Server启动超时，继续执行...")

        # 给server额外时间初始化
        time.sleep(5)

        # 4. 启动PSG_Nav
        print(f"[4/4] 启动PSG_Nav...")

        psgnav_cmd = [
            '/bin/bash', '-c',
            f"""
            source {CONDA_BASE}/etc/profile.d/conda.sh && \
            conda activate {PSGNAV_ENV} && \
            export CUDA_VISIBLE_DEVICES={gpu_id} && \
            python -W ignore PSG_Nav.py --split_l {split_l} --split_r {split_r} --server_port {server_port} --dataset {self.dataset} --dataset_split {self.dataset_split} --timestamp {self.timestamp}
            """
        ]

        with open(log_file, 'w') as log_f:
            log_f.write(f"=== PSG-Nav Parallel Run ===\n")
            log_f.write(f"GPU: {gpu_id}\n")
            log_f.write(f"Scenes: [{split_l}:{split_r}]\n")
            log_f.write(f"Server Port: {server_port}\n")
            log_f.write(f"Server PID: {server_pid}\n")
            log_f.write(f"Start Time: {datetime.now()}\n")
            log_f.write(f"{'='*50}\n\n")
            log_f.flush()

            psgnav_proc = subprocess.Popen(
                psgnav_cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=PSGNAV_WORK_DIR
            )

        print(f"     PSG_Nav PID: {psgnav_proc.pid}")
        print(f"     日志文件: {log_file.name}")

        return (gpu_id, server_pid, psgnav_proc, log_file, split_l, split_r)

    def run(self, scenes_per_gpu: List[int] = None):
        """运行所有任务"""
        # 创建日志目录
        log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(f"logs/parallel_run_{log_timestamp}")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print("PSG-Nav 并行运行 - 配置信息")
        print("="*70)
        print(f"数据集: {self.dataset.upper()}")
        print(f"数据集 split: {self.dataset_split}")
        print(f"运行时间戳: {self.timestamp}")
        print(f"总场景数: {self.total_scenes}")
        print(f"起始场景: {self.start_scene}")
        print(f"剩余场景数: {self.total_scenes - self.start_scene}")
        print(f"使用GPU: {self.gpus}")
        print(f"GPU数量: {len(self.gpus)}")
        print(f"日志目录: {self.log_dir}")
        print(f"结果保存至: data/results_{self.dataset}_{self.timestamp}/")
        print(f"可视化保存至: data/visualization_{self.dataset}_{self.timestamp}/")

        # 计算场景分配
        splits = self.calculate_scene_splits(scenes_per_gpu)

        print(f"\n场景分配:")
        for gpu_id, (split_l, split_r) in zip(self.gpus, splits):
            print(f"  GPU {gpu_id}: 场景 [{split_l:2d}:{split_r:2d}] ({split_r-split_l} scenes)")

        print("="*70)

        # 启动所有任务
        for gpu_id, (split_l, split_r) in zip(self.gpus, splits):
            try:
                job_info = self.start_job(gpu_id, split_l, split_r)
                self.processes.append(job_info)

                # 间隔启动，避免资源冲突
                time.sleep(3)
            except Exception as e:
                print(f"\n[错误] GPU {gpu_id} 启动失败: {e}")
                self.cleanup()
                sys.exit(1)

        # 保存任务信息
        self._save_job_info()

        # 显示监控信息
        self._print_monitoring_info()

        # 等待所有任务完成
        self._wait_for_completion()

    def _save_job_info(self):
        """保存任务信息到文件"""
        info_file = self.log_dir / "job_info.txt"
        with open(info_file, 'w') as f:
            f.write(f"PSG-Nav Parallel Run\n")
            f.write(f"{'='*50}\n")
            f.write(f"Start Time: {datetime.now()}\n")
            f.write(f"Total Scenes: {self.total_scenes}\n")
            f.write(f"GPUs: {self.gpus}\n")
            f.write(f"Number of Jobs: {len(self.processes)}\n\n")

            for gpu_id, server_pid, psgnav_proc, log_file, split_l, split_r in self.processes:
                f.write(f"\nGPU {gpu_id}:\n")
                f.write(f"  Scenes: [{split_l}:{split_r}]\n")
                f.write(f"  Server PID: {server_pid}\n")
                f.write(f"  PSG_Nav PID: {psgnav_proc.pid}\n")
                f.write(f"  Server Port: {5001 + gpu_id}\n")
                f.write(f"  Log File: {log_file.name}\n")

    def _print_monitoring_info(self):
        """打印监控信息"""
        print("\n" + "="*70)
        print("所有任务已启动!")
        print("="*70)

        print("\n📊 监控命令:")
        print(f"  # 查看所有日志")
        print(f"  tail -f {self.log_dir}/*.log")

        print(f"\n  # 查看单个GPU的日志")
        for gpu_id, _, _, log_file, split_l, split_r in self.processes:
            print(f"  tail -f {log_file}  # GPU {gpu_id}: [{split_l}:{split_r}]")

        print(f"\n  # 查看Server日志")
        for gpu_id, _, _, _, _, _ in self.processes:
            print(f"  tail -f {self.log_dir}/server_gpu{gpu_id}.log")

        print("\n🔍 检查运行状态:")
        print("  ps aux | grep 'PSG_Nav.py\\|server.py'")

        print("\n⚠️  紧急终止所有任务:")
        print("  pkill -f 'PSG_Nav.py'; pkill -f 'server.py'")

        print("\n📁 任务信息:")
        print(f"  cat {self.log_dir}/job_info.txt")

        print("\n" + "="*70)

    def _wait_for_completion(self):
        """等待所有任务完成"""
        print("\n⏳ 等待所有任务完成...")
        print("   (按 Ctrl+C 可以提前终止所有任务)\n")

        try:
            while True:
                all_done = True
                running_jobs = []

                for gpu_id, _, psgnav_proc, _, split_l, split_r in self.processes:
                    if psgnav_proc.poll() is None:
                        all_done = False
                        running_jobs.append(f"GPU{gpu_id}:[{split_l}:{split_r}]")

                if all_done:
                    break

                # 显示运行状态
                print(f"\r⏳ 运行中 ({len(running_jobs)}/{len(self.processes)}): {', '.join(running_jobs)}", end='', flush=True)
                time.sleep(5)

            print("\n\n✅ 所有任务已完成!")

            # 显示结果
            print("\n" + "="*70)
            print("任务完成情况:")
            print("="*70)
            for gpu_id, _, psgnav_proc, _, split_l, split_r in self.processes:
                return_code = psgnav_proc.returncode
                status = "✅ 成功" if return_code == 0 else f"❌ 失败 (code: {return_code})"
                print(f"GPU {gpu_id}: 场景[{split_l:2d}:{split_r:2d}] - {status}")

        finally:
            # 清理所有进程
            self.cleanup()


def get_available_gpus() -> List[int]:
    """自动检测可用GPU"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            gpu_id, mem_free = line.split(',')
            gpu_id = int(gpu_id.strip())
            mem_free = int(mem_free.strip())

            # 只选择空闲内存 > 10GB 的GPU
            if mem_free > 10000:
                gpus.append(gpu_id)

        return gpus
    except Exception as e:
        print(f"警告: 无法自动检测GPU: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description='并行运行PSG-Nav，为每个GPU启动独立的LLM server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # HSSD数据集
  python run_parallel_with_server.py --gpus 5,6,7 --dataset hssd

  # 手动指定场景分配
  python run_parallel_with_server.py --gpus 5,6,7 --dataset hssd --scenes-per-gpu 14,13,13

  # 自动检测空闲GPU并使用
  python run_parallel_with_server.py --auto-detect --dataset hssd

  # 从第6个场景开始运行 (跳过0-5)
  python run_parallel_with_server.py --gpus 2,3,4,5,6,7 --dataset hssd --scenes-per-gpu 1,1,1,1,1,1 --start-scene 6

  # 手动覆盖场景数量（不推荐，通常自动检测即可）
  python run_parallel_with_server.py --gpus 5,6 --dataset hssd --total-scenes 40
        """
    )

    parser.add_argument(
        '--gpus',
        type=str,
        help='使用的GPU ID列表，用逗号分隔 (例如: 5,6,7)'
    )

    parser.add_argument(
        '--auto-detect',
        action='store_true',
        help='自动检测并使用空闲GPU (内存 > 10GB)'
    )

    parser.add_argument(
        '--total-scenes',
        type=int,
        default=None,
        help='总场景数 (如果不指定，将优先从已下载数据集自动推断，否则回退到内置默认值)'
    )

    parser.add_argument(
        '--scenes-per-gpu',
        type=str,
        help='每个GPU处理的场景数，用逗号分隔 (例如: 4,4,3)。不指定则均匀分配'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='hssd',
        choices=list(SUPPORTED_DATASETS),
        help='使用的数据集: hssd'
    )

    parser.add_argument(
        '--dataset-split',
        type=str,
        default=None,
        help='数据集 split。默认 HSSD 为 val。'
    )

    parser.add_argument(
        '--start-scene',
        type=int,
        default=0,
        help='起始场景索引 (默认: 0)，用于跳过已完成的场景'
    )

    args = parser.parse_args()

    # 确定使用的GPU
    if args.auto_detect:
        gpus = get_available_gpus()
        if not gpus:
            print("错误: 未检测到可用GPU")
            sys.exit(1)
        print(f"自动检测到可用GPU: {gpus}")
    elif args.gpus:
        gpus = [int(x.strip()) for x in args.gpus.split(',')]
    else:
        print("错误: 请使用 --gpus 指定GPU，或使用 --auto-detect 自动检测")
        parser.print_help()
        sys.exit(1)

    # 解析场景分配
    scenes_per_gpu = None
    if args.scenes_per_gpu:
        scenes_per_gpu = [int(x.strip()) for x in args.scenes_per_gpu.split(',')]

    # 创建运行器并启动
    runner = ParallelRunner(
        gpus,
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        total_scenes=args.total_scenes,
        start_scene=args.start_scene,
    )
    runner.run(scenes_per_gpu)


if __name__ == "__main__":
    main()
