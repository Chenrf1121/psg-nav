#!/usr/bin/env python3
import os
import glob
import ast
import math

def parse_results_file(filepath):
    """解析单个 results.txt 文件"""
    episodes = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    # 解析字典
                    data = ast.literal_eval(line)
                    episodes.append(data)
                except:
                    continue
    return episodes

def main():
    # 查找所有 results.txt 文件
    # pattern = './data/results_hssd/*/results.txt'
    pattern = './hssd_tmp.txt'
    result_files = glob.glob(pattern)

    print(f"找到 {len(result_files)} 个场景的结果文件")
    print("=" * 60)

    all_episodes = []
    scene_stats = []

    # 按场景统计
    for result_file in sorted(result_files):
        scene_name = result_file.split('/')[-2]
        episodes = parse_results_file(result_file)

        if episodes:
            # 计算该场景的统计
            successes = [ep['success'] for ep in episodes]
            spls = []
            for ep in episodes:
                spl = ep['spl']
                # 将 nan 当作 0
                if isinstance(spl, float) and (math.isnan(spl) or math.isinf(spl)):
                    spls.append(0.0)
                else:
                    spls.append(spl)

            scene_success = sum(successes) / len(successes)
            scene_spl = sum(spls) / len(spls)

            scene_stats.append({
                'scene': scene_name,
                'episodes': len(episodes),
                'success': scene_success,
                'spl': scene_spl
            })

            all_episodes.extend(episodes)

            print(f"场景 {scene_name}: {len(episodes)} episodes, "
                  f"Success: {scene_success:.4f}, SPL: {scene_spl:.4f}")

    print("\n" + "=" * 60)
    print(f"总共 {len(scene_stats)} 个场景")
    print(f"总共 {len(all_episodes)} 个 episodes")
    print("=" * 60)

    # 计算总体平均（所有 episodes）
    total_success = sum(ep['success'] for ep in all_episodes)
    total_spls = []
    for ep in all_episodes:
        spl = ep['spl']
        if isinstance(spl, float) and (math.isnan(spl) or math.isinf(spl)):
            total_spls.append(0.0)
        else:
            total_spls.append(spl)

    avg_success = total_success / len(all_episodes)
    avg_spl = sum(total_spls) / len(total_spls)

    print(f"\n所有 episodes 的平均指标:")
    print(f"  平均 Success Rate: {avg_success:.4f} ({avg_success*100:.2f}%)")
    print(f"  平均 SPL: {avg_spl:.4f}")

    # 计算场景级别的平均（每个场景算一个单位）
    scene_avg_success = sum(s['success'] for s in scene_stats) / len(scene_stats)
    scene_avg_spl = sum(s['spl'] for s in scene_stats) / len(scene_stats)

    print(f"\n按场景平均的指标:")
    print(f"  平均 Success Rate: {scene_avg_success:.4f} ({scene_avg_success*100:.2f}%)")
    print(f"  平均 SPL: {scene_avg_spl:.4f}")

if __name__ == '__main__':
    main()
