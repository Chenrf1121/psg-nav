#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


def parse_tum_pose_line(line):
    parts = line.strip().split()
    if len(parts) != 8:
        raise ValueError(f"Invalid TUM pose line: {line}")
    ts, tx, ty, tz, qx, qy, qz, qw = parts
    return {
        "timestamp": ts,
        "tx": float(tx),
        "ty": float(ty),
        "tz": float(tz),
        "qx": float(qx),
        "qy": float(qy),
        "qz": float(qz),
        "qw": float(qw),
    }


def quat_to_rotmat(qx, qy, qz, qw):
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return [
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ]


def yaw_deg_from_optical_quat(qx, qy, qz, qw):
    rot = quat_to_rotmat(qx, qy, qz, qw)
    forward_x = rot[0][2]
    forward_y = rot[1][2]
    return math.degrees(math.atan2(forward_y, forward_x))


def timestamp_to_filename(timestamp):
    return timestamp.replace('.', '_') + '.png'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True, type=str)
    parser.add_argument('--pose_file', default='poses_map_tum.txt', type=str)
    parser.add_argument('--goal', required=True, type=str)
    parser.add_argument('--scene_id', default=None, type=str)
    parser.add_argument('--depth_scale', default=1000.0, type=float)
    parser.add_argument('--rgb_dir', default='rgb', type=str)
    parser.add_argument('--depth_dir', default=None, type=str,
                        help='Depth directory name. If omitted, auto-select depth_camera, depth, or lidar_depth.')
    parser.add_argument('--output', default=None, type=str)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    pose_path = dataset_dir / args.pose_file
    rgb_dir = dataset_dir / args.rgb_dir
    if args.depth_dir is not None:
        depth_dir = dataset_dir / args.depth_dir
    else:
        candidate_depth_dirs = ['depth_camera', 'depth', 'lidar_depth']
        depth_dir = None
        for candidate in candidate_depth_dirs:
            candidate_path = dataset_dir / candidate
            if candidate_path.exists():
                depth_dir = candidate_path
                break
        if depth_dir is None:
            raise FileNotFoundError(f'No depth directory found in {dataset_dir}. Tried: {candidate_depth_dirs}')

    if not pose_path.exists():
        raise FileNotFoundError(pose_path)
    if not rgb_dir.exists():
        raise FileNotFoundError(rgb_dir)
    if not depth_dir.exists():
        raise FileNotFoundError(depth_dir)

    frames = []
    with open(pose_path, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            pose = parse_tum_pose_line(line)
            values = [pose['tx'], pose['ty'], pose['tz'], pose['qx'], pose['qy'], pose['qz'], pose['qw']]
            if any(math.isnan(v) for v in values):
                continue
            filename = timestamp_to_filename(pose['timestamp'])
            rgb_path = rgb_dir / filename
            depth_path = depth_dir / filename
            if not rgb_path.exists() or not depth_path.exists():
                continue

            yaw_deg = yaw_deg_from_optical_quat(pose['qx'], pose['qy'], pose['qz'], pose['qw'])
            frames.append({
                'rgb': str(Path('rgb') / filename),
                'depth': str(Path(depth_dir.name) / filename),
                'depth_scale': args.depth_scale,
                'pose': [pose['tx'], pose['ty'], yaw_deg],
                'pose_debug_full': {
                    'timestamp': pose['timestamp'],
                    'tx': pose['tx'],
                    'ty': pose['ty'],
                    'tz': pose['tz'],
                    'qx': pose['qx'],
                    'qy': pose['qy'],
                    'qz': pose['qz'],
                    'qw': pose['qw'],
                }
            })

    if not frames:
        raise RuntimeError('No matched frames were found between pose file and rgb/depth folders.')

    output_path = Path(args.output).resolve() if args.output else dataset_dir / f'manifest_{Path(args.pose_file).stem}_{args.goal}.json'
    manifest = {
        'episode_id': '0',
        'scene_id': args.scene_id or dataset_dir.name,
        'object_category': args.goal,
        'frames': frames,
        'metrics': {
            'success': 0.0,
            'spl': 0.0,
            'softspl': 0.0,
            'distance_to_goal': 0.0,
        }
    }

    output_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(output_path)
    print(f'generated_frames={len(frames)}')


if __name__ == '__main__':
    main()
