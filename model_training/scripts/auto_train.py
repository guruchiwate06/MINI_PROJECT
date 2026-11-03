"""Explicit recovery launcher. Never evaluates the held-out test partition."""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/track_ot_recovery_fp32.json')
    parser.add_argument('--on-training-device', action='store_true')
    parser.add_argument('--warm-start')
    parser.add_argument('--wait-minutes', type=float, default=30)
    args = parser.parse_args(argv)
    if not args.on_training_device:
        print('Recovery requires --on-training-device on the GPU computer. Nothing was started.', file=sys.stderr)
        return 2
    if args.wait_minutes < 0:
        parser.error('--wait-minutes cannot be negative')
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from track_ot.config import load_config
    from railvivid.common import read_json
    from railvivid.vision import vision_path
    config = (root / args.config).resolve()
    c = load_config(config)
    cache = vision_path(c['_base'])
    deadline = time.monotonic() + args.wait_minutes * 60
    while not (cache / 'READY.json').exists():
        if (cache / 'FAILURE.json').exists():
            raise RuntimeError('Vision extraction failed; recovery training was not started')
        if time.monotonic() >= deadline:
            raise RuntimeError('Timed out waiting for the verified vision cache')
        time.sleep(min(20, max(deadline - time.monotonic(), 0.01)))
    output = root / 'artifacts' / ('recovery_preflight_' + time.strftime('%Y%m%d_%H%M%S'))
    command = [sys.executable, '-m', 'track_ot', 'diagnose', '--config', str(config), '--device', 'cuda',
               '--precisions', 'float32', '--batches', '4', '--component-gradients', '--output', str(output)]
    if args.warm_start:
        command += ['--from-run', str((root / args.warm_start).resolve())]
    check = subprocess.run(command, cwd=root)
    if check.returncode:
        return check.returncode
    report = read_json(output / 'diagnostics.json')
    if not report['precision_runs'] or not all(r.get('all_finite', False) for r in report['precision_runs']):
        raise RuntimeError('Preflight gradients were not finite; no optimizer was started')
    command = [sys.executable, '-m', 'track_ot', 'train', '--config', str(config), '--on-training-device']
    if args.warm_start:
        command += ['--warm-start', str((root / args.warm_start).resolve())]
    result = subprocess.run(command, cwd=root)
    print('Training stage ended. Test evaluation remains a separate, explicit command after model selection is frozen.')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
