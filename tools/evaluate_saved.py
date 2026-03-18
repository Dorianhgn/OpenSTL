# Copyright (c) CAIRI AI Lab. All rights reserved

"""Recompute test metrics from an existing saved/ folder.

This script is meant for cases where `preds.npy` and `trues.npy` already exist
and only the evaluation metrics need to be recomputed, for example after fixing
an incorrect detection threshold.

Example:
    python tools/evaluate_saved.py \
        --saved_dir work_dirs/mmnist/jvm/cosine/saved \
        --metric_threshold 0.5
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np

from openstl.core import metric, per_frame_metric


_ALLOWED_GLOBAL_METRICS = {
    'mae', 'mse', 'rmse', 'ssim', 'psnr', 'snr', 'lpips',
    'pod', 'sucr', 'csi', 'far'
}

_DEFAULT_GLOBAL_METRICS = [
    'mae', 'mse', 'ssim', 'psnr', 'lpips', 'pod', 'sucr', 'csi', 'far'
]

_DEFAULT_PER_FRAME_METRICS = [
    'mae', 'mse', 'ssim', 'psnr', 'pod', 'sucr', 'csi', 'far'
]


def _load_npy(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Missing file: {path}')
    return np.load(path, allow_pickle=True)


def _load_dict_npy(path: Path) -> dict:
    if not path.exists():
        return {}
    return np.load(path, allow_pickle=True).item()


def _infer_metric_list(saved_metrics: dict, default_metrics: list[str]) -> list[str]:
    if not saved_metrics:
        return default_metrics.copy()
    inferred = [k for k in saved_metrics.keys() if k in _ALLOWED_GLOBAL_METRICS]
    return inferred or default_metrics.copy()


def _format_threshold_tag(threshold: float | None) -> str:
    if threshold is None:
        return 'notset'
    return str(threshold).replace('.', 'p').replace('-', 'm')


def _save_dict(path: Path, value: dict) -> None:
    np.save(path, value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Recompute metrics from a saved/ directory containing preds.npy and trues.npy.'
    )
    parser.add_argument(
        '--saved_dir',
        type=str,
        required=True,
        help='Path to an existing saved directory containing inputs.npy, preds.npy and trues.npy.',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Optional output directory. If omitted, a sibling saved_... directory is created.',
    )
    parser.add_argument(
        '--metric_threshold',
        type=float,
        default=0.5,
        help='Threshold used for POD/FAR/CSI/SUCR. For normalized MNIST-like data, 0.5 is a good default.',
    )
    parser.add_argument(
        '--clip_min',
        type=float,
        default=0.0,
        help='Lower clipping bound used for SSIM/PSNR/SNR/LPIPS evaluation.',
    )
    parser.add_argument(
        '--clip_max',
        type=float,
        default=1.0,
        help='Upper clipping bound used for SSIM/PSNR/SNR/LPIPS evaluation.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        help='Allow overwriting the output directory if it already exists.',
    )

    args = parser.parse_args()

    saved_dir = Path(args.saved_dir).expanduser().resolve()
    if not saved_dir.exists():
        raise FileNotFoundError(f'saved_dir does not exist: {saved_dir}')

    inputs = _load_npy(saved_dir / 'inputs.npy')
    preds = _load_npy(saved_dir / 'preds.npy').astype(np.float32)
    trues = _load_npy(saved_dir / 'trues.npy').astype(np.float32)

    if preds.shape != trues.shape:
        raise ValueError(f'preds and trues must have the same shape, got {preds.shape} vs {trues.shape}')

    existing_metrics = _load_dict_npy(saved_dir / 'metrics.npy')
    existing_pf_metrics = _load_dict_npy(saved_dir / 'per_frame_metrics.npy')

    global_metrics = _infer_metric_list(existing_metrics, _DEFAULT_GLOBAL_METRICS)
    per_frame_metrics = [
        m for m in _infer_metric_list(existing_pf_metrics, _DEFAULT_PER_FRAME_METRICS)
        if m != 'lpips'
    ]

    if args.metric_threshold is None:
        global_metrics = [m for m in global_metrics if m not in {'pod', 'sucr', 'csi', 'far'}]
        per_frame_metrics = [m for m in per_frame_metrics if m not in {'pod', 'sucr', 'csi', 'far'}]

    eval_res, eval_log = metric(
        preds,
        trues,
        mean=None,
        std=None,
        metrics=global_metrics,
        clip_range=[args.clip_min, args.clip_max],
        threshold=args.metric_threshold,
        return_log=True,
    )

    pf_res = per_frame_metric(
        preds,
        trues,
        mean=None,
        std=None,
        metrics=per_frame_metrics,
        clip_range=[args.clip_min, args.clip_max],
        threshold=args.metric_threshold,
    )

    if 'ssim' in pf_res:
        ssim_pf = pf_res['ssim']
        t = len(ssim_pf)
        eval_res['ssim_start'] = float(ssim_pf[0])
        eval_res['ssim_mid'] = float(ssim_pf[t // 2])
        eval_res['ssim_end'] = float(ssim_pf[-1])

    if args.output_dir is None:
        threshold_tag = _format_threshold_tag(args.metric_threshold)
        output_dir = saved_dir.parent / f'{saved_dir.name}_recomputed_thr{threshold_tag}'
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f'Output directory already exists: {output_dir}. Use --overwrite to replace it.'
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / 'inputs.npy', inputs)
    np.save(output_dir / 'preds.npy', preds)
    np.save(output_dir / 'trues.npy', trues)
    _save_dict(output_dir / 'metrics.npy', eval_res)
    _save_dict(output_dir / 'per_frame_metrics.npy', pf_res)

    print('>' * 35 + ' recompute  ' + '<' * 35)
    print(f'Saved results to: {output_dir}')
    if eval_log:
        print(eval_log)
    print('Global metrics:')
    for key in sorted(eval_res.keys()):
        print(f'  {key}: {eval_res[key]}')
    print(f'Per-frame metrics: {", ".join(sorted(pf_res.keys()))}')


if __name__ == '__main__':
    main()
