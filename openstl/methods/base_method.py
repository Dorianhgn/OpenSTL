import numpy as np
import torch
import torch.nn as nn
import os.path as osp
import lightning as l
from torchmetrics.image import StructuralSimilarityIndexMeasure
from openstl.utils import print_log, check_dir
from openstl.core import get_optim_scheduler, timm_schedulers
from openstl.core import metric, per_frame_metric


class Base_method(l.LightningModule):

    def __init__(self, **args):
        print("ARGS METRICS IN INIT:", args.get("metrics"), args.get("metric_threshold"), "metric_threshold_in_keys?", "metric_threshold" in args)
        super().__init__()

        if 'weather' in args['dataname']:
            self.metric_list, self.spatial_norm = args['metrics'], True
            self.channel_names = args.data_name if 'mv' in args['data_name'] else None
        else:
            self.metric_list, self.spatial_norm, self.channel_names = args['metrics'], False, None

        if args.get('metric_threshold') is not None:
            for m in ['pod', 'far', 'csi']:
                if m not in self.metric_list:
                    self.metric_list.append(m)
        if 'lpips' not in self.metric_list:
            self.metric_list.append('lpips')

        self.save_hyperparameters()
        self.model = self._build_model(**args)
        self.criterion = nn.MSELoss()
        self.train_ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.test_ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.test_outputs = []

    def _build_model(self):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, 
            self.hparams.epoch, 
            self.model, 
            self.hparams.steps_per_epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch" if by_epoch else "step"
            },
        }
    
    def lr_scheduler_step(self, scheduler, metric):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def forward(self, batch):
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def _reshape_for_ssim(self, tensor):
        if tensor.ndim == 5:
            batch_size, steps, channels, height, width = tensor.shape
            return tensor.reshape(batch_size * steps, channels, height, width)
        return tensor

    def _compute_ssim(self, pred_y, batch_y, stage):
        metric_fn = getattr(self, f'{stage}_ssim_metric')
        pred_frames = self._reshape_for_ssim(pred_y.detach().float()).clamp(0, 1)
        true_frames = self._reshape_for_ssim(batch_y.detach().float()).clamp(0, 1)
        ssim = metric_fn(pred_frames, true_frames)
        metric_fn.reset()
        return ssim

    def _log_step_metrics(self, stage, metrics, on_step, on_epoch, prog_bar_keys=None, sync_dist=False):
        prog_bar_keys = set() if prog_bar_keys is None else set(prog_bar_keys)
        for key, value in metrics.items():
            if value is None:
                continue
            self.log(
                f'{stage}/{key}', value,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=key in prog_bar_keys,
                sync_dist=sync_dist,
            )

    def _get_metric_threshold(self):
        threshold = self.hparams.get('metric_threshold', None)
        if threshold is not None:
            return threshold

        data_name = str(self.hparams.get('dataname', self.hparams.get('data_name', ''))).lower()
        if 'mnist' in data_name:
            return 128.0
        return 74.0

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)
        metrics = {
            'loss': loss,
            'x_loss': loss,
            'ssim': self._compute_ssim(pred_y, batch_y, stage='val'),
        }
        self._log_step_metrics('val', metrics, on_step=True, on_epoch=True, prog_bar_keys={'loss', 'ssim'}, sync_dist=True)
        return metrics
    
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        outputs = {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}
        self.test_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        results_all = {}
        for k in self.test_outputs[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)
        
        threshold = self._get_metric_threshold()

        # Global metrics (existing)
        eval_res, eval_log = metric(results_all['preds'], results_all['trues'],
            self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list, 
            channel_names=self.channel_names, spatial_norm=self.spatial_norm,
            threshold=threshold)
        
        # Per-frame metrics (new)
        pf_res = per_frame_metric(
            results_all['preds'], results_all['trues'],
            mean=self.hparams.test_mean, std=self.hparams.test_std,
            metrics=self.metric_list, spatial_norm=self.spatial_norm,
            threshold=threshold)

        # Build comprehensive metrics dict
        results_all['metrics'] = eval_res  # full dict with all global metrics
        results_all['per_frame_metrics'] = pf_res  # dict of {metric: array(T,)}

        # Add summary SSIM at start/mid/end if available
        if 'ssim' in pf_res:
            ssim_pf = pf_res['ssim']
            T = len(ssim_pf)
            eval_res['ssim_start'] = float(ssim_pf[0])
            eval_res['ssim_mid'] = float(ssim_pf[T // 2])
            eval_res['ssim_end'] = float(ssim_pf[-1])
            eval_log += f", ssim_start:{ssim_pf[0]}, ssim_mid:{ssim_pf[T // 2]}, ssim_end:{ssim_pf[-1]}"

        self.log_dict({f'test/{k}': float(v) for k, v in eval_res.items()}, sync_dist=True)

        if self.trainer.is_global_zero:
            print_log(eval_log)
            folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

            # Save raw arrays
            for np_data in ['inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])

            # Save global metrics dict
            np.save(osp.join(folder_path, 'metrics.npy'), eval_res)

            # Save per-frame metrics dict
            np.save(osp.join(folder_path, 'per_frame_metrics.npy'), pf_res)

            print_log(f"Saved test results to {folder_path}")
        self.test_outputs = []
        return results_all