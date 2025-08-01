import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection

from .emp import EMP
from src.metrics import MR, brierMinFDE, minADE, minFDE
from src.utils.optim import WarmupCosLR
from src.utils.submission_av2 import SubmissionAv2


torch.set_printoptions(sci_mode=False)


class Trainer(pl.LightningModule):
    def __init__(
        self,
        dim=128,
        historical_steps=50,
        future_steps=60,
        encoder_depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
        decoder: str = "detr"
    ) -> None:
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        self.history_steps = historical_steps
        self.future_steps = future_steps
        self.submission_handler = SubmissionAv2()

        self.net = EMP(
            embed_dim=dim,
            encoder_depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_path=drop_path,
            decoder=decoder
        )  

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)

        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1),
                "minADE6": minADE(k=6),
                "minFDE1": minFDE(k=1),
                "minFDE6": minFDE(k=6),
                "MR": MR(),
                "brier-minFDE6": brierMinFDE(k=6)
            }
        )
        self.val_metrics = metrics.clone(prefix="val_")
        self.curr_ep = 0
        return


    def getNet(self):
        return self.net


    def forward(self, data):
        return self.net(data)


    def predict(self, data, full=False):
        with torch.no_grad():
            out = self.net(data)
        predictions, prob = self.submission_handler.format_data(
            data, out["y_hat"], out["pi"], inference=True
        )
        predictions = [predictions, out] if full else predictions
        return predictions, prob    


    def cal_loss(self, out, data, batch_idx=0):
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        # y, y_others = data["y"][:, 0], data["y"][:, 1:]
        y, y_others = data["y"], data["y"][:, 1:]
        # print(f"y_hat shape: {y_hat.shape}, pi shape: {pi.shape}, y_hat_others shape: {y_hat_others.shape}")
        # print(f"y shape: {y.shape}, y_others shape: {y_others.shape}")
        # AA
        loss = 0
        B = y_hat.shape[0]
        B_range = range(B)
        N = y_hat.shape[1]
        N_range = range(N)


        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(2), dim=-1).sum(-1)
        # print(f"l2_norm shape: {l2_norm.shape}")
        best_mode = torch.argmin(l2_norm, dim=-1)
        # print(f"best_mode shape: {best_mode.shape}")
        # y_hat_best = y_hat[B_range, best_mode]
        # Create indices for batch and sequence dimensions - REVISAR TODO
        batch_idx = torch.arange(y_hat.shape[0]).unsqueeze(1).expand(-1, y_hat.shape[1])
        seq_idx = torch.arange(y_hat.shape[1]).unsqueeze(0).expand(y_hat.shape[0], -1)
        y_hat_best = y_hat[batch_idx, seq_idx, best_mode]
        # print(f"best_mode shape: {best_mode.shape}, y_hat_best shape: {y_hat_best.shape}, y_hat_best_2:{y_hat_best[..., :2].shape}")
        # print(f"y: {y.shape}")
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)

        # print(f"agent_reg_loss shape: {agent_reg_loss}")
        # print(f"pi shape: {pi.shape}, best_mode shape: {best_mode.shape}")
        # agent_cls_loss = F.cross_entropy(pi, best_mode.detach())
        #REVISAR TODO
        agent_cls_loss = F.cross_entropy(pi.reshape(-1, pi.shape[-1]), best_mode.detach().reshape(-1))
        loss += agent_reg_loss + agent_cls_loss

        # print(f"agent_cls_loss shape: {agent_cls_loss}")

        others_reg_mask = ~data["x_padding_mask"][:, 1:, self.history_steps:]
        others_reg_loss = F.smooth_l1_loss(y_hat_others[others_reg_mask], y_others[others_reg_mask])
        loss += others_reg_loss
        # AA
        return {
            "loss": loss,
            "reg_loss": agent_reg_loss.item(),
            "cls_loss": agent_cls_loss.item(),
            "others_reg_loss": others_reg_loss.item(),
        }

    def training_step(self, data, batch_idx):
        out = self(data)
        losses = self.cal_loss(out, data)
        
        for k, v in losses.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return losses["loss"]

    def validation_step(self, data, batch_idx):
        out = self(data)

        losses = self.cal_loss(out, data, -1)
        #print(f"Output shape: {out['y_hat'].shape}, Data y shape: {data['y'].shape}")
        
        # metrics = self.val_metrics(out, data["y"][:, 0])

        # Aplanar batch y agentes
        pred = out["y_hat"].reshape(-1, 6, 60, 2)  # [B*A, 6, T, 2]
        gt = data["y"].reshape(-1, 60, 2)         # [B*A, T, 2]
        probs = out["pi"].reshape(-1, 6)          # [B*A, 6]

        # Construir outputs para las métricas
        outputs = {"y_hat": pred, "pi": probs}

        # Evaluar métricas
        metrics = self.val_metrics(outputs, gt)

        # pred = out["y_hat"].reshape(-1, 6, 60, 2)    # (96×48, 6, 60, 2)
        # gt = data["y"].reshape(-1, 60, 2)           # (96×48, 60, 2)

        # metrics = self.val_metrics(pred, gt)

        self.log(
            "val/reg_loss",
            losses["reg_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        for k in self.val_scores.keys(): self.val_scores[k].append(metrics[k].item())

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        # print(f"Metrics: {metrics}")

    def on_test_start(self) -> None:
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)

    def test_step(self, data, batch_idx) -> None:
        out = self(data)
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def on_validation_start(self) -> None:
        self.val_scores = {"val_MR": [], "val_minADE1": [], "val_minADE6": [], "val_minFDE1": [], "val_minFDE6": [], "val_brier-minFDE6": []}

    def on_validation_end(self) -> None:      
        print( " & ".join( ["{:5.3f}".format(np.mean(self.val_scores[k])) for k in ["val_MR", "val_minADE6", "val_minFDE6", "val_brier-minFDE6"]] ) )
        self.curr_ep += 1

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
            nn.GRUCell
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
            nn.Parameter
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]
