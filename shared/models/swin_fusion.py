#!/usr/bin/env python3
"""
SwinFusionModel — 4-stream multimodal model for antibiotic susceptibility testing.

Streams:
  1. original_tif  -> Swin Transformer (swin_tiny_patch4_window7_224)
  2. mask_overlay   -> ResNet18
  3. flow_tif       -> ResNet18
  4. tabular stats  -> MLP (LazyLinear)

Fusion: concatenation -> LayerNorm -> Dropout -> Linear(1)
"""

import torch
import torch.nn as nn
import timm


class SwinFusionModel(nn.Module):
    def __init__(self, swin_name="swin_tiny_patch4_window7_224", cnn_name="resnet18",
                 drop=0.2, pretrained=True, freeze_backbones=False):
        super().__init__()
        self.swin = timm.create_model(swin_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        d_swin = self.swin.num_features
        self.cnn_overlay = timm.create_model(cnn_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        d_ov = self.cnn_overlay.num_features
        self.cnn_flow = timm.create_model(cnn_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        d_flow = self.cnn_flow.num_features
        if freeze_backbones:
            for p in self.swin.parameters(): p.requires_grad = False
            for p in self.cnn_overlay.parameters(): p.requires_grad = False
            for p in self.cnn_flow.parameters(): p.requires_grad = False

        d_tab = 128
        self.tab = nn.Sequential(
            nn.LazyLinear(d_tab), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(d_tab, d_tab), nn.ReLU(inplace=True)
        )

        d_all = d_swin + d_ov + d_flow + d_tab
        self.head = nn.Sequential(nn.LayerNorm(d_all), nn.Dropout(drop), nn.Linear(d_all, 1))

    def forward(self, x_orig, x_overlay, x_flow, x_stats):
        f_orig    = self.swin(x_orig)
        f_overlay = self.cnn_overlay(x_overlay)
        f_flow    = self.cnn_flow(x_flow)
        f_tab     = self.tab(x_stats)
        h = torch.cat([f_orig, f_overlay, f_flow, f_tab], dim=1)
        return self.head(h).squeeze(1)
