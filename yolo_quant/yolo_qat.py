import argparse
import os
import sys
from pathlib import Path
import warnings
import yaml
import copy
from copy import deepcopy
import re
from typing import Dict

import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx,prepare_qat_fx, convert_fx,_convert_fx,convert_to_reference_fx,_prepare_fx
from torch.ao.quantization import QConfigMapping
import torch.ao as ao
import torchvision.transforms as transforms
import torch.nn as nn
from torch.ao.quantization.backend_config import (
    BackendConfig,
    get_native_backend_config,
    # get_evas_backend_config,
    get_tensorrt_backend_config_dict,
    get_qnnpack_backend_config
)
from torch.ao.quantization.observer import *
import torch.fx
import torch.fx.wrap
from torch.ao.quantization.qconfig import default_symmetric_qnnpack_qat_qconfig,get_default_qat_qconfig,default_per_channel_symmetric_qnnpack_qat_qconfig
import torch.nn.functional as F
import torch
import torch.fx
from torch.fx.node import Node


import numpy as np
from tqdm import tqdm
import logging
from accelerate import Accelerator

from ultralytics import YOLO
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils import RANK, colorstr
# from ultralytics.utils.loss import v8ClassificationLoss, v8DetectionLoss, v8PoseLoss, v8SegmentationLoss
from utils.dataloaders import create_dataloader, get_dataloader
from utils.general import (check_img_size, check_yaml, file_size, colorstr, check_dataset)
from utils.torch_utils import select_device
from utils.general import (LOGGER, check_img_size, check_yaml, file_size, colorstr, print_args, check_dataset, check_img_size, colorstr, init_seeds)
from yolov8 import yolo_model
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou

"""QAT:
prepare finetune dataset
finetune

#nn.op --> nnqat.op --> nnq.op 
"""
def get_log(log_name):
    logger = logging.getLogger(log_name)  # 设定logger的名字
    logger.setLevel(logging.INFO)  # 设定logger得等级

    ch = logging.StreamHandler()  # 输出流的hander，用与设定logger的各种信息
    ch.setLevel(logging.INFO)  # 设定输出hander的level

    fh = logging.FileHandler(log_name, mode='a')  # 文件流的hander，输出得文件名称，以及mode设置为覆盖模式
    fh.setLevel(logging.INFO)  # 设定文件hander得lever



    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)  # 两个hander设置个是，输出得信息包括，时间，信息得等级，以及message
    fh.setFormatter(formatter)
    logger.addHandler(fh)  # 将两个hander添加到我们声明的logger中去
    logger.addHandler(ch)
    return logger

train_logger = get_log("/home/wrq/yolo/train_resnet50_QAT")
test_logger = get_log("/home/wrq/yolo/test_resnet50_QAT")

class BboxLoss(nn.Module):

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl
    
    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)

class DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model,device):  # model must be de-paralleled

        # h = model.args  # hyperparameters      
        m = model.model.model[-1]  # Detect() module

        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        # self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch_idx,cls,bboxes):

        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]

        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch_idx.view(-1, 1), cls.view(-1, 1), bboxes), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        # loss[0] *= self.hyp.box  # box gain
        # loss[1] *= self.hyp.cls  # cls gain
        # loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

#TODO: split infer 
def predict_forward(model,x):
        """
        Perform a forward pass through the GraphModulenetwork.
        """
        shape = x[0].shape
        no = 144
        x = model(x)  # run
        reg_max  =16
        nc = 80
        x_cat = torch.cat([xi.view(shape[0], no, -1) for xi in x], 2)
        box, cls = x_cat.split((reg_max * 4, nc), 1)
        dbox = dist2bbox(dfl(box), anchors.unsqueeze(0), xywh=True, dim=1) * strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return (y, x)
    
def evaluate_fp_accuracy(yolo, val_loader):
    # set eval mode for model infer
    # in case that dropout/BN should be eval mode during infer
    yolo.eval()
    metrics = DetMetrics(save_dir="home/wrq/yolo")
    def preprocess(batch,device="cuda:1"):
        """Preprocesses batch of images for GPU training and evaluation"""
        batch['img'] = batch['img'].float() / 255
        batch['img'] = batch['img'].to(device)
        for k in ['batch_idx', 'cls', 'bboxes']:
            batch[k] = batch[k].to(device)
        return batch
    
    def postprocess(preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(preds,
                                       0.001,   # conf_thres default 0.25 predict, 0.001 val 
                                       0.7,
                                       multi_label=True,
                                    #    agnostic=args.single_cls,
                                       max_det=300)
        
    def match_predictions(pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().detach().numpy()
        for i, threshold in enumerate(iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands
                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)
        
    def _process_batch(detections, labels):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """

        iou = box_iou(labels[:, 1:], detections[:, :4])
        return match_predictions(detections[:, 5], labels[:, 0], iou)
    
    def update_metrics(preds, batch, stats):
        """Metrics."""
        
        iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        seen = 0
        for si, pred in enumerate(preds):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
    
            shape = batch['ori_shape'][si]
            correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool, device="cuda:1")  # init
            seen += 1
            if npr == 0:
                if nl:
                    stats.append((correct_bboxes, *torch.zeros((2, 0), device="cuda:1"), cls.squeeze(-1)))
                continue
            # Predictions
            predn = pred.clone().to("cuda:1")
            ops.scale_boxes(batch['img'][si].shape[1:], predn[:, :4], shape,
                            ratio_pad=batch['ratio_pad'][si])  # native-space pred
            # Evaluate
            if nl:
                height, width = batch['img'].shape[2:]
                tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                    (width, height, width, height), device="cuda:1")  # target boxes
                ops.scale_boxes(batch['img'][si].shape[1:], tbox, shape,
                                ratio_pad=batch['ratio_pad'][si])  # native-space labels
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                correct_bboxes = _process_batch(predn, labelsn)
            stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)
            
    def get_map(stats,metrics):
        """Returns metrics statistics and results dictionary."""
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
        # stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
        
        if len(stats) and stats[0].any():
            metrics.process(*stats)
        # nt_per_class = np.bincount(stats[-1].astype(int), minlength=self.nc)  # number of targets per class
        map50 = metrics.results_dict['metrics/mAP50(B)']
        map = metrics.results_dict['metrics/mAP50-95(B)']
        return map50, map
        # return metrics.results_dict

    pbar = tqdm(val_loader, desc="FP val")
    stats = []
    # cnt = 0  # for debug
    with torch.no_grad():
        for batch_i, batch in enumerate(pbar): 
            # Preprocess
            
            batch = preprocess(batch)
            # Inference
            preds = yolo(batch['img'])
            # Postprocess
            preds = postprocess(preds)
            update_metrics(preds, batch,stats)

    map50, map95 = get_map(stats,metrics)
    print('Torch evaluation:', "mAP@IoU=0.50:{:.5f}, mAP@IoU=0.50:0.95:{:.5f}".format(map50, map95))    
    
def evaluate_ptq_accuracy(yolo, val_loader):
    yolo.eval()
    metrics = DetMetrics(save_dir="home/wrq/yolo")
    def preprocess(batch,device="cpu"):
        """Preprocesses batch of images for YOLO training."""
        batch['img'] = batch['img'].float() / 255
        for k in ['batch_idx', 'cls', 'bboxes']:
            batch[k] = batch[k].to(device)

        return batch
    
    def postprocess(preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(preds,
                                       0.001,   # conf_thres default 0.25 predict, 0.001 val 
                                       0.7,
                                       multi_label=True,
                                    #    agnostic=args.single_cls,
                                       max_det=300)
        
    def match_predictions(pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands
                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)
        
    def _process_batch(detections, labels):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """

        iou = box_iou(labels[:, 1:], detections[:, :4])
        return match_predictions(detections[:, 5], labels[:, 0], iou)
    
    def update_metrics(preds, batch, stats):
        """Metrics."""
        
        iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        seen = 0
        
        for si, pred in enumerate(preds):
   
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
    
            shape = batch['ori_shape'][si]
            correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool, device="cpu")  # init
            seen += 1
            if npr == 0:
                if nl:
                    stats.append((correct_bboxes, *torch.zeros((2, 0), device="cpu"), cls.squeeze(-1)))
                continue
            # Predictions
            predn = pred.clone().to("cpu")
            ops.scale_boxes(batch['img'][si].shape[1:], predn[:, :4], shape,
                            ratio_pad=batch['ratio_pad'][si])  # native-space pred
            # Evaluate
            if nl:
                height, width = batch['img'].shape[2:]
                tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                    (width, height, width, height), device="cpu")  # target boxes
                ops.scale_boxes(batch['img'][si].shape[1:], tbox, shape,
                                ratio_pad=batch['ratio_pad'][si])  # native-space labels
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                correct_bboxes = _process_batch(predn, labelsn)
            
            stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)
            
    def get_map(stats,metrics):
            
        """Returns metrics statistics and results dictionary."""

        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
        # stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
        
        if len(stats) and stats[0].any():
            metrics.process(*stats)
        # nt_per_class = np.bincount(stats[-1].astype(int), minlength=self.nc)  # number of targets per class
    
        map50 = metrics.results_dict['metrics/mAP50(B)']
        map = metrics.results_dict['metrics/mAP50-95(B)']
        return map50, map
        # return metrics.results_dict

    pbar = tqdm(val_loader, desc="QAT val")
    stats = []
    # cnt = 0  # for debug
    for batch_i, batch in enumerate(pbar): 
        # Preprocess
        batch = preprocess(batch)
        # Inference
        preds = yolo(batch['img'])
        # Postprocess
        preds = postprocess(preds)
        update_metrics(preds, batch,stats)

    map50, map95 = get_map(stats,metrics)
    print('Torch evaluation:', "mAP@IoU=0.50:{:.5f}, mAP@IoU=0.50:0.95:{:.5f}".format(map50, map95))

def evaluate_qat_accuracy(yolo, val_loader):
    # bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
    yolo.eval()
    metrics = DetMetrics(save_dir="home/wrq/yolo")
    def preprocess(batch,device="cpu"):
        """Preprocesses batch of images for YOLO training."""
        batch['img'] = batch['img'].float() / 255
        batch['img'] = batch['img'].to(device)
        for k in ['batch_idx', 'cls', 'bboxes']:
            batch[k] = batch[k].to(device)

        return batch
    
    def postprocess(preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(preds,
                                       0.001,   # conf_thres default 0.25 predict, 0.001 val 
                                       0.7,
                                       multi_label=True,
                                    #    agnostic=args.single_cls,
                                       max_det=300)
        
    def match_predictions(pred_classes, true_classes, iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        correct = np.zeros((pred_classes.shape[0], iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands
                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)
        
    def _process_batch(detections, labels):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """

        iou = box_iou(labels[:, 1:], detections[:, :4])
        return match_predictions(detections[:, 5], labels[:, 0], iou)
    
    def update_metrics(preds, batch, stats):
        """Metrics."""
        
        iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        seen = 0
        
        for si, pred in enumerate(preds):
   
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
    
            shape = batch['ori_shape'][si]
            correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool, device="cpu")  # init
            seen += 1
            if npr == 0:
                if nl:
                    stats.append((correct_bboxes, *torch.zeros((2, 0), device="cpu"), cls.squeeze(-1)))
                continue
            
            # Predictions
            predn = pred.clone().to("cpu")
            ops.scale_boxes(batch['img'][si].shape[1:], predn[:, :4], shape,
                            ratio_pad=batch['ratio_pad'][si])  # native-space pred

            # Evaluate
            if nl:
                height, width = batch['img'].shape[2:]
                tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                    (width, height, width, height), device="cpu")  # target boxes
                ops.scale_boxes(batch['img'][si].shape[1:], tbox, shape,
                                ratio_pad=batch['ratio_pad'][si])  # native-space labels
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                correct_bboxes = _process_batch(predn, labelsn)
            
            stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)
            
    def get_map(stats,metrics):
        """Returns metrics statistics and results dictionary."""
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            metrics.process(*stats)
        # nt_per_class = np.bincount(stats[-1].astype(int), minlength=self.nc)  # number of targets per class
        map50 = metrics.results_dict['metrics/mAP50(B)']
        map = metrics.results_dict['metrics/mAP50-95(B)']
        return map50, map
        
    def finalize_metrics(nt_per_class):
        """Set final values for metrics speed and confusion matrix."""
        speed = {'preprocess': 0.0, 'inference': 0.0, 'loss': 0.0, 'postprocess': 0.0}
        metrics.speed = speed
        
        confusion_matrix = ConfusionMatrix(nc=80, conf=0.001)
        metrics.confusion_matrix = confusion_matrix
            
    # def print_results():
    #     """Prints training/validation set metrics per class."""
    #     pf = '%22s' + '%11i' * 2 + '%11.3g' * len(metrics.keys)  # print format
    #     # LOGGER.info(pf % ('all', self.seen, self.nt_per_class.sum(), *metrics.mean_results()))
    #     # Print results per class
    #     for i, c in enumerate(metrics.ap_class_index):
    #         LOGGER.info(pf % (self.names[c], self.seen, self.nt_per_class[c], *metrics.class_result(i)))

    pbar = tqdm(val_loader, desc="QAT val")
    stats = []
    # cnt = 0  # for debug
    for batch_i, batch in enumerate(pbar): 
        # Preprocess
        batch = preprocess(batch)
        # Inference
        preds = yolo(batch['img'])
        # Postprocess
        preds = postprocess(preds)
        update_metrics(preds, batch,stats)

    map50, map95 = get_map(stats,metrics)
    print('Torch evaluation:', "mAP@IoU=0.50:{:.5f}, mAP@IoU=0.50:0.95:{:.5f}".format(map50, map95))
    # finalize_metrics(nt_per_class)
    # print_results()

def is_int8_weight(model):
    """Check whether the input model is a int8 model.

    Args:
        model (torch.nn.Module): input model

    Returns:
        result(bool): Return True if the input model is a int8 model.
    """
    def _is_int8_value(value):
        """Check whether the input tensor is a qint8 tensor."""
        if hasattr(value, 'dtype') and 'int8'  in str(value.dtype):
            return True
        else:
            return False

    stat_dict = dict(model.state_dict())
    for name, value in stat_dict.items():
        if _is_int8_value(value):
            return True
        # value maybe a tuple, such as 'linear._packed_params._packed_params'
        if isinstance(value, tuple):
            for v in value:
                if _is_int8_value(v):
                    return True
    return False

def is_int8_activation(model):
    #check activation int8

    input_data = torch.randn(1, 3, 256, 256)
    # hook for getting the output shape ofeach layer
    def print_layer_output(module, input, output):
        # print(f"Output of {module.__class__.__name__}: {output.dtype}")
        assert output.dtype==torch.qint8, "activation should be int8"
        
    hooks = []
    for layer in model.named_children():
    
        hook = layer[1].register_forward_hook(print_layer_output)
        hooks.append(hook)
        if isinstance(layer, nn.Module):
            hook = layer.register_forward_hook(print_layer_output)
            hooks.append(hook)
    output = model(input_data)
    for hook in hooks:
        hook.remove()
    return True

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="/home/wrq/yolo/datasets/coco.yaml", help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default="yolov8n.pt", help='model.pt path(s)')
    parser.add_argument('--model-name', '-m', default='yolov8n', help='model name: default yolov5s')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--hyp', type=str, default="/home/wrq/anaconda3/envs/quant/lib/python3.9/site-packages/ultralytics/cfg/default.yaml", help='hyperparameters path')
    # train/val/export config
    
    # setting for calibration
    parser.add_argument('--calib-batch-size', type=int, default=32, help='calib batch size: default 64')
    # parser.add_argument('--sensitive-layer', default=[], help='skip sensitive layer: default detect head')

    parser.add_argument('--sensitive-layer', default=['model.15.cv1.conv',
                                                      'model.15.cv2.conv',
                                                      "model.15.m.0.cv1.conv",
                                                      "model.15.m.0.cv2.conv"], help='skip sensitive layer: default detect head')

    parser.add_argument('--num-calib-batch', default=32, type=int,
                        help='Number of batches for calibration. 0 will disable calibration. (default: 4)')
    parser.add_argument('--calibrator', type=str, choices=["max", "histogram"], default="max")
    parser.add_argument('--percentile', nargs='+', type=float, default=[99.9, 99.99, 99.999, 99.9999])
    parser.add_argument('--dynamic', default=False, help='dynamic ONNX axes')
    parser.add_argument('--simplify', default=True, help='simplify ONNX file')
    # parser.add_argument('--out-dir', '-o', default=ROOT / 'weights/', help='output folder: default ./runs/finetune')
    parser.add_argument('--batch-size-onnx', type=int, default=1, help='batch size for onnx: default 1')

    opt = parser.parse_args()
    # opt.data = check_yaml(opt.data)  # check YAML
    # print_args(vars(opt))
    return opt

def load_model():
    yolo = YOLO("/home/wrq/yolo/yolov8n.yaml",task='detect')   # load a pretrained model (recommended for training)
    # yolo = YOLO("/home/wrq/yolo/yolov8n.pt")
    model = yolo.model
    checkpoint = torch.load("/home/wrq/yolo/yolov8n.pt")
    state_dict =  checkpoint["model"].state_dict()
    model.load_state_dict(state_dict, strict=False)
    # model.to('cuda')
    model.eval()
    return model, yolo    # return yolo for loss and eval funvtion

def prepare_val_loader(model, opt, hyp):
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    with open(opt.data, encoding='utf-8') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  
        # data_dict = check_dataset(data_dict)
        
    train_path = data_dict['train']
    val_path = data_dict['val']
    calib_path = data_dict['val']
    
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints
    
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # gs = 32
    imgsz, _ = [check_img_size(x, gs) for x in [opt.imgsz, opt.imgsz]]  # verify imgsz are gs-multiples
    # val dataloader
    valset = "/data1/data/coco/val2017"
    val_loader = get_dataloader(model, valset, batch_size=32, rank=-1, mode='val')
    return val_loader

def prepare_model_dataloader(calibrator, opt, device):
    # yolov5 style dataloader
    with open(opt.data, encoding='utf-8') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  
        # print(data_dict)
        # data_dict = check_dataset(data_dict)
    calib_path = data_dict['val']
    model,yolo = load_model()
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # gs = 32
    imgsz, _ = [check_img_size(x, gs) for x in [opt.imgsz, opt.imgsz]]  # verify imgsz are gs-multiples
    # Calib dataloader
    calib_loader = create_dataloader(calib_path,
                                     opt.batch_size,
                                     imgsz,
                                     gs,
                                     hyp=None,
                                     cache=opt.cache,
                                     rect=True,
                                     rank=-1,
                                     workers=opt.workers * 2,
                                     pad=0.5,
                                     prefix=colorstr('calib: '))[0]
    return model, calib_loader,yolo

def prepare_data_loader(model, opt, device):
    
    # with open(hyp, errors='ignore') as f:
    #     hyp = yaml.safe_load(f)  # load hyps dict

    with open(opt.data, encoding='utf-8') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  
        # data_dict = check_dataset(data_dict)
        
    train_path = data_dict['train']
    test_path = data_dict['val']
    calib_path = data_dict['val']
    
    # LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    # opt.hyp = hyp.copy()  # for saving hyps to checkpoints
    
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # gs = 32
    imgsz, _ = [check_img_size(x, gs) for x in [opt.imgsz, opt.imgsz]]  # verify imgsz are gs-multiples
    # Train dataloader
    trainset = "/data1/data/coco/train2017"
    valset = "/data1/data/coco/val2017"
    train_loader = get_dataloader(model, trainset, batch_size=32, rank=-1, mode='train')
    val_loader = get_dataloader(model, valset, batch_size=32, rank=-1, mode='val')
    
    return train_loader, val_loader

def calibrate_model(model,  data_loader, num_batches, device):
    """
        Feed data to the network and calibrate.
        Arguments:
            model: detection model
            data_loader: calibration data set
            num_calib_batch: amount of calibration passes to perform
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        for i, (image, targets, paths, shapes) in tqdm(enumerate(data_loader), total=num_batches):
            image = image.to(device,non_blocking=True)
            image = image.float()  # uint8 to fp16/32
            image /= 255.0  # 0 - 255 to 0.0 - 1.0
            
            model(image)
            if i >= num_batches:
                break

def check_model(model,data_loader, num_batches):
    model.eval()
    model.to("cuda")
    with torch.no_grad():
        for i, (image, targets, paths, shapes) in tqdm(enumerate(data_loader), total=num_batches):
            image = image.to("cuda",non_blocking=True)
            image = image.float()  # uint8 to fp16/32
            image /= 255.0  # 0 - 255 to 0.0 - 1.0
            model(image)
            if i >= num_batches:
                break
        print("original model can be excuted successfully")
        
def _is_observer_script_module(mod, obs_type_name):
    """Returns true if given mod is an instance of Observer script module."""
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        # qualified name looks like '__torch__.torch.ao.quantization.observer.___torch_mangle_2.MinMaxObserver'
        suffix = mod._c.qualified_name.split(".", 1)[1]
        name = re.sub(r"\.___torch_mangle_\d+", "", suffix)
        return obs_type_name in name
    return False
        
def _is_activation_post_process(module):
    return (
        isinstance(module, (torch.ao.quantization.ObserverBase,
                            torch.ao.quantization.FakeQuantizeBase)) or _is_observer_script_module(module, "quantization.observer")
    )
def train_one_epoch(model, train_loader,criterion, optimizer, accelerator,nepoch,ntrain_batches=20,train_layers=None):
    # accelerator for multi gpu train
    # accelerator do not need specify a device
    total=len(train_loader)
    model, train_loader, optimizer = \
            accelerator.prepare(model, train_loader, optimizer)
    #set model.train for training 
    model.train()  
    pbar = tqdm(train_loader, desc="QAT Train",total=total)

    def preprocess_batch(batch,device="cuda"):
        """Preprocesses batch of images for GPU training and evaluation
           no need device set for accelerate
        """
        batch['img'] = batch['img'].float() / 255
        # batch['img'] = batch['img'].to(device)
        # for k in ['batch_idx', 'cls', 'bboxes']:
        #     batch[k] = batch[k].to(device)
        return batch
    

    
    for i, batch in enumerate(pbar):
        #no need device set for accelerate
        batch = preprocess_batch(batch)            
        # from torch.fx.passes.shape_prop import ShapeProp
        # ShapeProp(qat_model).propagate(batch["img"])
        img = batch["img"]
        batch_idx = batch["batch_idx"]
        cls = batch["cls"]
        bboxes = batch["bboxes"]
        pred= model(img)
        loss, _ = criterion(pred,batch_idx,cls,bboxes)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        train_logger.info("epoch:{} loss:{}".format(nepoch,loss))
        if i>3000:
            break
    return 
    
def train_model(prepared_model,yolo,train_loader,opt, val_loader, epoch,qconfig_mapping):
    # QAT takes time and one needs to train over a few epochs.
    # Train and check accuracy after each epoch
    # define training setting
    accelerator = Accelerator(even_batches=False)
    train_logger.info(accelerator.print(f'device {str(accelerator.device)} is used!'))
    criterion = DetectionLoss(model=yolo,device =accelerator.device )
    optimizer = torch.optim.Adam(prepared_model.parameters(), 1e-5)
    nepochs = epoch
    num_eval_batches=-1
    lrschedule = None
    if lrschedule is None:
        lrschedule = {
            0: 1e-4,
            5: 1e-5,
            20: 1e-6
        }
    # set backend_config for train                  
    backend_config=torch.ao.quantization.backend_config.qnnpack.get_qnnpack_backend_config()

    for nepoch in range(nepochs):
        train_logger.info("Training epoch: [{}/{}]\n".format(nepoch,nepochs))
        if nepoch in lrschedule:
            learningrate = lrschedule[nepoch]
            for g in optimizer.param_groups:
                g["lr"] = learningrate
        train_one_epoch(prepared_model, train_loader,criterion,optimizer,accelerator,nepoch)
        # Check the accuracy after each epoch
        # nnqat.op --> nnq.op 
        tmp_model = copy.deepcopy(prepared_model)
        tmp_model.to("cpu")
        convert_model = convert_fx(tmp_model,qconfig_mapping=qconfig_mapping,backend_config=backend_config)
        # check wether weight is int8
        assert is_int8_weight(convert_model)==True, "The exported model is not INT8 weight, "\
        "please reset 'dtype' to 'FP32' or check your model."
        # check wether activation is int8
        assert is_int8_activation(convert_model)==True, "The exported model is not INT8 activation, "\
        "please reset 'dtype' to 'FP32' or check your model."
        test_logger.info("Convert QAT_model to int_model Done...\n")
        test_logger.info("Epoch {} Evaluation\n".format(nepoch))
        with torch.no_grad():
            map50_calibrated_qat, map_calibrated_qat = evaluate_qat_accuracy(convert_model,val_loader)
            # check int8 model acc
            test_logger.info('QAT evaluation:', "mAP@IoU=0.50:{:.5f}, mAP@IoU=0.50:0.95:{:.5f}".format(map50_calibrated_qat, map_calibrated_qat))
        # TODO: return best_model
    return convert_model

if __name__ == "__main__":

    q_backend = "qnnpack"  # qnnpack  or fbgemm
    torch.backends.quantized.engine = q_backend

    qconfig = torch.ao.quantization.qconfig.QConfig(
        activation=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
        weight=torch.ao.quantization.observer.default_per_channel_weight_observer
    )
    qconfig_linear = torch.ao.quantization.qconfig.QConfig(
        activation=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
        weight=torch.ao.quantization.observer.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)   # qnnpakc not support perchannel for linear oop now
    )
    qconfig_mapping = (QConfigMapping()
        .set_global(qconfig)  
        .set_object_type(torch.nn.Linear, qconfig_linear)    
    )

    opt = parse_opt()
    train_logger.info(opt.device)
    
    fp_model, data_loader_calib,yolo = prepare_model_dataloader(calibrator=opt.calibrator, opt=opt, device=opt.device)
    train_loader,val_loader = prepare_data_loader(fp_model, opt=opt, device=opt.device)
    
    model = copy.deepcopy(fp_model)
    # check_model(model, data_loader_calib, num_batches=10)
    example_inputs = (next(iter(data_loader_calib))[0])
    example_inputs = example_inputs.float()  # uint8 to fp16/32
    example_inputs /= 255.0 
   
    prepared = prepare_qat_fx(model, qconfig_mapping,
                                example_inputs=(example_inputs),
                                backend_config= get_tensorrt_backend_config_dict()   # get_qnnpack_backend_config()    
                                )
    # set QAT epoch
    epoch=1 
    train_logger.info("Prepare QAT Model Done...")
    # train_logger.info(print(prepared))
 
    train_logger.info("Training QAT Model...")
    int8_model = train_model(prepared,yolo,train_loader, opt,val_loader, epoch,qconfig_mapping)
    

