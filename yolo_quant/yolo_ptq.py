import argparse
import os
import sys
from pathlib import Path
import warnings
import yaml
import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx,prepare_qat_fx, convert_fx,_convert_fx
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
from tqdm import tqdm
import numpy as np

from ultralytics import YOLO
from utils.dataloaders import create_dataloader
from utils.general import (check_img_size, check_yaml, file_size, colorstr, check_dataset)
from utils.torch_utils import select_device
import torch.fx
import torch.fx.wrap
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils import LOGGER, ops
from utils.dataloaders import create_dataloader, get_dataloader
################################################### pretrain yolo model #######################################################

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="/home/wrq/yolo/datasets/coco.yaml", help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default="yolov8n.pt", help='model.pt path(s)')
    parser.add_argument('--model-name', '-m', default='yolov8n', help='model name: default yolov5s')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='cuda:1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')

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

def evaluate_ptq_accuracy(yolo, val_loader):
    # set eval mode for model infer
    # in case that dropout/BN should be eval mode during infer
    '''int model eval should be on CPU'''
    yolo.eval()
    metrics = DetMetrics(save_dir="home/wrq/yolo")
    def preprocess(batch,device="cpu"):
        """Preprocesses batch of images for GPU training and evaluation"""
        batch['img'] = batch['img'].float() / 255
        # for k in ['batch_idx', 'cls', 'bboxes']:
        #     batch[k] = batch[k].to(device)
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
        Quantized tensor can not be execute on GPU,so device should be CPU
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

    pbar = tqdm(val_loader, desc="PTQ val")
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

def load_model():
    yolo = YOLO("/home/wrq/yolo/yolov8n.yaml",task='detect')   # load a pretrained model (recommended for training)
    # yolo = YOLO("/home/wrq/yolo/yolov8n.pt")
    model = yolo.model
    
    def replace_silu_with_relu(module):
        for name, child in module.named_children():
            if isinstance(child, nn.SiLU):
                setattr(module, name, nn.ReLU())
            else:
                replace_silu_with_relu(child)
                
    replace_silu_with_relu(model)
            
    
    checkpoint = torch.load("/home/wrq/yolo/yolov8n.pt")
    state_dict =  checkpoint["model"].state_dict()
    model.load_state_dict(state_dict, strict=False)
    model.to('cuda:1')
    model.eval()
    return model, yolo    # return yolo for loss and eval funvtion

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

def calibrate_model(model, calib_loader, num_batches, device):
    """
        Feed data to the network and calibrate.
        Arguments:
            model: detection model
            data_loader: calibration data set
            num_calib_batch: amount of calibration passes to perform
    """
    # model should be eval during infer
    model.eval()
    total=len(calib_loader)
    pbar = tqdm(calib_loader, desc="PTQ Cali",total=total)
    
    def preprocess_batch(batch,device=device):
        """Preprocesses batch of images for GPU training and evaluation"""
        batch['img'] = batch['img'].float() / 255
        batch['img'] = batch['img'].to(device)
        for k in ['batch_idx', 'cls', 'bboxes']:
            batch[k] = batch[k].to(device)
        return batch

    with torch.no_grad():
       for i, batch in enumerate(pbar):
            batch = preprocess_batch(batch)  
            model(batch["img"])
            return 
            # if i >= num_batches:
            #     break
            
def check_fp_model(model,data_loader, num_batches):
    model.eval()
    model.to("cuda:1")
    with torch.no_grad():
        for i, (image, targets, paths, shapes) in tqdm(enumerate(data_loader), total=num_batches):
            image = image.to("cuda:1",non_blocking=True)
            image = image.float()  # uint8 to fp16/32
            image /= 255.0  # 0 - 255 to 0.0 - 1.0
            model(image)
            if i >= num_batches:
                break
        print("original model can be excuted successfully")

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
    # val dataloader
    valset = "/data1/data/coco/val2017"
    val_loader = get_dataloader(model, valset, batch_size=32, rank=-1, mode='val')
    return val_loader


if __name__ == "__main__":
    import copy
    opt = parse_opt()
    print(opt.device)
    
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
    fp_model, data_loader_calib,yolo = prepare_model_dataloader(calibrator=opt.calibrator, opt=opt, device=opt.device)
    # val_loader for yolov8 eval
    val_loader = prepare_data_loader(fp_model, opt=opt, device=opt.device)
    
    # prepare and convert will do inplace,so fp_model should be preserved
    model = copy.deepcopy(fp_model)
       
    # check_model(model, data_loader_calib, num_batches=10)
    
    example_inputs = (next(iter(data_loader_calib))[0])
    example_inputs = example_inputs.float()  # uint8 to fp16/32
    example_inputs /= 255.0 
   
    prepared = prepare_fx(model, qconfig_mapping,
                                    example_inputs=(example_inputs),
                                    backend_config= get_tensorrt_backend_config_dict()   # get_qnnpack_backend_config()    
                                    )
  
    calibrate_model(prepared, val_loader, num_batches=100, device = opt.device)

    prepared.to("cpu")
    # evaluate_ptq_accuracy(prepared,val_loader)
    
    quantized_fx = _convert_fx(prepared, 
                is_reference=False,  # 选择reference模式
                backend_config= get_tensorrt_backend_config_dict()   # get_qnnpack_backend_config() 
                ) 
    with torch.no_grad():
        fp_output = fp_model(example_inputs.to(opt.device))
        
    torch.onnx.export(quantized_fx,example_inputs,"yolov8_int8_ptq_relu.onnx",opset_version=14)
    q_output = quantized_fx(example_inputs)
    
    # with open("output1.txt", 'a') as file:
    #     file.write(', '.join(map(str, fp_output)))
    
    # with open("output2.txt", 'a') as file:
    #     file.write(', '.join(map(str, q_output)))
        

    # print(quantized_fx)
    # # evaluate_fp_accuracy(fp_model,val_loader)
    # print(quantized_fx.code)
    
    # import torch.quantization._numeric_suite as ns
    # import pdb
    # pdb.set_trace()
    # evaluate_ptq_accuracy(quantized_fx,val_loader)
    # torch.onnx.export(quantized_fx, image_tensor, exported_model_path, verbose=False, input_names=["images"], output_names=["pred"], opset_version=13)
