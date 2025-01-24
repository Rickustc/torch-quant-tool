import torch




def make_anchors(x, strides, offset=0.5):
    anchors, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=x[i].device, dtype=x[i].dtype) + offset  # shift x
        sy = torch.arange(end=h, device=x[i].device, dtype=x[i].dtype) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchors.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=x[i].dtype, device=x[i].device))
    return torch.cat(anchors), torch.cat(stride_tensor)


class SiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.float_fn = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.float_fn.mul(self.sigmoid(x), x)


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, (k - 1) // 2, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        self.relu = SiLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class Residual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.conv1 = Conv(ch, ch, 3)
        self.conv2 = Conv(ch, ch, 3)
        self.quant = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return self.quant.add(x, y) if self.add_m else y


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch)
        self.conv2 = Conv((2 + n) * out_ch // 2, out_ch)
        self.quant = torch.nn.quantized.FloatFunctional()
        self.res_m = torch.nn.ModuleList(Residual(out_ch // 2, add) for _ in range(n))

    def forward(self, x):
    
        # y = list(self.conv1(x).chunk(2, 1))
        y = [self.conv1(x).chunk(2, 1)]
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(self.quant.cat(y, dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)
        self.quant = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(self.quant.cat([x, y1, y2, self.res_m(y2)], 1))


class DarkNet(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []
        # depth = [1, 2, 2]
        # width = [3, 16, 32, 64, 128, 256]
        # p1/2
        self.p1.append(Conv(width[0], width[1], 3, 2))
        # p2/4
        self.p2.append(Conv(width[1], width[2], 3, 2))
        self.p2.append(CSP(width[2], width[2], depth[0]))
        # p3/8
        self.p3.append(Conv(width[2], width[3], 3, 2))
        self.p3.append(CSP(width[3], width[3], depth[1]))
        # p4/16
        self.p4.append(Conv(width[3], width[4], 3, 2))
        self.p4.append(CSP(width[4], width[4], depth[2]))
        # p5/32
        self.p5.append(Conv(width[4], width[5], 3, 2))
        self.p5.append(CSP(width[5], width[5], depth[0]))
        self.p5.append(SPP(width[5], width[5]))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.fn = torch.nn.quantized.FloatFunctional()
        self.up = torch.nn.Upsample(size=None, scale_factor=2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False)

    def forward(self, p3, p4, p5):
        p4 = self.h1(self.fn.cat([self.up(p5), p4], 1))
        p3 = self.h2(self.fn.cat([self.up(p4), p3], 1))
        p4 = self.h4(self.fn.cat([self.h3(p3), p4], 1))
        p5 = self.h6(self.fn.cat([self.h5(p4), p5], 1))
        return p3, p4, p5


class Head(torch.nn.Module):

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 4  # number of outputs per anchor
        self.stride = torch.zeros(len(ch))  # strides computed during build

        box = max(64, ch[0] // 4)
        cls = max(80, ch[0], self.nc)

        self.fn = torch.nn.quantized.FloatFunctional()

        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, box, 3),
                                                           Conv(box, box, 3),
                                                           torch.nn.Conv2d(box, 4, 1)) for x in ch)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, cls, 3),
                                                           Conv(cls, cls, 3),
                                                           torch.nn.Conv2d(cls, self.nc, 1)) for x in ch)

    def forward(self, p3, p4, p5):
        x = [p3, p4, p5]
        for i, (box, cls) in enumerate(zip(self.box, self.cls)):
            x[i] = self.fn.cat((box(x[i]), cls(x[i])), 1)
        return x


class YOLO(torch.nn.Module):

    def __init__(self, width, depth, num_classes):
        super().__init__()
        self.net = DarkNet(width, depth)
        self.fpn = DarkFPN(width, depth)

        img_dummy = torch.zeros(1, width[0], 256, 256)
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride

    def forward(self, x):
        p3, p4, p5 = self.net(x)
        p3, p4, p5 = self.fpn(p3, p4, p5)
        return self.head(p3, p4, p5)


# class QAT(torch.nn.Module):

#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.quant = torch.quantization.QuantStub()
#         self.de_quant = torch.quantization.DeQuantStub()

#         self.nc = self.model.head.nc
#         self.no = self.model.head.no
#         self.stride = self.model.stride

#     def forward(self, x):
#         x = self.quant(x)
#         x = self.model(x)

#         for i in range(len(x)):
#             x[i] = self.de_quant(x[i])
#         return x


def yolo_v8_n(num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth, num_classes)


def yolo_v8_t(num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 24, 48, 96, 192, 384]
    return YOLO(width, depth, num_classes)


def yolo_v8_s(num_classes: int = 80):
    depth = [1, 2, 2]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth, num_classes)


def yolo_v8_m(num_classes: int = 80):
    depth = [2, 4, 4]
    width = [3, 48, 96, 192, 384, 576]
    return YOLO(width, depth, num_classes)


def yolo_v8_l(num_classes: int = 80):
    depth = [3, 6, 6]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, num_classes)


def yolo_v8_x(num_classes: int = 80):
    depth = [3, 6, 6]
    width = [3, 80, 160, 320, 640, 640]
    return YOLO(width, depth, num_classes)

# model = yolo_v8_n()
# def check_model(model, input):

#     model(input)
# input = torch.rand(32,3, 320, 672)    
# check_model(model,input)

# from torchsummary import summary
# summary(model, input_size=[(3, 320, 672)], batch_size=32, device="cpu")

from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                    Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
                                    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,
                                    RTDETRDecoder, Segment)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
import torch
import torch.nn as nn
from ultralytics.utils.torch_utils import (fuse_conv_and_bn, fuse_deconv_and_bn, initialize_weights, intersect_dicts,
                                           make_divisible, model_info, scale_img, time_sync)
import contextlib
from pathlib import Path
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml

def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale.
    The function uses regular expression matching to find the pattern of the model scale in the YAML file name,
    which is denoted by n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    with contextlib.suppress(AttributeError):
        import re
        return re.search(r'yolov\d+([nslmx])', Path(model_path).stem).group(1)  # n, s, m, l, or x
    return ''

def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg['head'][-1][-2].lower()  # output module name
        if m in ('classify', 'classifier', 'cls', 'fc'):
            return 'classify'
        if m == 'detect':
            return 'detect'
        if m == 'segment':
            return 'segment'
        if m == 'pose':
            return 'pose'

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)

    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model
        for x in 'model.args', 'model.model.args', 'model.model.model.args':
            with contextlib.suppress(Exception):
                return eval(x)['task']
        for x in 'model.yaml', 'model.model.yaml', 'model.model.model.yaml':
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))

        for m in model.modules():
            if isinstance(m, Detect):
                return 'detect'
            elif isinstance(m, Segment):
                return 'segment'
            elif isinstance(m, Classify):
                return 'classify'
            elif isinstance(m, Pose):
                return 'pose'

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if '-seg' in model.stem or 'segment' in model.parts:
            return 'segment'
        elif '-cls' in model.stem or 'classify' in model.parts:
            return 'classify'
        elif '-pose' in model.stem or 'pose' in model.parts:
            return 'pose'
        elif 'detect' in model.parts:
            return 'detect'

    # Unable to determine task from model
    LOGGER.warning("WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
                   "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify', or 'pose'.")
    return 'detect'  # assume detect

def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    import re

    path = Path(path)
    if path.stem in (f'yolov{d}{x}6' for x in 'nsmlx' for d in (5, 8)):
        new_stem = re.sub(r'(\d+)([nslmx])6(.+)?$', r'\1\2-p6\3', path.stem)
        LOGGER.warning(f'WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.')
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r'(\d+)([nslmx])(.+)?$', r'\1\3', str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d['scale'] = guess_model_scale(path)
    d['yaml_file'] = str(path)
    return d


def parse_model(d, ch=3, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in (HGStem, HGBlock):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (Detect, Segment, Pose):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def yolo_model(path):
    from copy import deepcopy
    yaml =  yaml_model_load(path)  # cfg dict
    ch = 3
    ch = yaml['ch'] = yaml.get('ch', ch)
    model,_ = parse_model(deepcopy(yaml), ch=ch, verbose=True)  # model, savelist
    print(model)
    return model