import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from PIL import Image  # for image loading/processing

try:
    import torchvision.models as models
except Exception as e:  # pragma: no cover
    raise ImportError("torchvision is required to build ResNet models") from e


device= torch.device("cpu")


def get_qmin_qmax(dtype: str = "int8"):
    if dtype == "int8":
        # symmetric int8 range [-127, 127]
        return -127, 127
    raise ValueError(f"Unsupported dtype: {dtype}")

def quant(x: torch.Tensor, scale: torch.Tensor | float, dtype: str):
    qmin, qmax = get_qmin_qmax(dtype=dtype)
    return torch.clamp(torch.round(x / scale), qmin, qmax)

def dequant(x: torch.Tensor, scale: torch.Tensor | float):
    return x * scale

def fake_quant(x: torch.Tensor, scale: torch.Tensor | float, dtype: str = "int8"):
    quant_x = quant(x, scale, dtype=dtype)
    return dequant(quant_x, scale)
    
    

class QuantConv2d(nn.Conv2d):
    """Conv2d with statistic collection and fake quantization modes.

    - do_collect=True: track input amax (per-tensor) and weight amax (per-out-channel).
    - do_quant=True:   apply symmetric fake quant to inputs and weights.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_abs_max_: torch.Tensor | None = None  # [Cout,1,1,1]
        self.input_abs_max_: float | None = None          # scalar
        self.do_quant: bool = False
        self.do_collect: bool = False

    def extra_repr(self):
        return (
            f"{super().extra_repr()}, do_collect={self.do_collect}, do_quant={self.do_quant}, "
            f"input_absmax={self.input_abs_max_}, weight_absmax={'set' if self.weight_abs_max_ is not None else 'None'}"
        )

    def _compute_weight_absmax(self) -> torch.Tensor:
        w = self.weight.detach()
        oc = w.size(0)
        return w.abs().view(oc, -1).amax(1).view(-1, 1, 1, 1).to(w.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        assert not (self.do_quant and self.do_collect), "do_quant and do_collect cannot be enabled simultaneously"

        if self.do_collect:
            if self.weight_abs_max_ is None:
                self.weight_abs_max_ = self._compute_weight_absmax()
            else:
                # track the maximum across runs
                self.weight_abs_max_ = torch.maximum(self.weight_abs_max_, self._compute_weight_absmax())

            amax_now = float(x.detach().abs().max().item())
            self.input_abs_max_ = amax_now if self.input_abs_max_ is None else max(self.input_abs_max_, amax_now)
            return super().forward(x)

        if self.do_quant:
            # Ensure scales are available
            if self.weight_abs_max_ is None:
                self.weight_abs_max_ = self._compute_weight_absmax()
            if self.input_abs_max_ is None:
                self.input_abs_max_ = float(x.detach().abs().max().item())

            wscale = (self.weight_abs_max_.clamp_min(1e-8) / 127.0).to(self.weight.device, self.weight.dtype)
            iscale = torch.tensor(max(self.input_abs_max_, 1e-8) / 127.0, device=x.device, dtype=x.dtype)

            x_q = fake_quant(x, iscale)
            w_q = fake_quant(self.weight, wscale)
            return F.conv2d(x_q, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return super().forward(x)


def replace_conv_with_quant(module: nn.Module) -> nn.Module:
    """Recursively replace nn.Conv2d with QuantConv2d in a module."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            quant_conv = QuantConv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(child.bias is not None),
                padding_mode=child.padding_mode,
            )
            # Copy weights/bias
            quant_conv.weight.data.copy_(child.weight.data)
            if child.bias is not None and quant_conv.bias is not None:
                quant_conv.bias.data.copy_(child.bias.data)
            setattr(module, name, quant_conv)
        else:
            replace_conv_with_quant(child)
    return module


def get_resnet(
    model: Literal[
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
    ] = "resnet18",
    pretrained: bool = False,
    quantize_conv: bool = False,
    device: str = "cpu",
) -> nn.Module:
    """Build a torchvision ResNet with optional Conv2d replacement.

    - model: which resnet variant
    - pretrained: load pretrained weights if available
    - quantize_conv: replace Conv2d with QuantConv2d
    - device: move model to device
    """

    builder = getattr(models, model, None)
    if builder is None:
        raise ValueError(f"Unsupported model '{model}'")

    # Handle new torchvision weights API gracefully
    kwargs = {}
    if pretrained:
        # Try best-effort to pick default weights enum if present
        weights_attr = f"{model.capitalize()}_Weights" if model != "resnet50" else "ResNet50_Weights"
        weights_enum = getattr(models, weights_attr, None)
        if weights_enum is not None and hasattr(weights_enum, "DEFAULT"):
            kwargs["weights"] = weights_enum.DEFAULT
        else:
            kwargs["pretrained"] = True  # for older torchvision
    # eval ensure dropout/BN to eval state not train state
    net = builder(**kwargs).eval()
    if quantize_conv:
        net = replace_conv_with_quant(net)
    net.to(device)
    return net


def set_collect_mode(model: nn.Module, enabled: bool = True, reset: bool = True):
    """Enable/disable do_collect across all QuantConv2d layers.

    - enabled=True: turn on collection and turn off do_quant.
    - reset=True: clear previous amax stats before collecting.
    """
    for _, m in model.named_modules():
        if isinstance(m, QuantConv2d):
            m.do_collect = enabled
            m.do_quant = False if enabled else m.do_quant
            if reset:
                m.input_abs_max_ = None
                m.weight_abs_max_ = None


def set_quant_mode(model: nn.Module, enabled: bool = True):
    """Enable/disable do_quant across all QuantConv2d layers (disables do_collect)."""
    for _, m in model.named_modules():
        if isinstance(m, QuantConv2d):
            m.do_quant = enabled
            if enabled:
                m.do_collect = False


def get_eval_transform(
    image_size: int = 224,
    crop_pct: float = 0.875,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """Return a fast, standard preprocessing transform for image classification.

    Pipeline (ImageNet style):
      - Resize shorter side to ``int(image_size / crop_pct)`` with bilinear + antialias
      - CenterCrop to ``image_size``
      - Convert to tensor in [0, 1]
      - Normalize with given mean/std

    Args:
        image_size: Final square crop size.
        crop_pct:   Inference crop ratio (0.875 for ResNet/ViT typical).
        mean:       Channel means (RGB) in [0, 1].
        std:        Channel stds (RGB) in [0, 1].

    Returns:
        A torchvision transform callable.
    """
    # Local import to avoid overhead at module import
    from torchvision import transforms as T
    from torchvision.transforms.functional import InterpolationMode

    resize_size = int(round(image_size / crop_pct))
    return T.Compose(
        [
            T.Resize(resize_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def get_train_transform(
    image_size: int = 224,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    scale=(0.08, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
):
    """Return a common training-time augmentation + preprocessing pipeline.

    Uses RandomResizedCrop + RandomHorizontalFlip for speed and robustness,
    followed by tensor conversion and normalization.
    """
    from torchvision import transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=scale, ratio=ratio, interpolation=InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


img_list = [torch.randn(1,3,224,224),torch.randn(1,3,224,224)]
model= get_resnet("resnet18")
# test_transform = get_train_transform()
# img = Image.open("dog.jpg")
# input_img_tensor = test_transform(img)


# original forward
# no grad means pytorch will not save grad during inference
with torch.no_grad():
    for img in img_list:
        logit = model(img)
        predict = torch.softmax(logit, dim=1)
        label =  predict.argmax(1).item()
        confidence = predict[0, label].item()
    
# collect mode
set_collect_mode(model)
with torch.no_grad():
    for img in img_list:
        logit = model(img)

# quant mode  
set_quant_mode(model)
with torch.no_grad():
    for img in img_list:
        logit = model(img)
        predict = torch.softmax(logit, dim=1)
        label =  predict.argmax(1).item()
        confidence = predict[0, label].item()

    
