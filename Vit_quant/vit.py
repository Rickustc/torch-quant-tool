"""
original code from rwightman:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

# import torch.nn.quantized.LayerNorm as QuantizedLayerNorm
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        
        # self.quant = torch.quantization.QuantStub()
        # self.dequant = torch.quantization.DeQuantStub()
        self.q_scaling_product = torch.ao.nn.quantized.FloatFunctional()
        self.softmax =  torch.nn.Softmax(dim=-1)

    def forward(self, x):
     
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape
        
        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
   
        # [1,197,768]  --> [1,197,2304] --[1,197,3,8,96] --> [3,1,8,197,96]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
       
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        
        # attn = self.float_fn.mul(q @ k.transpose(-2, -1),self.scale)
        # attn = torch.matmul(q , k.transpose(-2, -1))*self.scale
        # attn = torch.mul(torch.matmul(q , k.transpose(-2, -1)), self.scale)
        attn = torch.matmul(q , k.transpose(-2, -1))
        attn = self.q_scaling_product.mul_scalar(attn, self.scale)
        
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        
        # attn = self.quant(attn)
        # attn = self.dequant(attn)
        attn = self.softmax(attn)
        
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = torch.matmul(attn,v).transpose(1, 2).reshape(B, N, C)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
class DeQuant(nn.Module):
    r"""Dequantize stub module, before calibration, this is same as identity,
    this will be swapped as `nnq.DeQuantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """
    def __init__(self, qconfig=None):
        super().__init__()
        if qconfig:
            self.qconfig = qconfig

    def forward(self, x):
        return torch.dequantize(x)

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=None):
        super(Block, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
        # self.dequant = DeQuant()
        
        
    def forward(self, x):
        
        # x = torch.add(x,  self.drop_path(self.attn(self.dequant(self.norm1(x)))))
        # x = torch.add(x,  self.drop_path(self.mlp(self.dequant(self.norm2(x)))))
        x = torch.add(x,  self.drop_path(self.attn(self.norm1(x))))
        x = torch.add(x,  self.drop_path(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)    #    or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))      # [1, 197,768]
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        # self.norm100 = norm_layer(embed_dim)
        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
        # self.dequant = torch.ao.quantization.DeQuantStub()
        # self.quant = torch.ao.quantization.QuantStub()
    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            
        # [1, 197,768]
        x = torch.add(x,  self.pos_embed)
        x = self.pos_drop(x)
        x = self.blocks(x)
        
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,       # 1 for debug
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes,
                              norm_layer=nn.LayerNorm
                              )
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model
class traced_VIT( VisionTransformer):
    def __init__(self):
        super().__init__()
        
    def forward(self,x):
        patch_embed_proj_input_scale_0 = self.patch_embed_proj_input_scale_0
        patch_embed_proj_input_zero_point_0 = self.patch_embed_proj_input_zero_point_0
        quantize_per_tensor = torch.quantize_per_tensor(x, patch_embed_proj_input_scale_0, patch_embed_proj_input_zero_point_0, torch.qint8);  x = patch_embed_proj_input_scale_0 = patch_embed_proj_input_zero_point_0 = None
        patch_embed_proj = self.patch_embed.proj(quantize_per_tensor);  quantize_per_tensor = None
        dequantize_1 = patch_embed_proj.dequantize();  patch_embed_proj = None
        flatten = dequantize_1.flatten(2);  dequantize_1 = None
        patch_embed_scale_0 = self.patch_embed_scale_0
        patch_embed_zero_point_0 = self.patch_embed_zero_point_0
        quantize_per_tensor_2 = torch.quantize_per_tensor(flatten, patch_embed_scale_0, patch_embed_zero_point_0, torch.qint8);  flatten = patch_embed_scale_0 = patch_embed_zero_point_0 = None
        transpose = quantize_per_tensor_2.transpose(1, 2);  quantize_per_tensor_2 = None
        patch_embed_norm = self.patch_embed.norm(transpose);  transpose = None
        cls_token = self.cls_token
        getattr_2 = patch_embed_norm.shape
        getitem_4 = getattr_2[0];  getattr_2 = None
        expand = cls_token.expand(getitem_4, -1, -1);  cls_token = getitem_4 = None
        _scale_0 = self._scale_0
        _zero_point_0 = self._zero_point_0
        quantize_per_tensor_5 = torch.quantize_per_tensor(expand, _scale_0, _zero_point_0, torch.qint8);  expand = _scale_0 = _zero_point_0 = None
        cat = torch.cat((quantize_per_tensor_5, patch_embed_norm), dim = 1);  quantize_per_tensor_5 = patch_embed_norm = None
        pos_embed = self.pos_embed
        _scale_2 = self._scale_2
        _zero_point_2 = self._zero_point_2
        quantize_per_tensor_7 = torch.quantize_per_tensor(pos_embed, _scale_2, _zero_point_2, torch.qint8);  pos_embed = _scale_2 = _zero_point_2 = None
        _scale_3 = self._scale_3
        _zero_point_3 = self._zero_point_3
        add_3 = torch.ops.quantized.add(cat, quantize_per_tensor_7, _scale_3, _zero_point_3);  cat = quantize_per_tensor_7 = _scale_3 = _zero_point_3 = None
        pos_drop = self.pos_drop(add_3);  add_3 = None
        blocks_0_norm1 = getattr(self.blocks, "0").norm1(pos_drop)
        getattr_3 = blocks_0_norm1.shape
        getitem_5 = getattr_3[0]
        getitem_6 = getattr_3[1]
        getitem_7 = getattr_3[2];  getattr_3 = None
        blocks_0_attn_qkv = getattr(self.blocks, "0").attn.qkv(blocks_0_norm1);  blocks_0_norm1 = None
        floordiv = getitem_7 // 12
        reshape = blocks_0_attn_qkv.reshape(getitem_5, getitem_6, 3, 12, floordiv);  blocks_0_attn_qkv = floordiv = None
        permute = reshape.permute(2, 0, 3, 1, 4);  reshape = None
        dequantize_13 = permute.dequantize();  permute = None
        getitem_8 = dequantize_13[0]
        blocks_0_attn_scale_2 = self.blocks_0_attn_scale_2
        blocks_0_attn_zero_point_2 = self.blocks_0_attn_zero_point_2
        quantize_per_tensor_14 = torch.quantize_per_tensor(getitem_8, blocks_0_attn_scale_2, blocks_0_attn_zero_point_2, torch.qint8);  getitem_8 = blocks_0_attn_scale_2 = blocks_0_attn_zero_point_2 = None
        getitem_9 = dequantize_13[1]
        blocks_0_attn_scale_3 = self.blocks_0_attn_scale_3
        blocks_0_attn_zero_point_3 = self.blocks_0_attn_zero_point_3
        quantize_per_tensor_15 = torch.quantize_per_tensor(getitem_9, blocks_0_attn_scale_3, blocks_0_attn_zero_point_3, torch.qint8);  getitem_9 = blocks_0_attn_scale_3 = blocks_0_attn_zero_point_3 = None
        getitem_10 = dequantize_13[2];  dequantize_13 = None
        blocks_0_attn_scale_4 = self.blocks_0_attn_scale_4
        blocks_0_attn_zero_point_4 = self.blocks_0_attn_zero_point_4
        quantize_per_tensor_16 = torch.quantize_per_tensor(getitem_10, blocks_0_attn_scale_4, blocks_0_attn_zero_point_4, torch.qint8);  getitem_10 = blocks_0_attn_scale_4 = blocks_0_attn_zero_point_4 = None
        transpose_1 = quantize_per_tensor_15.transpose(-2, -1);  quantize_per_tensor_15 = None
        blocks_0_attn_scale_6 = self.blocks_0_attn_scale_6
        blocks_0_attn_zero_point_6 = self.blocks_0_attn_zero_point_6
        matmul_2 = torch.ops.quantized.matmul(quantize_per_tensor_14, transpose_1, blocks_0_attn_scale_6, blocks_0_attn_zero_point_6);  quantize_per_tensor_14 = transpose_1 = blocks_0_attn_scale_6 = blocks_0_attn_zero_point_6 = None
        mul_1 = torch.ops.quantized.mul(matmul_2, 0.125);  matmul_2 = None
        blocks_0_attn_softmax = getattr(self.blocks, "0").attn.softmax(mul_1);  mul_1 = None
        blocks_0_attn_attn_drop = getattr(self.blocks, "0").attn.attn_drop(blocks_0_attn_softmax);  blocks_0_attn_softmax = None
        blocks_0_attn_scale_8 = self.blocks_0_attn_scale_8
        blocks_0_attn_zero_point_8 = self.blocks_0_attn_zero_point_8
        matmul_3 = torch.ops.quantized.matmul(blocks_0_attn_attn_drop, quantize_per_tensor_16, blocks_0_attn_scale_8, blocks_0_attn_zero_point_8);  blocks_0_attn_attn_drop = quantize_per_tensor_16 = blocks_0_attn_scale_8 = blocks_0_attn_zero_point_8 = None
        transpose_2 = matmul_3.transpose(1, 2);  matmul_3 = None
        reshape_1 = transpose_2.reshape(getitem_5, getitem_6, getitem_7);  transpose_2 = getitem_5 = getitem_6 = getitem_7 = None
        blocks_0_attn_proj = getattr(self.blocks, "0").attn.proj(reshape_1);  reshape_1 = None
        blocks_0_attn_proj_drop = getattr(self.blocks, "0").attn.proj_drop(blocks_0_attn_proj);  blocks_0_attn_proj = None
        blocks_0_drop_path = getattr(self.blocks, "0").drop_path(blocks_0_attn_proj_drop);  blocks_0_attn_proj_drop = None
        blocks_0_scale_0 = self.blocks_0_scale_0
        blocks_0_zero_point_0 = self.blocks_0_zero_point_0
        add_4 = torch.ops.quantized.add(pos_drop, blocks_0_drop_path, blocks_0_scale_0, blocks_0_zero_point_0);  pos_drop = blocks_0_drop_path = blocks_0_scale_0 = blocks_0_zero_point_0 = None
        blocks_0_norm2 = getattr(self.blocks, "0").norm2(add_4)
        blocks_0_mlp_fc1 = getattr(self.blocks, "0").mlp.fc1(blocks_0_norm2);  blocks_0_norm2 = None
        dequantize_30 = blocks_0_mlp_fc1.dequantize();  blocks_0_mlp_fc1 = None
        blocks_0_mlp_act = getattr(self.blocks, "0").mlp.act(dequantize_30);  dequantize_30 = None
        blocks_0_mlp_drop = getattr(self.blocks, "0").mlp.drop(blocks_0_mlp_act);  blocks_0_mlp_act = None
        blocks_0_mlp_drop_scale_0 = self.blocks_0_mlp_drop_scale_0
        blocks_0_mlp_drop_zero_point_0 = self.blocks_0_mlp_drop_zero_point_0
        quantize_per_tensor_31 = torch.quantize_per_tensor(blocks_0_mlp_drop, blocks_0_mlp_drop_scale_0, blocks_0_mlp_drop_zero_point_0, torch.qint8);  blocks_0_mlp_drop = blocks_0_mlp_drop_scale_0 = blocks_0_mlp_drop_zero_point_0 = None
        blocks_0_mlp_fc2 = getattr(self.blocks, "0").mlp.fc2(quantize_per_tensor_31);  quantize_per_tensor_31 = None
        blocks_0_mlp_drop_1 = getattr(self.blocks, "0").mlp.drop(blocks_0_mlp_fc2);  blocks_0_mlp_fc2 = None
        blocks_0_drop_path_1 = getattr(self.blocks, "0").drop_path(blocks_0_mlp_drop_1);  blocks_0_mlp_drop_1 = None
        blocks_0_scale_1 = self.blocks_0_scale_1
        blocks_0_zero_point_1 = self.blocks_0_zero_point_1
        add_5 = torch.ops.quantized.add(add_4, blocks_0_drop_path_1, blocks_0_scale_1, blocks_0_zero_point_1);  add_4 = blocks_0_drop_path_1 = blocks_0_scale_1 = blocks_0_zero_point_1 = None
        norm = self.norm(add_5);  add_5 = None
        dequantize_36 = norm.dequantize();  norm = None
        getitem_11 = dequantize_36[(slice(None, None, None), 0)];  dequantize_36 = None
        _scale_4 = self._scale_4
        _zero_point_4 = self._zero_point_4
        quantize_per_tensor_37 = torch.quantize_per_tensor(getitem_11, _scale_4, _zero_point_4, torch.qint8);  getitem_11 = _scale_4 = _zero_point_4 = None
        pre_logits = self.pre_logits(quantize_per_tensor_37);  quantize_per_tensor_37 = None
        head = self.head(pre_logits);  pre_logits = None
        dequantize_39 = head.dequantize();  head = None
        return dequantize_39
        
        

def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model