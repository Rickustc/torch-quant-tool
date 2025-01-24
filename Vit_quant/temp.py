import math
import numpy as np
import torch



def float2fix_point(weights, Qmax=255):
    
    
    manti_quant = np.zeros_like(weights)
    right_shift = np.zeros_like(weights, dtype=int)
    
    # 使用 frexp 函数分解浮点数 s 为尾数（manti）和指数（exponent）
    for i, weight in enumerate(weights):
        # 使用math.frexp得到尾数和指数
        manti, exponent = math.frexp(weight)
        # 量化尾数，并四舍五入到最接近的整数
        manti_quant[i] = math.floor(manti * (Qmax + 1) + 0.5)    # 左移
        # 处理溢出情况
        if manti_quant[i] == Qmax + 1:
            exponent += 1
            manti_quant[i] = (Qmax + 1) / 2
        # 计算需要右移的位数
        right_shift[i] = exponent - math.log2(Qmax + 1)   #为什么要移位

    return manti_quant, right_shift

def fix2float_point(manti_quant, right_shift, Qmax=255):
    float_weights = np.zeros_like(manti_quant)

    for i, (quantized, shift) in enumerate(zip(manti_quant, right_shift)):
        # 通过移位操作反量化
        float_weights[i] = quantized * (2 ** shift)
    
    return float_weights

# 示例
weights = np.array([0.5])
Qmax = 255
manti_quant, right_shift = float2fix_point(weights, Qmax)
restored_s = fix2float_point(manti_quant, right_shift, Qmax)

print(f"Original Value: {weights}")
print(f"Quantized Manti: {manti_quant}")
print(f"Right Shift: {right_shift}")
print(f"Restored Original Value: {restored_s}")