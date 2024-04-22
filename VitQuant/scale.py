import math
import numpy as np



# frexp: 一个浮点数分解成一个正规化的小数和一个位移量 

def quantize_weights(weights, Qmax=255):
    manti_quant = np.zeros_like(weights)
    right_shift = np.zeros_like(weights, dtype=int)
    
    for i, weight in enumerate(weights):
        # 使用math.frexp得到尾数和指数
        manti, exponent = math.frexp(weight)
        # 量化尾数，并四舍五入到最接近的整数
        manti_quant[i] = math.floor(manti * (Qmax + 1) + 0.5)
        # 处理溢出情况
        if manti_quant[i] == Qmax + 1:
            exponent += 1
            manti_quant[i] = (Qmax + 1) / 2
        # 计算需要右移的位数
        right_shift[i] = -(exponent - math.log2(Qmax + 1))

    return manti_quant, right_shift

def dequantize(manti_quant, right_shift, Qmax=255):
    # 从量化值反量化得到近似小数值
    return manti_quant / (Qmax + 1) * (2 ** right_shift)

# 示例权重数组
weights = np.array([0.8, 0.65, 0.23, 0.005, 1.0])  
manti_quant, right_shift = quantize_weights(weights)

# # 反量化回浮点数
# approx_weights = dequantize(manti_quant, right_shift)

# print("Quantized Mantissas:", manti_quant)
# print("Right Shifts:", right_shift)
# print("Approximate Decimal Weights:", approx_weights)



# def quantize_and_dequantize(weights, Qmax=255):
#     # 量化
#     exponent = np.floor(np.log2(np.abs(weights)))  # Calculate the exponent
#     manti_quant = np.round((weights / np.power(2.0, exponent)) * Qmax)  # Quantize the mantissa
#     manti_quant = np.clip(manti_quant, -Qmax, Qmax)  # Clip to the range to avoid overflow

#     # 反量化
#     approx_weights = manti_quant * np.power(2.0, exponent) / Qmax  # Convert back to float
    
#     return approx_weights

# # # 示例权重数组
# # weights = np.array([0.8, 0.65, 0.23, 0.005, 0.76, 0,24])

# # # 计算近似值
# # approx_weights = quantize_and_dequantize(weights)

# # print("Original Weights:", weights)
# # print("Approximate Decimal Weights:", approx_weights)



# import numpy as np

def quantize_with_frexp(weights):
    quantized_weights = np.zeros_like(weights)
    shifts = np.zeros_like(weights)

    for i, weight in enumerate(weights):
        # 使用frexp分解为 fraction 尾数和二进制指数 exp
        mantissa, exponent = np.frexp(weight)
        # 使用8位量化尾数
        quantized = np.floor(mantissa * 256+0.5)
        shift = exponent - 8  # 因为256是2的8次方，所以这里是8
        quantized_weights[i] = quantized
        shifts[i] = shift
    
    return quantized_weights, shifts

def dequantize_with_shift(quantized_weights, shifts):
    float_weights = np.zeros_like(quantized_weights)

    for i, (quantized, shift) in enumerate(zip(quantized_weights, shifts)):
        # 通过移位操作反量化
        float_weights[i] = quantized * (2 ** shift)
    
    return float_weights


weights = np.array([0.000001, 0.65, 0.23, 0.005, 1.0, 0.42])

quantized_weights, shifts = quantize_with_frexp(weights)
approximated_float_weights = dequantize_with_shift(quantized_weights, shifts)

print("Quantized Weights:", quantized_weights)
print("Shifts:", shifts)
print("Approximated Float Weights:", approximated_float_weights)

# manti, exponent = math.frexp(0.005)

# '''
# 首先,和IEEE754标准一样, 我们需要把浮点数表示成 fraction 和 exp相乘的格式,也类似于科学计数法
# 例如
# 0.005 = 0.64 * 2^{-7}
# 其中0.64叫正规化小数 范围在[0.5,1)

# mantissa 为尾数，或者说精度
# exponent 为指数，是一个整数
# '''


# # [0.5,1)
# print(manti)  
# print(exponent)

