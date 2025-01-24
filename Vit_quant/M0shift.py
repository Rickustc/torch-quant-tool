M = 0.0072474273418460
P = 7091

def multiply(n, M, P):
    result = M * P
    Mo = int(round(2 ** n * M)) # 这里不一定要四舍五入截断，因为python定点数不好表示才这样处理

    approx_result = (Mo * P) >> n
    print("n=%d, Mo=%d, approx=%f, error=%f"%\
          (n, Mo, approx_result, result-approx_result))

for n in range(1, 16):
    multiply(n, M, P)

