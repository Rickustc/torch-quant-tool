

'''batch processing 并行化的好处'''
import torch
import time

def timing(x, w, batch = False, nb = 101):
    t = torch.zeros(nb)
    for u in range(nb):
        t0 = time.perf_counter()
        if batch:
            y = x.mm(w.t())
        else:
            y = torch.empty(x.size(0), w.size(0))
            for k in range(y.size(0)):
                y[k] = w.mv(x[k])
        y.is_cuda and torch.cuda.synchronize()
        t[u] = time.perf_counter() - t0
    return t.median().item()


x = torch.randn(2500, 1000)
w = torch.randn(1500, 1000)
print('Batch-processing speed-up on CPU %.1f' %
(timing(x, w, batch = False) / timing(x, w, batch = True)))
x, w = x.to('cuda'), w.to('cuda')
print('Batch-processing speed-up on GPU %.1f' %
(timing(x, w, batch = False) / timing(x, w, batch = True)))