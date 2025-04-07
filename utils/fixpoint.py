import torch


def approximate_float(M:torch.float32):
    '''
    
    
    return: Q31 format fixpoint number and shift'''
    significand, shift = torch.frexp(M)
    significand_q31 = torch.round(significand * (1 << 31))
    
    return significand_q31, shift
    