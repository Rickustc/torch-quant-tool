class LinearActivation(nn.Module):
    r"""Fused Linear and Activation Module.
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, act='gelu', bias=True, do_quant=True):
        super(LinearActivation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act_fn = nn.Identity()
        self.biased_act_fn = None
        
        if isinstance(act, str) or (sys.version_info[0] == 2 and isinstance(act, unicode)):
            if bias and not 'bias' in act:
                act = 'bias_' + act
                self.biased_act_fn = ACT2FN[act]
            else:
                self.act_fn = ACT2FN[act]
        else:
            self.act_fn = act
        
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.do_quant = do_quant
        if do_quant:
            self._input_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
            self._weight_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_weight)
            self._aftergemm_quantizer = TensorQuantizer(QuantLinear.default_quant_desc_input)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_normal_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        if self.do_quant:
            input = self._input_quantizer(input)
            weight = self._weight_quantizer(self.weight)
        else:
            weight = self.weight
        
        if not self.bias is None:
            if self.do_quant:
                return self.biased_act_fn(self.bias, self._aftergemm_quantizer(F.linear(input, weight, None)))
            else:
                return self.biased_act_fn(self.bias, F.linear(input, weight, None))
        else:
            if self.do_quant:
                return self.act_fn(self._aftergemm_quantizer(F.linear(input, weight, None)))
            else:
                return self.act_fn(F.linear(input, weight, None))
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )