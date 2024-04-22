def is_int8_model(model):
    """Check whether the input model is a int8 model.

    Args:
        model (torch.nn.Module): input model

    Returns:
        result(bool): Return True if the input model is a int8 model.
    """
    def _is_int8_value(value):
        """Check whether the input tensor is a int8 tensor."""
        if hasattr(value, 'dtype') and 'int8' in str(value.dtype):
            return True
        else:
            return False

    stat_dict = dict(model.state_dict())
    for name, value in stat_dict.items():
        if _is_int8_value(value):
            return True
        # value maybe a tuple, such as 'linear._packed_params._packed_params'
        if isinstance(value, tuple):
            for v in value:
                if _is_int8_value(v):
                    return True
    return False

import onnx
model = onnx.load("/home/wrq/quant_SDK/export/qlinear_resnet50_quant_int8_linear.onnx")
is_int8_model(model)