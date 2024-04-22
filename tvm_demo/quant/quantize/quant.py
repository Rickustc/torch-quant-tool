"""Automatic quantization toolkit."""
import tvm.ir
import tvm
from tvm.runtime import Object

from . import _quantize
from ._calibrate import calibrate
from ._partition_conversions import partition_conversions
from .. import expr as _expr
from .. import transform as _transform



class QAnnotateKind(object):
    """Denote the kind of annotation field, corresponding
    to different nbit configure."""

    IDENTITY = 0
    INPUT = 1
    WEIGHT = 2
    ACTIVATION = 3

def kind2str(kind):
    """Convert a `QAnnotateKind` to string"""
    str_map = {
        QAnnotateKind.INPUT: "input",
        QAnnotateKind.WEIGHT: "weight",
        QAnnotateKind.ACTIVATION: "activation",
        QAnnotateKind.IDENTITY: "identity",
    }
    assert kind in str_map
    return str_map[kind]


@tvm._ffi.register_object("relay.quantize.QConfig")
class QConfig(Object):
    """Configure the quantization behavior by setting config variables.

    Note
    ----
    This object is backed by node system in C++, with arguments that can be
    exchanged between python and C++.

    Do not construct directly, use qconfig instead.

    The fields that are backed by the C++ node are immutable once an instance
    is constructed. See _node_defaults for the fields.
    """

    _node_defaults = {
        "nbit_input": 8,
        "nbit_weight": 8,
        "nbit_activation": 32,
        "dtype_input": "int8",
        "dtype_weight": "int8",
        "dtype_activation": "int32",
        "calibrate_mode": "global_scale",
        "global_scale": 8.0,
        "weight_scale": "power2",
        "skip_dense_layer": True,
        "skip_conv_layers": [0],
        "do_simulation": False,
        "round_for_shift": True,
        "debug_enabled_ops": None,
        "rounding": "UPWARD",
        "calibrate_chunk_by": -1,
        "partition_conversions": "disabled",
    }

    # pylint: disable=no-member
    def __init__(self, handle):
        """Initialize the function with handle

        Parameters
        ----------
        handle : SymbolHandle
            the handle to the underlying C++ Symbol
        """
        super(QConfig, self).__init__(handle)
        self.handle = handle

    def guard(self, ref_call):
        """Return true if op is enabled, otherwise return false"""
        op_name = ref_call.op.name
        if self.debug_enabled_ops is not None:
            name_list = [x.value for x in self.debug_enabled_ops]
            if op_name not in name_list:
                return False
        return True

    def get_nbit_by_kind(self, kind):
        name = kind2str(kind)
        return getattr(self, "nbit_" + name)

    def get_dtype_by_kind(self, kind):
        name = kind2str(kind)
        return getattr(self, "dtype_" + name)

    def __enter__(self):
        # pylint: disable=protected-access
        _quantize._EnterQConfigScope(self)
        return self

    def __exit__(self, ptype, value, trace):
        _quantize._ExitQConfigScope()

    def __setattr__(self, name, value):
        if name in QConfig._node_defaults:
            raise AttributeError("'%s' object cannot set attribute '%s'" % (str(type(self)), name))
        return super(QConfig, self).__setattr__(name, value)

def quantize(mod, params=None, dataset=None):
    """The quantization procedure. Before running the three main
    procedure of quantization, "annotate", "calibrate" and "realize"
    , we need to do "SimplifyInference", "FoldScaleAxis", "FoldConstant"
    first for optimizing.

    Parameters
    ---------
    mod: Module
        The original module.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    dataset: list of dict of Var -> NDArray
        The calibration dataset.

    Returns
    -------
    ret: Function
        The graph after quantization
    """
    mod = prerequisite_optimize(mod, params)

    calibrate_pass = tvm.transform.module_pass(
        calibrate(dataset), opt_level=1, name="QuantizeCalibrate"
    )
    quant_passes = [partition(), annotate(), calibrate_pass, tvm.relay.transform.InferType()]
    if not current_qconfig().do_simulation:
        quant_passes.append(realize())
    quant_passes.append(_transform.FoldConstant())
    quantize_seq = tvm.transform.Sequential(quant_passes)
    with tvm.transform.PassContext(
        opt_level=3, required_pass=["QuantizeAnnotate", "QuantizeCalibrate", "QuantizeRealize"]
    ):
        with quantize_context():
            mod = quantize_seq(mod)

    q_cfg = current_qconfig()
    assert q_cfg.partition_conversions in ["disabled", "enabled", "fully_integral"]
    if q_cfg.partition_conversions != "disabled":
        quantized_dtypes = {q_cfg.dtype_input, q_cfg.dtype_weight, q_cfg.dtype_activation}
        ensure_fully_integral = q_cfg.partition_conversions == "fully_integral"
        return partition_conversions(mod, quantized_dtypes, ensure_fully_integral)

    return mod