import logging
import pathlib
from typing import Optional

import onnx
from onnx import version_converter

import onnxruntime as ort

def update_onnx_opset(
    model_path: pathlib.Path,
    opset: int,
    out_path: Optional[pathlib.Path] = None,
    logger: Optional[logging.Logger] = None,
):
    """
    Helper to update the opset of a model using onnx version_converter. Target opset must be greater than current opset.
    :param model_path: Path to model to update
    :param opset: Opset to update model to
    :param out_path: Optional output path for updated model to be saved to.
    :param logger: Optional logger for diagnostic output
    :returns: Updated onnx.ModelProto
    """

    model_path_str = str(model_path.resolve(strict=True))
    if logger:
        logger.info("Updating %s to opset %d", model_path_str, opset)

    model = onnx.load(model_path_str)

    new_model = version_converter.convert_version(model, opset)


    if out_path:
        onnx.save(new_model, str(out_path))
        if logger:
            logger.info("Saved updated model to %s", out_path)

    return new_model



def optimize_model(
    model_path: pathlib.Path,
    output_path: pathlib.Path,
    level: ort.GraphOptimizationLevel = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    log_level: int = 3,
    use_external_initializers: bool = False,
):
    """
    Optimize an ONNX model using ONNX Runtime to the specified level
    :param model_path: Path to ONNX model
    :param output_path: Path to save optimized model to.
    :param level: onnxruntime.GraphOptimizationLevel to use. Default is ORT_ENABLE_BASIC.
    :param log_level: Log level. Defaults to Error (3) so we don't get output about unused initializers being removed.
                      Warning (2) or Info (1) may be desirable in some scenarios.
    :param use_external_initializers: Set flag to write initializers to an external file. Required if model > 2GB.
                                      Requires onnxruntime 1.17+
    """
    so = ort.SessionOptions()
    so.optimized_model_filepath = str(output_path.resolve())
    so.graph_optimization_level = level
    so.log_severity_level = log_level

    # save using external initializers so models > 2 GB are handled
    if use_external_initializers:
        major, minor, rest = ort.__version__.split(".", 3)
        if (int(major), int(minor)) >= (1, 17):
            so.add_session_config_entry("session.optimized_model_external_initializers_file_name", "external_data.pb")
        else:
            raise ValueError(
                "ONNX Runtime 1.17 or higher required to save initializers as external data when optimizing model. "
                f"Current ONNX Runtime version is {ort.__version__}"
            )

    # create session to optimize. this will write the updated model to output_path
    _ = ort.InferenceSession(str(model_path.resolve(strict=True)), so, providers=["CPUExecutionProvider"])