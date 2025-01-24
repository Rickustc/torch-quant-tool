import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="onnx quant tool")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="The onnx model need to be quantized."
    )
    parser.add_argument(
        "--out_model",
        "-o",
        type=str,
        required=False,
        help="The location of quantized model need to be saved."
    )
    parser.add_argument(
        "--img_folder",
        "-i",
        type=str,
        required=True,
        help="The location of calibration image folder."
    )
    parser.add_argument(
        "--quant_format",
        "-q",
        type=str,
        default='qlinear',
        help="The quant format will be used, QDQ or QLinear(default)."
    )
    known_args = parser.parse_args()

    return known_args