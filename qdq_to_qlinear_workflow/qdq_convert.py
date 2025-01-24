import logging
import argparse
import onnxsim
from neural_compressor import PostTrainingQuantConfig
from neural_compressor.model.onnx_model import ONNXModel
from neural_compressor.model.model import Model
from neural_compressor.config import _Config
from neural_compressor.strategy.basic import BasicTuneStrategy



logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.DEBUG)

parser = argparse.ArgumentParser(
        description="Convert QDQ to QLinear",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
parser.add_argument(
        '--model_path',
        type=str,
        help="QDQ onnx model file"
)
parser.add_argument(
    '--output_model',
    type=str,
    help="converted model path"
)

if __name__ == "__main__":
    logger.info("Converting QDQ model to QLinear")    

    args = parser.parse_args()

    # do the simplify
    model_sim, check_ok = onnxsim.simplify(args.model_path)

    # create the model
    model = ONNXModel(model_sim)

    quant_dtype_setting_dict={}
    PTQConfig = PostTrainingQuantConfig(
        quant_format='QOperator',
        diagnosis=True,
        quant_level=1,
        op_type_dict=quant_dtype_setting_dict,
        excluded_precisions=[]
    )
    
    wrapped_model = Model(model, conf=PTQConfig)

    config = _Config(quantization=PTQConfig, benchmark=None, pruning=None, distillation=None, nas=None)

    strategy = BasicTuneStrategy(
        model=wrapped_model,
        conf=config,
        q_dataloader=None,
        q_func=None,
        eval_func=None,
        eval_dataloader=None,
        eval_metric=None,
        resume=None,
        q_hooks=None,
    )

    qlinear_model = strategy.traverse_for_qdq_converter()
    qlinear_model.save(args.output_model)  

    
