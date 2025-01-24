
from PIL import Image
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
import os
from scipy import spatial
from utils import parse_args
from quant_utils import (DataReader, get_cosine_dist, get_average_err,
                          create_weight_matching_QOperator, create_activation_matching_QOperator,
                          compute_activation_error_QOperator)
from onnx_utils import get_io_node_info, load_model_with_shape_infer
import onnxruntime as rt
import glob
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.quantization.qdq_loss_debug import (
    collect_activations, compute_activation_error, compute_weight_error,
    create_activation_matching, create_weight_matching,
    modify_model_output_intermediate_tensors)

QUANT_OP_TYPE = ['Split', 'Squeeze', 'Pad', 'MaxPool',
                'GlobalAveragePool', 'AveragePool',
                'LeakyRelu', 'ConvTranspose',
                'MatMul', 'Resize', 'Add',
                'InstanceNormalization', 'Conv',
                'Sigmoid', 'Gather', 'EmbedLayerNormalization',
                'Gemm', 'Concat', 'Clip', 'Unsqueeze', 'Relu',
                'Softmax', 'ArgMax', 'Reshape', 'Where',
                'Transpose', 'Mul']

SKIP_QUANT_OP_TYPE = ['Add', 'Mul', 'Softmax', 'Concat']
# SKIP_QUANT_OP_TYPE = []

def quant_pre_processing(model_path):
    save_path = model_path.replace('.onnx', '_infer.onnx')
 
    quant_pre_process(
        input_model_path=model_path,
        output_model_path=save_path,
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=True,
        auto_merge=False,
        int_max=2**31 - 1,
        guess_output_rank=False,
        verbose=0,
        save_as_external_data=False,
        all_tensors_to_one_file=False,
        external_data_location=None,
        external_data_size_threshold=1024)
    
    if os.path.isfile(save_path):
        return save_path
    else:
        return None



def quant(src_model_path, quant_model_path, calib_folder, input_info, q_format): 

    dr = DataReader(calib_folder, input_info)
    
    quant_op_list = QUANT_OP_TYPE.copy()
    if SKIP_QUANT_OP_TYPE != []:
        for node in SKIP_QUANT_OP_TYPE:
            quant_op_list.remove(node)
    
    if 'qdq' in q_format:
        q_format = QuantFormat.QDQ
    else:
        q_format = QuantFormat.QOperator  


    quantize_static(
        src_model_path,
        quant_model_path,
        dr,
        op_types_to_quantize=quant_op_list,
        quant_format=q_format,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.Percentile,
        per_channel=True,
        reduce_range=False,
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )

    # print(f'[{src_model_path}] ONNX full precision model size (MB):', os.path.getsize(src_model_path) / (1024 * 1024))
    # print(f'[{src_model_path}] ONNX quantized model size (MB):', os.path.getsize(quant_model_path) / (1024 * 1024))
    


def check_similarity(src_model_path, quant_model_path, input_info, test_img_folder = None):
    session1 = rt.InferenceSession(src_model_path,providers=['CUDAExecutionProvider'])
    input_name1 = session1.get_inputs()[0].name

    session2 = rt.InferenceSession(quant_model_path,providers=['CUDAExecutionProvider'])
    input_name2 = session2.get_inputs()[0].name

    if(test_img_folder != None):
        reader = DataReader(test_img_folder, input_info)
        onnx_input = reader.get_next()[input_info[0].name]
    else:
        onnx_input = np.random.random(input_info[0].shape).astype(np.float32)

    result1 = session1.run([], {input_name1: onnx_input})
    result2 = session2.run([], {input_name2: onnx_input})

    print(f'[{src_model_path}] src_result:', result1[0].flatten()[:10])
    print(f'[{src_model_path}] quant_result:', result2[0].flatten()[:10])
    print(f'[{src_model_path}] cosine_dist: ', get_cosine_dist(result1[0],result2[0]))
    print(f'[{src_model_path}] average error: ', get_average_err(result1[0],result2[0]))


def _generate_aug_model_path(model_path: str) -> str:
    aug_model_path = (
        model_path[: -len(".onnx")] if model_path.endswith(".onnx") else model_path
    )
    return aug_model_path + ".save_tensors.onnx"

def accuracy_debug(float_model_path, quant_model_path, calib_folder, input_info):
    print("------------------------------------------------\n")
    print("Comparing weights of float model vs qdq model.....")

    matched_weights = create_weight_matching(float_model_path, quant_model_path)
    weights_error = compute_weight_error(matched_weights)
    for weight_name, err in weights_error.items():
        print(f"Cross model error of '{weight_name}': {err}\n")

    print("------------------------------------------------\n")
    print("Augmenting models to save intermediate activations......")

    aug_float_model_path = _generate_aug_model_path(float_model_path)
    modify_model_output_intermediate_tensors(float_model_path, aug_float_model_path)

    aug_qdq_model_path = _generate_aug_model_path(quant_model_path)
    modify_model_output_intermediate_tensors(quant_model_path, aug_qdq_model_path)

    print("------------------------------------------------\n")
    print("Running the augmented floating point model to collect activations......")
    # input_data_reader = resnet50_data_reader.ResNet50DataReader(
    #     calibration_dataset_path, float_model_path
    # )
    input_data_reader = DataReader(calib_folder, input_info)
    float_activations = collect_activations(aug_float_model_path, input_data_reader)

    print("------------------------------------------------\n")
    print("Running the augmented qdq model to collect activations......")
    input_data_reader.rewind()
    qdq_activations = collect_activations(aug_qdq_model_path, input_data_reader)

    print("------------------------------------------------\n")
    print("Comparing activations of float model vs qdq model......")

    act_matching = create_activation_matching_QOperator(qdq_activations, float_activations)
    act_error = compute_activation_error_QOperator(act_matching)

    # sort the dict
    sorted_list = sorted(act_error.items(), key=lambda x: x[1]['qdq_err'], reverse=True)


    for act_name, err in sorted_list:
        # print(f"Cross model error of '{act_name}': {err['xmodel_err']} \n")
        print(f"QDQ error of '{act_name}': {err['qdq_err']}")

def quant_pipeline(model_path, out_model_path, img_folder, q_format):
    print(f'[{model_path}] QUANT BEGIN')
    if out_model_path == None:
        out_model_path = model_path.replace('.onnx', '_quant.onnx')

    # onnx_tool.model_profile(args.model, saveshapesmodel='shapes.onnx')

    # check src model & get input_info
    input_info, output_info = get_io_node_info(model_path)
    
    if len(input_info) != 1 :
        print(f'[{model_path}] ERROR: only support input num==1, while num is {len(input_info)}!!!')
        return

    # assert len(input_info) == 1, f"only support input num==1, while num is {len(input_info)}"

    # load_model_with_shape_infer(args.model)

    pre_model_path = quant_pre_processing(model_path)

    if pre_model_path != None:
        model_path = pre_model_path

    # quant the model
    quant(model_path, out_model_path, img_folder, input_info, q_format)  

    # check the cosine similarity 
    check_similarity(model_path, out_model_path, input_info, img_folder)

    # accuracy_debug(model_path, out_model_path, img_folder, input_info)



if __name__ == "__main__":
    
    args = parse_args()

    if os.path.isfile(args.model):

        quant_pipeline(args.model, args.out_model, args.img_folder, args.quant_format)

    elif os.path.isdir(args.model):

        assert os.path.isdir(args.out_model), f'out_model path should be folder when input_model path is folder'
        model_list = glob.glob(os.path.join(args.model, '*.onnx'), recursive=True)

        for model in model_list:
            if 'convnext' in model:
                continue
            if model == '/home/fiery/work/models/pytorch-image-models/export_13/correct/deit_tiny_distilled_patch16_224.onnx':
                continue
            out_model_path = os.path.join(args.out_model, os.path.basename(model))
            quant_pipeline(model, out_model_path, args.img_folder, args.quant_format)

            


        

    


