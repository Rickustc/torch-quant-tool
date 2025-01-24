import onnxruntime as rt
from pathlib import Path
from onnx import ModelProto
import onnx

def get_io_node_info(model_path):
    try:
        sess = rt.InferenceSession(model_path)
    except:
        print(f'can not load {model_path}')
    sess = rt.InferenceSession(model_path)
    input_nodes = sess.get_inputs()
    output_nodes = sess.get_outputs()
    for i in range(0, len(input_nodes)):
        print("[INFO] Model input name <{}>:".format(i), input_nodes[i].name, "input shape :",
              input_nodes[i].shape, input_nodes[i].type)
    for i in range(0, len(output_nodes)):
        print("[INFO] Model output name <{}>:".format(i), output_nodes[i].name, 'output shape: ', output_nodes[i].shape)
    return input_nodes, output_nodes

def generate_identified_filename(filename: Path, identifier: str) -> Path:
    """
    Helper function to generate a identifiable filepath by concatenating the given identifier as a suffix.
    """
    return filename.replace('.onnx', f'_{identifier}.onnx')

def add_infer_metadata(model: ModelProto):
    metadata_props = {"onnx.infer": "onnxruntime.quant"}
    if model.metadata_props:
        for p in model.metadata_props:
            metadata_props.update({p.key: p.value})
    onnx.helper.set_model_props(model, metadata_props)

def load_model_with_shape_infer(model_path: Path) -> ModelProto:
    inferred_model_path = generate_identified_filename(model_path, "inferred")
    onnx.shape_inference.infer_shapes_path(str(model_path), str(inferred_model_path))
    # model = onnx.load(inferred_model_path.as_posix())
    # add_infer_metadata(model)
    # inferred_model_path.unlink()
    