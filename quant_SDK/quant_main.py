import torch
import logging
logger = logging.getLogger("evas_quant")
import onnx
import onnxruntime as ort
import torchvision.transforms as transforms
import torchvision
from utils import prepare_data_loaders, calibrate, print_size_of_model, evaluate
from quant_api import create_model,quant,torch_to_int8_onnx




# set quant_methods
QUANTMAPPING = {
    "ptq": "post_training_static_quant",
    "qat": "quant_aware_training",
}





if __name__ == "__main__":


    import onnxruntime
    print(onnxruntime.get_available_providers())
    # set dataloader
    data_path = '/data1/data/imagenet2012/'
    train_batch_size = 128
    eval_batch_size = 128     
    data_loader, data_loader_test = prepare_data_loaders(data_path, train_batch_size, eval_batch_size)
    criterion = torch.nn.CrossEntropyLoss()
    example_inputs = (next(iter(data_loader_test))[0]) # get an example input
    
    # create fp32 model 
    model = create_model("resnet50")
    ## evaluate the original model

    # print("The original accuracy:")
    # evaluate(model.cuda(), criterion, data_loader_test, torch.device('cuda'))

    # do ptq/qat quant
    quant_mode = "ptq"
    quant_model = quant(model,quant_mode,example_inputs,data_loader_test,1)
    print('quant done')
    
    # check the size of model
    print("fp32 model size is: ")
    print_size_of_model(model)
    print("quantized model size is: ")
    print_size_of_model(quant_model)
    
    # evaluate the int8 model
    # print("The accuracy after the quantization")
    # evaluate(quant_model, criterion, data_loader_test, torch.device('cpu'))
    
    # export int8model to onnx
    image_tensor = torch.rand([1,3,224,224])
    onnx_quant_model = torch_to_int8_onnx(quant_model,example_input=image_tensor,save_path="quant_int_resnet.onnx")
    print('export done')
            