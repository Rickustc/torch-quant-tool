import copy
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torchvision.models import resnet50
import torch

fp32_model = resnet50().eval()
model = copy.deepcopy(fp32_model)

qconfig = get_default_qconfig("fbgemm")
qconfig_dict = {"": qconfig}
# `prepare_fx` inserts observers in the model based on the configuration in `qconfig_dict`
model_prepared = prepare_fx(model, qconfig_dict, example_inputs=(torch.randn(1, 3, 256, 256),),)
# calibration runs the model with some sample data, which allows observers to record the statistics of
# the activation and weigths of the operators
calibration_data = [torch.randn(1, 3, 224, 224) for _ in range(100)]
for i in range(len(calibration_data)):
   model_prepared(calibration_data[i])
# `convert_fx` converts a calibrated model to a quantized model, this includes inserting
# quantize, dequantize operators to the model and swap floating point operators with quantized operators
model_quantized = convert_fx(copy.deepcopy(model_prepared))
# benchmark
x = torch.randn(1, 3, 224, 224)
# %timeit fp32_model(x)
# %timeit model_quantized(x)




import torch.ao.ns._numeric_suite_fx as ns

resnet50_wt_compare_dict = ns.extract_weights(
    'fp32',  # string name for model A
    model_prepared,  # model A
    'int8',  # string name for model B
    model_quantized,  # model B
)

ns.extend_logger_results_with_comparison(
    resnet50_wt_compare_dict,  # results object to modify inplace
    'fp32',  # string name of model A (from previous step)
    'int8',  # string name of model B (from previous step)
    torch.ao.ns.fx.utils.compute_sqnr,  # the function to use to compare two tensors
    'sqnr',  # the name to use to store the results under
)

resnet50_wt_to_print = []
for idx, (layer_name, v) in enumerate(resnet50_wt_compare_dict.items()):
    resnet50_wt_to_print.append([
        idx,
        layer_name,                                                   
        v['weight']['int8'][0]['prev_node_target_type'],                      
        v['weight']['int8'][0]['values'][0].shape,
        v['weight']['int8'][0]['sqnr'][0],
    ])


import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
# a simple line graph
def plot(xdata, ydata, xlabel, ylabel, title):
    fig = plt.figure(figsize=(10, 5), dpi=100)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    ax = plt.axes()
    ax.plot(xdata, ydata)  
    plt.savefig("ns.png")   
# plot the SQNR between fp32 and int8 weights for each layer
# Note: we may explore easier to read charts (bar chart, etc) at a later time, for now
# line chart + table is good enough.
plot([x[0] for x in resnet50_wt_to_print], [x[4] for x in resnet50_wt_to_print], 'idx', 'sqnr', 'weights, idx to sqnr')



# Comparing activations, with error propagation
#

# add loggers
mp_ns, mq_ns = ns.add_loggers(
    'a', copy.deepcopy(model_prepared),
    'b', copy.deepcopy(model_quantized),
    ns.OutputLogger)

# send an example datum to capture intermediate activations
datum = torch.randn(1, 1, 1, 1)
mp_ns(datum)
mq_ns(datum)

# extract intermediate activations
act_comparison = ns.extract_logger_info(
    mp_ns, mq_ns, ns.OutputLogger, 'b')

# add SQNR for each comparison, inplace
ns.extend_logger_results_with_comparison(
    act_comparison, 'a', 'b', torch.ao.ns.fx.utils.compute_sqnr,
    'sqnr')


# act_comparison contains the activations from `mp_ns` and `mq_ns` stored
# in pairs, and can be used for further analysis.

#
# Comparing activations, without error propagation
#

# create shadow model
mp_shadows_mq = ns.add_shadow_loggers(
    'a', copy.deepcopy(model_prepared),
    'b', copy.deepcopy(model_quantized),
    ns.OutputLogger)

# send an example datum to capture intermediate activations
datum = torch.randn(1, 1, 1, 1)
mp_shadows_mq(datum)

# extract intermediate activations
shadow_act_comparison = ns.extract_shadow_logger_info(
    mp_shadows_mq, ns.OutputLogger, 'b')

# add SQNR for each comparison, inplace
ns.extend_logger_results_with_comparison(
    shadow_act_comparison, 'a', 'b', torch.ao.ns.fx.utils.compute_sqnr,
    'sqnr')

# shadow_act_comparison contains the activations from `mp_ns` and `mq_ns` stored
# in pairs, and can be used for further analysis.