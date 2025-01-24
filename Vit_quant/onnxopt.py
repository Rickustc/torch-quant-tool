import onnx
import onnx.checker
import argparse
import sys
import onnxoptimizer
import pathlib


# SPDX-License-Identifier: Apache-2.0

# ATTENTION: The code in this file is highly EXPERIMENTAL.
# Adventurous users should note that the APIs will probably change.

"""onnx optimizer

This enables users to optimize their models.
"""

import onnx
import onnx.checker
import argparse
import sys
import onnxoptimizer
import pathlib

usage = 'python -m onnxoptimizer input_model.onnx output_model.onnx '


def format_argv(argv):
    argv_ = argv[1:]
    if len(argv_) == 1:
        return argv_
    elif len(argv_) >= 2:
        return argv_[2:]
    else:
        print('please check arguments!')
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog='onnxoptimizer',
        usage=usage,
        description='onnxoptimizer command-line api')
    parser.add_argument('--print_all_passes', action='store_true', default=False, help='print all available passes')
    parser.add_argument('--print_fuse_elimination_passes', action='store_true', default=False, help='print all fuse and elimination passes')
    parser.add_argument('-p', '--passes', nargs='*', default=None, help='list of optimization passes name, if no set, fuse_and_elimination_passes will be used')
    parser.add_argument('--fixed_point', action='store_true', default=False, help='fixed point')
    argv = sys.argv.copy()
    args = parser.parse_args(format_argv(sys.argv))

    all_available_passes = onnxoptimizer.get_available_passes()
    fuse_and_elimination_passes = onnxoptimizer.get_fuse_and_elimination_passes()

    if args.print_all_passes:
        print(*all_available_passes)
        sys.exit(0)

    if args.print_fuse_elimination_passes:
        print(*fuse_and_elimination_passes)
        sys.exit(0)

    passes = args.passes
    if args.passes is None:
        passes = fuse_and_elimination_passes

    if len(argv[1:]) < 2:
        print('usage:{}'.format(usage))
        print('please check arguments!')
        sys.exit(1)

    input_file = argv[1]
    output_file = argv[2]

    if not pathlib.Path(input_file).exists():
        print("input file: {0} no exist!".format(input_file))
        sys.exit(1)

    model = onnx.load(input_file)

    # when model size large than 2G bytes, onnx.checker.check_model(model) will fail.
    # we use onnx.check.check_model(input_file) as workaround
    onnx.checker.check_model(input_file)
    model = onnxoptimizer.optimize(model=model, passes=passes, fixed_point=args.fixed_point)
    if model is None:
        print('onnxoptimizer failed')
        sys.exit(1)
    try:
        onnx.save(proto=model, f=output_file)
    except:
        onnx.save(proto=model, f=output_file, save_as_external_data=True)
    onnx.checker.check_model(output_file)
    
    
    
# all_available_passes = onnxoptimizer.get_available_passes()

# fuse_and_elimination_passes = onnxoptimizer.get_fuse_and_elimination_passes()



all_available_passes = onnxoptimizer.get_available_passes()
input_path = "/home/wrq/VitQuant/vit_fp32.onnx"
output_path = "fuse_matmul_add_vit.onnx"


model = onnx.load(input_path)
onnx.checker.check_model(model)

model = onnxoptimizer.optimize(model=model, passes=["fuse_matmul_add_bias_into_gemm"])
onnx.checker.check_model(output_path)
onnx.save(model, output_path)

'''
PASS:
['rename_input_output',
 'set_unique_name_for_nodes',
 'nop',
 
 # ELIMINATE PASS
 'eliminate_nop_cast', 
 'eliminate_nop_dropout', 
 'eliminate_nop_flatten',
 'extract_constant_to_initializer', 
 'eliminate_if_with_const_cond', 
 'eliminate_nop_monotone_argmax',
 'eliminate_nop_pad', 
 'eliminate_nop_concat', 
 'eliminate_nop_split', 
 'eliminate_nop_expand', 
 'eliminate_shape_gather', 
 'eliminate_slice_after_shape', 
 'eliminate_nop_transpose', 
 
 # FUSE PASS
 'fuse_add_bias_into_conv', 
 'fuse_bn_into_conv', 
 'fuse_consecutive_concats', 
 'fuse_consecutive_log_softmax', 
 'fuse_consecutive_reduce_unsqueeze', 
 'fuse_consecutive_squeezes', 
 'fuse_consecutive_transposes',
 'fuse_matmul_add_bias_into_gemm', 
 'fuse_pad_into_conv',
 'fuse_pad_into_pool', 
 'fuse_transpose_into_gemm', 
 
 
 'replace_einsum_with_matmul', 
 'lift_lexical_references', 
 'split_init',
 'split_predict', 
 'fuse_concat_into_reshape', 
 
 'eliminate_nop_reshape', 
 'eliminate_deadend', 
 'eliminate_identity', 
 'eliminate_shape_op', 
 'eliminate_unused_initializer', 
 'eliminate_duplicate_initializer']
'''
