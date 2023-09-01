from enum import Enum
import os
import re

import numpy as np

######################
#   DATA STRUCTURES
######################


class Dtype(Enum):
    FP8x23 = 'FP8x23'
    FP16x16 = 'FP16x16'
    I8 = 'I8'
    I32 = 'I32'
    U32 = 'U32'


class FixedImpl(Enum):
    FP8x23 = 'FP8x23'
    FP16x16 = 'FP16x16'


class Tensor:
    def __init__(self, dtype: Dtype, shape: [], data: [], extra_fp=FixedImpl.FP16x16):
        self.dtype = dtype
        self.shape = shape
        self.data = data
        self.extra_fp = extra_fp


class Trait(Enum):
    TENSOR = 'TENSOR'
    NN = 'NN'

################
#   EXTERNALS
################


def make_node(inputs: [Tensor], outputs: [Tensor], dir_name, path="src/tests/nodes/"):

    path = path + dir_name

    for i, input in enumerate(inputs):
        __generate_data(input, path, f"input_{i}")

    for i, output in enumerate(outputs):
        __generate_data(output, path, f"output_{i}")


def make_test(inputs: [Tensor], output: Tensor, func_sig: str, file_name: str, trait_type: Trait = Trait.TENSOR):

    code = []
    type_of_first_input = inputs[0].dtype
    type_of_output = output.dtype
    func_sig = re.sub("^[^.]*", "input_0",
                      func_sig) if trait_type == Trait.TENSOR else func_sig

    match  trait_type:
        case Trait.TENSOR:
            code.append("\n\nuse array::{ArrayTrait, SpanTrait};\n")
            code.append("use orion::operators::tensor::TensorTrait;\n")
            match type_of_first_input:
                case Dtype.U32:
                    match inputs[0].extra_fp:
                        case FixedImpl.FP8x23:
                            code.append(
                                "use orion::operators::tensor::Tensor_u32_fp8x23;\n")
                        case FixedImpl.FP16x16:
                            code.append(
                                "use orion::operators::tensor::Tensor_u32_fp16x16;\n")
                case Dtype.I32:
                    match inputs[0].extra_fp:
                        case FixedImpl.FP8x23:
                            code.append(
                                "use orion::operators::tensor::Tensor_i32_fp8x23;\n")
                        case FixedImpl.FP16x16:
                            code.append(
                                "use orion::operators::tensor::Tensor_i32_fp16x16;\n")
                case Dtype.I8:
                    match inputs[0].extra_fp:
                        case FixedImpl.FP8x23:
                            code.append(
                                "use orion::operators::tensor::Tensor_i8_fp8x23;\n")
                        case FixedImpl.FP16x16:
                            code.append(
                                "use orion::operators::tensor::Tensor_i8_fp16x16;\n")
                case Dtype.FP8x23:
                    code.append(
                        "use orion::operators::tensor::Tensor_fp8x23;\n")
                case Dtype.FP16x16:
                    code.append(
                        "use orion::operators::tensor::Tensor_fp16x16;\n")
            match type_of_output:
                case Dtype.U32:
                    match inputs[0].extra_fp:
                        case FixedImpl.FP8x23:
                            code.append(
                                "use orion::operators::tensor::implementations::tensor_u32_fp8x23::u32TensorPartialEq;\n")
                        case FixedImpl.FP16x16:
                            code.append(
                                "use orion::operators::tensor::implementations::tensor_u32_fp16x16::u32TensorPartialEq;\n")
                case Dtype.I32:
                    match inputs[0].extra_fp:
                        case FixedImpl.FP8x23:
                            code.append(
                                "use orion::operators::tensor::implementations::tensor_i32_fp8x23::i32TensorPartialEq;\n")
                        case FixedImpl.FP16x16:
                            code.append(
                                "use orion::operators::tensor::implementations::tensor_i32_fp16x16::i32TensorPartialEq;\n")
                case Dtype.I8:
                    match inputs[0].extra_fp:
                        case FixedImpl.FP8x23:
                            code.append(
                                "use orion::operators::tensor::implementations::tensor_i8_fp8x23::i8TensorPartialEq;\n")
                        case FixedImpl.FP16x16:
                            code.append(
                                "use orion::operators::tensor::implementations::tensor_i8_fp16x16::i8TensorPartialEq;\n")
                case Dtype.FP16x16:
                    code.append(
                        "use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16TensorPartialEq;\n")
                case Dtype.FP8x23:
                    code.append(
                        "use orion::operators::tensor::implementations::tensor_fp8x23::FP8x23TensorPartialEq;\n")
        case Trait.NN:
            code.append("\n\nuse orion::operators::nn::core::NNTrait;\n")
            code.append("use orion::numbers::FixedTrait;\n")
            match type_of_first_input:
                case Dtype.I32:
                    match inputs[0].extra_fp:
                        case FixedImpl.FP8x23:
                            code.append(
                                "use orion::operators::nn::NN_i32_fp8x23;\n")
                        case FixedImpl.FP16x16:
                            code.append(
                                "use orion::operators::nn::NN_i32_fp16x16;\n")
                case Dtype.I8:
                    match inputs[0].extra_fp:
                        case FixedImpl.FP8x23:
                            code.append(
                                "use orion::operators::nn::NN_i8_fp8x23;\n")
                        case FixedImpl.FP16x16:
                            code.append(
                                "use orion::operators::nn::NN_i8_fp16x16;\n")
                case Dtype.U32:
                    match inputs[0].extra_fp:
                        case FixedImpl.FP8x23:
                            code.append(
                                "use orion::operators::nn::NN_u32_fp8x23;\n")
                        case FixedImpl.FP16x16:
                            code.append(
                                "use orion::operators::nn::NN_u32_fp16x16;\n")
                case Dtype.FP8x23:
                    code.append(
                        "use orion::operators::nn::NN_fp8x23;\n")
                case Dtype.FP16x16:
                    code.append(
                        "use orion::operators::nn::NN_fp16x16;\n")
            match inputs[0].extra_fp:
                case FixedImpl.FP8x23:
                    code.append(
                        "use orion::numbers::FP8x23Impl;\n")
                case FixedImpl.FP16x16:
                    code.append(
                        "use orion::numbers::FP16x16Impl;\n")
            match type_of_output:
                case Dtype.U32:
                    match inputs[0].extra_fp:
                        case FixedImpl.FP8x23:
                            code.append(
                                "use orion::operators::tensor::implementations::tensor_u32_fp8x23::u32TensorPartialEq;\n")
                        case FixedImpl.FP16x16:
                            code.append(
                                "use orion::operators::tensor::implementations::tensor_u32_fp16x16::u32TensorPartialEq;\n")
                case Dtype.I32:
                    match inputs[0].extra_fp:
                        case FixedImpl.FP8x23:
                            code.append(
                                "use orion::operators::tensor::implementations::tensor_i32_fp8x23::i32TensorPartialEq;\n")
                        case FixedImpl.FP16x16:
                            code.append(
                                "use orion::operators::tensor::implementations::tensor_i32_fp16x16::i32TensorPartialEq;\n")
                case Dtype.I8:
                    match inputs[0].extra_fp:
                        case FixedImpl.FP8x23:
                            code.append(
                                "use orion::operators::tensor::implementations::tensor_i8_fp8x23::i8TensorPartialEq;\n")
                        case FixedImpl.FP16x16:
                            code.append(
                                "use orion::operators::tensor::implementations::tensor_i8_fp16x16::i8TensorPartialEq;\n")
                case Dtype.FP16x16:
                    code.append(
                        "use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16TensorPartialEq;\n")
                case Dtype.FP8x23:
                    code.append(
                        "use orion::operators::tensor::implementations::tensor_fp8x23::FP8x23TensorPartialEq;\n")

    code.append("use orion::utils::assert_eq;\n\n")
    code.append("#[test]\n")
    code.append("#[available_gas(2000000000)]\n")
    code.append(f"fn test_{file_name}() {{\n")

    for i, input in enumerate(inputs):
        code.append(f"    let input_{i} = input_{i}::input_{i}();\n")

    code.append("    let z = output_0::output_0();\n\n")

    code.append(f"    let y = {func_sig};\n\n")
    code.append("    assert_eq(y, z);\n")
    code.append("}")

    with open(os.path.join("src/tests/nodes", f"{file_name}.cairo"), "a") as f:
        f.write(
            ''.join(code)
        )


def to_fp(x: np.array, fp_impl: FixedImpl):

    match fp_impl:
        case FixedImpl.FP8x23:
            return (x * 2**23).astype(np.int64)
        case FixedImpl.FP16x16:
            return (x * 2**16).astype(np.int64)

################
#   INTERNALS
################


def __build_tensor_code(tensor: Tensor, name: str, type_string: str, is_fixed: bool = False, is_signed_int: bool = False) -> []:
    result = [
        "use array::{ArrayTrait, SpanTrait};\n",
        "use orion::operators::tensor::{TensorTrait, Tensor};\n",
    ]

    match tensor.dtype:
        case Dtype.U32:
            match tensor.extra_fp:
                case FixedImpl.FP8x23:
                    result.append(
                        "use orion::operators::tensor::Tensor_u32_fp8x23;\n")
                case FixedImpl.FP16x16:
                    result.append(
                        "use orion::operators::tensor::Tensor_u32_fp16x16;\n")
        case Dtype.I32:
            match tensor.extra_fp:
                case FixedImpl.FP8x23:
                    result.append(
                        "use orion::operators::tensor::Tensor_i32_fp8x23;\n")
                case FixedImpl.FP16x16:
                    result.append(
                        "use orion::operators::tensor::Tensor_i32_fp16x16;\n")
            result.append(
                "use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};\n")
        case Dtype.I8:
            match tensor.extra_fp:
                case FixedImpl.FP8x23:
                    result.append(
                        "use orion::operators::tensor::Tensor_i8_fp8x23;\n")
                case FixedImpl.FP16x16:
                    result.append(
                        "use orion::operators::tensor::Tensor_i8_fp16x16;\n")
            result.append(
                "use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};\n")
        case Dtype.FP8x23:
            result.append(
                "use orion::operators::tensor::Tensor_fp8x23;\n")
            result.append(
                "use orion::numbers::fixed_point::core::{FixedTrait};\n")
            result.append(
                "use orion::numbers::FP8x23;\n")
        case Dtype.FP16x16:
            result.append(
                "use orion::operators::tensor::Tensor_fp16x16;\n")
            result.append(
                "use orion::numbers::fixed_point::core::{FixedTrait};\n")
            result.append(
                "use orion::numbers::FP16x16;\n")

    result.append(f"\nfn {name}() -> Tensor<{type_string}> {{\n")
    result.append("    let mut shape = ArrayTrait::<usize>::new();\n")
    for dim in tensor.shape:
        result.append(f"    shape.append({dim});\n")
    result.append("\n    let mut data = ArrayTrait::new();\n")
    if is_signed_int | is_fixed:
        for val in tensor.data:
            result.append(
                f"    data.append({type_string} {{ mag: {abs(int(val))}, sign: {str(val < 0).lower()} }});\n")
    else:
        for val in tensor.data:
            result.append(f"    data.append({abs(int(val))});\n")
    result.append(
        "    TensorTrait::new(shape.span(), data.span())\n")
    result.append("}")
    return result


def __convert_tensor_to_cairo(tensor: Tensor, name: str) -> []:
    dtype_mapping = {
        Dtype.FP8x23: ('FP8x23',  True, False),
        Dtype.FP16x16: ('FP16x16', True, False),
        Dtype.I32: ('i32', False, True),
        Dtype.I8: ('i8', False, True),
        Dtype.U32: ('u32', False, False),
    }

    dtype_info = dtype_mapping.get(tensor.dtype)
    if dtype_info is None:
        raise ValueError(f"Invalid dtype: {tensor.dtype}")

    return __build_tensor_code(tensor, name, *dtype_info)


def __generate_data(tensor: Tensor, path: str, name: str):

    # If path not exist:
    # Create directory
    # Add mod parent to nodes.cairo
    if not os.path.exists(path) or not os.listdir(path):
        os.makedirs(path, exist_ok=True)
        parent = path.replace("src/tests/nodes/", "")
        with open("src/tests/nodes.cairo", "a") as f:
            f.write(f"mod {parent}; \n")

    # Add tensor mod in parent file
    parent = path.replace("src/tests/nodes/", "")
    if not __module_exists(os.path.join("src/tests/nodes/", f"{parent}.cairo"), name):
        with open(os.path.join("src/tests/nodes/", f"{parent}.cairo"), "a") as f:
            f.write(f"mod {name}; \n")

    # Convert tensor to cairo
    content = __convert_tensor_to_cairo(tensor, name)
    # Create tensor cairo file
    with open(os.path.join(path, f"{name}.cairo"), "w") as f:
        f.write(
            ''.join(content)
        )


def __module_exists(filepath: str, mod_name: str) -> bool:
    """
    Checks if a module already exists in a file.

    Parameters:
    - filepath: The path to the file to check.
    - mod_name: The module to look for.

    Returns:
    - True if the module exists, False otherwise.
    """
    if not os.path.exists(filepath):
        return False

    with open(filepath, 'r') as f:
        for line in f:
            if f"mod {mod_name};" in line:
                return True

    return False
