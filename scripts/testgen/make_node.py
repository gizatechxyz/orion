from enum import Enum
import os


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
    def __init__(self, dtype: Dtype, shape: [], data: [], extra_fp: FixedImpl):
        self.dtype = dtype
        self.shape = shape
        self.data = data
        self.extra_fp = extra_fp


def _build_tensor_code(tensor: Tensor, name: str, type_string: str, is_fixed: bool = False, is_signed_int: bool = False) -> []:
    result = [
        f"use array::ArrayTrait;\n",
        f"use orion::operators::tensor::core::{{TensorTrait, Tensor, ExtraParams}};\n",
    ]

    if is_fixed:
        result.append(
            f"use orion::numbers::fixed_point::core::{{FixedTrait, FixedType, FixedImpl}};\n")
        result.append(
            f"use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;\n")
    else:
        result.append(
            f"use orion::numbers::fixed_point::core::FixedImpl;\n")
        result.append(
            f"use orion::operators::tensor::implementations::impl_tensor_{type_string.lower()}::Tensor_{type_string.lower()};\n")
        if is_signed_int:
            result.append(
                f"use orion::numbers::signed_integer::{{integer_trait::IntegerTrait, {type_string}::{type_string}}};\n")

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
        f"\n    let extra = ExtraParams {{ fixed_point: Option::Some(FixedImpl::{tensor.extra_fp.name}) }};\n")
    result.append(
        "    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))\n")
    result.append("}")
    return result


def convert_tensor_to_cairo(tensor: Tensor, name: str) -> []:
    dtype_mapping = {
        Dtype.FP8x23: ('FixedType',  True, False),
        Dtype.FP16x16: ('FixedType', True, False),
        Dtype.I32: ('i32', False, True),
        Dtype.I8: ('i8', False, True),
        Dtype.U32: ('u32', False, False),
    }

    dtype_info = dtype_mapping.get(tensor.dtype)
    if dtype_info is None:
        raise ValueError(f"Invalid dtype: {tensor.dtype}")

    return _build_tensor_code(tensor, name, *dtype_info)


def generate_data(tensor: Tensor, path: str, name: str):

    # If path not exist:
    # Create directory
    # Add mod parent to node.cairo
    if not os.path.exists(path):
        os.makedirs(path)
        parent = path.replace("src/tests/data/node/", "")
        with open("src/tests/data/node.cairo", "a") as f:
            f.write(f"mod {parent}; \n")

    # Add tensor mod in parent file
    parent = path.replace("src/tests/data/node/", "")
    with open(os.path.join("src/tests/data/node/", f"{parent}.cairo"), "a") as f:
        f.write(f"mod {name}; \n")

    # Convert tensor to cairo
    content = convert_tensor_to_cairo(tensor, name)
    # Create tensor cairo file
    with open(os.path.join(path, f"{name}.cairo"), "w") as f:
        f.write(
            ''.join(content)
        )


def make_node(inputs: [Tensor], outputs: [Tensor], dir_name):

    path = "src/tests/data/node/" + dir_name

    for i, input in enumerate(inputs):
        generate_data(input, path, f"input_{i}")

    for i, output in enumerate(outputs):
        generate_data(output, path, f"output_{i}")


# def test():
#     # Create tensor with Dtype FP8x23
#     input = Tensor(
#         Dtype.FP8x23, [2, 2], [1.23, -4.56, 7.89, 0.12], FixedImpl.FP16x16
#     )

#     output = Tensor(
#         Dtype.FP8x23, [2, 2], [10, 20, 30, 40], FixedImpl.FP16x16
#     )

#     make_node([input], [output], "relu_2D")


# test()
