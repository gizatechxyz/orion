from enum import Enum
import os

from .file_manager import Test, Data, Mod

import numpy as np


class FixedImpl(Enum):
    FP8x23 = 'FP8x23'
    FP16x16 = 'FP16x16'


def to_fp(x: np.ndarray, fp_impl: FixedImpl):
    match fp_impl:
        case FixedImpl.FP8x23:
            return (x * 2**23).astype(np.int64)
        case FixedImpl.FP16x16:
            return (x * 2**16).astype(np.int64)


class Dtype(Enum):
    FP8x23 = 'FP8x23'
    FP16x16 = 'FP16x16'
    I8 = 'i8'
    I32 = 'i32'
    U32 = 'u32'


class Tensor:
    def __init__(self, dtype: Dtype, shape: tuple, data: np.ndarray):
        self.dtype = dtype
        self.shape = shape
        self.data = data


class Trait(Enum):
    TENSOR = 'TENSOR'
    NN = 'NN'


def make_test(inputs: list[Tensor | list[Tensor]], output: Tensor | list[Tensor], func_sig: str, name: str, trait: Trait = Trait.TENSOR):
    Mod().update(name)

    for i, input in enumerate(inputs):
        input_data = Data(os.path.join(name, f"input_{i}.cairo"))
        if isinstance(input, list):
            input_data.buffer = Data.template_sequence(
                func=f"input_{i}",
                dtype=input[0].dtype.value,
                refs=get_data_refs(input[0].dtype),
                data=get_data_data_sequence(input, input[0].dtype),
                shape=[x.shape for x in input],
            )
        else:
            input_data.buffer = Data.template(
                func=f"input_{i}",
                dtype=input.dtype.value,
                refs=get_data_refs(input.dtype),
                data=get_data_data(input.data, input.dtype),
                shape=input.shape,
            )
        input_data.dump()

    output_data = Data(os.path.join(name, "output_0.cairo"))
    if isinstance(output, list):
        output_data.buffer = Data.template_sequence(
            func="output_0",
            dtype=output[0].dtype.value,
            refs=get_data_refs(output[0].dtype),
            data=get_data_data_sequence(output, output[0].dtype),
            shape=[x.shape for x in output],
        )
    else:
        output_data.buffer = Data.template(
            func="output_0",
            dtype=output.dtype.value,
            refs=get_data_refs(output.dtype),
            data=get_data_data(output.data, output.dtype),
            shape=output.shape,
        )
    output_data.dump()

    test_file = Test(f"{name}.cairo")
    test_file.buffer = Test.template(
        name=name,
        arg_cnt=len(inputs),
        refs=get_all_use(find_all_types([*inputs, output]), trait),
        func_sig=func_sig,
    )
    test_file.dump()


def find_all_types(tensors: list[Tensor | list[Tensor]]) -> list[Dtype]:
    dtypes = []
    for tensor in tensors:
        if isinstance(tensor, list):
            dtypes += [x.dtype for x in tensor]
        else:
            dtypes.append(tensor.dtype)

    return list(set(dtypes))


def get_all_use(dtypes: list[Dtype], trait: Trait) -> list[str]:
    refs = []
    for dtype in dtypes:
        refs += get_test_refs(dtype, trait)

    return list(set(refs))


def get_data_data(data: np.ndarray, dtype: Dtype) -> list[str]:
    match dtype:
        case Dtype.U32:
            return [f"{int(x)}" for x in data.flatten()]
        case Dtype.I32:
            return ["i32 { "+f"mag: {abs(int(x))}, sign: {str(x < 0).lower()} "+"}" for x in data.flatten()]
        case Dtype.I8:
            return ["i8 { "+f"mag: {abs(int(x))}, sign: {str(x < 0).lower()} "+"}" for x in data.flatten()]
        case Dtype.FP8x23:
            return ["FP8x23 { "+f"mag: {abs(int(x))}, sign: {str(x < 0).lower()} "+"}" for x in data.flatten()]
        case Dtype.FP16x16:
            return ["FP16x16 { "+f"mag: {abs(int(x))}, sign: {str(x < 0).lower()} "+"}" for x in data.flatten()]


def get_data_data_sequence(data: list[Tensor], dtype: Dtype) -> list[list[str]]:
    return [get_data_data(x.data, dtype) for x in data]


def get_data_refs(dtype: Dtype) -> list[str]:
    refs = [
        *trait2use[Trait.TENSOR],
        *dtype2tensor[dtype],
        *dtype2numbers[dtype],
    ]

    return refs


def get_test_refs(dtype: Dtype, trait: Trait) -> list[str]:
    dtype_ref = dtype2nn[dtype] if trait == Trait.NN else dtype2tensor[dtype]
    refs = [
        *trait2use[trait],
        *dtype_ref,
        *dtype2partial[dtype],
        "orion::utils::assert_eq",
        ]

    return refs


trait2use = {
    Trait.TENSOR: [
        "array::{ArrayTrait, SpanTrait}",
        "orion::operators::tensor::{TensorTrait, Tensor}",
    ],
    Trait.NN: [
        "orion::numbers::FixedTrait",
        "orion::operators::nn::NNTrait",
    ],
}


dtype2tensor = {
    Dtype.U32: ["orion::operators::tensor::U32Tensor",],
    Dtype.I32: ["orion::operators::tensor::I32Tensor",],
    Dtype.I8: ["orion::operators::tensor::I8Tensor",],
    Dtype.FP8x23: ["orion::operators::tensor::FP8x23Tensor",],
    Dtype.FP16x16: ["orion::operators::tensor::FP16x16Tensor",],
}


dtype2nn = {
    Dtype.U32: ["orion::operators::nn::U32NN",],
    Dtype.I32: ["orion::operators::nn::I32NN",],
    Dtype.I8: ["orion::operators::nn::I8NN",],
    Dtype.FP8x23: ["orion::operators::nn::FP8x23NN",],
    Dtype.FP16x16: ["orion::operators::nn::FP16x16NN",],
}


dtype2partial = {
    Dtype.U32: ["orion::operators::tensor::U32TensorPartialEq",],
    Dtype.I32: ["orion::operators::tensor::I32TensorPartialEq",],
    Dtype.I8: ["orion::operators::tensor::I8TensorPartialEq",],
    Dtype.FP8x23: ["orion::operators::tensor::FP8x23TensorPartialEq",],
    Dtype.FP16x16: ["orion::operators::tensor::FP16x16TensorPartialEq",],
}


dtype2numbers = {
    Dtype.U32: [],
    Dtype.I32: ["orion::numbers::{IntegerTrait, i32}",],
    Dtype.I8: ["orion::numbers::{IntegerTrait, i8}",],
    Dtype.FP8x23: ["orion::numbers::{FixedTrait, FP8x23}",],
    Dtype.FP16x16: ["orion::numbers::{FixedTrait, FP16x16}",],
}
