from enum import Enum
import os
from typing import List

from .file_manager import CairoTest, CairoData, ModFile

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
    BOOL = 'bool'


class Tensor:
    def __init__(self, dtype: Dtype, shape: tuple, data: np.ndarray):
        self.dtype = dtype
        self.shape = shape
        self.data = data


Sequence = List[Tensor]


class Trait(Enum):
    TENSOR = 'TENSOR'
    NN = 'NN'


def make_test(inputs: list[Tensor | Sequence], output: Tensor | Sequence, func_sig: str, name: str, trait: Trait = Trait.TENSOR):
    """
    Generate and write Cairo tests based on the provided inputs and output.

    Args:
        inputs (list[Tensor | list[Tensor]]): A list of input tensors or tensor sequences.
        output (Tensor | list[Tensor]): The expected output tensor or tensor sequences.
        func_sig (str): The signature of the function to be tested.
        name (str): The name of the test.
        trait (Trait, optional): The trait of the tensors. Defaults to Trait.TENSOR.
    """
    ModFile().update(name)

    for i, input in enumerate(inputs):
        input_data = CairoData(os.path.join(name, f"input_{i}.cairo"))
        match input:
            case list():
                input_data.buffer = CairoData.sequence_template(
                    func=f"input_{i}",
                    dtype=input[0].dtype.value,
                    refs=get_data_refs(input[0].dtype),
                    data=get_data_statement_for_sequences(input, input[0].dtype),
                    shape=[x.shape for x in input],
                )
            case Tensor():
                input_data.buffer = CairoData.base_template(
                    func=f"input_{i}",
                    dtype=input.dtype.value,
                    refs=get_data_refs(input.dtype),
                    data=get_data_statement(input.data, input.dtype),
                    shape=input.shape,
                )

        input_data.dump()

    output_data = CairoData(os.path.join(name, "output_0.cairo"))
    match output:
        case list():
            output_data.buffer = CairoData.sequence_template(
                func="output_0",
                dtype=output[0].dtype.value,
                refs=get_data_refs(output[0].dtype),
                data=get_data_statement_for_sequences(output, output[0].dtype),
                shape=[x.shape for x in output],
            )
        case Tensor():
            output_data.buffer = CairoData.base_template(
                func="output_0",
                dtype=output.dtype.value,
                refs=get_data_refs(output.dtype),
                data=get_data_statement(output.data, output.dtype),
                shape=output.shape,
            )

    output_data.dump()

    test_file = CairoTest(f"{name}.cairo")
    match output:
        case list():
            test_file.buffer = CairoTest.sequence_template(
                name=name,
                arg_cnt=len(inputs),
                refs=get_all_test_refs(find_all_types([*inputs, *output]), trait),
                func_sig=func_sig,
            )
        case Tensor():
            test_file.buffer = CairoTest.base_template(
                name=name,
                arg_cnt=len(inputs),
                refs=get_all_test_refs(find_all_types([*inputs, output]), trait),
                func_sig=func_sig,
            )

    test_file.dump()


def get_data_refs(dtype: Dtype) -> list[str]:
    refs = [
        *trait_to_ref[Trait.TENSOR],
        *dtype_to_tensor[dtype],
        *dtype_to_numbers[dtype],
    ]

    return refs


def get_data_statement(data: np.ndarray, dtype: Dtype) -> list[str]:
    match dtype:
        case Dtype.U32:
            return [f"{int(x)}" for x in data.flatten()]
        case Dtype.I32:
            return [f"{int(x)}" for x in data.flatten()]
        case Dtype.I8:
            return [f"{int(x)}" for x in data.flatten()]
        case Dtype.FP8x23:
            return ["FP8x23 { "+f"mag: {abs(int(x))}, sign: {str(x < 0).lower()} "+"}" for x in data.flatten()]
        case Dtype.FP16x16:
            return ["FP16x16 { "+f"mag: {abs(int(x))}, sign: {str(x < 0).lower()} "+"}" for x in data.flatten()]
        case Dtype.BOOL:
            return [str(x).lower() for x in data.flatten()]


def get_data_statement_for_sequences(data: Sequence, dtype: Dtype) -> list[list[str]]:
    return [get_data_statement(x.data, dtype) for x in data]


def get_all_test_refs(dtypes: list[Dtype], trait: Trait) -> list[str]:
    refs = []
    for dtype in dtypes:
        # refs += [*dtype_to_numbers[dtype]]
        refs += get_test_refs(dtype, trait)

    return list(set(refs))


def get_test_refs(dtype: Dtype, trait: Trait) -> list[str]:
    if trait == Trait.NN and dtype == Dtype.BOOL:
        raise Exception("NN trait does not support bool dtype")

    dtype_ref = dtype_to_nn[dtype] if trait == Trait.NN else dtype_to_tensor[dtype]
    refs = [
        *trait_to_ref[trait],
        *dtype_ref,
        *dtype_to_partial_eq[dtype],
        "orion::utils::{assert_eq, assert_seq_eq}",
        ]

    return refs


def find_all_types(tensors: list[Tensor | Sequence]) -> list[Dtype]:
    dtypes = []
    for tensor in tensors:
        if isinstance(tensor, list):
            dtypes += [x.dtype for x in tensor]
        else:
            dtypes.append(tensor.dtype)

    return list(set(dtypes))


trait_to_ref = {
    Trait.TENSOR: [
        "core::array::{ArrayTrait, SpanTrait}",
        "orion::operators::tensor::{TensorTrait, Tensor}",
    ],
    Trait.NN: [
        "orion::numbers::FixedTrait",
        "orion::operators::nn::NNTrait",
    ],
}


dtype_to_tensor = {
    Dtype.U32: ["orion::operators::tensor::{U32Tensor, U32TensorAdd}",],
    Dtype.I32: ["orion::operators::tensor::{I32Tensor, I32TensorAdd}",],
    Dtype.I8: ["orion::operators::tensor::{I8Tensor, I8TensorAdd}",],
    Dtype.FP8x23: ["orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd}",],
    Dtype.FP16x16: ["orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd}",],
    Dtype.BOOL: ["orion::operators::tensor::BoolTensor",],
}


dtype_to_nn = {
    Dtype.U32: ["orion::operators::nn::U32NN",],
    Dtype.I32: ["orion::operators::nn::I32NN",],
    Dtype.I8: ["orion::operators::nn::I8NN",],
    Dtype.FP8x23: ["orion::operators::nn::FP8x23NN",],
    Dtype.FP16x16: ["orion::operators::nn::FP16x16NN",],
}


dtype_to_partial_eq = {
    Dtype.U32: ["orion::operators::tensor::U32TensorPartialEq",],
    Dtype.I32: ["orion::operators::tensor::I32TensorPartialEq",],
    Dtype.I8: ["orion::operators::tensor::I8TensorPartialEq",],
    Dtype.FP8x23: ["orion::operators::tensor::FP8x23TensorPartialEq",],
    Dtype.FP16x16: ["orion::operators::tensor::FP16x16TensorPartialEq",],
    Dtype.BOOL: ["orion::operators::tensor::BoolTensorPartialEq",],
}


dtype_to_numbers = {
    Dtype.U32: [],
    Dtype.I32: [],
    Dtype.I8: [],
    Dtype.FP8x23: ["orion::numbers::{FixedTrait, FP8x23}",],
    Dtype.FP16x16: ["orion::numbers::{FixedTrait, FP16x16}",],
    Dtype.BOOL: [],
}
