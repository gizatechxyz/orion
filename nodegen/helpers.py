from enum import Enum
import os
from typing import List

from .file_manager import CairoTest, CairoData, ModFile

import numpy as np


class FixedImpl(Enum):
    FP8x23 = 'FP8x23'
    FP16x16 = 'FP16x16'
    FP64x64 = 'FP64x64'
    


def to_fp(x: np.ndarray, fp_impl: FixedImpl):
    match fp_impl:
        case FixedImpl.FP8x23:
            return (x * 2**23).astype(np.int64)
        case FixedImpl.FP16x16:
            return (x * 2**16).astype(np.int64)
        case FixedImpl.FP64x64:
            return (x * 2**64)


class Dtype(Enum):
    FP8x23 = 'FP8x23'
    FP16x16 = 'FP16x16'
    FP64x64 = 'FP64x64'
    I8 = 'i8'
    I32 = 'i32'
    U32 = 'u32'
    BOOL = 'bool'
    COMPLEX64 = 'complex64'


class Tensor:
    def __init__(self, dtype: Dtype, shape: tuple, data: np.ndarray):
        self.dtype = dtype
        self.shape = shape
        self.data = data


Sequence = List[Tensor]


class Trait(Enum):
    TENSOR = 'TENSOR'
    NN = 'NN'
    SEQUENCE = 'SEQUENCE'


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
                    data=get_data_statement_for_sequences(
                        input, input[0].dtype),
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

    match output:
        case list():
            output_data = CairoData(os.path.join(name, "output_0.cairo"))
            output_data.buffer = CairoData.sequence_template(
                func="output_0",
                dtype=output[0].dtype.value,
                refs=get_data_refs(output[0].dtype),
                data=get_data_statement_for_sequences(output, output[0].dtype),
                shape=[x.shape for x in output],
            )
            output_data.dump()

        case tuple():
            for i, out in enumerate(output):
                output_data = CairoData(
                    os.path.join(name, f"output_{i}.cairo"))
                output_data.buffer = CairoData.base_template(
                    func=f"output_{i}",
                    dtype=out.dtype.value,
                    refs=get_data_refs(out.dtype),
                    data=get_data_statement(out.data, out.dtype),
                    shape=out.shape,
                )
                output_data.dump()

        case Tensor():
            output_data = CairoData(os.path.join(name, "output_0.cairo"))
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
        case tuple():
            test_file.buffer = CairoTest.base_template(
                name=name,
                arg_cnt=len(inputs),
                out_cnt=len(output),
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
        case Dtype.FP64x64:
            return ["FP64x64 { "+f"mag: {abs(int(x))}, sign: {str(x < 0).lower()} "+"}" for x in data.flatten()]
        case Dtype.BOOL:
            return [str(x).lower() for x in data.flatten()]
        case Dtype.COMPLEX64:
            return ["complex64 { "+"real: FP64x64 { "+f"mag: {abs(int(np.real(x)))}, sign: {str(np.real(x) < 0).lower()} "+"} , img: FP64x64 { "+f"mag: {abs(int(np.imag(x)))}, sign: {str(np.imag(x) < 0).lower()} "+"} }" for x in data.flatten()]
        




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

    if trait == Trait.NN:
        dtype_ref = dtype_to_nn[dtype]
    elif trait == Trait.SEQUENCE:
        dtype_ref = dtype_to_sequence[dtype]
    else:
        dtype_ref = dtype_to_tensor[dtype]

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
        if isinstance(tensor, list) or isinstance(tensor, tuple):
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
    Trait.SEQUENCE: [
        "array::{ArrayTrait, SpanTrait}",
        "orion::operators::sequence::SequenceTrait",
    ],
}


dtype_to_tensor = {
    Dtype.U32: ["orion::operators::tensor::{U32Tensor, U32TensorAdd}",],
    Dtype.I32: ["orion::operators::tensor::{I32Tensor, I32TensorAdd}",],
    Dtype.I8: ["orion::operators::tensor::{I8Tensor, I8TensorAdd}",],
    Dtype.FP8x23: ["orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd}",],
    Dtype.FP16x16: ["orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd}",],
    Dtype.BOOL: ["orion::operators::tensor::BoolTensor",],
    Dtype.COMPLEX64: ["orion::operators::tensor::Complex64Tensor",],
}


dtype_to_nn = {
    Dtype.U32: ["orion::operators::nn::U32NN",],
    Dtype.I32: ["orion::operators::nn::I32NN",],
    Dtype.I8: ["orion::operators::nn::I8NN",],
    Dtype.FP8x23: ["orion::operators::nn::FP8x23NN",],
    Dtype.FP16x16: ["orion::operators::nn::FP16x16NN",],
}


dtype_to_sequence = {
    Dtype.U32: ["orion::operators::sequence::U32Sequence",],
    Dtype.I32: ["orion::operators::sequence::I32Sequence",],
    Dtype.I8: ["orion::operators::sequence::I8Sequence",],
    Dtype.FP8x23: ["orion::operators::sequence::FP8x23Sequence",],
    Dtype.FP16x16: ["orion::operators::sequence::FP16x16Sequence",],
}


dtype_to_partial_eq = {
    Dtype.U32: ["orion::operators::tensor::U32TensorPartialEq",],
    Dtype.I32: ["orion::operators::tensor::I32TensorPartialEq",],
    Dtype.I8: ["orion::operators::tensor::I8TensorPartialEq",],
    Dtype.FP8x23: ["orion::operators::tensor::FP8x23TensorPartialEq",],
    Dtype.FP16x16: ["orion::operators::tensor::FP16x16TensorPartialEq",],
    Dtype.BOOL: ["orion::operators::tensor::BoolTensorPartialEq",],
    Dtype.COMPLEX64: ["orion::operators::tensor::Complex64TensorPartialEq",],
}


dtype_to_numbers = {
    Dtype.U32: ["orion::numbers::NumberTrait"],
    Dtype.I32: ["orion::numbers::NumberTrait"],
    Dtype.I8: ["orion::numbers::NumberTrait"],
    Dtype.FP8x23: ["orion::numbers::{FixedTrait, FP8x23}",],
    Dtype.FP16x16: ["orion::numbers::{FixedTrait, FP16x16}",],
    Dtype.BOOL: [],
    Dtype.COMPLEX64: ["orion::numbers::{NumberTrait, complex64}",],
}