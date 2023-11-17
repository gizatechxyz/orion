import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait


class Sequence_construct(RunAll):

    @staticmethod
    def sequence_construct_u32():
        sequence = []
        tensor_cnt = np.random.randint(1, 10)
        shape = np.random.randint(1, 4, 2)

        for _ in range(tensor_cnt):
            values = np.random.randint(0, 6, shape).astype(np.uint32)
            tensor = Tensor(Dtype.U32, values.shape, values.flatten())

            sequence.append(tensor)

        name = "sequence_construct_u32"
        make_test([sequence], sequence, "SequenceTrait::sequence_construct(input_0)", name, Trait.SEQUENCE)


    @staticmethod
    def sequence_construct_i32():
        sequence = []
        tensor_cnt = np.random.randint(1, 10)
        shape = np.random.randint(1, 4, 2)

        for _ in range(tensor_cnt):
            values = np.random.randint(-6, 6, shape).astype(np.int32)
            tensor = Tensor(Dtype.I32, values.shape, values.flatten())

            sequence.append(tensor)

        name = "sequence_construct_i32"
        make_test([sequence], sequence, "SequenceTrait::sequence_construct(input_0)", name, Trait.SEQUENCE)


    @staticmethod
    def sequence_construct_i8():
        sequence = []
        tensor_cnt = np.random.randint(1, 10)
        shape = np.random.randint(1, 4, 2)

        for _ in range(tensor_cnt):
            values = np.random.randint(-6, 6, shape).astype(np.int8)
            tensor = Tensor(Dtype.I8, values.shape, values.flatten())

            sequence.append(tensor)

        name = "sequence_construct_i8"
        make_test([sequence], sequence, "SequenceTrait::sequence_construct(input_0)", name, Trait.SEQUENCE)


    @staticmethod
    def sequence_construct_fp8x23():
        sequence = []
        tensor_cnt = np.random.randint(1, 10)
        shape = np.random.randint(1, 4, 2)

        for _ in range(tensor_cnt):
            values = np.random.randint(-6, 6, shape).astype(np.float64)
            tensor = Tensor(Dtype.FP8x23, values.shape, to_fp(values.flatten(), FixedImpl.FP8x23))

            sequence.append(tensor)

        name = "sequence_construct_fp8x23"
        make_test([sequence], sequence, "SequenceTrait::sequence_construct(input_0)", name, Trait.SEQUENCE)


    @staticmethod
    def sequence_construct_fp16x16():
        sequence = []
        tensor_cnt = np.random.randint(1, 10)
        shape = np.random.randint(1, 4, 2)

        for _ in range(tensor_cnt):
            values = np.random.randint(-6, 6, shape).astype(np.float64)
            tensor = Tensor(Dtype.FP16x16, values.shape, to_fp(values.flatten(), FixedImpl.FP16x16))

            sequence.append(tensor)

        name = "sequence_construct_fp16x16"
        make_test([sequence], sequence, "SequenceTrait::sequence_construct(input_0)", name, Trait.SEQUENCE)
