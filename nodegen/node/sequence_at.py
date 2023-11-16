import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


scalar = lambda x: Tensor(Dtype.I32, (), np.array([x]).astype(np.int32).flatten())


class Sequence_at(RunAll):

    @staticmethod
    def sequence_at_u32():
        def positive_index():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(0, 6, shape).astype(np.uint32)
                tensor = Tensor(Dtype.U32, values.shape, values.flatten())

                sequence.append(tensor)

            index = scalar(2)

            name = "sequence_at_u32_positive"
            make_test([sequence, index], sequence[2], "TensorTrait::sequence_at(input_0, input_1)", name)

        def negative_index():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(0, 6, shape).astype(np.uint32)
                tensor = Tensor(Dtype.U32, values.shape, values.flatten())

                sequence.append(tensor)

            index = scalar(-2)

            name = "sequence_at_u32_negative"
            make_test([sequence, index], sequence[-2], "TensorTrait::sequence_at(input_0, input_1)", name)

        positive_index()
        negative_index()


    @staticmethod
    def sequence_at_i32():
        def positive_index():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.int32)
                tensor = Tensor(Dtype.I32, values.shape, values.flatten())

                sequence.append(tensor)

            index = scalar(2)

            name = "sequence_at_i32_positive"
            make_test([sequence, index], sequence[2], "TensorTrait::sequence_at(input_0, input_1)", name)

        def negative_index():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.int32)
                tensor = Tensor(Dtype.I32, values.shape, values.flatten())

                sequence.append(tensor)

            index = scalar(-2)

            name = "sequence_at_i32_negative"
            make_test([sequence, index], sequence[-2], "TensorTrait::sequence_at(input_0, input_1)", name)

        positive_index()
        negative_index()


    @staticmethod
    def sequence_at_i8():
        def positive_index():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.int8)
                tensor = Tensor(Dtype.I8, values.shape, values.flatten())

                sequence.append(tensor)

            index = scalar(2)

            name = "sequence_at_i8_positive"
            make_test([sequence, index], sequence[2], "TensorTrait::sequence_at(input_0, input_1)", name)

        def negative_index():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.int8)
                tensor = Tensor(Dtype.I8, values.shape, values.flatten())

                sequence.append(tensor)

            index = scalar(-2)

            name = "sequence_at_i8_negative"
            make_test([sequence, index], sequence[-2], "TensorTrait::sequence_at(input_0, input_1)", name)

        positive_index()
        negative_index()


    @staticmethod
    def sequence_at_fp8x23():
        def positive_index():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP8x23, values.shape, to_fp(values.flatten(), FixedImpl.FP8x23))

                sequence.append(tensor)

            index = scalar(2)

            name = "sequence_at_fp8x23_positive"
            make_test([sequence, index], sequence[2], "TensorTrait::sequence_at(input_0, input_1)", name)

        def negative_index():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP8x23, values.shape, to_fp(values.flatten(), FixedImpl.FP8x23))

                sequence.append(tensor)

            index = scalar(-2)

            name = "sequence_at_fp8x23_negative"
            make_test([sequence, index], sequence[-2], "TensorTrait::sequence_at(input_0, input_1)", name)

        positive_index()
        negative_index()
    

    @staticmethod
    def sequence_at_fp16x16():
        def positive_index():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP16x16, values.shape, to_fp(values.flatten(), FixedImpl.FP16x16))

                sequence.append(tensor)

            index = scalar(2)

            name = "sequence_at_fp16x16_positive"
            make_test([sequence, index], sequence[2], "TensorTrait::sequence_at(input_0, input_1)", name)

        def negative_index():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP16x16, values.shape, to_fp(values.flatten(), FixedImpl.FP16x16))

                sequence.append(tensor)

            index = scalar(-2)

            name = "sequence_at_fp16x16_negative"
            make_test([sequence, index], sequence[-2], "TensorTrait::sequence_at(input_0, input_1)", name)

        positive_index()
        negative_index()
