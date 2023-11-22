import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


scalar = lambda x: Tensor(Dtype.U32, (), np.array([x]).astype(np.uint32).flatten())


class Sequence_length(RunAll):
    @staticmethod
    def sequence_length_u32():
        def default():
            sequence = []
            tensor_cnt = np.random.randint(1, 10)
            shape = np.random.randint(1, 4, 2)

            for _ in range(tensor_cnt):
                values = np.random.randint(0, 6, shape).astype(np.uint32)
                tensor = Tensor(Dtype.U32, values.shape, values.flatten())

                sequence.append(tensor)

            name = "sequence_length_u32"
            make_test([sequence], scalar(len(sequence)), "input_0.sequence_length()", name)

        def broadcast():
            sequence = []
            tensor_cnt = np.random.randint(1, 10)

            for _ in range(tensor_cnt):
                shape = np.random.randint(1, 4, 2)
                values = np.random.randint(0, 6, shape).astype(np.uint32)
                tensor = Tensor(Dtype.U32, values.shape, values.flatten())

                sequence.append(tensor)

            name = "sequence_length_u32_broadcast"
            make_test([sequence], scalar(len(sequence)), "input_0.sequence_length()", name)

        default()
        broadcast()

    @staticmethod
    def sequence_length_i32():
        def default():
            sequence = []
            tensor_cnt = np.random.randint(1, 10)
            shape = np.random.randint(1, 4, 2)

            for _ in range(tensor_cnt):
                values = np.random.randint(-6, 6, shape).astype(np.int32)
                tensor = Tensor(Dtype.I32, values.shape, values.flatten())

                sequence.append(tensor)

            name = "sequence_length_i32"
            make_test([sequence], scalar(len(sequence)), "input_0.sequence_length()", name)

        def broadcast():
            sequence = []
            tensor_cnt = np.random.randint(1, 10)

            for _ in range(tensor_cnt):
                shape = np.random.randint(1, 4, 2)
                values = np.random.randint(-6, 6, shape).astype(np.int32)
                tensor = Tensor(Dtype.I32, values.shape, values.flatten())

                sequence.append(tensor)

            name = "sequence_length_i32_broadcast"
            make_test([sequence], scalar(len(sequence)), "input_0.sequence_length()", name)

        default()
        broadcast()

    @staticmethod
    def sequence_length_i8():
        def default():
            sequence = []
            tensor_cnt = np.random.randint(1, 10)
            shape = np.random.randint(1, 4, 2)

            for _ in range(tensor_cnt):
                values = np.random.randint(-6, 6, shape).astype(np.int8)
                tensor = Tensor(Dtype.I8, values.shape, values.flatten())

                sequence.append(tensor)

            name = "sequence_length_i8"
            make_test([sequence], scalar(len(sequence)), "input_0.sequence_length()", name)

        def broadcast():
            sequence = []
            tensor_cnt = np.random.randint(1, 10)

            for _ in range(tensor_cnt):
                shape = np.random.randint(1, 4, 2)
                values = np.random.randint(-6, 6, shape).astype(np.int8)
                tensor = Tensor(Dtype.I8, values.shape, values.flatten())

                sequence.append(tensor)

            name = "sequence_length_i8_broadcast"
            make_test([sequence], scalar(len(sequence)), "input_0.sequence_length()", name)

        default()
        broadcast()

    @staticmethod
    def sequence_length_fp8x23():
        def default():
            sequence = []
            tensor_cnt = np.random.randint(1, 10)
            shape = np.random.randint(1, 4, 2)

            for _ in range(tensor_cnt):
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP8x23, values.shape, to_fp(values.flatten(), FixedImpl.FP8x23))

                sequence.append(tensor)

            name = "sequence_length_fp8x23"
            make_test([sequence], scalar(len(sequence)), "input_0.sequence_length()", name)

        def broadcast():
            sequence = []
            tensor_cnt = np.random.randint(1, 10)

            for _ in range(tensor_cnt):
                shape = np.random.randint(1, 4, 2)
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP8x23, values.shape, to_fp(values.flatten(), FixedImpl.FP8x23))

                sequence.append(tensor)

            name = "sequence_length_fp8x23_broadcast"
            make_test([sequence], scalar(len(sequence)), "input_0.sequence_length()", name)

        default()
        broadcast()

    @staticmethod
    def sequence_length_fp16x16():
        def default():
            sequence = []
            tensor_cnt = np.random.randint(1, 10)
            shape = np.random.randint(1, 4, 2)

            for _ in range(tensor_cnt):
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP16x16, values.shape, to_fp(values.flatten(), FixedImpl.FP16x16))

                sequence.append(tensor)

            name = "sequence_length_fp16x16"
            make_test([sequence], scalar(len(sequence)), "input_0.sequence_length()", name)

        def broadcast():
            sequence = []
            tensor_cnt = np.random.randint(1, 10)

            for _ in range(tensor_cnt):
                shape = np.random.randint(1, 4, 2)
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP16x16, values.shape, to_fp(values.flatten(), FixedImpl.FP16x16))

                sequence.append(tensor)

            name = "sequence_length_fp16x16_broadcast"
            make_test([sequence], scalar(len(sequence)), "input_0.sequence_length()", name)

        default()
        broadcast()
