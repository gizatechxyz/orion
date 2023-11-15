import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


scalar = lambda x: Tensor(Dtype.I32, (), np.array([x]).astype(np.int32).flatten())


class Sequence_insert(RunAll):

    @staticmethod
    def sequence_insert_u32():
        def default():
            sequence = []
            tensor_cnt = 3
            shape = np.random.randint(1, 4, 2)

            for _ in range(tensor_cnt):
                val = np.random.randint(0, 6, shape).astype(np.uint32)
                t = Tensor(Dtype.U32, val.shape, val.flatten())

                sequence.append(t)

            val = np.random.randint(0, 6, shape).astype(np.uint32)
            tensor = Tensor(Dtype.U32, val.shape, val.flatten())

            position = np.random.randint(-2, 2)

            expected_sequence = sequence.copy()
            expected_sequence.insert(position, tensor)

            name = "sequence_insert_u32"
            make_test([sequence, tensor, scalar(position)], expected_sequence, "input_0.sequence_insert(@input_1,@input_2)", name)

        default()

    @staticmethod
    def sequence_insert_i32():
        def default():
            sequence = []
            tensor_cnt = 3
            shape = np.random.randint(1, 4, 2)

            for _ in range(tensor_cnt):
                val = np.random.randint(0, 6, shape).astype(np.int32)
                t = Tensor(Dtype.I32, val.shape, val.flatten())

                sequence.append(t)

            val = np.random.randint(0, 6, shape).astype(np.int32)
            tensor = Tensor(Dtype.I32, val.shape, val.flatten())

            position = np.random.randint(-2, 2)

            expected_sequence = sequence.copy()
            expected_sequence.insert(position, tensor)

            name = "sequence_insert_i32"
            make_test([sequence, tensor, scalar(position)], expected_sequence, "input_0.sequence_insert(@input_1,@input_2)", name)

        default()

    @staticmethod
    def sequence_insert_i8():
        def default():
            sequence = []
            tensor_cnt = 3
            shape = np.random.randint(1, 4, 2)

            for _ in range(tensor_cnt):
                val = np.random.randint(0, 6, shape).astype(np.int8)
                t = Tensor(Dtype.I8, val.shape, val.flatten())

                sequence.append(t)

            val = np.random.randint(0, 6, shape).astype(np.int8)
            tensor = Tensor(Dtype.I8, val.shape, val.flatten())

            position = np.random.randint(-2, 2)

            expected_sequence = sequence.copy()
            expected_sequence.insert(position, tensor)

            name = "sequence_insert_i8"
            make_test([sequence, tensor, scalar(position)], expected_sequence, "input_0.sequence_insert(@input_1,@input_2)", name)

        default()

    @staticmethod
    def sequence_insert_fp8x23():
        def default():
            sequence = []
            tensor_cnt = 3
            shape = np.random.randint(1, 4, 2)

            for _ in range(tensor_cnt):
                val = np.random.randint(0, 6, shape).astype(np.float64)
                t = Tensor(Dtype.FP8x23, val.shape, to_fp(
                    val.flatten(), FixedImpl.FP8x23))

                sequence.append(t)

            val = np.random.randint(0, 6, shape).astype(np.float64)
            tensor = Tensor(Dtype.FP8x23, val.shape, to_fp(
                val.flatten(), FixedImpl.FP8x23))

            position = np.random.randint(-2, 2)

            expected_sequence = sequence.copy()
            expected_sequence.insert(position, tensor)

            name = "sequence_insert_fp8x23"
            make_test([sequence, tensor, scalar(position)], expected_sequence, "input_0.sequence_insert(@input_1,@input_2)", name)

        default()

    @staticmethod
    def sequence_insert_fp16x16():
        def default():
            sequence = []
            tensor_cnt = 3
            shape = np.random.randint(1, 4, 2)

            for _ in range(tensor_cnt):
                val = np.random.randint(0, 6, shape).astype(np.float64)
                t = Tensor(Dtype.FP16x16, val.shape, to_fp(
                    val.flatten(), FixedImpl.FP16x16))

                sequence.append(t)

            val = np.random.randint(0, 6, shape).astype(np.float64)
            tensor = Tensor(Dtype.FP16x16, val.shape, to_fp(
                val.flatten(), FixedImpl.FP16x16))

            position = np.random.randint(-2, 2)

            expected_sequence = sequence.copy()
            expected_sequence.insert(position, tensor)

            name = "sequence_insert_fp16x16"
            make_test([sequence, tensor, scalar(position)], expected_sequence, "input_0.sequence_insert(@input_1,@input_2)", name)

        default()
