import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait


scalar = lambda x: Tensor(Dtype.I32, (), np.array([x]).astype(np.int32).flatten())


class Sequence_erase(RunAll):

    @staticmethod
    def sequence_erase_u32():
        def positive_position():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(0, 6, shape).astype(np.uint32)
                tensor = Tensor(Dtype.U32, values.shape, values.flatten())

                sequence.append(tensor)

            position = scalar(2)

            output_sequence = sequence.copy()
            output_sequence.pop(2)

            name = "sequence_erase_u32_positive"
            make_test([sequence, position], output_sequence, "SequenceTrait::sequence_erase(input_0, Option::Some(input_1))", name, Trait.SEQUENCE)

        def negative_position():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(0, 6, shape).astype(np.uint32)
                tensor = Tensor(Dtype.U32, values.shape, values.flatten())

                sequence.append(tensor)

            position = scalar(-2)
            
            output_sequence = sequence.copy()
            output_sequence.pop(-2)

            name = "sequence_erase_u32_negative"
            make_test([sequence, position], output_sequence, "SequenceTrait::sequence_erase(input_0, Option::Some(input_1))", name, Trait.SEQUENCE)

        def empty_position():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(0, 6, shape).astype(np.uint32)
                tensor = Tensor(Dtype.U32, values.shape, values.flatten())

                sequence.append(tensor)
            
            output_sequence = sequence.copy()
            output_sequence.pop(-1)

            name = "sequence_erase_u32_empty"
            make_test([sequence], output_sequence, "SequenceTrait::sequence_erase(input_0, Option::None(()))", name, Trait.SEQUENCE)

        positive_position()
        negative_position()
        empty_position()


    @staticmethod
    def sequence_erase_i32():
        def positive_position():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.int32)
                tensor = Tensor(Dtype.I32, values.shape, values.flatten())

                sequence.append(tensor)

            position = scalar(2)

            output_sequence = sequence.copy()
            output_sequence.pop(2)

            name = "sequence_erase_i32_positive"
            make_test([sequence, position], output_sequence, "SequenceTrait::sequence_erase(input_0, Option::Some(input_1))", name, Trait.SEQUENCE)

        def negative_position():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.int32)
                tensor = Tensor(Dtype.I32, values.shape, values.flatten())

                sequence.append(tensor)

            position = scalar(-2)

            output_sequence = sequence.copy()
            output_sequence.pop(-2)

            name = "sequence_erase_i32_negative"
            make_test([sequence, position], output_sequence, "SequenceTrait::sequence_erase(input_0, Option::Some(input_1))", name, Trait.SEQUENCE)

        def empty_position():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.int32)
                tensor = Tensor(Dtype.I32, values.shape, values.flatten())

                sequence.append(tensor)

            output_sequence = sequence.copy()
            output_sequence.pop(-1)

            name = "sequence_erase_i32_empty"
            make_test([sequence], output_sequence, "SequenceTrait::sequence_erase(input_0, Option::None(()))", name, Trait.SEQUENCE)

        positive_position()
        negative_position()
        empty_position()


    @staticmethod
    def sequence_erase_i8():
        def positive_position():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.int8)
                tensor = Tensor(Dtype.I8, values.shape, values.flatten())

                sequence.append(tensor)

            position = scalar(2)

            output_sequence = sequence.copy()
            output_sequence.pop(2)

            name = "sequence_erase_i8_positive"
            make_test([sequence, position], output_sequence, "SequenceTrait::sequence_erase(input_0, Option::Some(input_1))", name, Trait.SEQUENCE)

        def negative_position():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.int8)
                tensor = Tensor(Dtype.I8, values.shape, values.flatten())

                sequence.append(tensor)

            position = scalar(-2)

            output_sequence = sequence.copy()
            output_sequence.pop(-2)

            name = "sequence_erase_i8_negative"
            make_test([sequence, position], output_sequence, "SequenceTrait::sequence_erase(input_0, Option::Some(input_1))", name, Trait.SEQUENCE)

        def empty_position():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.int8)
                tensor = Tensor(Dtype.I8, values.shape, values.flatten())

                sequence.append(tensor)

            output_sequence = sequence.copy()
            output_sequence.pop(-1)

            name = "sequence_erase_i8_empty"
            make_test([sequence], output_sequence, "SequenceTrait::sequence_erase(input_0, Option::None(()))", name, Trait.SEQUENCE)

        positive_position()
        negative_position()
        empty_position()


    @staticmethod
    def sequence_erase_fp8x23():
        def positive_position():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP8x23, values.shape, to_fp(values.flatten(), FixedImpl.FP8x23))

                sequence.append(tensor)

            position = scalar(2)

            output_sequence = sequence.copy()
            output_sequence.pop(2)

            name = "sequence_erase_fp8x23_positive"
            make_test([sequence, position], output_sequence, "SequenceTrait::sequence_erase(input_0, Option::Some(input_1))", name, Trait.SEQUENCE)

        def negative_position():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP8x23, values.shape, to_fp(values.flatten(), FixedImpl.FP8x23))

                sequence.append(tensor)

            position = scalar(-2)

            output_sequence = sequence.copy()
            output_sequence.pop(-2)

            name = "sequence_erase_fp8x23_negative"
            make_test([sequence, position], output_sequence, "SequenceTrait::sequence_erase(input_0, Option::Some(input_1))", name, Trait.SEQUENCE)

        def empty_position():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP8x23, values.shape, to_fp(values.flatten(), FixedImpl.FP8x23))

                sequence.append(tensor)

            output_sequence = sequence.copy()
            output_sequence.pop(-1)

            name = "sequence_erase_fp8x23_empty"
            make_test([sequence], output_sequence, "SequenceTrait::sequence_erase(input_0, Option::None(()))", name, Trait.SEQUENCE)

        positive_position()
        negative_position()
        empty_position()


    @staticmethod
    def sequence_erase_fp16x16():
        def positive_position():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP16x16, values.shape, to_fp(values.flatten(), FixedImpl.FP16x16))

                sequence.append(tensor)

            position = scalar(2)

            output_sequence = sequence.copy()
            output_sequence.pop(2)

            name = "sequence_erase_fp16x16_positive"
            make_test([sequence, position], output_sequence, "SequenceTrait::sequence_erase(input_0, Option::Some(input_1))", name, Trait.SEQUENCE)

        def negative_position():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP16x16, values.shape, to_fp(values.flatten(), FixedImpl.FP16x16))

                sequence.append(tensor)

            position = scalar(-2)

            output_sequence = sequence.copy()
            output_sequence.pop(-2)

            name = "sequence_erase_fp16x16_negative"
            make_test([sequence, position], output_sequence, "SequenceTrait::sequence_erase(input_0, Option::Some(input_1))", name, Trait.SEQUENCE)

        def empty_position():
            sequence = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP16x16, values.shape, to_fp(values.flatten(), FixedImpl.FP16x16))

                sequence.append(tensor)

            output_sequence = sequence.copy()
            output_sequence.pop(-1)

            name = "sequence_erase_fp16x16_empty"
            make_test([sequence], output_sequence, "SequenceTrait::sequence_erase(input_0, Option::None(()))", name, Trait.SEQUENCE)

        positive_position()
        negative_position()
        empty_position()