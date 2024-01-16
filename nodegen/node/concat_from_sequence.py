import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait


class Concat_from_sequence(RunAll):

    @staticmethod
    def concat_from_sequence_u32():
        def new_axis_zero():
            sequence = []
            values_array = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(0, 6, shape).astype(np.uint32)
                tensor = Tensor(Dtype.U32, values.shape, values.flatten())
                sequence.append(tensor)
                values_array.append(values)

            axis = np.int32(1)
            new_axis = np.uint32(0)

            concatenated_tensor = np.concatenate(values_array, axis)
            concatenated_tensor = Tensor(Dtype.U32, concatenated_tensor.shape, concatenated_tensor.flatten())

            name = "concat_from_sequence_u32_new_axis_zero"
            make_test([sequence], concatenated_tensor, "SequenceTrait::concat_from_sequence(input_0, 1_i32, Option::Some(0))", name, Trait.SEQUENCE)

        def new_axis_one():
            sequence = []
            values_array = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(0, 6, shape).astype(np.uint32)
                tensor = Tensor(Dtype.U32, values.shape, values.flatten())
                sequence.append(tensor)
                values_array.append(values)

            axis = np.int32(1)
            new_axis = np.uint32(1)

            concatenated_tensor = np.stack(values_array, axis)
            concatenated_tensor = Tensor(Dtype.U32, concatenated_tensor.shape, concatenated_tensor.flatten())

            name = "concat_from_sequence_u32_new_axis_one"
            make_test([sequence], concatenated_tensor, "SequenceTrait::concat_from_sequence(input_0, 1_i32, Option::Some(1))", name, Trait.SEQUENCE)

        def new_axis_default():
            sequence = []
            values_array = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(0, 6, shape).astype(np.uint32)
                tensor = Tensor(Dtype.U32, values.shape, values.flatten())
                sequence.append(tensor)
                values_array.append(values)

            axis = np.int32(1)
            new_axis = np.uint32(0)

            concatenated_tensor = np.concatenate(values_array, axis)
            concatenated_tensor = Tensor(Dtype.U32, concatenated_tensor.shape, concatenated_tensor.flatten())

            name = "concat_from_sequence_u32_new_axis_default"
            make_test([sequence], concatenated_tensor, "SequenceTrait::concat_from_sequence(input_0, 1_i32, Option::None(()))", name, Trait.SEQUENCE)

        new_axis_zero()
        new_axis_one()
        new_axis_default()


    @staticmethod
    def concat_from_sequence_i32():
        def new_axis_zero():
            sequence = []
            values_array = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.int32)
                tensor = Tensor(Dtype.I32, values.shape, values.flatten())
                sequence.append(tensor)
                values_array.append(values)

            axis = np.int32(1)
            new_axis = np.uint32(0)

            concatenated_tensor = np.concatenate(values_array, axis)
            concatenated_tensor = Tensor(Dtype.I32, concatenated_tensor.shape, concatenated_tensor.flatten())

            name = "concat_from_sequence_i32_new_axis_zero"
            make_test([sequence], concatenated_tensor, "SequenceTrait::concat_from_sequence(input_0, 1_i32, Option::Some(0))", name, Trait.SEQUENCE)

        def new_axis_one():
            sequence = []
            values_array = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.int32)
                tensor = Tensor(Dtype.I32, values.shape, values.flatten())
                sequence.append(tensor)
                values_array.append(values)

            axis = np.int32(1)
            new_axis = np.uint32(1)

            concatenated_tensor = np.stack(values_array, axis)
            concatenated_tensor = Tensor(Dtype.I32, concatenated_tensor.shape, concatenated_tensor.flatten())

            name = "concat_from_sequence_i32_new_axis_one"
            make_test([sequence], concatenated_tensor, "SequenceTrait::concat_from_sequence(input_0, 1_i32, Option::Some(1))", name, Trait.SEQUENCE)

        def new_axis_default():
            sequence = []
            values_array = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.int32)
                tensor = Tensor(Dtype.I32, values.shape, values.flatten())
                sequence.append(tensor)
                values_array.append(values)

            axis = np.int32(1)
            new_axis = np.uint32(0)

            concatenated_tensor = np.concatenate(values_array, axis)
            concatenated_tensor = Tensor(Dtype.I32, concatenated_tensor.shape, concatenated_tensor.flatten())

            name = "concat_from_sequence_i32_new_axis_default"
            make_test([sequence], concatenated_tensor, "SequenceTrait::concat_from_sequence(input_0, 1_i32, Option::None(()))", name, Trait.SEQUENCE)

        new_axis_zero()
        new_axis_one()
        new_axis_default()


    @staticmethod
    def concat_from_sequence_i8():
        def new_axis_zero():
            sequence = []
            values_array = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.int8)
                tensor = Tensor(Dtype.I8, values.shape, values.flatten())
                sequence.append(tensor)
                values_array.append(values)

            axis = np.int32(1)
            new_axis = np.uint32(0)

            concatenated_tensor = np.concatenate(values_array, axis)
            concatenated_tensor = Tensor(Dtype.I8, concatenated_tensor.shape, concatenated_tensor.flatten())

            name = "concat_from_sequence_i8_new_axis_zero"
            make_test([sequence], concatenated_tensor, "SequenceTrait::concat_from_sequence(input_0, 1_i32, Option::Some(0))", name, Trait.SEQUENCE)

        def new_axis_one():
            sequence = []
            values_array = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.int8)
                tensor = Tensor(Dtype.I8, values.shape, values.flatten())
                sequence.append(tensor)
                values_array.append(values)

            axis = np.int32(1)
            new_axis = np.uint32(1)

            concatenated_tensor = np.stack(values_array, axis)
            concatenated_tensor = Tensor(Dtype.I8, concatenated_tensor.shape, concatenated_tensor.flatten())

            name = "concat_from_sequence_i8_new_axis_one"
            make_test([sequence], concatenated_tensor, "SequenceTrait::concat_from_sequence(input_0, 1_i32, Option::Some(1))", name, Trait.SEQUENCE)

        def new_axis_default():
            sequence = []
            values_array = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.int8)
                tensor = Tensor(Dtype.I8, values.shape, values.flatten())
                sequence.append(tensor)
                values_array.append(values)

            axis = np.int32(1)
            new_axis = np.uint32(0)

            concatenated_tensor = np.concatenate(values_array, axis)
            concatenated_tensor = Tensor(Dtype.I8, concatenated_tensor.shape, concatenated_tensor.flatten())

            name = "concat_from_sequence_i8_new_axis_default"
            make_test([sequence], concatenated_tensor, "SequenceTrait::concat_from_sequence(input_0, 1_i32, Option::None(()))", name, Trait.SEQUENCE)

        new_axis_zero()
        new_axis_one()
        new_axis_default()


    @staticmethod
    def concat_from_sequence_fp8x23():
        def new_axis_zero():
            sequence = []
            values_array = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP8x23, values.shape, to_fp(values.flatten(), FixedImpl.FP8x23))
                sequence.append(tensor)
                values_array.append(values)

            axis = np.int32(1)
            new_axis = np.uint32(0)

            concatenated_tensor = np.concatenate(values_array, axis)
            concatenated_tensor = Tensor(Dtype.FP8x23, concatenated_tensor.shape, to_fp(concatenated_tensor.flatten(), FixedImpl.FP8x23))

            name = "concat_from_sequence_fp8x23_new_axis_zero"
            make_test([sequence], concatenated_tensor, "SequenceTrait::concat_from_sequence(input_0, 1_i32, Option::Some(0))", name, Trait.SEQUENCE)

        def new_axis_one():
            sequence = []
            values_array = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP8x23, values.shape, to_fp(values.flatten(), FixedImpl.FP8x23))
                sequence.append(tensor)
                values_array.append(values)

            axis = np.int32(1)
            new_axis = np.uint32(1)

            concatenated_tensor = np.stack(values_array, axis)
            concatenated_tensor = Tensor(Dtype.FP8x23, concatenated_tensor.shape, to_fp(concatenated_tensor.flatten(), FixedImpl.FP8x23))

            name = "concat_from_sequence_fp8x23_new_axis_one"
            make_test([sequence], concatenated_tensor, "SequenceTrait::concat_from_sequence(input_0, 1_i32, Option::Some(1))", name, Trait.SEQUENCE)

        def new_axis_default():
            sequence = []
            values_array = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP8x23, values.shape, to_fp(values.flatten(), FixedImpl.FP8x23))
                sequence.append(tensor)
                values_array.append(values)

            axis = np.int32(1)
            new_axis = np.uint32(0)

            concatenated_tensor = np.concatenate(values_array, axis)
            concatenated_tensor = Tensor(Dtype.FP8x23, concatenated_tensor.shape, to_fp(concatenated_tensor.flatten(), FixedImpl.FP8x23))

            name = "concat_from_sequence_fp8x23_new_axis_default"
            make_test([sequence], concatenated_tensor, "SequenceTrait::concat_from_sequence(input_0, 1_i32, Option::None(()))", name, Trait.SEQUENCE)

        new_axis_zero()
        new_axis_one()
        new_axis_default()


    @staticmethod
    def concat_from_sequence_fp16x16():
        def new_axis_zero():
            sequence = []
            values_array = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP16x16, values.shape, to_fp(values.flatten(), FixedImpl.FP16x16))
                sequence.append(tensor)
                values_array.append(values)

            axis = np.int32(1)
            new_axis = np.uint32(0)

            concatenated_tensor = np.concatenate(values_array, axis)
            concatenated_tensor = Tensor(Dtype.FP16x16, concatenated_tensor.shape, to_fp(concatenated_tensor.flatten(), FixedImpl.FP16x16))

            name = "concat_from_sequence_fp16x16_new_axis_zero"
            make_test([sequence], concatenated_tensor, "SequenceTrait::concat_from_sequence(input_0, 1_i32, Option::Some(0))", name, Trait.SEQUENCE)

        def new_axis_one():
            sequence = []
            values_array = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP16x16, values.shape, to_fp(values.flatten(), FixedImpl.FP16x16))
                sequence.append(tensor)
                values_array.append(values)

            axis = np.int32(1)
            new_axis = np.uint32(1)

            concatenated_tensor = np.stack(values_array, axis)
            concatenated_tensor = Tensor(Dtype.FP16x16, concatenated_tensor.shape, to_fp(concatenated_tensor.flatten(), FixedImpl.FP16x16))

            name = "concat_from_sequence_fp16x16_new_axis_one"
            make_test([sequence], concatenated_tensor, "SequenceTrait::concat_from_sequence(input_0, 1_i32, Option::Some(1))", name, Trait.SEQUENCE)

        def new_axis_default():
            sequence = []
            values_array = []
            shape = np.random.randint(1, 4, 2)

            for _ in range(5):
                values = np.random.randint(-6, 6, shape).astype(np.float64)
                tensor = Tensor(Dtype.FP16x16, values.shape, to_fp(values.flatten(), FixedImpl.FP16x16))
                sequence.append(tensor)
                values_array.append(values)

            axis = np.int32(1)
            new_axis = np.uint32(0)

            concatenated_tensor = np.concatenate(values_array, axis)
            concatenated_tensor = Tensor(Dtype.FP16x16, concatenated_tensor.shape, to_fp(concatenated_tensor.flatten(), FixedImpl.FP16x16))

            name = "concat_from_sequence_fp16x16_new_axis_default"
            make_test([sequence], concatenated_tensor, "SequenceTrait::concat_from_sequence(input_0, 1_i32, Option::None(()))", name, Trait.SEQUENCE)

        new_axis_zero()
        new_axis_one()
        new_axis_default()