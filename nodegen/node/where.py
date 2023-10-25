import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl


class Where(RunAll):

    @staticmethod
    def where_u32():
        def default():
            x = np.random.randint(0, 6, (3, 3, 3)).astype(np.uint32)
            y = np.random.randint(0, 6, (3, 3, 3)).astype(np.uint32)
            condition = np.random.choice([1, 0], x.shape).astype(np.uint32)
            
            z = np.where(condition, x, y).astype(x.dtype)

            condition = Tensor(Dtype.U32, condition.shape, condition.flatten())
            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())
            z = Tensor(Dtype.U32, z.shape, z.flatten())

            name = "where_u32"
            make_node([condition, x, y], [z], name)
            make_test([condition, x, y], z, "input_0.where(@input_1,@input_2)", name)

        def broadcast():
            x = np.random.randint(0, 6, (2, 2)).astype(np.uint32)
            y = np.random.randint(0, 6, (1, 2)).astype(np.uint32)
            condition = np.random.choice([1, 0], x.shape).astype(np.uint32)
            
            z = np.where(condition, x, y).astype(x.dtype)

            condition = Tensor(Dtype.U32, condition.shape, condition.flatten())
            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())
            z = Tensor(Dtype.U32, z.shape, z.flatten())

            name = "where_u32_broadcast"
            make_node([condition, x, y], [z], name)
            make_test([condition, x, y], z, "input_0.where(@input_1,@input_2)", name)

        default()
        broadcast()

    @staticmethod
    def where_i32():
        def default():
            x = np.random.randint(0, 6, (3, 3, 3)).astype(np.int32)
            y = np.random.randint(0, 6, (3, 3, 3)).astype(np.int32)
            condition = np.random.choice([1, 0], x.shape).astype(np.int32)
            
            z = np.where(condition, x, y).astype(x.dtype)

            condition = Tensor(Dtype.I32, condition.shape, condition.flatten())
            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())
            z = Tensor(Dtype.I32, z.shape, z.flatten())

            name = "where_i32"
            make_node([condition, x, y], [z], name)
            make_test([condition, x, y], z, "input_0.where(@input_1,@input_2)", name)

        def broadcast():
            x = np.random.randint(0, 6, (2, 2)).astype(np.int32)
            y = np.random.randint(0, 6, (1, 2)).astype(np.int32)
            condition = np.random.choice([1, 0], x.shape).astype(np.int32)

            z = np.where(condition, x, y).astype(x.dtype)

            condition = Tensor(Dtype.I32, condition.shape, condition.flatten())
            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())
            z = Tensor(Dtype.I32, z.shape, z.flatten())

            name = "where_i32_broadcast"
            make_node([condition, x, y], [z], name)
            make_test([condition, x, y], z, "input_0.where(@input_1,@input_2)", name)

        default()
        broadcast()

    @staticmethod
    def where_i8():
        def default():
            x = np.random.randint(0, 6, (3, 3, 3)).astype(np.int8)
            y = np.random.randint(0, 6, (3, 3, 3)).astype(np.int8)
            condition = np.random.choice([1, 0], x.shape).astype(np.int8)

            z = np.where(condition, x, y).astype(x.dtype)

            condition = Tensor(Dtype.I8, condition.shape, condition.flatten())
            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())
            z = Tensor(Dtype.I8, z.shape, z.flatten())

            name = "where_i8"
            make_node([condition, x, y], [z], name)
            make_test([condition, x, y], z, "input_0.where(@input_1,@input_2)", name)

        def broadcast():
            x = np.random.randint(0, 6, (2, 2)).astype(np.int8)
            y = np.random.randint(0, 6, (1, 2)).astype(np.int8)
            condition = np.random.choice([1, 0], x.shape).astype(np.int8)
            
            z = np.where(condition, x, y).astype(x.dtype)

            condition = Tensor(Dtype.I8, condition.shape, condition.flatten())
            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())
            z = Tensor(Dtype.I8, z.shape, z.flatten())

            name = "where_i8_broadcast"
            make_node([condition, x, y], [z], name)
            make_test([condition, x, y], z, "input_0.where(@input_1,@input_2)", name)

        default()
        broadcast()

    @staticmethod
    def where_fp8x23():
        def default():
            x = np.random.randint(0, 6, (3, 3, 3)).astype(np.float64)
            y = np.random.randint(0, 6, (3, 3, 3)).astype(np.float64)
            condition = np.random.choice([1, 0], x.shape).astype(np.float64)
            
            z = np.where(condition, x, y).astype(x.dtype)

            condition = Tensor(Dtype.FP8x23, condition.shape, to_fp(
                condition.flatten(), FixedImpl.FP8x23))
            x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                x.flatten(), FixedImpl.FP8x23))
            y = Tensor(Dtype.FP8x23, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP8x23))
            z = Tensor(Dtype.FP8x23, z.shape,  to_fp(
                z.flatten(), FixedImpl.FP8x23))

            name = "where_fp8x23"
            make_node([condition, x, y], [z], name)
            make_test([condition, x, y], z, "input_0.where(@input_1,@input_2)", name)

        def broadcast():
            x = np.random.randint(0, 6, (2, 2)).astype(np.float64)
            y = np.random.randint(0, 6, (1, 2)).astype(np.float64)
            condition = np.random.choice([1, 0], x.shape).astype(np.float64)
            
            z = np.where(condition, x, y).astype(x.dtype)

            condition = Tensor(Dtype.FP8x23, condition.shape, to_fp(
                condition.flatten(), FixedImpl.FP8x23))
            x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                x.flatten(), FixedImpl.FP8x23))
            y = Tensor(Dtype.FP8x23, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP8x23))
            z = Tensor(Dtype.FP8x23, z.shape,  to_fp(
                z.flatten(), FixedImpl.FP8x23))

            name = "where_fp8x23_broadcast"
            make_node([condition, x, y], [z], name)
            make_test([condition, x, y], z, "input_0.where(@input_1,@input_2)", name)

        default()
        broadcast()

    @staticmethod
    def where_fp16x16():
        def default():
            x = np.random.randint(0, 6, (3, 3, 3)).astype(np.float64)
            y = np.random.randint(0, 6, (3, 3, 3)).astype(np.float64)
            condition = np.random.choice([1, 0], x.shape).astype(np.float64)
            
            z = np.where(condition, x, y).astype(x.dtype)

            condition = Tensor(Dtype.FP16x16, condition.shape, to_fp(
                condition.flatten(), FixedImpl.FP16x16))
            x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                x.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP16x16))
            z = Tensor(Dtype.FP16x16, z.shape,  to_fp(
                z.flatten(), FixedImpl.FP16x16))
            
            name = "where_fp16x16"
            make_node([condition, x, y], [z], name)
            make_test([condition, x, y], z, "input_0.where(@input_1,@input_2)", name)

        def broadcast():
            x = np.random.randint(0, 6, (2, 2)).astype(np.float64)
            y = np.random.randint(0, 6, (1, 2)).astype(np.float64)
            condition = np.random.choice([1, 0], x.shape).astype(np.float64)
            
            z = np.where(condition, x, y).astype(x.dtype)

            condition = Tensor(Dtype.FP16x16, condition.shape, to_fp(
                condition.flatten(), FixedImpl.FP16x16))
            x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                x.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP16x16))
            z = Tensor(Dtype.FP16x16, z.shape,  to_fp(
                z.flatten(), FixedImpl.FP16x16))

            name = "where_fp16x16_broadcast"
            make_node([condition, x, y], [z], name)
            make_test([condition, x, y], z, "input_0.where(@input_1,@input_2)", name)

        default()
        broadcast()
