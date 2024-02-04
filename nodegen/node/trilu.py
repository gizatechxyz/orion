import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


class Trilu(RunAll):
    @staticmethod
    def trilu_u32():
        def tril():
            x = np.random.randint(0, 255, (4, 5)).astype(np.uint32)
            y = np.tril(x)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "tril_u32"
            
            make_test(
                [x], y, "input_0.trilu(false, 0)", name)

        def tril_neg():
            x = np.random.randint(0, 255, (4, 5)).astype(np.uint32)
            y = np.tril(x, k=-1)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "tril_u32_neg"
            
            make_test(
                [x], y, "input_0.trilu(false, -1)", name)

        def tril_one_row():
            x = np.random.randint(0, 255, (3, 1, 5)).astype(np.uint32)
            y = np.tril(x)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "tril_u32_one_row"
            
            make_test(
                [x], y, "input_0.trilu(false, 0)", name)

        def tril_out_neg():
            x = np.random.randint(0, 255, (4, 5)).astype(np.uint32)
            y = np.tril(x, k=-7)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "tril_u32_out_neg"
            
            make_test(
                [x], y, "input_0.trilu(false, -7)", name)


        def tril_out_pos():
            x = np.random.randint(0, 255, (4, 5)).astype(np.uint32)
            y = np.tril(x, k=6)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "tril_u32_out_pos"
            
            make_test(
                [x], y, "input_0.trilu(false, 6)", name)


        def tril_pos():
            x = np.random.randint(0, 255, (4, 5)).astype(np.uint32)
            y = np.tril(x, k=2)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "tril_u32_pos"
            
            make_test(
                [x], y, "input_0.trilu(false, 2)", name)


        def tril_square():
            x = np.random.randint(0, 255, (2, 3, 3)).astype(np.uint32)
            y = np.tril(x, k=0)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "tril_u32_square"
            
            make_test(
                [x], y, "input_0.trilu(false, 0)", name)


        def tril_square_neg():
            x = np.random.randint(0, 255, (2, 3, 3)).astype(np.uint32)
            y = np.tril(x, k=-1)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "tril_u32_square_neg"
            
            make_test(
                [x], y, "input_0.trilu(false, -1)", name)


        def tril_zero():
            x = np.random.randint(0, 255, (3, 0, 5)).astype(np.uint32)
            y = np.tril(x, k=6)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "tril_u32_zero"
            
            make_test(
                [x], y, "input_0.trilu(false, 6)", name)


        def triu():
            x = np.random.randint(0, 255, (4, 5)).astype(np.uint32)
            y = np.triu(x)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "triu_u32"
            
            make_test(
                [x], y, "input_0.trilu(true, 0)", name)

        def triu_neg():
            x = np.random.randint(0, 255, (4, 5)).astype(np.uint32)
            y = np.triu(x, k=-1)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "triu_u32_neg"
            
            make_test(
                [x], y, "input_0.trilu(true, -1)", name)

        def triu_one_row():
            x = np.random.randint(0, 255, (3, 1, 5)).astype(np.uint32)
            y = np.triu(x)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "triu_u32_one_row"
            
            make_test(
                [x], y, "input_0.trilu(true, 0)", name)

        def triu_out_neg():
            x = np.random.randint(0, 255, (4, 5)).astype(np.uint32)
            y = np.triu(x, k=-7)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "triu_u32_out_neg"
            
            make_test(
                [x], y, "input_0.trilu(true, -7)", name)


        def triu_out_pos():
            x = np.random.randint(0, 255, (4, 5)).astype(np.uint32)
            y = np.triu(x, k=6)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "triu_u32_out_pos"
            
            make_test(
                [x], y, "input_0.trilu(true, 6)", name)


        def triu_pos():
            x = np.random.randint(0, 255, (4, 5)).astype(np.uint32)
            y = np.triu(x, k=2)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "triu_u32_pos"
            
            make_test(
                [x], y, "input_0.trilu(true, 2)", name)


        def triu_square():
            x = np.random.randint(0, 255, (2, 3, 3)).astype(np.uint32)
            y = np.triu(x, k=0)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "triu_u32_square"
            
            make_test(
                [x], y, "input_0.trilu(true, 0)", name)


        def triu_square_neg():
            x = np.random.randint(0, 255, (2, 3, 3)).astype(np.uint32)
            y = np.triu(x, k=-1)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "triu_u32_square_neg"
            
            make_test(
                [x], y, "input_0.trilu(true, -1)", name)


        def triu_zero():
            x = np.random.randint(0, 255, (3, 0, 5)).astype(np.uint32)
            y = np.triu(x, k=6)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())

            name = "triu_u32_zero"
            
            make_test(
                [x], y, "input_0.trilu(true, 6)", name)

        tril()
        tril_neg()
        tril_one_row()
        tril_out_neg()
        tril_out_pos()
        tril_pos()
        tril_square()
        tril_square_neg()
        tril_zero()
        triu()
        triu_neg()
        triu_one_row()
        triu_out_neg()
        triu_out_pos()
        triu_pos()
        triu_square()
        triu_square_neg()
        triu_zero()


    @staticmethod
    def trilu_i32():
        def tril():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int32)
            y = np.tril(x)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "tril_i32"
            
            make_test(
                [x], y, "input_0.trilu(false, 0)", name)

        def tril_neg():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int32)
            y = np.tril(x, k=-1)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "tril_neg_i32"
            
            make_test(
                [x], y, "input_0.trilu(false, -1)", name)

        def tril_one_row():
            x = np.random.randint(-127, 127, (3, 1, 5)).astype(np.int32)
            y = np.tril(x)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "tril_i32_one_row"
            
            make_test(
                [x], y, "input_0.trilu(false, 0)", name)

        def tril_out_neg():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int32)
            y = np.tril(x, k=-7)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "tril_i32_out_neg"
            
            make_test(
                [x], y, "input_0.trilu(false, -7)", name)


        def tril_out_pos():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int32)
            y = np.tril(x, k=6)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "tril_i32_out_pos"
            
            make_test(
                [x], y, "input_0.trilu(false, 6)", name)


        def tril_pos():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int32)
            y = np.tril(x, k=2)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "tril_i32_pos"
            
            make_test(
                [x], y, "input_0.trilu(false, 2)", name)


        def tril_square():
            x = np.random.randint(-127, 127, (2, 3, 3)).astype(np.int32)
            y = np.tril(x, k=0)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "tril_i32_square"
            
            make_test(
                [x], y, "input_0.trilu(false, 0)", name)


        def tril_square_neg():
            x = np.random.randint(-127, 127, (2, 3, 3)).astype(np.int32)
            y = np.tril(x, k=-1)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "tril_i32_square_neg"
            
            make_test(
                [x], y, "input_0.trilu(false, -1)", name)


        def tril_zero():
            x = np.random.randint(-127, 127, (3, 0, 5)).astype(np.int32)
            y = np.tril(x, k=6)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "tril_i32_zero"
            
            make_test(
                [x], y, "input_0.trilu(false, 6)", name)


        def triu():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int32)
            y = np.triu(x)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "triu_i32"
            
            make_test(
                [x], y, "input_0.trilu(true, 0)", name)

        def triu_neg():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int32)
            y = np.triu(x, k=-1)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "triu_i32_neg"
            
            make_test(
                [x], y, "input_0.trilu(true, -1)", name)

        def triu_one_row():
            x = np.random.randint(-127, 127, (3, 1, 5)).astype(np.int32)
            y = np.triu(x)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "triu_i32_one_row"
            
            make_test(
                [x], y, "input_0.trilu(true, 0)", name)

        def triu_out_neg():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int32)
            y = np.triu(x, k=-7)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "triu_i32_out_neg"
            
            make_test(
                [x], y, "input_0.trilu(true, -7)", name)


        def triu_out_pos():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int32)
            y = np.triu(x, k=6)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "triu_i32_out_pos"
            
            make_test(
                [x], y, "input_0.trilu(true, 6)", name)


        def triu_pos():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int32)
            y = np.triu(x, k=2)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "triu_i32_pos"
            
            make_test(
                [x], y, "input_0.trilu(true, 2)", name)


        def triu_square():
            x = np.random.randint(-127, 127, (2, 3, 3)).astype(np.int32)
            y = np.triu(x, k=0)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "triu_i32_square"
            
            make_test(
                [x], y, "input_0.trilu(true, 0)", name)


        def triu_square_neg():
            x = np.random.randint(-127, 127, (2, 3, 3)).astype(np.int32)
            y = np.triu(x, k=-1)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "triu_i32_square_neg"
            
            make_test(
                [x], y, "input_0.trilu(true, -1)", name)


        def triu_zero():
            x = np.random.randint(-127, 127, (3, 0, 5)).astype(np.int32)
            y = np.triu(x, k=6)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())

            name = "triu_i32_zero"
            
            make_test(
                [x], y, "input_0.trilu(true, 6)", name)

        tril()
        tril_neg()
        tril_one_row()
        tril_out_neg()
        tril_out_pos()
        tril_pos()
        tril_square()
        tril_square_neg()
        tril_zero()
        triu()
        triu_neg()
        triu_one_row()
        triu_out_neg()
        triu_out_pos()
        triu_pos()
        triu_square()
        triu_square_neg()
        triu_zero()


    @staticmethod
    def trilu_i8():
        def tril():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int8)
            y = np.tril(x)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "tril_i8"
            
            make_test(
                [x], y, "input_0.trilu(false, 0)", name)

        def tril_neg():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int8)
            y = np.tril(x, k=-1)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "tril_i8_neg"
            
            make_test(
                [x], y, "input_0.trilu(false, -1)", name)

        def tril_one_row():
            x = np.random.randint(-127, 127, (3, 1, 5)).astype(np.int8)
            y = np.tril(x)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "tril_i8_one_row"
            
            make_test(
                [x], y, "input_0.trilu(false, 0)", name)

        def tril_out_neg():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int8)
            y = np.tril(x, k=-7)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "tril_i8_out_neg"
            
            make_test(
                [x], y, "input_0.trilu(false, -7)", name)


        def tril_out_pos():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int8)
            y = np.tril(x, k=6)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "tril_i8_out_pos"
            
            make_test(
                [x], y, "input_0.trilu(false, 6)", name)


        def tril_pos():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int8)
            y = np.tril(x, k=2)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "tril_i8_pos"
            
            make_test(
                [x], y, "input_0.trilu(false, 2)", name)


        def tril_square():
            x = np.random.randint(-127, 127, (2, 3, 3)).astype(np.int8)
            y = np.tril(x, k=0)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "tril_i8_square"
            
            make_test(
                [x], y, "input_0.trilu(false, 0)", name)


        def tril_square_neg():
            x = np.random.randint(-127, 127, (2, 3, 3)).astype(np.int8)
            y = np.tril(x, k=-1)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "tril_i8_square_neg"
            
            make_test(
                [x], y, "input_0.trilu(false, -1)", name)


        def tril_zero():
            x = np.random.randint(-127, 127, (3, 0, 5)).astype(np.int8)
            y = np.tril(x, k=6)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "tril_i8_zero"
            
            make_test(
                [x], y, "input_0.trilu(false, 6)", name)


        def triu():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int8)
            y = np.triu(x)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "triu_i8"
            
            make_test(
                [x], y, "input_0.trilu(true, 0)", name)

        def triu_neg():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int8)
            y = np.triu(x, k=-1)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "triu_i8_neg"
            
            make_test(
                [x], y, "input_0.trilu(true, -1)", name)

        def triu_one_row():
            x = np.random.randint(-127, 127, (3, 1, 5)).astype(np.int8)
            y = np.triu(x)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "triu_i8_one_row"
            
            make_test(
                [x], y, "input_0.trilu(true, 0)", name)

        def triu_out_neg():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int8)
            y = np.triu(x, k=-7)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "triu_i8_out_neg"
            
            make_test(
                [x], y, "input_0.trilu(true, -7)", name)


        def triu_out_pos():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int8)
            y = np.triu(x, k=6)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "triu_i8_out_pos"
            
            make_test(
                [x], y, "input_0.trilu(true, 6)", name)


        def triu_pos():
            x = np.random.randint(-127, 127, (4, 5)).astype(np.int8)
            y = np.triu(x, k=2)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "triu_i8_pos"
            
            make_test(
                [x], y, "input_0.trilu(true, 2)", name)


        def triu_square():
            x = np.random.randint(-127, 127, (2, 3, 3)).astype(np.int8)
            y = np.triu(x, k=0)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "triu_i8_square"
            
            make_test(
                [x], y, "input_0.trilu(true, 0)", name)


        def triu_square_neg():
            x = np.random.randint(-127, 127, (2, 3, 3)).astype(np.int8)
            y = np.triu(x, k=-1)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "triu_i8_square_neg"
            
            make_test(
                [x], y, "input_0.trilu(true, -1)", name)


        def triu_zero():
            x = np.random.randint(-127, 127, (3, 0, 5)).astype(np.int8)
            y = np.triu(x, k=6)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())

            name = "triu_i8_zero"
            
            make_test(
                [x], y, "input_0.trilu(true, 6)", name)

        tril()
        tril_neg()
        tril_one_row()
        tril_out_neg()
        tril_out_pos()
        tril_pos()
        tril_square()
        tril_square_neg()
        tril_zero()
        triu()
        triu_neg()
        triu_one_row()
        triu_out_neg()
        triu_out_pos()
        triu_pos()
        triu_square()
        triu_square_neg()
        triu_zero()


    @staticmethod
    def trilu_fp8x23():
        def tril():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP8x23)
            y = np.tril(x)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "tril_fp8x23"
            
            make_test(
                [x], y, "input_0.trilu(false, 0)", name)

        def tril_neg():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP8x23)
            y = np.tril(x, k=-1)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "tril_fp8x23_neg"
            
            make_test(
                [x], y, "input_0.trilu(false, -1)", name)

        def tril_one_row():
            x = to_fp(np.random.randint(-127, 127, (3, 1, 5)).astype(np.int64), FixedImpl.FP8x23)
            y = np.tril(x)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "tril_fp8x23_one_row"
            
            make_test(
                [x], y, "input_0.trilu(false, 0)", name)

        def tril_out_neg():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP8x23)
            y = np.tril(x, k=-7)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "tril_fp8x23_out_neg"
            
            make_test(
                [x], y, "input_0.trilu(false, -7)", name)


        def tril_out_pos():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP8x23)
            y = np.tril(x, k=6)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "tril_fp8x23_out_pos"
            
            make_test(
                [x], y, "input_0.trilu(false, 6)", name)


        def tril_pos():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP8x23)
            y = np.tril(x, k=2)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "tril_fp8x23_pos"
            
            make_test(
                [x], y, "input_0.trilu(false, 2)", name)


        def tril_square():
            x = to_fp(np.random.randint(-127, 127, (2, 3, 3)).astype(np.int64), FixedImpl.FP8x23)
            y = np.tril(x, k=0)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "tril_fp8x23_square"
            
            make_test(
                [x], y, "input_0.trilu(false, 0)", name)


        def tril_square_neg():
            x = to_fp(np.random.randint(-127, 127, (2, 3, 3)).astype(np.int64), FixedImpl.FP8x23)
            y = np.tril(x, k=-1)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "tril_fp8x23_square_neg"
            
            make_test(
                [x], y, "input_0.trilu(false, -1)", name)


        def tril_zero():
            x = to_fp(np.random.randint(-127, 127, (3, 0, 5)).astype(np.int64), FixedImpl.FP8x23)
            y = np.tril(x, k=6)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "tril_fp8x23_zero"
            
            make_test(
                [x], y, "input_0.trilu(false, 6)", name)


        def triu():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP8x23)
            y = np.triu(x)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "triu_fp8x23"
            
            make_test(
                [x], y, "input_0.trilu(true, 0)", name)

        def triu_neg():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP8x23)
            y = np.triu(x, k=-1)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "triu_fp8x23_neg"
            
            make_test(
                [x], y, "input_0.trilu(true, -1)", name)

        def triu_one_row():
            x = to_fp(np.random.randint(-127, 127, (3, 1, 5)).astype(np.int64), FixedImpl.FP8x23)
            y = np.triu(x)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "triu_fp8x23_one_row"
            
            make_test(
                [x], y, "input_0.trilu(true, 0)", name)

        def triu_out_neg():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP8x23)
            y = np.triu(x, k=-7)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "triu_fp8x23_out_neg"
            
            make_test(
                [x], y, "input_0.trilu(true, -7)", name)


        def triu_out_pos():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP8x23)
            y = np.triu(x, k=6)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "triu_fp8x23_out_pos"
            
            make_test(
                [x], y, "input_0.trilu(true, 6)", name)


        def triu_pos():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP8x23)
            y = np.triu(x, k=2)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "triu_fp8x23_pos"
            
            make_test(
                [x], y, "input_0.trilu(true, 2)", name)


        def triu_square():
            x = to_fp(np.random.randint(-127, 127, (2, 3, 3)).astype(np.int64), FixedImpl.FP8x23)
            y = np.triu(x, k=0)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "triu_fp8x23_square"
            
            make_test(
                [x], y, "input_0.trilu(true, 0)", name)


        def triu_square_neg():
            x = to_fp(np.random.randint(-127, 127, (2, 3, 3)).astype(np.int64), FixedImpl.FP8x23)
            y = np.triu(x, k=-1)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "triu_fp8x23_square_neg"
            
            make_test(
                [x], y, "input_0.trilu(true, -1)", name)


        def triu_zero():
            x = to_fp(np.random.randint(-127, 127, (3, 0, 5)).astype(np.int64), FixedImpl.FP8x23)
            y = np.triu(x, k=6)

            x = Tensor(Dtype.FP8x23, x.shape, x.flatten())
            y = Tensor(Dtype.FP8x23, y.shape, y.flatten())

            name = "triu_fp8x23_zero"
            
            make_test(
                [x], y, "input_0.trilu(true, 6)", name)

        tril()
        tril_neg()
        tril_one_row()
        tril_out_neg()
        tril_out_pos()
        tril_pos()
        tril_square()
        tril_square_neg()
        tril_zero()
        triu()
        triu_neg()
        triu_one_row()
        triu_out_neg()
        triu_out_pos()
        triu_pos()
        triu_square()
        triu_square_neg()
        triu_zero()

        
    @staticmethod
    def trilu_fp16x16():
        def tril():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP16x16)
            y = np.tril(x)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "tril_fp16x16"
            
            make_test(
                [x], y, "input_0.trilu(false, 0)", name)

        def tril_neg():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP16x16)
            y = np.tril(x, k=-1)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "tril_fp16x16_neg"
            
            make_test(
                [x], y, "input_0.trilu(false, -1)", name)

        def tril_one_row():
            x = to_fp(np.random.randint(-127, 127, (3, 1, 5)).astype(np.int64), FixedImpl.FP16x16)
            y = np.tril(x)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "tril_fp16x16_one_row"
            
            make_test(
                [x], y, "input_0.trilu(false, 0)", name)

        def tril_out_neg():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP16x16)
            y = np.tril(x, k=-7)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "tril_fp16x16_out_neg"
            
            make_test(
                [x], y, "input_0.trilu(false, -7)", name)


        def tril_out_pos():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP16x16)
            y = np.tril(x, k=6)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "tril_fp16x16_out_pos"
            
            make_test(
                [x], y, "input_0.trilu(false, 6)", name)


        def tril_pos():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP16x16)
            y = np.tril(x, k=2)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "tril_fp16x16_pos"
            
            make_test(
                [x], y, "input_0.trilu(false, 2)", name)


        def tril_square():
            x = to_fp(np.random.randint(-127, 127, (2, 3, 3)).astype(np.int64), FixedImpl.FP16x16)
            y = np.tril(x, k=0)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "tril_fp16x16_square"
            
            make_test(
                [x], y, "input_0.trilu(false, 0)", name)


        def tril_square_neg():
            x = to_fp(np.random.randint(-127, 127, (2, 3, 3)).astype(np.int64), FixedImpl.FP16x16)
            y = np.tril(x, k=-1)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "tril_fp16x16_square_neg"
            
            make_test(
                [x], y, "input_0.trilu(false, -1)", name)


        def tril_zero():
            x = to_fp(np.random.randint(-127, 127, (3, 0, 5)).astype(np.int64), FixedImpl.FP16x16)
            y = np.tril(x, k=6)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "tril_fp16x16_zero"
            
            make_test(
                [x], y, "input_0.trilu(false, 6)", name)


        def triu():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP16x16)
            y = np.triu(x)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "triu_fp16x16"
            
            make_test(
                [x], y, "input_0.trilu(true, 0)", name)

        def triu_neg():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP16x16)
            y = np.triu(x, k=-1)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "triu_fp16x16_neg"
            
            make_test(
                [x], y, "input_0.trilu(true, -1)", name)

        def triu_one_row():
            x = to_fp(np.random.randint(-127, 127, (3, 1, 5)).astype(np.int64), FixedImpl.FP16x16)
            y = np.triu(x)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "triu_fp16x16_one_row"
            
            make_test(
                [x], y, "input_0.trilu(true, 0)", name)

        def triu_out_neg():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP16x16)
            y = np.triu(x, k=-7)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "triu_fp16x16_out_neg"
            
            make_test(
                [x], y, "input_0.trilu(true, -7)", name)


        def triu_out_pos():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP16x16)
            y = np.triu(x, k=6)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "triu_fp16x16_out_pos"
            
            make_test(
                [x], y, "input_0.trilu(true, 6)", name)


        def triu_pos():
            x = to_fp(np.random.randint(-127, 127, (4, 5)).astype(np.int64), FixedImpl.FP16x16)
            y = np.triu(x, k=2)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "triu_fp16x16_pos"
            
            make_test(
                [x], y, "input_0.trilu(true, 2)", name)


        def triu_square():
            x = to_fp(np.random.randint(-127, 127, (2, 3, 3)).astype(np.int64), FixedImpl.FP16x16)
            y = np.triu(x, k=0)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "triu_fp16x16_square"
            
            make_test(
                [x], y, "input_0.trilu(true, 0)", name)


        def triu_square_neg():
            x = to_fp(np.random.randint(-127, 127, (2, 3, 3)).astype(np.int64), FixedImpl.FP16x16)
            y = np.triu(x, k=-1)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "triu_fp16x16_square_neg"
            
            make_test(
                [x], y, "input_0.trilu(true, -1)", name)


        def triu_zero():
            x = to_fp(np.random.randint(-127, 127, (3, 0, 5)).astype(np.int64), FixedImpl.FP16x16)
            y = np.triu(x, k=6)

            x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            y = Tensor(Dtype.FP16x16, y.shape, y.flatten())

            name = "triu_fp16x16_zero"
            
            make_test(
                [x], y, "input_0.trilu(true, 6)", name)

        tril()
        tril_neg()
        tril_one_row()
        tril_out_neg()
        tril_out_pos()
        tril_pos()
        tril_square()
        tril_square_neg()
        tril_zero()
        triu()
        triu_neg()
        triu_one_row()
        triu_out_neg()
        triu_out_pos()
        triu_pos()
        triu_square()
        triu_square_neg()
        triu_zero()
