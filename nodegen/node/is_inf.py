import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl

INF = 2**32 - 1

class Is_inf(RunAll):

    @staticmethod
    def is_inf_u32():
        def default():
            input_0 = np.array([1, 0, INF, 8, -INF, INF], dtype=np.uint32)
            output = np.array([False, False, True, False, True, True], dtype=bool)

            input_0 = Tensor(Dtype.U32, input_0.shape, input_0.flatten())
            output = Tensor(Dtype.BOOL, output.shape, output.flatten())

            name = "is_inf_u32"
            make_test([input_0], output, "TensorTrait::is_inf(@input_0, Option::None, Option::None)", name)

        default()

    @staticmethod
    def is_inf_i32():
        def default():
            input_0 = np.array([-1, 0, INF, 8, -INF, INF], dtype=np.int32)
            output = np.array([False, False, True, False, True, True], dtype=bool)

            input_0 = Tensor(Dtype.I32, input_0.shape, input_0.flatten())
            output = Tensor(Dtype.BOOL, output.shape, output.flatten())
            
            name = "is_inf_i32"
            make_test([input_0], output, "TensorTrait::is_inf(@input_0, Option::None, Option::None)", name)

        def positive():
            input_0 = np.array([-1, 0, INF, 8, -INF, INF], dtype=np.int32)
            output = np.array([False, False, True, False, False, True], dtype=bool)

            input_0 = Tensor(Dtype.I32, input_0.shape, input_0.flatten())
            output = Tensor(Dtype.BOOL, output.shape, output.flatten())
            
            name = "is_pos_inf_i32"
            make_test([input_0], output, "TensorTrait::is_inf(@input_0, Option::Some(0), Option::Some(1))", name)

        def negative():
            input_0 = np.array([-1, 0, INF, 8, -INF, INF], dtype=np.int32)
            output = np.array([False, False, False, False, True, False], dtype=bool)

            input_0 = Tensor(Dtype.I32, input_0.shape, input_0.flatten())
            output = Tensor(Dtype.BOOL, output.shape, output.flatten())
            
            name = "is_neg_inf_i32"
            make_test([input_0], output, "TensorTrait::is_inf(@input_0, Option::Some(1), Option::Some(0))", name)

        default()
        positive()
        negative()

    @staticmethod
    def is_inf_i8():
        def default():
            input_0 = np.array([-1, 0, INF, 8, -INF, INF], dtype=np.int8)
            output = np.array([False, False, True, False, True, True], dtype=bool)

            input_0 = Tensor(Dtype.I8, input_0.shape, input_0.flatten())
            output = Tensor(Dtype.BOOL, output.shape, output.flatten())
            
            name = "is_inf_i8"
            make_test([input_0], output, "TensorTrait::is_inf(@input_0, Option::None, Option::None)", name)

        def positive():
            input_0 = np.array([-1, 0, INF, 8, -INF, INF], dtype=np.int32)
            output = np.array([False, False, True, False, False, True], dtype=bool)

            input_0 = Tensor(Dtype.I8, input_0.shape, input_0.flatten())
            output = Tensor(Dtype.BOOL, output.shape, output.flatten())
            
            name = "is_pos_inf_i8"
            make_test([input_0], output, "TensorTrait::is_inf(@input_0, Option::Some(0), Option::Some(1))", name)

        def negative():
            input_0 = np.array([-1, 0, INF, 8, -INF, INF], dtype=np.int32)
            output = np.array([False, False, False, False, True, False], dtype=bool)

            input_0 = Tensor(Dtype.I8, input_0.shape, input_0.flatten())
            output = Tensor(Dtype.BOOL, output.shape, output.flatten())
            
            name = "is_neg_inf_i8"
            make_test([input_0], output, "TensorTrait::is_inf(@input_0, Option::Some(1), Option::Some(0))", name)

        default()
        positive()
        negative()

    @staticmethod
    def is_inf_fp8x23():
        def default():
            input_0 = np.array([-1.2, 0, INF, 2.8, -INF, INF], dtype=np.float64)
            output = np.array([False, False, True, False, True, True], dtype=bool)

            input_0 = Tensor(Dtype.FP8x23, input_0.shape, to_fp(
                input_0.flatten(), FixedImpl.FP8x23))
            output = Tensor(Dtype.BOOL, output.shape, output.flatten())
            
            name = "is_inf_fp8x23"
            make_test([input_0], output, "TensorTrait::is_inf(@input_0, Option::None, Option::None)", name)

        def positive():
            input_0 = np.array([-1.2, 0, INF, 2.8, -INF, INF], dtype=np.float64)
            output = np.array([False, False, True, False, False, True], dtype=bool)

            input_0 = Tensor(Dtype.FP8x23, input_0.shape, input_0.flatten())
            output = Tensor(Dtype.BOOL, output.shape, output.flatten())
            
            name = "is_pos_inf_fp8x23"
            make_test([input_0], output, "TensorTrait::is_inf(@input_0, Option::Some(0), Option::Some(1))", name)

        def negative():
            input_0 = np.array([-1.2, 0, INF, 2.8, -INF, INF], dtype=np.float64)
            output = np.array([False, False, False, False, True, False], dtype=bool)

            input_0 = Tensor(Dtype.FP8x23, input_0.shape, input_0.flatten())
            output = Tensor(Dtype.BOOL, output.shape, output.flatten())
            
            name = "is_neg_inf_fp8x23"
            make_test([input_0], output, "TensorTrait::is_inf(@input_0, Option::Some(1), Option::Some(0))", name)

        default()
        positive()
        negative()

    @staticmethod
    def is_inf_fp16x16():
        def default():
            input_0 = np.array([-1.2, 0, INF, 2.8, -INF, INF], dtype=np.float64)
            output = np.array([False, False, True, False, True, True], dtype=bool)

            input_0 = Tensor(Dtype.FP16x16, input_0.shape, to_fp(
                input_0.flatten(), FixedImpl.FP16x16))
            output = Tensor(Dtype.BOOL, output.shape, output.flatten())
            
            name = "is_inf_fp16x16"
            make_test([input_0], output, "TensorTrait::is_inf(@input_0, Option::None, Option::None)", name)

        def positive():
            input_0 = np.array([-1.2, 0, INF, 2.8, -INF, INF], dtype=np.float64)
            output = np.array([False, False, True, False, False, True], dtype=bool)

            input_0 = Tensor(Dtype.FP16x16, input_0.shape, input_0.flatten())
            output = Tensor(Dtype.BOOL, output.shape, output.flatten())
            
            name = "is_pos_inf_fp16x16"
            make_test([input_0], output, "TensorTrait::is_inf(@input_0, Option::Some(0), Option::Some(1))", name)

        def negative():
            input_0 = np.array([-1.2, 0, INF, 2.8, -INF, INF], dtype=np.float64)
            output = np.array([False, False, False, False, True, False], dtype=bool)

            input_0 = Tensor(Dtype.FP16x16, input_0.shape, input_0.flatten())
            output = Tensor(Dtype.BOOL, output.shape, output.flatten())
            
            name = "is_neg_inf_fp16x16"
            make_test([input_0], output, "TensorTrait::is_inf(@input_0, Option::Some(1), Option::Some(0))", name)

        default()
        positive()
        negative()
