import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl

INF = 2**32 - 1

class Is_inf(RunAll):

    @staticmethod
    def is_inf_i32():
        def default():
            input_0 = np.array([-1, 0, INF, 8, -INF, INF], dtype=np.int32)
            output = np.array([0, 0, 1, 0, 1, 1], dtype=np.uint32)

            input_0 = Tensor(Dtype.I32, input_0.shape, input_0.flatten())
            output = Tensor(Dtype.U32, output.shape, output.flatten())
            
            name = "is_inf_i32"
            make_test([input_0], output, "TensorTrait::is_inf(@input_0, Option::None, Option::None)", name)

        def positive():
            input_0 = np.array([-1, 0, INF, 8, -INF, INF], dtype=np.int32)
            output = np.array([0, 0, 1, 0, 0, 1], dtype=np.uint32)

            input_0 = Tensor(Dtype.I32, input_0.shape, input_0.flatten())
            output = Tensor(Dtype.U32, output.shape, output.flatten())
            
            name = "is_pos_inf_i32"
            make_test([input_0], output, "TensorTrait::is_inf(@input_0, Option::Some(0), Option::Some(1))", name)

        def negative():
            input_0 = np.array([-1, 0, INF, 8, -INF, INF], dtype=np.int32)
            output = np.array([0, 0, 0, 0, 1, 0], dtype=np.uint32)

            input_0 = Tensor(Dtype.I32, input_0.shape, input_0.flatten())
            output = Tensor(Dtype.U32, output.shape, output.flatten())
            
            name = "is_neg_inf_i32"
            make_test([input_0], output, "TensorTrait::is_inf(@input_0, Option::Some(1), Option::Some(0))", name)

        default()
        positive()
        negative()