import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl


class Array_feature_extractor(RunAll):

    @staticmethod
    def array_feature_extractor_i32():
        x = np.random.randint(-3, 3, (2, 3, 4)).astype(np.int32)
        y = np.array([1, 3]).astype(np.uint32)
        z = (x[..., y])

        x = Tensor(Dtype.I32, x.shape, x.flatten())
        y = Tensor(Dtype.U32, y.shape, y.flatten())
        z = Tensor(Dtype.I32, z.shape, z.flatten())

        name = "array_feature_extractor_i32"
        make_node([x, y], [z], name)
        make_test([x, y], z, "TensorTrait::array_feature_extractor(@input_0, input_1);", name)


    @staticmethod
    def array_feature_extractor_fp8x23():
        x = np.random.randint(-3, 3, (2, 3, 4)).astype(np.float64)
        y = np.array([1, 3]).astype(np.uint32)
        z = (x[..., y])

        x = Tensor(Dtype.FP8x23, x.shape, to_fp(
            x.flatten(), FixedImpl.FP8x23))
        y = Tensor(Dtype.U32, y.shape, y.flatten())
        z = Tensor(Dtype.FP8x23, z.shape, to_fp(
            z.flatten(), FixedImpl.FP8x23))

        name = "array_feature_extractor_fp8x23"
        make_node([x, y], [z], name)
        make_test([x, y], z, "TensorTrait::array_feature_extractor(@input_0, input_1);", name)

    
    @staticmethod
    def array_feature_extractor_fp16x16():
        x = np.random.randint(-3, 3, (2, 3, 4)).astype(np.float64)
        y = np.array([1, 3]).astype(np.uint32)
        z = (x[..., y])

        x = Tensor(Dtype.FP16x16, x.shape, to_fp(
            x.flatten(), FixedImpl.FP16x16))
        y = Tensor(Dtype.U32, y.shape, y.flatten())
        z = Tensor(Dtype.FP16x16, z.shape, to_fp(
            z.flatten(), FixedImpl.FP16x16))

        name = "array_feature_extractor_fp16x16"
        make_node([x, y], [z], name)
        make_test([x, y], z, "TensorTrait::array_feature_extractor(@input_0, input_1);", name)