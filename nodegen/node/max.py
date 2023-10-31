import numpy as np
from nodegen.node import RunAll
from ..helpers import make_node, make_test, to_fp, Tensor, Dtype, FixedImpl, Trait

class Max(RunAll):

    @staticmethod
    def max_u32_two_tensors():
        def default():
            x = np.random.randint(0, 6, (3, 3, 3)).astype(np.uint32)
            y = np.random.randint(0, 6, (3, 3, 3)).astype(np.uint32)
            z = np.maximum(x, y)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())
            z = Tensor(Dtype.U32, z.shape, z.flatten())

            name = "max_u32_two_tensors"
            make_node([x, y], [z], name)
            make_test([x, y], z, "TensorTrait::max(array![input_0, input_1].span());", name)

        def broadcast():
            x = np.random.randint(0, 6, (2, 2)).astype(np.uint32)
            y = np.random.randint(0, 6, (1, 2)).astype(np.uint32)
            z = np.maximum(x, y)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())
            z = Tensor(Dtype.U32, z.shape, z.flatten())

            name = "max_u32_broadcast_two_tensors"
            make_node([x, y], [z], name)
            make_test([x, y], z, "TensorTrait::max(array![input_0, input_1].span());", name)

        default()
        broadcast()

    @staticmethod
    def max_i32_two_tensors():
        def default():
            x = np.random.randint(0, 6, (3, 3, 3)).astype(np.int32)
            y = np.random.randint(0, 6, (3, 3, 3)).astype(np.int32)
            z = np.maximum(x, y)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())
            z = Tensor(Dtype.I32, z.shape, z.flatten())

            name = "max_i32_two_tensors"
            make_node([x, y], [z], name)
            make_test([x, y], z, "TensorTrait::max(array![input_0, input_1].span());", name)

        def broadcast():
            x = np.random.randint(0, 6, (2, 2)).astype(np.int32)
            y = np.random.randint(0, 6, (1, 2)).astype(np.int32)
            z = np.maximum(x, y)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())
            z = Tensor(Dtype.I32, z.shape, z.flatten())

            name = "max_i32_broadcast_two_tensors"
            make_node([x, y], [z], name)
            make_test([x, y], z, "TensorTrait::max(array![input_0, input_1].span());", name)

        default()
        broadcast()

    @staticmethod
    def max_i8_two_tensors():
        def default():
            x = np.random.randint(0, 6, (3, 3, 3)).astype(np.int8)
            y = np.random.randint(0, 6, (3, 3, 3)).astype(np.int8)
            z = np.maximum(x, y)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())
            z = Tensor(Dtype.I8, z.shape, z.flatten())

            name = "max_i8_two_tensors"
            make_node([x, y], [z], name)
            make_test([x, y], z, "TensorTrait::max(array![input_0, input_1].span());", name)

        def broadcast():
            x = np.random.randint(0, 6, (2, 2)).astype(np.int8)
            y = np.random.randint(0, 6, (1, 2)).astype(np.int8)
            z = np.maximum(x, y)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())
            z = Tensor(Dtype.I8, z.shape, z.flatten())

            name = "max_i8_broadcast_two_tensors"
            make_node([x, y], [z], name)
            make_test([x, y], z, "TensorTrait::max(array![input_0, input_1].span());", name)

        default()
        broadcast()

    @staticmethod
    def max_fp8x23_two_tensors():
        def default():
            x = np.random.randint(-3, 3, (3, 3, 3)).astype(np.float64)
            y = np.random.randint(-3, 3, (3, 3, 3)).astype(np.float64)
            z = np.maximum(x, y)

            x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                x.flatten(), FixedImpl.FP8x23))
            y = Tensor(Dtype.FP8x23, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP8x23))
            z = Tensor(Dtype.FP8x23, z.shape,  to_fp(
                z.flatten(), FixedImpl.FP8x23))

            name = "max_fp8x23_two_tensors"
            make_node([x, y], [z], name)
            make_test([x, y], z, "TensorTrait::max(array![input_0, input_1].span());", name)

        def broadcast():
            x = np.random.randint(-3, 3, (2, 2)).astype(np.float64)
            y = np.random.randint(-3, 3, (1, 2)).astype(np.float64)
            z = np.maximum(x, y)

            x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                x.flatten(), FixedImpl.FP8x23))
            y = Tensor(Dtype.FP8x23, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP8x23))
            z = Tensor(Dtype.FP8x23, z.shape,  to_fp(
                z.flatten(), FixedImpl.FP8x23))

            name = "max_fp8x23_broadcast_two_tensors"
            make_node([x, y], [z], name)
            make_test([x, y], z, "TensorTrait::max(array![input_0, input_1].span());", name)

        default()
        broadcast()

    @staticmethod
    def max_fp16x16_two_tensors():
        def default():
            x = np.random.randint(-3, 3, (3, 3, 3)).astype(np.float64)
            y = np.random.randint(-3, 3, (3, 3, 3)).astype(np.float64)
            z = np.maximum(x, y)

            x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                x.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP16x16))
            z = Tensor(Dtype.FP16x16, z.shape,  to_fp(
                z.flatten(), FixedImpl.FP16x16))

            name = "max_fp16x16_two_tensors"
            make_node([x, y], [z], name)
            make_test([x, y], z, "TensorTrait::max(array![input_0, input_1].span());", name)

        def broadcast():
            x = np.random.randint(-3, 3, (2, 2)).astype(np.float64)
            y = np.random.randint(-3, 3, (1, 2)).astype(np.float64)
            z = np.maximum(x, y)

            x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                x.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP16x16))
            z = Tensor(Dtype.FP16x16, z.shape,  to_fp(
                z.flatten(), FixedImpl.FP16x16))

            name = "max_fp16x16_broadcast_two_tensors"
            make_node([x, y], [z], name)
            make_test([x, y], z, "TensorTrait::max(array![input_0, input_1].span());", name)

        default()
        broadcast()


    @staticmethod
    def max_u32_three_tensors():
        def default():
            x = np.random.randint(0, 6, (3, 3, 3)).astype(np.uint32)
            y = np.random.randint(0, 6, (3, 3, 3)).astype(np.uint32)
            z = np.random.randint(0, 6, (3, 3, 3)).astype(np.uint32)
            m = np.maximum(np.maximum(x, y), z)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())
            z = Tensor(Dtype.U32, z.shape, z.flatten())
            m = Tensor(Dtype.U32, m.shape, m.flatten())

            name = "max_u32_three_tensors"
            make_node([x, y, z], [m], name)
            make_test([x, y, z], m, "TensorTrait::max(array![input_0, input_1, input_2].span());", name)

        def broadcast():
            x = np.random.randint(0, 6, (2, 2)).astype(np.uint32)
            y = np.random.randint(0, 6, (1, 2)).astype(np.uint32)
            z = np.random.randint(0, 6, (1, 1)).astype(np.uint32)
            m = np.maximum(np.maximum(x, y), z)

            x = Tensor(Dtype.U32, x.shape, x.flatten())
            y = Tensor(Dtype.U32, y.shape, y.flatten())
            z = Tensor(Dtype.U32, z.shape, z.flatten())
            m = Tensor(Dtype.U32, m.shape, m.flatten())

            name = "max_u32_broadcast_three_tensors"
            make_node([x, y, z], [m], name)
            make_test([x, y, z], m, "TensorTrait::max(array![input_0, input_1, input_2].span());", name)

        default()
        broadcast()

    @staticmethod
    def max_i32_three_tensors():
        def default():
            x = np.random.randint(0, 6, (3, 3, 3)).astype(np.int32)
            y = np.random.randint(0, 6, (3, 3, 3)).astype(np.int32)
            z = np.random.randint(0, 6, (3, 3, 3)).astype(np.int32)
            m = np.maximum(np.maximum(x, y), z)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())
            z = Tensor(Dtype.I32, z.shape, z.flatten())
            m = Tensor(Dtype.I32, m.shape, m.flatten())

            name = "max_i32_three_tensors"
            make_node([x, y, z], [m], name)
            make_test([x, y, z], m, "TensorTrait::max(array![input_0, input_1, input_2].span());", name)

        def broadcast():
            x = np.random.randint(0, 6, (2, 2)).astype(np.int32)
            y = np.random.randint(0, 6, (1, 2)).astype(np.int32)
            z = np.random.randint(0, 6, (1, 1)).astype(np.int32)
            m = np.maximum(np.maximum(x, y), z)

            x = Tensor(Dtype.I32, x.shape, x.flatten())
            y = Tensor(Dtype.I32, y.shape, y.flatten())
            z = Tensor(Dtype.I32, z.shape, z.flatten())
            m = Tensor(Dtype.I32, m.shape, m.flatten())

            name = "max_i32_broadcast_three_tensors"
            make_node([x, y, z], [m], name)
            make_test([x, y, z], m, "TensorTrait::max(array![input_0, input_1, input_2].span());", name)

        default()
        broadcast()

    @staticmethod
    def max_i8_three_tensors():
        def default():
            x = np.random.randint(0, 6, (3, 3, 3)).astype(np.int8)
            y = np.random.randint(0, 6, (3, 3, 3)).astype(np.int8)
            z = np.random.randint(0, 6, (3, 3, 3)).astype(np.int8)
            m = np.maximum(np.maximum(x, y), z)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())
            z = Tensor(Dtype.I8, z.shape, z.flatten())
            m = Tensor(Dtype.I8, m.shape, m.flatten())

            name = "max_i8_three_tensors"
            make_node([x, y, z], [m], name)
            make_test([x, y, z], m, "TensorTrait::max(array![input_0, input_1, input_2].span());", name)

        def broadcast():
            x = np.random.randint(0, 6, (2, 2)).astype(np.int8)
            y = np.random.randint(0, 6, (1, 2)).astype(np.int8)
            z = np.random.randint(0, 6, (1, 1)).astype(np.int8)
            m = np.maximum(np.maximum(x, y), z)

            x = Tensor(Dtype.I8, x.shape, x.flatten())
            y = Tensor(Dtype.I8, y.shape, y.flatten())
            z = Tensor(Dtype.I8, z.shape, z.flatten())
            m = Tensor(Dtype.I8, m.shape, m.flatten())

            name = "max_i8_broadcast_three_tensors"
            make_node([x, y, z], [m], name)
            make_test([x, y, z], m, "TensorTrait::max(array![input_0, input_1, input_2].span());", name)

        default()
        broadcast()

    @staticmethod
    def max_fp8x23_three_tensors():
        def default():
            x = np.random.randint(-3, 3, (3, 3, 3)).astype(np.float64)
            y = np.random.randint(-3, 3, (3, 3, 3)).astype(np.float64)
            z = np.random.randint(-3, 3, (3, 3, 3)).astype(np.float64)
            m = np.maximum(np.maximum(x, y), z)

            x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                x.flatten(), FixedImpl.FP8x23))
            y = Tensor(Dtype.FP8x23, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP8x23))
            z = Tensor(Dtype.FP8x23, z.shape,  to_fp(
                z.flatten(), FixedImpl.FP8x23))  
            m = Tensor(Dtype.FP8x23, m.shape,  to_fp(
                m.flatten(), FixedImpl.FP8x23)) 

            name = "max_fp8x23_three_tensors"
            make_node([x, y, z], [m], name)
            make_test([x, y, z], m, "TensorTrait::max(array![input_0, input_1, input_2].span());", name)

        def broadcast():
            x = np.random.randint(-3, 3, (2, 2)).astype(np.float64)
            y = np.random.randint(-3, 3, (1, 2)).astype(np.float64)
            z = np.random.randint(-3, 3, (1, 1)).astype(np.float64)
            m = np.maximum(np.maximum(x, y), z)

            x = Tensor(Dtype.FP8x23, x.shape, to_fp(
                x.flatten(), FixedImpl.FP8x23))
            y = Tensor(Dtype.FP8x23, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP8x23))
            z = Tensor(Dtype.FP8x23, z.shape,  to_fp(
                z.flatten(), FixedImpl.FP8x23))  
            m = Tensor(Dtype.FP8x23, m.shape,  to_fp(
                m.flatten(), FixedImpl.FP8x23))

            name = "max_fp8x23_broadcast_three_tensors"
            make_node([x, y, z], [m], name)
            make_test([x, y, z], m, "TensorTrait::max(array![input_0, input_1, input_2].span());", name)

        default()
        broadcast()

    @staticmethod
    def max_fp16x16_three_tensors():
        def default():
            x = np.random.randint(-3, 3, (3, 3, 3)).astype(np.float64)
            y = np.random.randint(-3, 3, (3, 3, 3)).astype(np.float64)
            z = np.random.randint(-3, 3, (3, 3, 3)).astype(np.float64)
            m = np.maximum(np.maximum(x, y), z)

            x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                x.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP16x16))
            z = Tensor(Dtype.FP16x16, z.shape,  to_fp(
                z.flatten(), FixedImpl.FP16x16))    
            m = Tensor(Dtype.FP16x16, m.shape,  to_fp(
                m.flatten(), FixedImpl.FP16x16))   

            name = "max_fp16x16_three_tensors"
            make_node([x, y, z], [m], name)
            make_test([x, y, z], m, "TensorTrait::max(array![input_0, input_1, input_2].span());", name)

        def broadcast():
            x = np.random.randint(-3, 3, (2, 2)).astype(np.float64)
            y = np.random.randint(-3, 3, (1, 2)).astype(np.float64)
            z = np.random.randint(-3, 3, (1, 1)).astype(np.float64)
            m = np.maximum(np.maximum(x, y), z)

            x = Tensor(Dtype.FP16x16, x.shape, to_fp(
                x.flatten(), FixedImpl.FP16x16))
            y = Tensor(Dtype.FP16x16, y.shape,  to_fp(
                y.flatten(), FixedImpl.FP16x16))
            z = Tensor(Dtype.FP16x16, z.shape,  to_fp(
                z.flatten(), FixedImpl.FP16x16))    
            m = Tensor(Dtype.FP16x16, m.shape,  to_fp(
                m.flatten(), FixedImpl.FP16x16)) 

            name = "max_fp16x16_broadcast_three_tensors"
            make_node([x, y, z], [m], name)
            make_test([x, y, z], m, "TensorTrait::max(array![input_0, input_1, input_2].span());", name)

        default()
        broadcast()