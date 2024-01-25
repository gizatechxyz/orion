import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


class Reverse_sequence(RunAll):
    @staticmethod
    def Reverse_sequence_u32():
        def reverse_sequence_2d_batch():
            x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.uint32).reshape((4, 4))
            y = np.array([0, 1, 2, 3, 5, 4, 6, 7, 10, 9, 8, 11, 15, 14, 13, 12], dtype=np.uint32).reshape((4, 4))
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = Tensor(Dtype.U32, y.shape, y.flatten())
            name = "reverse_sequence_u32_2d_batch_equal_parts"
        
            make_test([_x], _y, "input_0.reverse_sequence(TensorTrait::<usize>::new(array![4].span(), array![1,2,3,4].span()), Option::Some(0), Option::Some(1))", name)

        def reverse_sequence_2d_time():    
            x = np.array([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], dtype=np.uint32).reshape((4, 4))
            y = np.array([3, 6, 9, 12, 2, 5, 8, 13, 1, 4, 10, 14, 0, 7, 11, 15], dtype=np.uint32).reshape((4, 4))
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = Tensor(Dtype.U32, y.shape, y.flatten())
            name = "reverse_sequence_u32_2d_time_equal_parts"
            make_test([_x], _y, "input_0.reverse_sequence(TensorTrait::<usize>::new(array![4].span(), array![4,3,2,1].span()), Option::Some(1), Option::Some(0))", name)
        reverse_sequence_2d_batch()
        reverse_sequence_2d_time()
    
    @staticmethod
    def Reverse_sequence_i32():
        def reverse_sequence_2d_batch():
            x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int32).reshape((4, 4))
            y = np.array([0, 1, 2, 3, 5, 4, 6, 7, 10, 9, 8, 11, 15, 14, 13, 12], dtype=np.int32).reshape((4, 4))
            _x = Tensor(Dtype.I32, x.shape, x.flatten())
            _y = Tensor(Dtype.I32, y.shape, y.flatten())
            name = "reverse_sequence_i32_2d_batch_equal_parts"
        
            make_test([_x], _y, "input_0.reverse_sequence(TensorTrait::<usize>::new(array![4].span(), array![1,2,3,4].span()), Option::Some(0), Option::Some(1))", name)

        def reverse_sequence_2d_time():    
            x = np.array([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], dtype=np.int32).reshape((4, 4))
            y = np.array([3, 6, 9, 12, 2, 5, 8, 13, 1, 4, 10, 14, 0, 7, 11, 15], dtype=np.int32).reshape((4, 4))
            _x = Tensor(Dtype.I32, x.shape, x.flatten())
            _y = Tensor(Dtype.I32, y.shape, y.flatten())
            name = "reverse_sequence_i32_2d_time_equal_parts"
            make_test([_x], _y, "input_0.reverse_sequence(TensorTrait::<usize>::new(array![4].span(), array![4,3,2,1].span()), Option::Some(1), Option::Some(0))", name)
        reverse_sequence_2d_batch()
        reverse_sequence_2d_time()
    
    @staticmethod
    def Reverse_sequence_i8():
        def reverse_sequence_2d_batch():
            x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int8).reshape((4, 4))
            y = np.array([0, 1, 2, 3, 5, 4, 6, 7, 10, 9, 8, 11, 15, 14, 13, 12], dtype=np.int8).reshape((4, 4))
            _x = Tensor(Dtype.I8, x.shape, x.flatten())
            _y = Tensor(Dtype.I8, y.shape, y.flatten())
            name = "reverse_sequence_i8_2d_batch_equal_parts"
        
            make_test([_x], _y, "input_0.reverse_sequence(TensorTrait::<usize>::new(array![4].span(), array![1,2,3,4].span()), Option::Some(0), Option::Some(1))", name)

        def reverse_sequence_2d_time():    
            x = np.array([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], dtype=np.uint32).reshape((4, 4))
            y = np.array([3, 6, 9, 12, 2, 5, 8, 13, 1, 4, 10, 14, 0, 7, 11, 15], dtype=np.uint32).reshape((4, 4))
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = Tensor(Dtype.U32, y.shape, y.flatten())
            name = "reverse_sequence_i8_2d_time_equal_parts"
            make_test([_x], _y, "input_0.reverse_sequence(TensorTrait::<usize>::new(array![4].span(), array![4,3,2,1].span()), Option::Some(1), Option::Some(0))", name)
        reverse_sequence_2d_batch()
        reverse_sequence_2d_time()

    def Reverse_sequence_fp16x16():
        def reverse_sequence_2d_batch():
            x = to_fp(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int64).reshape(4, 4), FixedImpl.FP16x16)
            y = to_fp(np.array([0, 1, 2, 3, 5, 4, 6, 7, 10, 9, 8, 11, 15, 14, 13, 12], dtype=np.int64).reshape(4, 4), FixedImpl.FP16x16)
            _x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            _y = Tensor(Dtype.FP16x16, y.shape, y.flatten())
            name = "reverse_sequence_fp16x16_2d_batch_equal_parts"
            make_test([_x], _y, "input_0.reverse_sequence(TensorTrait::<usize>::new(array![4].span(), array![1,2,3,4].span()), Option::Some(0), Option::Some(1))", name)
        def reverse_sequence_2d_time():
            x = to_fp(np.array([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], dtype=np.int64).reshape(4, 4), FixedImpl.FP16x16)
            y = to_fp(np.array([3, 6, 9, 12, 2, 5, 8, 13, 1, 4, 10, 14, 0, 7, 11, 15], dtype=np.int64).reshape(4, 4), FixedImpl.FP16x16)
            _x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            _y = Tensor(Dtype.FP16x16, y.shape, y.flatten())
            name = "reverse_sequence_fp16x16_2d_time_equal_parts"
            make_test([_x], _y, "input_0.reverse_sequence(TensorTrait::<usize>::new(array![4].span(), array![4,3,2,1].span()), Option::Some(1), Option::Some(0))", name)
        reverse_sequence_2d_batch()
        reverse_sequence_2d_time()

    def reverse_sequence_zero_size():
        x = np.array([]).astype(np.uint32)
        y = np.array([]).astype(np.uint32)
        _x = Tensor(Dtype.U32, x.shape, y.flatten())
        _y = Tensor(Dtype.U32, x.shape, y.flatten())
        name = "reverse_sequence_u32_zero_size"
        make_test([_x], _y, "input_0.reverse_sequence(TensorTrait::<usize>::new(array![].span(), array![].span()), Option::Some(1), Option::Some(0))", name)