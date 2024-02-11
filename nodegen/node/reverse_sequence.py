import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


class Reverse_sequence(RunAll):
    @staticmethod
    def Reverse_sequence_u32():
        def reverse_sequence_u32_4x4_batch():
            x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.uint32).reshape((4, 4))
            y = np.array([0, 1, 2, 3, 5, 4, 6, 7, 10, 9, 8, 11, 15, 14, 13, 12], dtype=np.uint32).reshape((4, 4))
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = Tensor(Dtype.U32, y.shape, y.flatten())
            name = "reverse_sequence_u32_4x4_batch"
        
            make_test(
                [_x], 
                _y, 
                "input_0.reverse_sequence(TensorTrait::<usize>::new(array![4].span(), array![1,2,3,4].span()), Option::Some(0), Option::Some(1))", 
                name
            )

        def reverse_sequence_u32_4x4_time():    
            x = np.array([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], dtype=np.uint32).reshape((4, 4))
            y = np.array([3, 6, 9, 12, 2, 5, 8, 13, 1, 4, 10, 14, 0, 7, 11, 15], dtype=np.uint32).reshape((4, 4))
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = Tensor(Dtype.U32, y.shape, y.flatten())
            name = "reverse_sequence_u32_4x4_time"
            make_test(
                [_x], 
                _y, 
                "input_0.reverse_sequence(TensorTrait::<usize>::new(array![4].span(), array![4,3,2,1].span()), Option::Some(1), Option::Some(0))", 
                name
            )
        def reverse_sequence_u32_3x3_batch():
            x = np.array([0,1,2,3,4,5,6,7,8], dtype=np.uint32).reshape(3,3)
            y = np.array([2,1,0,3,4,5,7,6,8], dtype=np.uint32).reshape(3,3)
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = Tensor(Dtype.U32, y.shape, y.flatten())
            name = "reverse_sequence_u32_3x3_batch"
            make_test(
                [_x],
                _y,
                "input_0.reverse_sequence(TensorTrait::<usize>::new(array![3].span(), array![3,1,2].span()), Option::Some(0), Option::Some(1))",
                name
                )
        def reverse_sequence_u32_3x3_time():
            x = np.array([0,1,2,3,4,5,6,7,8], dtype=np.uint32).reshape(3,3)
            y = np.array([0,7,8,3,4,5,6,1,2], dtype=np.uint32).reshape(3,3)
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = Tensor(Dtype.U32, y.shape, y.flatten())
            name = "reverse_sequence_u32_3x3_time"
            make_test(
                [_x],
                _y,
                "input_0.reverse_sequence(TensorTrait::<usize>::new(array![3].span(), array![1,3,3].span()), Option::Some(1), Option::Some(0))",
                name
                )
            
        reverse_sequence_u32_4x4_batch()
        reverse_sequence_u32_4x4_time()
        reverse_sequence_u32_3x3_batch()
        reverse_sequence_u32_3x3_time()

    
    @staticmethod
    def Reverse_sequence_i32():
        def reverse_sequence_i32_batch():
            x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int32).reshape((4, 4))
            y = np.array([0, 1, 2, 3, 5, 4, 6, 7, 10, 9, 8, 11, 15, 14, 13, 12], dtype=np.int32).reshape((4, 4))
            _x = Tensor(Dtype.I32, x.shape, x.flatten())
            _y = Tensor(Dtype.I32, y.shape, y.flatten())
            name = "reverse_sequence_i32_batch_equal_parts"
        
            make_test(
                [_x], 
                _y, 
                "input_0.reverse_sequence(TensorTrait::<usize>::new(array![4].span(), array![1,2,3,4].span()), Option::Some(0), Option::Some(1))", 
                name
            )

        def reverse_sequence_i32_time():    
            x = np.array([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], dtype=np.int32).reshape((4, 4))
            y = np.array([3, 6, 9, 12, 2, 5, 8, 13, 1, 4, 10, 14, 0, 7, 11, 15], dtype=np.int32).reshape((4, 4))
            _x = Tensor(Dtype.I32, x.shape, x.flatten())
            _y = Tensor(Dtype.I32, y.shape, y.flatten())
            name = "reverse_sequence_i32_time_equal_parts"
            make_test(
                [_x], 
                _y, 
                "input_0.reverse_sequence(TensorTrait::<usize>::new(array![4].span(), array![4,3,2,1].span()), Option::Some(1), Option::Some(0))", 
                name
            )
        
        reverse_sequence_i32_batch()
        reverse_sequence_i32_time()
    
    @staticmethod
    def Reverse_sequence_i8():
        def reverse_sequence_batch():
            x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int8).reshape((4, 4))
            y = np.array([0, 1, 2, 3, 5, 4, 6, 7, 10, 9, 8, 11, 15, 14, 13, 12], dtype=np.int8).reshape((4, 4))
            _x = Tensor(Dtype.I8, x.shape, x.flatten())
            _y = Tensor(Dtype.I8, y.shape, y.flatten())
            name = "reverse_sequence_i8_batch_equal_parts"
        
            make_test(
                [_x], 
                _y, 
                "input_0.reverse_sequence(TensorTrait::<usize>::new(array![4].span(), array![1,2,3,4].span()), Option::Some(0), Option::Some(1))", 
                name
            )

        def reverse_sequence_time():    
            x = np.array([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], dtype=np.uint32).reshape((4, 4))
            y = np.array([3, 6, 9, 12, 2, 5, 8, 13, 1, 4, 10, 14, 0, 7, 11, 15], dtype=np.uint32).reshape((4, 4))
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = Tensor(Dtype.U32, y.shape, y.flatten())
            name = "reverse_sequence_i8_time_equal_parts"
            make_test(
                [_x], 
                _y, 
                "input_0.reverse_sequence(TensorTrait::<usize>::new(array![4].span(), array![4,3,2,1].span()), Option::Some(1), Option::Some(0))", 
                name
            )
        reverse_sequence_batch()
        reverse_sequence_time()

    def Reverse_sequence_fp16x16():
        def reverse_sequence_batch():
            x = to_fp(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int64).reshape(4, 4), FixedImpl.FP16x16)
            y = to_fp(np.array([0, 1, 2, 3, 5, 4, 6, 7, 10, 9, 8, 11, 15, 14, 13, 12], dtype=np.int64).reshape(4, 4), FixedImpl.FP16x16)
            _x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            _y = Tensor(Dtype.FP16x16, y.shape, y.flatten())
            name = "reverse_sequence_fp16x16_batch_equal_parts"
            make_test(
                [_x],
                _y, "input_0.reverse_sequence(TensorTrait::<usize>::new(array![4].span(), array![1,2,3,4].span()), Option::Some(0), Option::Some(1))", 
                name
            )
        def reverse_sequence_time():
            x = to_fp(np.array([0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], dtype=np.int64).reshape(4, 4), FixedImpl.FP16x16)
            y = to_fp(np.array([3, 6, 9, 12, 2, 5, 8, 13, 1, 4, 10, 14, 0, 7, 11, 15], dtype=np.int64).reshape(4, 4), FixedImpl.FP16x16)
            _x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            _y = Tensor(Dtype.FP16x16, y.shape, y.flatten())
            name = "reverse_sequence_fp16x16_time_equal_parts"
            make_test(
                [_x],
                _y, 
                "input_0.reverse_sequence(TensorTrait::<usize>::new(array![4].span(), array![4,3,2,1].span()), Option::Some(1), Option::Some(0))", 
                name
            )
        reverse_sequence_batch()
        reverse_sequence_time()

    def reverse_sequence_different_dimensions():
        def reverse_sequence_different_dimensions_4_5():
            x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], dtype=np.uint32).reshape(4,5)
            y = np.array([5,4,3,2,1,9,8,7,6,10,13,12,11,14,15,17,16,18,19,20], dtype=np.uint32).reshape(4,5)
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = Tensor(Dtype.U32, y.shape, y.flatten())
            name = "reverse_sequence_different_dimensions_4_5"
            make_test(
                [_x],
                _y,
                "input_0.reverse_sequence(TensorTrait::<usize>::new(array![4].span(), array![5,4,3,2].span()), Option::Some(0), Option::Some(1))",
                name
                )
        
        def reverse_sequence_different_dimensions_2_4():
            x = np.array([1,2,3,4,5,6,7,8], dtype=np.uint32).reshape(2,4)
            y = np.array([5,6,7,8,1,2,3,4], dtype=np.uint32).reshape(2,4)
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = Tensor(Dtype.U32, y.shape, y.flatten())
            name = "reverse_sequence_different_dimensions_2_4"
            make_test(
                [_x],
                _y,
                "input_0.reverse_sequence(TensorTrait::<usize>::new(array![4].span(), array![2,2,2,2].span()), Option::Some(1), Option::Some(0))",
                name
                )
        def reverse_sequence_different_dimensions_1_6():
            x = np.array([0,1,2,3,4,5], dtype=np.uint32).reshape(1,6)
            y = np.array([4,3,2,1,0,5], dtype=np.uint32).reshape(1,6)
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = Tensor(Dtype.U32, y.shape, y.flatten())
            name = "reverse_sequence_different_dimensions_1_6"
            make_test(
                [_x],
                _y,
                "input_0.reverse_sequence(TensorTrait::<usize>::new(array![1].span(), array![5].span()), Option::Some(0), Option::Some(1))",
                name
                )
            
        def reverse_sequence_different_dimensions_3x9_batch():
            x = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26], dtype=np.uint32).reshape(3,9)
            y = np.array([6,5,4,3,2,1,0,7,8,16,15,14,13,12,11,10,9,17,26,25,24,23,22,21,20,19,18], dtype=np.uint32).reshape(3,9)
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = Tensor(Dtype.U32, y.shape, y.flatten())
            name = "reverse_sequence_different_dimensions_3x9_batch"
            make_test(
                [_x],
                _y,
                "input_0.reverse_sequence(TensorTrait::<usize>::new(array![3].span(), array![7,8,9].span()), Option::Some(0), Option::Some(1))",
                name
                )
        def reverse_sequence_different_dimensions_3x9_time():
            x = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26], dtype=np.uint32).reshape(3,9)
            y = np.array([18,10,20,12,22,14,24,16,8,9,1,11,3,13,5,15,7,17,0,19,2,21,4,23,6,25,26], dtype=np.uint32).reshape(3,9)
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = Tensor(Dtype.U32, y.shape, y.flatten())
            name = "reverse_sequence_different_dimensions_3x9_time"
            make_test(
                [_x],
                _y,
                "input_0.reverse_sequence(TensorTrait::<usize>::new(array![9].span(), array![3,2,3,2,3,2,3,2,1].span()), Option::Some(1), Option::Some(0))",
                name
                )
            
        reverse_sequence_different_dimensions_4_5()
        reverse_sequence_different_dimensions_2_4()
        reverse_sequence_different_dimensions_1_6()
        reverse_sequence_different_dimensions_3x9_batch()
        reverse_sequence_different_dimensions_3x9_time()
            