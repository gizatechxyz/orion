import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


class Split_to_sequence(RunAll):
    @staticmethod
    def split_to_sequence_u32():
        def split_to_sequence_1D():
            x = np.random.randint(0, 255, 6).astype(np.uint32)
            y = [
                np.array(x[0:2]).astype(np.uint32), 
                np.array(x[2:4]).astype(np.uint32),
                np.array(x[4:6]).astype(np.uint32),
            ]

            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = [
                Tensor(Dtype.U32, y[0].shape, y[0].flatten()),
                Tensor(Dtype.U32, y[1].shape, y[1].flatten()),
                Tensor(Dtype.U32, y[2].shape, y[2].flatten()),
            ]

            name = "split_to_sequence_u32_1d_equal_parts"
            make_test(
                [_x], _y, "input_0.split_to_sequence(0, 1, Option::Some(TensorTrait::<u32>::new(shape: array![1].span(), data: array![3].span(),)))", name)
            y = [
                np.array(x[0:2]).astype(np.uint32), 
                np.array(x[2:6]).astype(np.uint32),
            ]
            _y = [
                Tensor(Dtype.U32, y[0].shape, y[0].flatten()),
                Tensor(Dtype.U32, y[1].shape, y[1].flatten()),
            ]
            name = "split_to_sequence_u32_1d_variable_parts"
            make_test(
                [_x], _y, "input_0.split_to_sequence(0, 1, Option::Some(TensorTrait::<u32>::new(shape: array![2].span(), data: array![2, 4].span(),)))", name)
        def split_to_sequence_2D():
            x = np.random.randint(0, 255, (2, 6)).astype(np.uint32)
            y = [
                np.array(x[0:2, 0:3]).astype(np.uint32),
                np.array(x[0:2, 3:6]).astype(np.uint32),
            ]
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = [
                Tensor(Dtype.U32, y[0].shape, y[0].flatten()),
                Tensor(Dtype.U32, y[1].shape, y[1].flatten()),
            ]
            name = "split_to_sequence_u32_2d_equal_parts"
            make_test(
                [_x], _y, "input_0.split_to_sequence(1, 1, Option::Some(TensorTrait::<u32>::new(shape: array![1].span(), data: array![2].span(),)))", name)

            y = [
                np.array(x[0:2, 0:2]).astype(np.uint32), 
                np.array(x[0:2, 2:6]).astype(np.uint32)
            ]
            _y = [
                Tensor(Dtype.U32, y[0].shape, y[0].flatten()),
                Tensor(Dtype.U32, y[1].shape, y[1].flatten()),
            ]
            name = "split_to_sequence_u32_2d_variable_parts"
            make_test(
                [_x], _y, "input_0.split_to_sequence(1, 1, Option::Some(TensorTrait::<u32>::new(shape: array![2].span(), data: array![2, 4].span(),)))", name)
        
        def split_to_sequence_zero_size():
            # 1-dimensional tensor with dimension_size=0
            x = np.array([]).astype(np.uint32)
            y = [
                np.array([]).astype(np.uint32),
                np.array([]).astype(np.uint32),
                np.array([]).astype(np.uint32),
            ]
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = [
                Tensor(Dtype.U32, y[0].shape, y[0].flatten()),
                Tensor(Dtype.U32, y[1].shape, y[1].flatten()),
                Tensor(Dtype.U32, y[2].shape, y[2].flatten()),
            ]
            # Split emtpy tensor to tensors of size zero
            name = "split_to_sequence_u32_zero_size"
            make_test(
                [_x], _y, "input_0.split_to_sequence(0, 1, Option::Some(TensorTrait::<u32>::new(shape: array![3].span(), data: array![0, 0, 0].span(),)))", name)
            
        
        def split_to_sequence_1d_uneven():
            x = np.random.randint(0, 255, 7).astype(np.uint32)
            y = [
                np.array(x[0:2]).astype(np.uint32), 
                np.array(x[2:4]).astype(np.uint32),
                np.array(x[4:6]).astype(np.uint32),
                np.array(x[6:7]).astype(np.uint32),
            ]
            
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = [
                Tensor(Dtype.U32, y[0].shape, y[0].flatten()),
                Tensor(Dtype.U32, y[1].shape, y[1].flatten()),
                Tensor(Dtype.U32, y[2].shape, y[2].flatten()),
                Tensor(Dtype.U32, y[3].shape, y[3].flatten()),
            ]

            name = "split_to_sequence_u32_1d_uneven"
            make_test(
                [_x], _y, "input_0.split_to_sequence(0, 1, Option::Some(TensorTrait::<u32>::new(shape: array![1].span(), data: array![4].span(),)))", name)
            

        def split_to_sequence_2d_uneven():
            x = np.random.randint(0, 255, (2, 8)).astype(np.uint32)
            y = [
                np.array(x[0:2, 0:3]).astype(np.uint32), 
                np.array(x[0:2, 3:6]).astype(np.uint32), 
                np.array(x[0:2, 6:8]).astype(np.uint32)
            ]
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = [
                Tensor(Dtype.U32, y[0].shape, y[0].flatten()),
                Tensor(Dtype.U32, y[1].shape, y[1].flatten()),
                Tensor(Dtype.U32, y[2].shape, y[2].flatten()),
            ]

            name = "split_to_sequence_u32_2d_uneven"
            make_test(
                [_x], _y, "input_0.split_to_sequence(1, 1, Option::Some(TensorTrait::<u32>::new(shape: array![1].span(), data: array![3].span(),)))", name)
        
        def split_to_sequence_2d_scalar():
            x = np.random.randint(0, 255, (2, 8)).astype(np.uint32)
            y = [
                np.array(x[0:2, 0:1]).astype(np.uint32), 
                np.array(x[0:2, 1:2]).astype(np.uint32), 
                np.array(x[0:2, 2:3]).astype(np.uint32), 
                np.array(x[0:2, 3:4]).astype(np.uint32), 
                np.array(x[0:2, 4:5]).astype(np.uint32), 
                np.array(x[0:2, 5:6]).astype(np.uint32), 
                np.array(x[0:2, 6:7]).astype(np.uint32),
                np.array(x[0:2, 7:8]).astype(np.uint32)
            ]
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = [
                Tensor(Dtype.U32, y[0].shape, y[0].flatten()),
                Tensor(Dtype.U32, y[1].shape, y[1].flatten()),
                Tensor(Dtype.U32, y[2].shape, y[2].flatten()),
                Tensor(Dtype.U32, y[3].shape, y[3].flatten()),
                Tensor(Dtype.U32, y[4].shape, y[4].flatten()),
                Tensor(Dtype.U32, y[5].shape, y[5].flatten()),
                Tensor(Dtype.U32, y[6].shape, y[6].flatten()),
                Tensor(Dtype.U32, y[7].shape, y[7].flatten()),
            ]

            name = "split_to_sequence_2d_scalar"
            make_test(
                [_x], _y, "input_0.split_to_sequence(1, 1, Option::None(()))", name)
        
        def split_to_sequence_2d_nokeepdims():
            x = np.random.randint(0, 255, (2, 8)).astype(np.uint32)
            y = [
                np.array(x[0:2, 0:1]).astype(np.uint32), 
                np.array(x[0:2, 1:2]).astype(np.uint32), 
                np.array(x[0:2, 2:3]).astype(np.uint32), 
                np.array(x[0:2, 3:4]).astype(np.uint32), 
                np.array(x[0:2, 4:5]).astype(np.uint32), 
                np.array(x[0:2, 5:6]).astype(np.uint32), 
                np.array(x[0:2, 6:7]).astype(np.uint32),
                np.array(x[0:2, 7:8]).astype(np.uint32)
            ]
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = [
                Tensor(Dtype.U32, y[0].shape, y[0].flatten()),
                Tensor(Dtype.U32, y[1].shape, y[1].flatten()),
                Tensor(Dtype.U32, y[2].shape, y[2].flatten()),
                Tensor(Dtype.U32, y[3].shape, y[3].flatten()),
                Tensor(Dtype.U32, y[4].shape, y[4].flatten()),
                Tensor(Dtype.U32, y[5].shape, y[5].flatten()),
                Tensor(Dtype.U32, y[6].shape, y[6].flatten()),
                Tensor(Dtype.U32, y[7].shape, y[7].flatten()),
            ]

            name = "split_to_sequence_2d_nokeepdims"
            make_test(
                [_x], _y, "input_0.split_to_sequence(1, 0, Option::None(()))", name)
            
        def split_to_sequence_1d_nokeepdims():
            x = np.random.randint(0, 255, 8).astype(np.uint32)
            y = [
                np.array(x[0:1]).astype(np.uint32), 
                np.array(x[1:2]).astype(np.uint32), 
                np.array(x[2:3]).astype(np.uint32), 
                np.array(x[3:4]).astype(np.uint32), 
                np.array(x[4:5]).astype(np.uint32), 
                np.array(x[5:6]).astype(np.uint32), 
                np.array(x[6:7]).astype(np.uint32),
                np.array(x[7:8]).astype(np.uint32)
            ]
            _x = Tensor(Dtype.U32, x.shape, x.flatten())
            _y = [
                Tensor(Dtype.U32, y[0].shape, y[0].flatten()),
                Tensor(Dtype.U32, y[1].shape, y[1].flatten()),
                Tensor(Dtype.U32, y[2].shape, y[2].flatten()),
                Tensor(Dtype.U32, y[3].shape, y[3].flatten()),
                Tensor(Dtype.U32, y[4].shape, y[4].flatten()),
                Tensor(Dtype.U32, y[5].shape, y[5].flatten()),
                Tensor(Dtype.U32, y[6].shape, y[6].flatten()),
                Tensor(Dtype.U32, y[7].shape, y[7].flatten()),
            ]

            name = "split_to_sequence_1d_nokeepdims"
            make_test(
                [_x], _y, "input_0.split_to_sequence(0, 0, Option::None(()))", name)

        
        split_to_sequence_1D()
        split_to_sequence_2D()
        split_to_sequence_zero_size()
        split_to_sequence_1d_uneven()
        split_to_sequence_2d_uneven()
        split_to_sequence_2d_scalar()
        split_to_sequence_1d_nokeepdims()
        split_to_sequence_2d_nokeepdims()

    @staticmethod
    def split_to_sequence_fp16x16():
        def split_to_sequence_1D():
            x = to_fp(np.random.randint(-127, 127, 6
                                        ).astype(np.int64), FixedImpl.FP16x16)
            y = [
                np.array(x[0:2]).astype(np.int64), 
                np.array(x[2:4]).astype(np.int64),
                np.array(x[4:6]).astype(np.int64),
            ]

            _x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            _y = [
                Tensor(Dtype.FP16x16, y[0].shape, y[0].flatten()),
                Tensor(Dtype.FP16x16, y[1].shape, y[1].flatten()),
                Tensor(Dtype.FP16x16, y[2].shape, y[2].flatten()),
            ]

            name = "split_to_sequence_fp16x16_1d_equal_parts"
            make_test(
                [_x], _y, "input_0.split_to_sequence(0, 1, Option::Some(TensorTrait::<u32>::new(shape: array![1].span(), data: array![3].span(),)))", name)
            y = [
                np.array(x[0:2]).astype(np.int64), 
                np.array(x[2:6]).astype(np.int64),
            ]
            _y = [
                Tensor(Dtype.FP16x16, y[0].shape, y[0].flatten()),
                Tensor(Dtype.FP16x16, y[1].shape, y[1].flatten()),
            ]
            name = "split_to_sequence_fp16x16_1d_variable_parts"
            make_test(
                [_x], _y, "input_0.split_to_sequence(0, 1, Option::Some(TensorTrait::<u32>::new(shape: array![2].span(), data: array![2, 4].span(),)))", name)
        def split_to_sequence_2D():
            x = to_fp(np.random.randint(-127, 127, (2, 6)
                                        ).astype(np.int64), FixedImpl.FP16x16)
            y = [
                np.array(x[0:2, 0:3]).astype(np.int64),
                np.array(x[0:2, 3:6]).astype(np.int64),
            ]
            _x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            _y = [
                Tensor(Dtype.FP16x16, y[0].shape, y[0].flatten()),
                Tensor(Dtype.FP16x16, y[1].shape, y[1].flatten()),
            ]
            name = "split_to_sequence_fp16x16_2d_equal_parts"
            make_test(
                [_x], _y, "input_0.split_to_sequence(1, 1, Option::Some(TensorTrait::<u32>::new(shape: array![1].span(), data: array![2].span(),)))", name)

            y = [
                np.array(x[0:2, 0:2]).astype(np.int64), 
                np.array(x[0:2, 2:6]).astype(np.int64)
            ]
            _y = [
                Tensor(Dtype.FP16x16, y[0].shape, y[0].flatten()),
                Tensor(Dtype.FP16x16, y[1].shape, y[1].flatten()),
            ]
            name = "split_to_sequence_fp16x16_2d_variable_parts"
            make_test(
                [_x], _y, "input_0.split_to_sequence(1, 1, Option::Some(TensorTrait::<u32>::new(shape: array![2].span(), data: array![2, 4].span(),)))", name)
        
        def split_to_sequence_zero_size():
            # 1-dimensional tensor with dimension_size=0
            x = to_fp(np.array([]).astype(np.int64
                                        ).astype(np.int64), FixedImpl.FP16x16)
            y = [
                np.array([]).astype(np.int64),
                np.array([]).astype(np.int64),
                np.array([]).astype(np.int64),
            ]
            _x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            _y = [
                Tensor(Dtype.FP16x16, y[0].shape, y[0].flatten()),
                Tensor(Dtype.FP16x16, y[1].shape, y[1].flatten()),
                Tensor(Dtype.FP16x16, y[2].shape, y[2].flatten()),
            ]
            # Split emtpy tensor to tensors of size zero
            name = "split_to_sequence_fp16x16_zero_size"
            make_test(
                [_x], _y, "input_0.split_to_sequence(0, 1, Option::Some(TensorTrait::<u32>::new(shape: array![3].span(), data: array![0, 0, 0].span(),)))", name)
            
        
        def split_to_sequence_1d_uneven():
            x = to_fp(np.random.randint(-127, 127, 7
                                        ).astype(np.int64), FixedImpl.FP16x16)
            y = [
                np.array(x[0:2]).astype(np.int64), 
                np.array(x[2:4]).astype(np.int64),
                np.array(x[4:6]).astype(np.int64),
                np.array(x[6:7]).astype(np.int64),
            ]

            _x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            _y = [
                Tensor(Dtype.FP16x16, y[0].shape, y[0].flatten()),
                Tensor(Dtype.FP16x16, y[1].shape, y[1].flatten()),
                Tensor(Dtype.FP16x16, y[2].shape, y[2].flatten()),
                Tensor(Dtype.FP16x16, y[3].shape, y[3].flatten()),
            ]

            name = "split_to_sequence_fp16x16_1d_uneven"
            make_test(
                [_x], _y, "input_0.split_to_sequence(0, 1, Option::Some(TensorTrait::<u32>::new(shape: array![1].span(), data: array![4].span())))", name)
            

        def split_to_sequence_2d_uneven():
            x = to_fp(np.random.randint(-127, 127, (2, 8)
                                        ).astype(np.int64), FixedImpl.FP16x16)
            y = [
                np.array(x[0:2, 0:3]).astype(np.int64), 
                np.array(x[0:2, 3:6]).astype(np.int64), 
                np.array(x[0:2, 6:8]).astype(np.int64)
            ]
            _x = Tensor(Dtype.FP16x16, x.shape, x.flatten())
            _y = [
                Tensor(Dtype.FP16x16, y[0].shape, y[0].flatten()),
                Tensor(Dtype.FP16x16, y[1].shape, y[1].flatten()),
                Tensor(Dtype.FP16x16, y[2].shape, y[2].flatten()),
            ]

            name = "split_to_sequence_fp16x16_2d_uneven"
            make_test(
                [_x], _y, "input_0.split_to_sequence(1, 1, Option::Some(TensorTrait::<u32>::new(shape: array![1].span(), data: array![3].span(),)))", name)

        
        split_to_sequence_1D()
        split_to_sequence_2D()
        split_to_sequence_zero_size()
        split_to_sequence_1d_uneven()
        split_to_sequence_2d_uneven()
    