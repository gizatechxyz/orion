import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


class Split(RunAll):
    @staticmethod
    def split_u32():
        def split_1D():
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

            name = "split_u32_1d_equal_parts"
            make_test(
                [_x], _y, "input_0.split(0, Option::Some(3), Option::None(()))", name)
            y = [
                np.array(x[0:2]).astype(np.uint32), 
                np.array(x[2:6]).astype(np.uint32),
            ]
            _y = [
                Tensor(Dtype.U32, y[0].shape, y[0].flatten()),
                Tensor(Dtype.U32, y[1].shape, y[1].flatten()),
            ]
            name = "split_u32_1d_variable_parts"
            make_test(
                [_x], _y, "input_0.split(0, Option::None(()), Option::Some(TensorTrait::<u32>::new(shape: array![2].span(), data: array![2, 4].span(),)))", name)
        def split_2D():
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
            name = "split_u32_2d_equal_parts"
            make_test(
                [_x], _y, "input_0.split(1, Option::Some(2), Option::None(()))", name)

            y = [
                np.array(x[0:2, 0:2]).astype(np.uint32), 
                np.array(x[0:2, 2:6]).astype(np.uint32)
            ]
            _y = [
                Tensor(Dtype.U32, y[0].shape, y[0].flatten()),
                Tensor(Dtype.U32, y[1].shape, y[1].flatten()),
            ]
            name = "split_u32_2d_variable_parts"
            make_test(
                [_x], _y, "input_0.split(1, Option::None(()), Option::Some(TensorTrait::<u32>::new(shape: array![2].span(), data: array![2, 4].span(),)))", name)
        
        def split_zero_size():
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
            name = "split_u32_zero_size"
            make_test(
                [_x], _y, "input_0.split(0, Option::None(()), Option::Some(TensorTrait::<u32>::new(shape: array![3].span(), data: array![0, 0, 0].span(),)))", name)
            
        
        def split_1d_uneven():
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

            name = "split_u32_1d_uneven"
            make_test(
                [_x], _y, "input_0.split(0, Option::Some(4), Option::None(()))", name)
            

        def split_2d_uneven():
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

            name = "split_u32_2d_uneven"
            make_test(
                [_x], _y, "input_0.split(1, Option::Some(3), Option::None(()))", name)

        
        split_1D()
        split_2D()
        split_zero_size()
        split_1d_uneven()
        split_2d_uneven()

    @staticmethod
    def split_fp16x16():
        def split_1D():
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

            name = "split_fp16x16_1d_equal_parts"
            make_test(
                [_x], _y, "input_0.split(0, Option::Some(3), Option::None(()))", name)
            y = [
                np.array(x[0:2]).astype(np.int64), 
                np.array(x[2:6]).astype(np.int64),
            ]
            _y = [
                Tensor(Dtype.FP16x16, y[0].shape, y[0].flatten()),
                Tensor(Dtype.FP16x16, y[1].shape, y[1].flatten()),
            ]
            name = "split_fp16x16_1d_variable_parts"
            make_test(
                [_x], _y, "input_0.split(0, Option::None(()), Option::Some(TensorTrait::<u32>::new(shape: array![2].span(), data: array![2, 4].span(),)))", name)
        def split_2D():
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
            name = "split_fp16x16_2d_equal_parts"
            make_test(
                [_x], _y, "input_0.split(1, Option::Some(2), Option::None(()))", name)

            y = [
                np.array(x[0:2, 0:2]).astype(np.int64), 
                np.array(x[0:2, 2:6]).astype(np.int64)
            ]
            _y = [
                Tensor(Dtype.FP16x16, y[0].shape, y[0].flatten()),
                Tensor(Dtype.FP16x16, y[1].shape, y[1].flatten()),
            ]
            name = "split_fp16x16_2d_variable_parts"
            make_test(
                [_x], _y, "input_0.split(1, Option::None(()), Option::Some(TensorTrait::<u32>::new(shape: array![2].span(), data: array![2, 4].span(),)))", name)
        
        def split_zero_size():
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
            name = "split_fp16x16_zero_size"
            make_test(
                [_x], _y, "input_0.split(0, Option::None(()), Option::Some(TensorTrait::<u32>::new(shape: array![3].span(), data: array![0, 0, 0].span(),)))", name)
            
        
        def split_1d_uneven():
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

            name = "split_fp16x16_1d_uneven"
            make_test(
                [_x], _y, "input_0.split(0, Option::Some(4), Option::None(()))", name)
            

        def split_2d_uneven():
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

            name = "split_fp16x16_2d_uneven"
            make_test(
                [_x], _y, "input_0.split(1, Option::Some(3), Option::None(()))", name)

        
        split_1D()
        split_2D()
        split_zero_size()
        split_1d_uneven()
        split_2d_uneven()
    