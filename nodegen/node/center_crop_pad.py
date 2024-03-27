import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl


class Center_crop_pad(RunAll):
    @staticmethod
    def export_center_crop_pad_crop():
        x = np.array(range(600), dtype=np.complex64).reshape((20,10,3))
        _x = Tensor(Dtype.COMPLEX64, x.shape, x.flatten())
        y = x[5:15, 1:8, :]
        _y = Tensor(Dtype.COMPLEX64, y.shape, y.flatten())

        name = "export_center_crop_pad_crop"
        make_test([_x], _y, "input_0.center_crop_pad(TensorTrait::new(array![3].span(), array![10,7,3].span())), Option::None(()))", name)

    @staticmethod
    def export_center_crop_pad_pad():
        x = np.array(range(210), dtype=np.complex64).reshape((10,7,3))
        _x = Tensor(Dtype.COMPLEX64, x.shape, x.flatten())
        y = np.zeros([20,10,3], dtype=np.complex64)
        y[5:15, 1:8, :] = x
        _y = Tensor(Dtype.COMPLEX64, y.shape, y.flatten())

        name = "export_center_crop_pad_pad"
        make_test([_x], _y, "input_0.center_crop_pad(TensorTrait::new(array![3].span(), array![20,10,3].span()), Option::None(()))", name)
    
    @staticmethod
    def export_center_crop_pad_crop_and_pad():
        # x = np.random.randn(20, 8, 3).astype(np.complex64)
        x = np.array(np.random.randn(20,8,3), dtype=np.complex64)
        _x = Tensor(Dtype.COMPLEX64, x.shape, x.flatten())
        y = np.zeros([10,10,3], dtype=np.complex64)
        y[:, 1:9, :] = x[5:15, :, :]
        _y = Tensor(Dtype.COMPLEX64, y.shape, y.flatten())

        name = "export_center_crop_pad_crop_and_pad"
        make_test([_x], _y, "input_0.center_crop_pad(TensorTrait::new(array![3].span(), array![10,10,3].span()), Option::None(()))", name)

    @staticmethod
    def export_center_crop_pad_crop_axes_hwc():
        x = np.array(np.random.randn(20,8,3), dtype=np.complex64)
        _x = Tensor(Dtype.COMPLEX64, x.shape, x.flatten())
        y = np.zeros([10,9,3], dtype=np.complex64)
        y[:, :8, :] = x[5:15, :, :]
        _y = Tensor(Dtype.COMPLEX64, y.shape, y.flatten())

        name = "export_center_crop_pad_crop_axes_hwc"
        make_test([_x], _y, "input_0.center_crop_pad(TensorTrait::new(array![2].span(), array![10,9].span()), Option::Some(array![0,1]))", name)
    
    @staticmethod
    def export_center_crop_pad_crop_negative_axes_hwc():
        x = np.array(np.random.randn(20,8,3), dtype=np.complex64)
        _x = Tensor(Dtype.COMPLEX64, x.shape, x.flatten())
        y = np.zeros([10,9,3], dtype=np.complex64)
        y[:, :8, :] = x[5:15, :, :]
        _y = Tensor(Dtype.COMPLEX64, y.shape, y.flatten())
        name = "export_center_crop_pad_crop_negative_axes_hwc"
        make_test([_x], _y, "input_0.center_crop_pad(TensorTrait::new(array![2].span(), array![10,9].span()), Option::Some(array![-3,-2]))", name)

    @staticmethod
    def export_center_crop_pad_crop_axes_chw():
        x = np.array(np.random.randn(3,20,8), dtype=np.complex64)
        _x = Tensor(Dtype.COMPLEX64, x.shape, x.flatten())
        y = np.zeros([3,10,9], dtype=np.complex64)
        y[:, :, :8] = x[:, 5:15, :]
        _y = Tensor(Dtype.COMPLEX64, y.shape, y.flatten())
        name = "export_center_crop_pad_crop_axes_chw"
        make_test([_x], _y, "input_0.center_crop_pad(TensorTrait::new(array![2].span(), array![10,9].span()), Option::Some(array![1,2]))", name)