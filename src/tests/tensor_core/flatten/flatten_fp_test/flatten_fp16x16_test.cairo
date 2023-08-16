// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use core::traits::Into;
    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_1x3_helper;
    use orion::operators::tensor::core::TensorTrait;

    #[test]
    #[available_gas(200000)]
    fn axis_0() {
        let tensor = fp_tensor_1x3_helper();
        let result = tensor.flatten(0);
        assert((*result.shape[0]).into() == 1, 'result[0] = 1');
        assert((*result.shape[1]).into() == 3, 'result[1] = 3');
    }
}


// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use core::traits::Into;
    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2_helper;
    use orion::operators::tensor::core::TensorTrait;

    #[test]
    #[available_gas(200000)]
    fn axis_0() {
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.flatten(0);
        assert((*result.shape[0]).into() == 1, 'result[0] = 1');
        assert((*result.shape[1]).into() == 4, 'result[1] = 4');
    }

    #[test]
    #[available_gas(200000)]
    fn axis_1() {
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.flatten(1);
        assert((*result.shape[0]).into() == 2, 'result[0] = 2');
        assert((*result.shape[1]).into() == 2, 'result[1] = 2');
    }
}


// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use core::traits::Into;
    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2x2_helper;
    use orion::operators::tensor::core::TensorTrait;

    #[test]
    #[available_gas(200000)]
    fn axis_0() {
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.flatten(0);
        assert((*result.shape[0]).into() == 1, 'result[0] = 1');
        assert((*result.shape[1]).into() == 8, 'result[1] = 8');
    }

    #[test]
    #[available_gas(200000)]
    fn axis_1() {
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.flatten(1);
        assert((*result.shape[0]).into() == 2, 'result[0] = 2');
        assert((*result.shape[1]).into() == 4, 'result[1] = 4');
    }

    #[test]
    #[available_gas(200000)]
    fn axis_2() {
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.flatten(2);
        assert((*result.shape[0]).into() == 4, 'result[0] = 4');
        assert((*result.shape[1]).into() == 2, 'result[1] = 2');
    }
}
