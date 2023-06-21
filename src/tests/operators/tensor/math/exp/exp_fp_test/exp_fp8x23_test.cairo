// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::SpanTrait;
    use traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, };
    use orion::tests::helpers::tensor::fixed_point::fp8x23::fp_tensor_1x3_helper;
    use orion::numbers::fixed_point::implementations::impl_8x23::FP8x23Impl;

    #[test]
    #[available_gas(20000000)]
    fn tensor_exp_test() {
        let tensor = fp_tensor_1x3_helper();
        let result = tensor.exp().data;

        assert((*result.at(0).mag).into() == 8388608, 'result[0] = 1');
        assert((*result.at(1).mag).into() == 22802598, 'result[1] = 2.7182...');
        assert((*result.at(2).mag).into() == 61983880, 'result[2] = 7.389...');
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::SpanTrait;
    use traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, };
    use orion::tests::helpers::tensor::fixed_point::fp8x23::fp_tensor_2x2_helper;
    use orion::numbers::fixed_point::implementations::impl_8x23::FP8x23Impl;

    #[test]
    #[available_gas(20000000)]
    fn tensor_exp_test() {
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.exp().data;

        assert((*result.at(0).mag).into() == 8388608, 'result[0] = 1');
        assert((*result.at(1).mag).into() == 22802598, 'result[1] = 2.7182...');
        assert((*result.at(2).mag).into() == 61983880, 'result[2] = 7.389...');
        assert((*result.at(3).mag).into() == 168489680, 'result[3] = 20.085...');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::SpanTrait;
    use traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, };
    use orion::tests::helpers::tensor::fixed_point::fp8x23::fp_tensor_2x2x2_helper;
    use orion::numbers::fixed_point::implementations::impl_8x23::FP8x23Impl;

    #[test]
    #[available_gas(20000000)]
    fn tensor_exp_test() {
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.exp().data;

        assert((*result.at(0).mag).into() == 8388608, 'result[0] = 1');
        assert((*result.at(1).mag).into() == 22802598, 'result[1] = 2.7182...');
        assert((*result.at(2).mag).into() == 61983880, 'result[2] = 7.389...');
        assert((*result.at(3).mag).into() == 168489680, 'result[3] = 20.085...');
        assert((*result.at(4).mag).into() == 458002400, 'result[3] = 54.5981...');
        assert((*result.at(5).mag).into() == 1244979584, 'result[3] = 148.4131...');
        assert((*result.at(6).mag).into() == 3384205312, 'result[3] = 403.4287...');
        assert((*result.at(7).mag).into() == 9199223808, 'result[3] = 1096.6331...');
    }
}

