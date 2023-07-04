// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::SpanTrait;
    use traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, };
    use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_1x3_helper;
    use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;

    #[test]
    #[available_gas(20000000)]
    fn tensor_tanh_test() {
        let tensor = fp_tensor_1x3_helper();
        let result = tensor.tanh().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 49911, 'result[1] = 0.761594...');
        assert((*result.at(2).mag).into() == 63178, 'result[2] = 0.964028...');
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::SpanTrait;
    use traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, };
    use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2_helper;
    use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;

    #[test]
    #[available_gas(20000000)]
    fn tensor_tanh_test() {
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.tanh().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 49911, 'result[1] = 0.761594...');
        assert((*result.at(2).mag).into() == 63178, 'result[2] = 0.964028...');
        assert((*result.at(3).mag).into() == 65211, 'result[3] = 0.99505...');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::SpanTrait;
    use traits::Into;
    use debug::PrintTrait;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, };
    use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2x2_helper;
    use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;

    #[test]
    #[available_gas(20000000)]
    fn tensor_tanh_test() {
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.tanh().data;
        
        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 49911, 'result[1] = 0.761594...');
        assert((*result.at(2).mag).into() == 63178, 'result[2] = 0.964028...');
        assert((*result.at(3).mag).into() == 65211, 'result[3] = 0.99505...');
        assert((*result.at(4).mag).into() == 65492, 'result[4] = 0.99933...');
        assert((*result.at(5).mag).into() == 65530, 'result[5] = 0.99991...');
        assert((*result.at(6).mag).into() == 65535, 'result[6] = 0.99999...');
        assert((*result.at(7).mag).into() == 65535, 'result[7] = 0.99999...');
    }
}

