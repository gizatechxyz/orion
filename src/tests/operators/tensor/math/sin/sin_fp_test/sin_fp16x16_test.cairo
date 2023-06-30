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
    fn tensor_sin_test() {
        let tensor = fp_tensor_1x3_helper();
        let result = tensor.sin().data;
        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 55147, 'result[1] = 0.8414...');
        assert((*result.at(2).mag).into() == 59592, 'result[2] = 0.909...');
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::SpanTrait;
    use traits::Into;
    use debug::PrintTrait;


    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, };
    use orion::tests::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2_helper;
    use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;

    #[test]
    #[available_gas(20000000)]
    fn tensor_sin_test() {
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.sin().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 55147, 'result[1] = 0.8414...');
        assert((*result.at(2).mag).into() == 59592, 'result[2] = 0.909...');
        assert((*result.at(3).mag).into() == 9246, 'result[3] = 0.141...');
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
    fn tensor_sin_test() {
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.sin().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 55147, 'result[1] = 0.8414...');
        assert((*result.at(2).mag).into() == 59592, 'result[2] = 0.909...');
        assert((*result.at(3).mag).into() == 9246, 'result[3] = 0.141...');
        assert((*result.at(4).mag).into() == 49598, 'result[3] = -0.7568...');
        assert((*result.at(4).sign).into() == true, 'result[3].sign = true');
        assert((*result.at(5).mag).into() == 62844, 'result[3] = -0.9589...');
        assert((*result.at(5).sign).into() == true, 'result[3].sign = true');
        assert((*result.at(6).mag).into() == 18310, 'result[3] = -0.2794...');
        assert((*result.at(6).sign).into() == true, 'result[3].sign = true...');
        assert((*result.at(7).mag).into() == 43056, 'result[3] = 0.6569...');
        assert((*result.at(7).sign).into() == false, 'result[3].sign = false');
    }
}
