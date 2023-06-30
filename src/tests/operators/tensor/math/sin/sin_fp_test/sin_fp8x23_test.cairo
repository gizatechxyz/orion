use core::clone::Clone;

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
    fn tensor_sin_test() {
        let tensor = fp_tensor_1x3_helper();
        let result = tensor.sin().data;
        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 7058771, 'result[1] = 0.8414...');
        assert((*result.at(2).mag).into() == 7627740, 'result[2] = 0.9092...');
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
    fn tensor_sin_test() {
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.sin().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 7058771, 'result[1] = 0.8414...');
        assert((*result.at(2).mag).into() == 7627740, 'result[2] = 0.9092...');
        assert((*result.at(3).mag).into() == 1183800, 'result[3] = 0.1411...');
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
    use orion::tests::helpers::tensor::fixed_point::fp8x23::fp_tensor_2x2x2_helper;
    use orion::numbers::fixed_point::implementations::impl_8x23::FP8x23Impl;

    #[test]
    #[available_gas(20000000)]
    fn tensor_sin_test() {
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.sin().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 7058771, 'result[1] = 0.8414...');
        assert((*result.at(2).mag).into() == 7627740, 'result[2] = 0.9092...');
        assert((*result.at(3).mag).into() == 1183800, 'result[3] = 0.1411...');
        assert((*result.at(4).mag).into() == 6348520, 'result[3] = -0.7568...');
        assert((*result.at(4).sign).into() == true, 'result[3].sign = true');
        assert((*result.at(5).mag).into() == 8044040, 'result[3] = -0.9589...');
        assert((*result.at(5).sign).into() == true, 'result[3].sign = true');
        assert((*result.at(6).mag).into() == 2343908, 'result[3] = -0.2794...');
        assert((*result.at(6).sign).into() == true, 'result[3].sign = true...');
        assert((*result.at(7).mag).into() == 5511203, 'result[3] = 0.6569...');
        assert((*result.at(7).sign).into() == false, 'result[3].sign = false');
    }
}
