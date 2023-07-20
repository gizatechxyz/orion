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
    use debug::PrintTrait;

    #[test]
    #[available_gas(20000000)]
    fn tensor_atan_test() {
        let tensor = fp_tensor_1x3_helper();
        let result = tensor.atan().data;
        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 6588397, 'result[1] = 0.7853');
        assert((*result.at(2).mag).into() == 9287436, 'result[2] = 1.1071');
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
    fn tensor_atan_test() {
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.atan().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 6588397, 'result[1] = 0.7853');
        assert((*result.at(2).mag).into() == 9287436, 'result[2] = 1.1071');
        assert((*result.at(3).mag).into() == 10477755, 'result[3] = 1.2490');
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
    fn tensor_atan_test() {
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.atan().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 6588397, 'result[1] = 0.7853');
        assert((*result.at(2).mag).into() == 9287436, 'result[2] = 1.1071');
        assert((*result.at(3).mag).into() == 10477755, 'result[3] = 1.2490');
        assert((*result.at(4).mag).into() == 11121764, 'result[4] = 1.3258');
        assert((*result.at(5).mag).into() == 11520921, 'result[5] = 1.3734');
        assert((*result.at(6).mag).into() == 11791427, 'result[6] = 1.4056');
        assert((*result.at(7).mag).into() == 11986476, 'result[7] = 1.4288');
    }
}
