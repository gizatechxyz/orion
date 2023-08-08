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
    fn tensor_sqrt_test() {
        let tensor = fp_tensor_1x3_helper();
        let result = tensor.sqrt().data;
        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 8388608, 'result[1] = 1');
        assert((*result.at(2).mag).into() == 11864550, 'result[2] = 1.4142');
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
    fn tensor_sqrt_test() {
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.sqrt().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 8388608, 'result[1] = 1');
        assert((*result.at(2).mag).into() == 11864550, 'result[2] = 1.4142');
        assert((*result.at(3).mag).into() == 14529439, 'result[3] = 1.7320');
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
    fn tensor_sqrt_test() {
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.sqrt().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 8388608, 'result[1] = 1');
        assert((*result.at(2).mag).into() == 11864550, 'result[2] = 1.4142');
        assert((*result.at(3).mag).into() == 14529439, 'result[3] = 1.7320');
        assert((*result.at(4).mag).into() == 16777216, 'result[4] = 2');
        assert((*result.at(5).mag).into() == 18758503, 'result[5] = 2.2360');
        assert((*result.at(6).mag).into() == 20548613, 'result[6] = 2.4494');
        assert((*result.at(7).mag).into() == 22193893, 'result[7] = 2.6457');
    }
}
