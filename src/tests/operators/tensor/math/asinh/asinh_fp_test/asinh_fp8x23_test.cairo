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
    fn tensor_asinh_test() {
        let tensor = fp_tensor_1x3_helper();
        let result = tensor.asinh().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 7394022, 'result[1] = 0.881374...');
        assert((*result.at(2).mag).into() == 12110330, 'result[2] = 1.44364...');
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
    fn tensor_asinh_test() {
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.asinh().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 7394022, 'result[1] = 0.881374...');
        assert((*result.at(2).mag).into() == 12110330, 'result[2] = 1.44364...');
        assert((*result.at(3).mag).into() == 15254269, 'result[3] = 1.81845...');
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
    fn tensor_asinh_test() {
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.asinh().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 7394022, 'result[1] = 0.881374...');
        assert((*result.at(2).mag).into() == 12110330, 'result[2] = 1.44364...');
        assert((*result.at(3).mag).into() == 15254269, 'result[3] = 1.81845...');
        assert((*result.at(4).mag).into() == 17571895, 'result[4] = 2.09471');
        assert((*result.at(5).mag).into() == 19398496, 'result[5] = 2.31244...');
        assert((*result.at(6).mag).into() == 20902886, 'result[6] = 2.49178...');
        assert((*result.at(7).mag).into() == 22180969, 'result[7] = 2.64412...');
    }
}

