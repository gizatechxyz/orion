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
    fn tensor_sinh_test() {
        let tensor = fp_tensor_1x3_helper();
        let result = tensor.sinh().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 9858301, 'result[1] = 1.175201...');
        assert((*result.at(2).mag).into() == 30424303, 'result[2] = 3.62686...');
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
    fn tensor_sinh_test() {
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.sinh().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 9858301, 'result[1] = 1.175201...');
        assert((*result.at(2).mag).into() == 30424303, 'result[2] = 3.62686...');
        assert((*result.at(3).mag).into() == 84036018, 'result[3] = 10.0179...');
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
    fn tensor_sinh_test() {
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.sinh().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 9858301, 'result[1] = 1.175201...');
        assert((*result.at(2).mag).into() == 30424303, 'result[2] = 3.62686...');
        assert((*result.at(3).mag).into() == 84036018, 'result[3] = 10.0179...');
        assert((*result.at(4).mag).into() == 228924379, 'result[4] = 27.2899...');
        assert((*result.at(5).mag).into() == 622461531, 'result[5] = 74.2032...');
        assert((*result.at(6).mag).into() == 1692092259, 'result[6] = 201.7132...');
        assert((*result.at(7).mag).into() == 4599608079, 'result[7] = 548.3161...');
    }
}

