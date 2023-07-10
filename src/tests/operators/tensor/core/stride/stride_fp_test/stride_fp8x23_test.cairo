// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::ArrayTrait;
    use array::SpanTrait;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::fixed_point::fp8x23::fp_tensor_1x3_helper;


    #[test]
    #[available_gas(2000000)]
    fn tensor_stride() {
        let tensor = fp_tensor_1x3_helper();
        let result = tensor.stride();
        assert(*result[0] == 1, 'stride x = 1');
        assert(result.len() == 1, 'len = 1');
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::ArrayTrait;
    use array::SpanTrait;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::fixed_point::fp8x23::fp_tensor_2x2_helper;


    #[test]
    #[available_gas(2000000)]
    fn tensor_stride() {
        let tensor = fp_tensor_2x2_helper();
        let result = tensor.stride();
        assert(*result[0] == 2, 'stride x = 2');
        assert(*result[1] == 1, 'stride y = 1');
        assert(result.len() == 2, 'len = 2');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::ArrayTrait;
    use array::SpanTrait;

    use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::fixed_point::fp8x23::fp_tensor_2x2x2_helper;


    #[test]
    #[available_gas(2000000)]
    fn tensor_stride() {
        let tensor = fp_tensor_2x2x2_helper();
        let result = tensor.stride();
        assert(*result[0] == 4, 'stride x = 4');
        assert(*result[1] == 2, 'stride y = 2');
        assert(*result[2] == 1, 'stride z = 1');
        assert(result.len() == 3, 'len = 3');
    }
}
