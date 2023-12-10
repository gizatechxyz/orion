// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use core::array::ArrayTrait;
    use core::array::SpanTrait;

    use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
    use orion::operators::tensor::core::{TensorTrait};
    use orion::test_helper::tensor::fixed_point::fp16x16::fp_tensor_1x3_helper;


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
    use core::array::ArrayTrait;
    use core::array::SpanTrait;

    use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
    use orion::operators::tensor::core::{TensorTrait};
    use orion::test_helper::tensor::fixed_point::fp16x16::fp_tensor_2x2_helper;


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
    use core::array::ArrayTrait;
    use core::array::SpanTrait;

    use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
    use orion::operators::tensor::core::{TensorTrait};
    use orion::test_helper::tensor::fixed_point::fp16x16::fp_tensor_2x2x2_helper;


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
