// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::ArrayTrait;
    use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
    use orion::operators::tensor::core::{TensorTrait};
    use orion::helpers::tensor::fixed_point::fp16x16::fp_tensor_1x3_helper;
    use orion::numbers::fixed_point::core::FixedTrait;
    use orion::numbers::fixed_point::implementations::fp16x16::core::{
        FP16x16Impl, FP16x16PartialEq
    };

    #[test]
    #[available_gas(2000000)]
    fn tensor_at() {
        let tensor = fp_tensor_1x3_helper();
        let mut indices = ArrayTrait::new();
        indices.append(1);

        let result = tensor.at(indices.span());

        assert(result == FixedTrait::new_unscaled(1, false), 'result[2] = 1');
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::ArrayTrait;
    use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
    use orion::operators::tensor::core::{TensorTrait};
    use orion::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2_helper;
    use orion::numbers::fixed_point::core::FixedTrait;
    use orion::numbers::fixed_point::implementations::fp16x16::core::{
        FP16x16Impl, FP16x16PartialEq
    };


    #[test]
    #[available_gas(2000000)]
    fn tensor_at() {
        let tensor = fp_tensor_2x2_helper();

        let mut indices = ArrayTrait::new();
        indices.append(1);
        indices.append(1);

        let result = tensor.at(indices.span());

        assert(result == FixedTrait::new_unscaled(3, false), 'result[4] = 3');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::ArrayTrait;
    use orion::operators::tensor::implementations::tensor_fp16x16::FP16x16Tensor;
    use orion::operators::tensor::core::{TensorTrait};
    use orion::helpers::tensor::fixed_point::fp16x16::fp_tensor_2x2x2_helper;
    use orion::numbers::fixed_point::core::FixedTrait;
    use orion::numbers::fixed_point::implementations::fp16x16::core::{
        FP16x16Impl, FP16x16PartialEq
    };

    #[test]
    #[available_gas(2000000)]
    fn tensor_at() {
        let tensor = fp_tensor_2x2x2_helper();

        let mut indices = ArrayTrait::new();
        indices.append(0);
        indices.append(1);
        indices.append(1);

        let result = tensor.at(indices.span());

        assert(result == FixedTrait::new_unscaled(3, false), 'result[3] = 3');
    }
}
