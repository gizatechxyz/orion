// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::ArrayTrait;
    use orion::operators::tensor::I32Tensor;
    use orion::operators::tensor::core::{TensorTrait};
    use orion::test_helper::tensor::i32::i32_tensor_1x3_helper;


    #[test]
    #[available_gas(2000000)]
    fn tensor_at() {
        let tensor = i32_tensor_1x3_helper();
        let mut indices = ArrayTrait::new();
        indices.append(1);

        let result = tensor.at(indices.span()).mag;

        assert(result == 1, 'result[2] = 1');
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::ArrayTrait;
    use orion::operators::tensor::I32Tensor;
    use orion::operators::tensor::core::{TensorTrait};
    use orion::test_helper::tensor::i32::i32_tensor_2x2_helper;


    #[test]
    #[available_gas(2000000)]
    fn tensor_at() {
        let tensor = i32_tensor_2x2_helper();

        let mut indices = ArrayTrait::new();
        indices.append(1);
        indices.append(1);

        let result = tensor.at(indices.span()).mag;

        assert(result == 3, 'result[4] = 3');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::ArrayTrait;
    use orion::operators::tensor::I32Tensor;
    use orion::operators::tensor::core::{TensorTrait};
    use orion::test_helper::tensor::i32::i32_tensor_2x2x2_helper;


    #[test]
    #[available_gas(2000000)]
    fn tensor_at() {
        let tensor = i32_tensor_2x2x2_helper();

        let mut indices = ArrayTrait::new();
        indices.append(0);
        indices.append(1);
        indices.append(1);

        let result = tensor.at(indices.span()).mag;

        assert(result == 3, 'result[3] = 3');
    }
}
