// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use core::traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::i32::i32_tensor_1x3_neg_helper;

    #[test]
    #[available_gas(20000000)]
    fn tensor_abs() {
        let tensor = i32_tensor_1x3_neg_helper();
        let result = tensor.abs();
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use core::traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::i32::i32_tensor_2x2_neg_helper;

    #[test]
    #[available_gas(20000000)]
    fn tensor_abs() {
        let tensor = i32_tensor_2x2_neg_helper();
        let result = tensor.abs();
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 3, 'result[3] = 3');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use core::traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::i32::i32_tensor_2x2x2_neg_helper;

    #[test]
    #[available_gas(20000000)]
    fn tensor_abs() {
        let tensor = i32_tensor_2x2x2_neg_helper();
        let result = tensor.abs();
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 3, 'result[3] = 3');
        assert((*result.data[4]).into() == 4, 'result[4] = 4');
        assert((*result.data[5]).into() == 5, 'result[5] = 5');
        assert((*result.data[6]).into() == 6, 'result[6] = 6');
        assert((*result.data[7]).into() == 7, 'result[7] = 7');
    }
}
