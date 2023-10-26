// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::{ArrayTrait, SpanTrait};
    use core::traits::Into;

    use orion::operators::tensor::I32Tensor;
    use orion::operators::tensor::core::{TensorTrait};
    use orion::helpers::tensor::i32::i32_tensor_1x3_helper;

    #[test]
    #[available_gas(200000)]
    fn axis_0() {
        let tensor = i32_tensor_1x3_helper();
        let result = tensor.flatten(0);
        assert((*result.shape[0]).into() == 1, 'result[0] = 1');
        assert((*result.shape[1]).into() == 3, 'result[1] = 3');
    }
}


// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::{ArrayTrait, SpanTrait};
    use core::traits::Into;

    use orion::operators::tensor::I32Tensor;
    use orion::operators::tensor::core::{TensorTrait};
    use orion::helpers::tensor::i32::i32_tensor_2x2_helper;

    #[test]
    #[available_gas(200000)]
    fn axis_0() {
        let tensor = i32_tensor_2x2_helper();
        let result = tensor.flatten(0);
        assert((*result.shape[0]).into() == 1, 'result[0] = 1');
        assert((*result.shape[1]).into() == 4, 'result[1] = 4');
    }

    #[test]
    #[available_gas(200000)]
    fn axis_1() {
        let tensor = i32_tensor_2x2_helper();
        let result = tensor.flatten(1);
        assert((*result.shape[0]).into() == 2, 'result[0] = 2');
        assert((*result.shape[1]).into() == 2, 'result[1] = 2');
    }
}


// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::{ArrayTrait, SpanTrait};
    use core::traits::Into;

    use orion::operators::tensor::I32Tensor;
    use orion::operators::tensor::core::{TensorTrait};
    use orion::helpers::tensor::i32::i32_tensor_2x2x2_helper;


    #[test]
    #[available_gas(200000)]
    fn axis_0() {
        let tensor = i32_tensor_2x2x2_helper();
        let result = tensor.flatten(0);
        assert((*result.shape[0]).into() == 1, 'result[0] = 1');
        assert((*result.shape[1]).into() == 8, 'result[1] = 8');
    }

    #[test]
    #[available_gas(200000)]
    fn axis_1() {
        let tensor = i32_tensor_2x2x2_helper();
        let result = tensor.flatten(1);
        assert((*result.shape[0]).into() == 2, 'result[0] = 2');
        assert((*result.shape[1]).into() == 4, 'result[1] = 4');
    }

    #[test]
    #[available_gas(200000)]
    fn axis_2() {
        let tensor = i32_tensor_2x2x2_helper();
        let result = tensor.flatten(2);
        assert((*result.shape[0]).into() == 4, 'result[0] = 4');
        assert((*result.shape[1]).into() == 2, 'result[1] = 2');
    }
}
