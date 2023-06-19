#[cfg(test)]
mod tensor_1D {
    use array::SpanTrait;
    use array::ArrayTrait;
    use core::traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_i32;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::i32::i32_tensor_1x3_neg_helper;

    #[test]
    #[available_gas(2000000000)]
    fn abs_1D() {
        let tensor = i32_tensor_1x3_neg_helper();
        let result = tensor.abs();
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
    }
}

#[cfg(test)]
mod tensor_2D {
    use array::SpanTrait;
    use array::ArrayTrait;
    use core::traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_i32;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::i32::i32_tensor_2x2_neg_helper;

    #[test]
    #[available_gas(2000000000)]
    fn abs_2D() {
        let tensor = i32_tensor_2x2_neg_helper();
        let result = tensor.abs();
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[1]).into() == 1, 'result[1] = 1');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 3, 'result[3] = 3');
    }
}

#[cfg(test)]
mod tensor_3D {
    use array::SpanTrait;
    use array::ArrayTrait;
    use core::traits::Into;
    use core::option::OptionTrait;

    use orion::operators::tensor::implementations::impl_tensor_i32;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::i32::i32_tensor_2x2x2_neg_helper;

    #[test]
    #[available_gas(2000000000)]
    fn abs_3D() {
        let tensor = i32_tensor_2x2x2_neg_helper();
        let result = tensor.abs();
        
        assert((*result.data[0]).into() == 0, 'result[0] = 0');
        assert((*result.data[2]).into() == 2, 'result[2] = 2');
        assert((*result.data[3]).into() == 3, 'result[3] = 3');
        assert((*result.data[7]).into() == 7, 'result[7] = 7');
    }
}

