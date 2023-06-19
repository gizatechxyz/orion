#[cfg(test)]
mod tensor_1D {
    use array::SpanTrait;
    use array::ArrayTrait;
    use orion::operators::tensor::implementations::impl_tensor_u32;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::u32::u32_tensor_1x3_helper;

    #[test]
    #[available_gas(20000000)]
    fn abs_1D() {
        let tensor = u32_tensor_1x3_helper();
        let result = tensor.abs();
        assert(*result.data[0] == 0, 'result[0] = 0');
        assert(*result.data[1] == 1, 'result[1] = 1');
        assert(*result.data[2] == 2, 'result[2] = 2');
    }
}

#[cfg(test)]
mod tensor_2D {
    use array::SpanTrait;
    use array::ArrayTrait;
    use orion::operators::tensor::implementations::impl_tensor_u32;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::u32::u32_tensor_2x2_helper;

    #[test]
    #[available_gas(20000000)]
    fn abs_2D() {
        let tensor = u32_tensor_2x2_helper();
        let result = tensor.abs();
        assert(*result.data[0] == 0, 'result[0] = 0');
        assert(*result.data[1] == 1, 'result[1] = 1');
        assert(*result.data[2] == 2, 'result[2] = 2');
        assert(*result.data[3] == 3, 'result[3] = 3');
    }
}

#[cfg(test)]
mod tensor_3D {
    use array::SpanTrait;
    use array::ArrayTrait;
    use orion::operators::tensor::implementations::impl_tensor_u32;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::u32::u32_tensor_2x2x2_helper;

    #[test]
    #[available_gas(20000000)]
    fn abs_3D() {
        let tensor = u32_tensor_2x2x2_helper();
        let result = tensor.abs();
        assert(*result.data[0] == 0, 'result[0] = 0');
        assert(*result.data[1] == 1, 'result[1] = 1');
        assert(*result.data[2] == 2, 'result[2] = 2');
        assert(*result.data[3] == 3, 'result[3] = 3');
        assert(*result.data[4] == 4, 'result[4] = 4');
        assert(*result.data[5] == 5, 'result[5] = 5');
        assert(*result.data[6] == 6, 'result[6] = 6');
        assert(*result.data[7] == 7, 'result[7] = 7');
    }
}

