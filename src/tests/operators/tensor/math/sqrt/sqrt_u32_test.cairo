// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::SpanTrait;
    use traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
    use orion::operators::tensor::core::{TensorTrait, };
    use orion::tests::helpers::tensor::u32::u32_tensor_1x3_helper;
    use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;

    #[test]
    #[available_gas(20000000)]
    fn tensor_sqrt_test() {
        let tensor = u32_tensor_1x3_helper();
        let result = tensor.sqrt().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 65536, 'result[1] = 1');
        assert((*result.at(2).mag).into() == 92672, 'result[2] = 1.4142');
    }
}
// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::SpanTrait;
    use traits::Into;
    use debug::PrintTrait;

    use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
    use orion::operators::tensor::core::{TensorTrait, };
    use orion::tests::helpers::tensor::u32::u32_tensor_2x2_helper;
    use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;

    #[test]
    #[available_gas(20000000)]
    fn tensor_sqrt_test() {
        let tensor = u32_tensor_2x2_helper();
        let result = tensor.sqrt().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 65536, 'result[1] = 1');
        assert((*result.at(2).mag).into() == 92672, 'result[2] = 1.4142');
        assert((*result.at(3).mag).into() == 113408, 'result[3] = 1.7320');
    }
}
// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::SpanTrait;
    use traits::Into;
    use debug::PrintTrait;

    use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
    use orion::operators::tensor::core::{TensorTrait, };
    use orion::tests::helpers::tensor::u32::u32_tensor_2x2x2_helper;
    use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;

    #[test]
    #[available_gas(20000000)]
    fn tensor_sqrt_test() {
        let tensor = u32_tensor_2x2x2_helper();
        let result = tensor.sqrt().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 65536, 'result[1] = 1');
        assert((*result.at(2).mag).into() == 92672, 'result[2] = 1.4142');
        assert((*result.at(3).mag).into() == 113408, 'result[3] = 1.7320');
        assert((*result.at(4).mag).into() == 131072, 'result[4] = 2');
        assert((*result.at(5).mag).into() == 146432, 'result[5] = 2.2360');
        assert((*result.at(6).mag).into() == 160512, 'result[6] = 2.4494');
        assert((*result.at(7).mag).into() == 173312, 'result[7] = 2.6457');
    }
}
