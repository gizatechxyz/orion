// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::SpanTrait;
    use traits::Into;
    use debug::PrintTrait;

    use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
    use orion::operators::tensor::core::{TensorTrait, };
    use orion::tests::helpers::tensor::i32::i32_tensor_1x3_helper;
    use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;

    #[test]
    #[available_gas(20000000)]
    fn tensor_cos_test() {
        let tensor = i32_tensor_1x3_helper();
        let result = tensor.cos().data;

        assert((*result.at(0).mag).into() == 65536, 'result[0] = 1');
        assert((*result.at(1).mag).into() == 35409, 'result[1] = 0.5403...');
        assert((*result.at(2).mag).into() == 27273, 'result[2] = -0.4161...');
    }
}
// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    use array::SpanTrait;
    use traits::Into;
    use debug::PrintTrait;

    use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
    use orion::operators::tensor::core::{TensorTrait, };
    use orion::tests::helpers::tensor::i32::i32_tensor_2x2_helper;
    use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;

    #[test]
    #[available_gas(20000000)]
    fn tensor_cos_test() {
        let tensor = i32_tensor_2x2_helper();
        let result = tensor.cos().data;

        assert((*result.at(0).mag).into() == 65536, 'result[0] = 1');
        assert((*result.at(1).mag).into() == 35409, 'result[1] = 0.5403...');
        assert((*result.at(2).mag).into() == 27273, 'result[2] = -0.4161...');
        assert((*result.at(3).mag).into() == 64880, 'result[3] = -0.9899...');
    }
}
// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::SpanTrait;
    use traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
    use orion::operators::tensor::core::{TensorTrait, };
    use orion::tests::helpers::tensor::i32::i32_tensor_2x2x2_helper;
    use orion::numbers::fixed_point::implementations::impl_16x16::FP16x16Impl;

    #[test]
    #[available_gas(20000000)]
    fn tensor_cos_test() {
        let tensor = i32_tensor_2x2x2_helper();
        let result = tensor.cos().data;

        assert((*result.at(0).mag).into() == 65536, 'result[0] = 1');
        assert((*result.at(1).mag).into() == 35409, 'result[1] = 0.5403...');
        assert((*result.at(2).mag).into() == 27273, 'result[2] = -0.4161...');
        assert((*result.at(3).mag).into() == 64880, 'result[3] = -0.9899...');
        assert((*result.at(3).sign).into() == true, 'result[3].sign = true');
        assert((*result.at(4).mag).into() == 42839, 'result[4] = -0.6536...');
        assert((*result.at(4).sign).into() == true, 'result[4].sign = true');
        assert((*result.at(5).mag).into() == 18591, 'result[5] = 02836...');
        assert((*result.at(6).mag).into() == 62926, 'result[6] = 0.9601...');
        assert((*result.at(7).mag).into() == 49405, 'result[7] = 0.7539...');
    }
}

