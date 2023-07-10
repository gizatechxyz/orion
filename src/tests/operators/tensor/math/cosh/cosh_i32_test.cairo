// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::SpanTrait;
    use traits::Into;
    use array::ArrayTrait;

    use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
    use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
    use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};

    #[test]
    #[available_gas(20000000)]
    fn tensor_cosh_test() {
        let mut sizes = ArrayTrait::new();
        sizes.append(4);

        let mut data = ArrayTrait::new();
        data.append(IntegerTrait::new(0, false));
        data.append(IntegerTrait::new(1, false));
        data.append(IntegerTrait::new(2, false));
        data.append(IntegerTrait::new(3, false));
        let extra = Option::<ExtraParams>::None(());
        let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

        let result = tensor.cosh().data;

        assert((*result.at(0).mag).into() == 65536, 'result[0] = 1');
        assert((*result.at(1).mag).into() == 101125, 'result[1] = 1.5431...');
        assert((*result.at(2).mag).into() == 246550, 'result[2] = 3.7622...');
        assert((*result.at(3).mag).into() == 659775, 'result[3] = 10.0677...');
    }
}

// ===== 2D ===== //

mod tensor_2D {
    use array::SpanTrait;
    use traits::Into;
    use array::ArrayTrait;

    use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
    use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
    use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};

    #[test]
    #[available_gas(20000000)]
    fn tensor_cosh_test() {
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(IntegerTrait::new(0, false));
        data.append(IntegerTrait::new(1, false));
        data.append(IntegerTrait::new(2, false));
        data.append(IntegerTrait::new(3, false));
        let extra = Option::<ExtraParams>::None(());
        let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

        let result = tensor.cosh().data;

        assert((*result.at(0).mag).into() == 65536, 'result[0] = 1');
        assert((*result.at(1).mag).into() == 101125, 'result[1] = 1.5431...');
        assert((*result.at(2).mag).into() == 246550, 'result[2] = 3.7622...');
        assert((*result.at(3).mag).into() == 659775, 'result[3] = 10.0677...');
    }
}
// // ===== 3D ===== //

mod tensor_3D {
    use array::SpanTrait;
    use traits::Into;
    use array::ArrayTrait;

    use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
    use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
    use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};

    #[test]
    #[available_gas(20000000)]
    fn tensor_cosh_test() {
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(IntegerTrait::new(0, false));
        data.append(IntegerTrait::new(1, false));
        data.append(IntegerTrait::new(2, false));
        data.append(IntegerTrait::new(3, false));
        data.append(IntegerTrait::new(4, false));
        data.append(IntegerTrait::new(5, false));
        data.append(IntegerTrait::new(6, false));
        data.append(IntegerTrait::new(7, false));
        let extra = Option::<ExtraParams>::None(());
        let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

        let result = tensor.cosh().data;

        assert((*result.at(0).mag).into() == 65536, 'result[0] = 1');
        assert((*result.at(1).mag).into() == 101125, 'result[1] = 1.5431...');
        assert((*result.at(2).mag).into() == 246550, 'result[2] = 3.7622...');
        assert((*result.at(3).mag).into() == 659775, 'result[3] = 10.0677...');
        assert((*result.at(4).mag).into() == 1789592, 'result[4] = 27.3082...');
        assert((*result.at(5).mag).into() == 4863260, 'result[5] = 74.20995...');
        assert((*result.at(6).mag).into() == 13219025, 'result[6] = 201.7156...');
        assert((*result.at(7).mag).into() == 35933213, 'result[7] = 548.3170...');
    }
}

