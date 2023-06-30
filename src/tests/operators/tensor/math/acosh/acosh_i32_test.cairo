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
    fn tensor_acosh_test() {
        let mut sizes = ArrayTrait::new();
        sizes.append(3);

        let mut data = ArrayTrait::new();

        data.append(IntegerTrait::new(1, false));
        data.append(IntegerTrait::new(2, false));
        data.append(IntegerTrait::new(3, false));
        let extra = Option::<ExtraParams>::None(());
        let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

        let result = tensor.acosh().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 1');
        assert((*result.at(1).mag).into() == 86255, 'result[1] = 1.31696...');
        assert((*result.at(2).mag).into() == 115516, 'result[2] = 1.76275...');
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
    fn tensor_acosh_test() {
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(IntegerTrait::new(1, false));
        data.append(IntegerTrait::new(2, false));
        data.append(IntegerTrait::new(3, false));
        data.append(IntegerTrait::new(4, false));
        let extra = Option::<ExtraParams>::None(());
        let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

        let result = tensor.acosh().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 1');
        assert((*result.at(1).mag).into() == 86255, 'result[1] = 1.31696...');
        assert((*result.at(2).mag).into() == 115516, 'result[2] = 1.76275...');
        assert((*result.at(3).mag).into() == 135220, 'result[3] = 2.06344...');
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
    fn tensor_acosh_test() {
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(IntegerTrait::new(1, false));
        data.append(IntegerTrait::new(2, false));
        data.append(IntegerTrait::new(3, false));
        data.append(IntegerTrait::new(4, false));
        data.append(IntegerTrait::new(5, false));
        data.append(IntegerTrait::new(6, false));
        data.append(IntegerTrait::new(7, false));
        data.append(IntegerTrait::new(8, false));
        let extra = Option::<ExtraParams>::None(());
        let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);

        let result = tensor.acosh().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 1');
        assert((*result.at(1).mag).into() == 86255, 'result[1] = 1.31696...');
        assert((*result.at(2).mag).into() == 115516, 'result[2] = 1.76275...');
        assert((*result.at(3).mag).into() == 135220, 'result[3] = 2.06344...');
        assert((*result.at(4).mag).into() == 150233, 'result[4] = 2.29243...');
        assert((*result.at(5).mag).into() == 162372, 'result[5] = 2.47789...');
        assert((*result.at(6).mag).into() == 172594, 'result[6] = 2.63392..');
        assert((*result.at(7).mag).into() == 181354, 'result[7] = 2.76866...');
    }
}

