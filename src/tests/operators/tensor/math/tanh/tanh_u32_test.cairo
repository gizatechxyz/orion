// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    use array::SpanTrait;
    use traits::Into;
    use array::ArrayTrait;

    use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
    use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};

    #[test]
    #[available_gas(20000000)]
    fn tensor_tanh_test() {
        let mut sizes = ArrayTrait::new();
        sizes.append(4);

        let mut data = ArrayTrait::new();
        data.append(0);
        data.append(1);
        data.append(2);
        data.append(3);
        let extra = Option::<ExtraParams>::None(());
        let tensor = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

        let result = tensor.tanh().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 49911, 'result[1] = 0.761594...');
        assert((*result.at(2).mag).into() == 63178, 'result[2] = 0.964028...');
        assert((*result.at(3).mag).into() == 65211, 'result[3] = 0.99505...');
    }
}

// ===== 2D ===== //

mod tensor_2D {
    use array::SpanTrait;
    use traits::Into;
    use array::ArrayTrait;

    use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
    use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};

    #[test]
    #[available_gas(20000000)]
    fn tensor_tanh_test() {
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(0);
        data.append(1);
        data.append(2);
        data.append(3);
        let extra = Option::<ExtraParams>::None(());
        let tensor = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

        let result = tensor.tanh().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 49911, 'result[1] = 0.761594...');
        assert((*result.at(2).mag).into() == 63178, 'result[2] = 0.964028...');
        assert((*result.at(3).mag).into() == 65211, 'result[3] = 0.99505...');
    }
}
// // ===== 3D ===== //

mod tensor_3D {
    use array::SpanTrait;
    use traits::Into;
    use array::ArrayTrait;

    use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
    use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};

    #[test]
    #[available_gas(20000000)]
    fn tensor_tanh_test() {
        let mut sizes = ArrayTrait::new();
        sizes.append(2);
        sizes.append(2);
        sizes.append(2);

        let mut data = ArrayTrait::new();
        data.append(0);
        data.append(1);
        data.append(2);
        data.append(3);
        data.append(4);
        data.append(5);
        data.append(6);
        data.append(7);
        let extra = Option::<ExtraParams>::None(());
        let tensor = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

        let result = tensor.tanh().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 49911, 'result[1] = 0.761594...');
        assert((*result.at(2).mag).into() == 63178, 'result[2] = 0.964028...');
        assert((*result.at(3).mag).into() == 65211, 'result[3] = 0.99505...');
        assert((*result.at(4).mag).into() == 65492, 'result[4] = 0.99933...');
        assert((*result.at(5).mag).into() == 65530, 'result[5] = 0.99991...');
        assert((*result.at(6).mag).into() == 65535, 'result[6] = 0.99999...');
        assert((*result.at(7).mag).into() == 65535, 'result[7] = 0.99999...');
    }
}

