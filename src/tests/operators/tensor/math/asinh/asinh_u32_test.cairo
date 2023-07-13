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
    fn tensor_asinh_test() {
        let mut sizes = ArrayTrait::new();
        sizes.append(4);

        let mut data = ArrayTrait::new();
        data.append(0);
        data.append(1);
        data.append(2);
        data.append(3);
        let extra = Option::<ExtraParams>::None(());
        let tensor = TensorTrait::<u32>::new(sizes.span(), data.span(), extra);

        let result = tensor.asinh().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 57756, 'result[1] = 0.881374...');
        assert((*result.at(2).mag).into() == 94583, 'result[2] = 1.44364...');
        assert((*result.at(3).mag).into() == 119152, 'result[3] = 1.81845...');
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
    fn tensor_asinh_test() {
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

        let result = tensor.asinh().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 57756, 'result[1] = 0.881374...');
        assert((*result.at(2).mag).into() == 94583, 'result[2] = 1.44364...');
        assert((*result.at(3).mag).into() == 119152, 'result[3] = 1.81845...');
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
    fn tensor_asinh_test() {
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

        let result = tensor.asinh().data;

        assert((*result.at(0).mag).into() == 0, 'result[0] = 0');
        assert((*result.at(1).mag).into() == 57756, 'result[1] = 0.881374...');
        assert((*result.at(2).mag).into() == 94583, 'result[2] = 1.44364...');
        assert((*result.at(3).mag).into() == 119152, 'result[3] = 1.81845...');
        assert((*result.at(4).mag).into() == 137261, 'result[4] = 2.09471');
        assert((*result.at(5).mag).into() == 151536, 'result[5] = 2.31244...');
        assert((*result.at(6).mag).into() == 163283, 'result[6] = 2.49178...');
        assert((*result.at(7).mag).into() == 173267, 'result[7] = 2.64412...');
    }
}

