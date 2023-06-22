// ===== 1D ===== //

#[cfg(test)]
mod input_1D {
    use array::ArrayTrait;
    use array::SpanTrait;

    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
    use orion::numbers::signed_integer::{integer_trait::IntegerTrait};
    use orion::operators::nn::core::NNTrait;
    use orion::operators::nn::implementations::impl_nn_u32::NN_u32;

    #[test]
    #[available_gas(2000000)]
    fn relu() {
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(4);

        let mut data = ArrayTrait::<u32>::new();
        let val_1 = 1;
        let val_2 = 2;
        let val_3 = 3;
        let val_4 = 4;

        data.append(val_1);
        data.append(val_2);
        data.append(val_3);
        data.append(val_4);

        let extra = Option::<ExtraParams>::None(());

        let mut tensor = TensorTrait::new(shape.span(), data.span(), extra);
        let threshold = 3;
        let mut result = NNTrait::relu(@tensor, threshold);

        let data_0 = *result.data.at(0);
        assert(data_0 == 0, 'result[0] == 0');

        let data_1 = *result.data.at(1);
        assert(data_1 == 0, 'result[1] == 0');

        let data_2 = *result.data.at(2);
        assert(data_2 == 3, 'result[2] == 3');

        let data_3 = *result.data.at(3);
        assert(data_3 == 4, 'result[3] == 4');
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod input_2D {
    use array::ArrayTrait;
    use array::SpanTrait;

    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
    use orion::numbers::signed_integer::{integer_trait::IntegerTrait};
    use orion::operators::nn::core::NNTrait;
    use orion::operators::nn::implementations::impl_nn_u32::NN_u32;

    #[test]
    #[available_gas(2000000)]
    fn relu() {
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(2);
        shape.append(2);

        let mut data = ArrayTrait::<u32>::new();
        let val_1 = 1;
        let val_2 = 2;
        let val_3 = 3;
        let val_4 = 4;

        data.append(val_1);
        data.append(val_2);
        data.append(val_3);
        data.append(val_4);

        let extra = Option::<ExtraParams>::None(());

        let mut tensor = TensorTrait::new(shape.span(), data.span(), extra);
        let threshold = 3;
        let mut result = NNTrait::relu(@tensor, threshold);

        let data_0 = *result.data.at(0);
        assert(data_0 == 0, 'result[0] == 0');

        let data_1 = *result.data.at(1);
        assert(data_1 == 0, 'result[1] == 0');

        let data_2 = *result.data.at(2);
        assert(data_2 == 3, 'result[2] == 3');

        let data_3 = *result.data.at(3);
        assert(data_3 == 4, 'result[3] == 4');
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod input_3D {
    use array::ArrayTrait;
    use array::SpanTrait;

    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
    use orion::numbers::signed_integer::{integer_trait::IntegerTrait};
    use orion::operators::nn::core::NNTrait;
    use orion::operators::nn::implementations::impl_nn_u32::NN_u32;

    #[test]
    #[available_gas(2000000)]
    fn relu() {
        let mut shape = ArrayTrait::<usize>::new();
        shape.append(2);
        shape.append(2);
        shape.append(2);

        let mut data = ArrayTrait::<u32>::new();
        let val_1 = 1;
        let val_2 = 2;
        let val_3 = 3;
        let val_4 = 4;
        let val_5 = 5;
        let val_6 = 6;
        let val_7 = 7;
        let val_8 = 8;

        data.append(val_1);
        data.append(val_2);
        data.append(val_3);
        data.append(val_4);
        data.append(val_5);
        data.append(val_6);
        data.append(val_7);
        data.append(val_8);
        let extra = Option::<ExtraParams>::None(());

        let mut tensor = TensorTrait::new(shape.span(), data.span(), extra);
        let threshold = 3;
        let mut result = NNTrait::relu(@tensor, threshold);

        let data_0 = *result.data.at(0);
        assert(data_0 == 0, 'result[0] == 0');

        let data_1 = *result.data.at(1);
        assert(data_1 == 0, 'result[1] == 0');

        let data_2 = *result.data.at(2);
        assert(data_2 == 3, 'result[2] == 3');

        let data_3 = *result.data.at(3);
        assert(data_3 == 4, 'result[3] == 4');

        let data_4 = *result.data.at(4);
        assert(data_4 == 5, 'result[4] == 5');

        let data_5 = *result.data.at(5);
        assert(data_5 == 6, 'result[5] == 6');

        let data_6 = *result.data.at(6);
        assert(data_6 == 7, 'result[6] == 7');

        let data_7 = *result.data.at(7);
        assert(data_7 == 8, 'result[7] == 8');
    }
}

