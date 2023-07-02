// ===== 1D ===== //

#[cfg(test)]
mod input_1D {
    #[cfg(test)]
    mod fp8x23 {
        use array::ArrayTrait;
        use array::SpanTrait;
        use traits::Into;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl, FP8x23Into};

        #[test]
        #[available_gas(50000000)]
        fn softplus() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(4);

            let mut data = ArrayTrait::<u32>::new();
            let val_1 = 0;
            let val_2 = 1;
            let val_3 = 2;
            let val_4 = 3;

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let mut result = NNTrait::softplus(@tensor);

            assert((*result.data[0]).into() == 5814556, 'result[0] == 5814556'); // 0.6931452
            assert((*result.data[1]).into() == 11016447, 'result[1] == 11016447'); // 1.31326096
            assert((*result.data[2]).into() == 17841964, 'result[2] == 17841964'); // 2.1269
            assert((*result.data[3]).into() == 25573406, 'result[3] == 25573406'); // 3.0485875
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use array::ArrayTrait;
        use array::SpanTrait;
        use traits::Into;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16Into};

        #[test]
        #[available_gas(50000000)]
        fn softplus() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(4);

            let mut data = ArrayTrait::<u32>::new();
            let val_1 = 0;
            let val_2 = 1;
            let val_3 = 2;
            let val_4 = 3;

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let mut result = NNTrait::softplus(@tensor);

            assert((*result.data[0]).into() == 45355, 'result[0] == 45355'); // 0.6931452
            assert((*result.data[1]).into() == 86022, 'result[1] == 86022'); // 1.31326096
            assert(
                (*result.data[2]).into() == 139388, 'result[2] == 139388'
            ); // 2.12689208984375
            assert((*result.data[3]).into() == 199788, 'result[3] == 199788'); // 3.04852294
        }
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod input_2D {
    #[cfg(test)]
    mod fp8x23 {
        use array::ArrayTrait;
        use array::SpanTrait;
        use traits::Into;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl, FP8x23Into};

        #[test]
        #[available_gas(50000000)]
        fn softplus() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(2);

            let mut data = ArrayTrait::<u32>::new();
            let val_1 = 0;
            let val_2 = 1;
            let val_3 = 2;
            let val_4 = 3;

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let mut result = NNTrait::softplus(@tensor);

            assert((*result.data[0]).into() == 5814556, 'result[0] == 5814556'); // 0.6931452
            assert((*result.data[1]).into() == 11016447, 'result[1] == 11016447'); // 1.31326096
            assert(
                (*result.data[2]).into() == 17841964, 'result[2] == 17841964'
            ); // 2.12689208984375
            assert((*result.data[3]).into() == 25573406, 'result[3] == 25573406'); // 3.0485875
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use array::ArrayTrait;
        use array::SpanTrait;
        use traits::Into;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16Into};

        #[test]
        #[available_gas(50000000)]
        fn softplus() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(2);

            let mut data = ArrayTrait::<u32>::new();
            let val_1 = 0;
            let val_2 = 1;
            let val_3 = 2;
            let val_4 = 3;

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let mut result = NNTrait::softplus(@tensor);

            assert((*result.data[0]).into() == 45355, 'result[0] == 45355'); // 0.6931452
            assert((*result.data[1]).into() == 86022, 'result[1] == 86022'); // 1.31326096
            assert(
                (*result.data[2]).into() == 139388, 'result[2] == 139388'
            ); // 2.12689208984375
            assert((*result.data[3]).into() == 199788, 'result[3] == 199788'); // 3.04852294
        }
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod input_3D {
    #[cfg(test)]
    mod fp8x23 {
        use array::ArrayTrait;
        use array::SpanTrait;
        use traits::Into;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl, FP8x23Into};

        #[test]
        #[available_gas(50000000)]
        fn softplus() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(2);
            shape.append(2);

            let mut data = ArrayTrait::<u32>::new();
            let val_1 = 0;
            let val_2 = 1;
            let val_3 = 2;
            let val_4 = 3;
            let val_5 = 0;
            let val_6 = 1;
            let val_7 = 2;
            let val_8 = 3;

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);
            data.append(val_5);
            data.append(val_6);
            data.append(val_7);
            data.append(val_8);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let mut result = NNTrait::softplus(@tensor);

            assert((*result.data[0]).into() == 5814556, 'result[0] == 5814556'); // 0.6931452
            assert((*result.data[1]).into() == 11016447, 'result[1] == 11016447'); // 1.31326096
            assert(
                (*result.data[2]).into() == 17841964, 'result[2] == 17841964'
            ); // 2.12689208984375
            assert((*result.data[3]).into() == 25573406, 'result[3] == 25573406'); // 3.0485875
            assert((*result.data[4]).into() == 5814556, 'result[4] == 5814556'); // 0.6931452
            assert((*result.data[5]).into() == 11016447, 'result[5] == 11016447'); // 1.31326096
            assert((*result.data[6]).into() == 17841964, 'result[6] == 17841964'); // 2.1269278
            assert((*result.data[7]).into() == 25573406, 'result[7] == 25573406'); // 3.0485875
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use array::ArrayTrait;
        use array::SpanTrait;
        use traits::Into;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16Into};

        #[test]
        #[available_gas(50000000)]
        fn softplus() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(2);
            shape.append(2);

            let mut data = ArrayTrait::<u32>::new();
            let val_1 = 0;
            let val_2 = 1;
            let val_3 = 2;
            let val_4 = 3;
            let val_5 = 0;
            let val_6 = 1;
            let val_7 = 2;
            let val_8 = 3;

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);
            data.append(val_5);
            data.append(val_6);
            data.append(val_7);
            data.append(val_8);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let mut result = NNTrait::softplus(@tensor);

            assert((*result.data[0]).into() == 45355, 'result[0] == 45355'); // 0.6931452
            assert((*result.data[1]).into() == 86022, 'result[1] == 86022'); // 1.31326096
            assert(
                (*result.data[2]).into() == 139388, 'result[2] == 139388'
            ); // 2.12689208984375
            assert((*result.data[3]).into() == 199788, 'result[3] == 199788'); // 3.04852294
            assert((*result.data[4]).into() == 45355, 'result[4] == 45355'); // 0.6931452
            assert((*result.data[5]).into() == 86022, 'result[5] == 86022'); // 1.31326096
            assert(
                (*result.data[6]).into() == 139388, 'result[6] == 139388'
            ); // 2.12689208984375
            assert((*result.data[7]).into() == 199788, 'result[7] == 199788'); // 3.04852294
        }
    }
}

