// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    #[cfg(test)]
    mod fp8x23 {
        use array::ArrayTrait;
        use array::SpanTrait;
        use traits::Into;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
        use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl, FP8x23Into};

        #[test]
        #[available_gas(50000000)]
        fn softplus() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(4);

            let mut data = ArrayTrait::<i32>::new();
            let val_1 = IntegerTrait::new(0, false);
            let val_2 = IntegerTrait::new(1, false);
            let val_3 = IntegerTrait::new(2, true);
            let val_4 = IntegerTrait::new(3, true);

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let mut result = NNTrait::softplus(@tensor);

            assert((*result.data.at(0)).into() == 5814556, 'result[0] == 5814556'); // 0.6931452
            assert((*result.data.at(1)).into() == 11016447, 'result[1] == 11016447'); // 1.31326096
            assert((*result.data.at(2)).into() == 1064751, 'result[2] == 1064751'); // 0.12692797
            assert((*result.data.at(3)).into() == 407580, 'result[3] == 407580'); // 0.04858729
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use array::ArrayTrait;
        use array::SpanTrait;
        use traits::Into;
        use debug::PrintTrait;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
        use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_16x16::{
            FP16x16Impl, FP16x16Into, FP16x16Print
        };

        #[test]
        #[available_gas(50000000)]
        fn softplus() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(4);

            let mut data = ArrayTrait::<i32>::new();
            let val_1 = IntegerTrait::new(0, false);
            let val_2 = IntegerTrait::new(1, false);
            let val_3 = IntegerTrait::new(2, true);
            let val_4 = IntegerTrait::new(3, true);

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let mut result = NNTrait::softplus(@tensor);

            assert((*result.data.at(0)).into() == 45355, 'result[0] == 45355'); // 0.6931452
            assert((*result.data.at(1)).into() == 86022, 'result[1] == 86022'); // 1.31326096
            assert((*result.data.at(2)).into() == 8315, 'result[2] == 8315'); // 0.12692797
            assert((*result.data.at(3)).into() == 3182, 'result[3] == 3182'); // 0.04858729
        }
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    #[cfg(test)]
    mod fp8x23 {
        use array::ArrayTrait;
        use array::SpanTrait;
        use traits::Into;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
        use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl, FP8x23Into};

        #[test]
        #[available_gas(50000000)]
        fn softplus() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(2);

            let mut data = ArrayTrait::<i32>::new();
            let val_1 = IntegerTrait::new(0, false);
            let val_2 = IntegerTrait::new(1, false);
            let val_3 = IntegerTrait::new(2, true);
            let val_4 = IntegerTrait::new(3, true);

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let mut result = NNTrait::softplus(@tensor);

            assert((*result.data.at(0)).into() == 5814556, 'result[0] == 5814556'); // 0.6931452
            assert((*result.data.at(1)).into() == 11016447, 'result[1] == 11016447'); // 1.31326096
            assert((*result.data.at(2)).into() == 1064751, 'result[2] == 1064751'); // 0.12692797
            assert((*result.data.at(3)).into() == 407580, 'result[3] == 407580'); // 0.04858729
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use array::ArrayTrait;
        use array::SpanTrait;
        use traits::Into;
        use debug::PrintTrait;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
        use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_16x16::{
            FP16x16Impl, FP16x16Into, FP16x16Print
        };

        #[test]
        #[available_gas(50000000)]
        fn softplus() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(2);

            let mut data = ArrayTrait::<i32>::new();
            let val_1 = IntegerTrait::new(0, false);
            let val_2 = IntegerTrait::new(1, false);
            let val_3 = IntegerTrait::new(2, true);
            let val_4 = IntegerTrait::new(3, true);

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let mut result = NNTrait::softplus(@tensor);

            assert((*result.data.at(0)).into() == 45355, 'result[0] == 45355'); // 0.6931452
            assert((*result.data.at(1)).into() == 86022, 'result[1] == 86022'); // 1.31326096
            assert((*result.data.at(2)).into() == 8315, 'result[2] == 8315'); // 0.12692797
            assert((*result.data.at(3)).into() == 3182, 'result[3] == 3182'); // 0.04858729
        }
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    #[cfg(test)]
    mod fp8x23 {
        use array::ArrayTrait;
        use array::SpanTrait;
        use traits::Into;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
        use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl, FP8x23Into};

        #[test]
        #[available_gas(50000000)]
        fn softplus() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(2);
            shape.append(2);

            let mut data = ArrayTrait::<i32>::new();
            let val_1 = IntegerTrait::new(0, false);
            let val_2 = IntegerTrait::new(1, false);
            let val_3 = IntegerTrait::new(2, true);
            let val_4 = IntegerTrait::new(3, true);
            let val_5 = IntegerTrait::new(0, false);
            let val_6 = IntegerTrait::new(1, false);
            let val_7 = IntegerTrait::new(2, true);
            let val_8 = IntegerTrait::new(3, true);

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

            assert((*result.data.at(0)).into() == 5814556, 'result[0] == 5814556'); // 0.6931452
            assert((*result.data.at(1)).into() == 11016447, 'result[1] == 11016447'); // 1.31326096
            assert((*result.data.at(2)).into() == 1064751, 'result[2] == 1064751'); // 0.12692797
            assert((*result.data.at(3)).into() == 407580, 'result[3] == 407580'); // 0.04858729
            assert((*result.data.at(4)).into() == 5814556, 'result[4] == 5814556'); // 0.6931452
            assert((*result.data.at(5)).into() == 11016447, 'result[5] == 11016447'); // 1.31326096
            assert((*result.data.at(6)).into() == 1064751, 'result[6] == 1064751'); // 0.12692797
            assert((*result.data.at(7)).into() == 407580, 'result[7] == 407580'); // 0.04858729
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use array::ArrayTrait;
        use array::SpanTrait;
        use traits::Into;
        use debug::PrintTrait;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
        use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_16x16::{
            FP16x16Impl, FP16x16Into, FP16x16Print
        };

        #[test]
        #[available_gas(50000000)]
        fn softplus() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(2);
            shape.append(2);

            let mut data = ArrayTrait::<i32>::new();
            let val_1 = IntegerTrait::new(0, false);
            let val_2 = IntegerTrait::new(1, false);
            let val_3 = IntegerTrait::new(2, true);
            let val_4 = IntegerTrait::new(3, true);
            let val_5 = IntegerTrait::new(0, false);
            let val_6 = IntegerTrait::new(1, false);
            let val_7 = IntegerTrait::new(2, true);
            let val_8 = IntegerTrait::new(3, true);

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

            assert((*result.data.at(0)).into() == 45355, 'result[0] == 45355'); // 0.6931452
            assert((*result.data.at(1)).into() == 86022, 'result[1] == 86022'); // 1.31326096
            assert((*result.data.at(2)).into() == 8315, 'result[2] == 8315'); // 0.12692797
            assert((*result.data.at(3)).into() == 3182, 'result[3] == 3182'); // 0.04858729
            assert((*result.data.at(4)).into() == 45355, 'result[4] == 45355'); // 0.6931452
            assert((*result.data.at(5)).into() == 86022, 'result[5] == 86022'); // 1.31326096
            assert((*result.data.at(6)).into() == 8315, 'result[6] == 8315'); // 0.12692797
            assert((*result.data.at(7)).into() == 3182, 'result[7] == 3182'); // 0.04858729
        }
    }
}

