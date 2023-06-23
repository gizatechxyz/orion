// ===== 1D ===== //

#[cfg(test)]
mod input_1D {
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
        use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};
        use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl, ONE, FP8x23Into};

        #[test]
        #[available_gas(2000000)]
        fn leaky_relu() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(6);

            let mut data = ArrayTrait::<i32>::new();
            let val_1 = IntegerTrait::new(1, false);
            let val_2 = IntegerTrait::new(2, false);
            let val_3 = IntegerTrait::new(1, true);
            let val_4 = IntegerTrait::new(2, true);
            let val_5 = IntegerTrait::new(0, false);
            let val_6 = IntegerTrait::new(0, false);

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);
            data.append(val_5);
            data.append(val_6);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let alpha = FixedTrait::new(838861, false); // 0.1
            let threshold = IntegerTrait::new(0, false);

            let mut result = NNTrait::leaky_relu(@tensor, @alpha, threshold);

            assert((*result.data[0]).into() == 8388608, 'result[0] == 8388608'); // 1
            assert(
                (*result.data[3]).into() == -1677722, 'result[3] == - 1677722'
            ); // 2 * 0.1 = - 0.2
            assert((*result.data[5]).into() == 0, 'result[5] == 0');
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use array::ArrayTrait;
        use array::SpanTrait;
        use traits::Into;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
        use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};
        use orion::numbers::fixed_point::implementations::impl_16x16::{
            FP16x16Impl, ONE, FP16x16Into
        };

        #[test]
        #[available_gas(2000000)]
        fn leaky_relu() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(6);

            let mut data = ArrayTrait::<i32>::new();
            let val_1 = IntegerTrait::new(1, false);
            let val_2 = IntegerTrait::new(2, false);
            let val_3 = IntegerTrait::new(1, true);
            let val_4 = IntegerTrait::new(2, true);
            let val_5 = IntegerTrait::new(0, false);
            let val_6 = IntegerTrait::new(0, false);

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);
            data.append(val_5);
            data.append(val_6);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let alpha = FixedTrait::new(6554, false); // 0.1
            let threshold = IntegerTrait::new(0, false);

            let mut result = NNTrait::leaky_relu(@tensor, @alpha, threshold);

            assert((*result.data[0]).into() == 65536, 'result[0] == 65536'); // 1
            assert(
                (*result.data[3]).into() == -13108, 'result[3] == - 13108'
            ); // 2 * 0.1 = - 0.2
            assert((*result.data[5]).into() == 0, 'result[5] == 0');
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
        use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
        use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};
        use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl, ONE, FP8x23Into};

        #[test]
        #[available_gas(2000000)]
        fn leaky_relu() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(3);

            let mut data = ArrayTrait::<i32>::new();
            let val_1 = IntegerTrait::new(1, false);
            let val_2 = IntegerTrait::new(2, false);
            let val_3 = IntegerTrait::new(1, true);
            let val_4 = IntegerTrait::new(2, true);
            let val_5 = IntegerTrait::new(0, false);
            let val_6 = IntegerTrait::new(0, false);

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);
            data.append(val_5);
            data.append(val_6);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let alpha = FixedTrait::new(838861, false); // 0.1
            let threshold = IntegerTrait::new(0, false);

            let mut result = NNTrait::leaky_relu(@tensor, @alpha, threshold);

            assert((*result.data[0]).into() == 8388608, 'result[0] == 8388608'); // 1
            assert(
                (*result.data[3]).into() == -1677722, 'result[3] == - 1677722'
            ); // 2 * 0.1 = - 0.2
            assert((*result.data[5]).into() == 0, 'result[5] == 0');
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use array::ArrayTrait;
        use array::SpanTrait;
        use traits::Into;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
        use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};
        use orion::numbers::fixed_point::implementations::impl_16x16::{
            FP16x16Impl, ONE, FP16x16Into
        };

        #[test]
        #[available_gas(2000000)]
        fn leaky_relu() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(3);

            let mut data = ArrayTrait::<i32>::new();
            let val_1 = IntegerTrait::new(1, false);
            let val_2 = IntegerTrait::new(2, false);
            let val_3 = IntegerTrait::new(1, true);
            let val_4 = IntegerTrait::new(2, true);
            let val_5 = IntegerTrait::new(0, false);
            let val_6 = IntegerTrait::new(0, false);

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);
            data.append(val_5);
            data.append(val_6);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let alpha = FixedTrait::new(6554, false); // 0.1
            let threshold = IntegerTrait::new(0, false);

            let mut result = NNTrait::leaky_relu(@tensor, @alpha, threshold);

            assert((*result.data[0]).into() == 65536, 'result[0] == 65536'); // 1
            assert(
                (*result.data[3]).into() == -13108, 'result[3] == - 13108'
            ); // 2 * 0.1 = - 0.2
            assert((*result.data[5]).into() == 0, 'result[5] == 0');
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
        use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
        use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};
        use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl, ONE, FP8x23Into};

        #[test]
        #[available_gas(2000000)]
        fn leaky_relu() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(2);
            shape.append(2);

            let mut data = ArrayTrait::<i32>::new();
            let val_1 = IntegerTrait::new(1, false);
            let val_2 = IntegerTrait::new(2, false);
            let val_3 = IntegerTrait::new(1, true);
            let val_4 = IntegerTrait::new(2, true);
            let val_5 = IntegerTrait::new(0, false);
            let val_6 = IntegerTrait::new(0, false);
            let val_7 = IntegerTrait::new(0, false);
            let val_8 = IntegerTrait::new(0, false);

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
            let alpha = FixedTrait::new(838861, false); // 0.1
            let threshold = IntegerTrait::new(0, false);

            let mut result = NNTrait::leaky_relu(@tensor, @alpha, threshold);

            assert((*result.data[0]).into() == 8388608, 'result[0] == 8388608'); // 1
            assert(
                (*result.data[3]).into() == -1677722, 'result[3] == - 1677722'
            ); // 2 * 0.1 = - 0.2
            assert((*result.data[5]).into() == 0, 'result[5] == 0');
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use array::ArrayTrait;
        use array::SpanTrait;
        use traits::Into;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
        use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};
        use orion::numbers::fixed_point::implementations::impl_16x16::{
            FP16x16Impl, ONE, FP16x16Into
        };

        #[test]
        #[available_gas(2000000)]
        fn leaky_relu() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(2);
            shape.append(2);

            let mut data = ArrayTrait::<i32>::new();
            let val_1 = IntegerTrait::new(1, false);
            let val_2 = IntegerTrait::new(2, false);
            let val_3 = IntegerTrait::new(1, true);
            let val_4 = IntegerTrait::new(2, true);
            let val_5 = IntegerTrait::new(0, false);
            let val_6 = IntegerTrait::new(0, false);
            let val_7 = IntegerTrait::new(0, false);
            let val_8 = IntegerTrait::new(0, false);

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
            let alpha = FixedTrait::new(6554, false); // 0.1
            let threshold = IntegerTrait::new(0, false);

            let mut result = NNTrait::leaky_relu(@tensor, @alpha, threshold);

            assert((*result.data[0]).into() == 65536, 'result[0] == 65536'); // 1
            assert(
                (*result.data[3]).into() == -13108, 'result[3] == - 13108'
            ); // 2 * 0.1 = - 0.2
            assert((*result.data[5]).into() == 0, 'result[5] == 0');
        }
    }
}

