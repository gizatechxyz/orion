// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    #[cfg(test)]
    mod fp8x23 {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::signed_integer::{integer_trait::IntegerTrait};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;
        use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
        use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl, FP8x23PartialEq};

        #[test]
        #[available_gas(2000000)]
        fn leaky_relu() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(6);

            let mut data = ArrayTrait::<u32>::new();
            let val_1 = 4;
            let val_2 = 3;
            let val_3 = 2;
            let val_4 = 1;
            let val_5 = 0;
            let val_6 = 0;

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);
            data.append(val_5);
            data.append(val_6);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let alpha = FixedTrait::new(838861, false); // 0.1
            let threshold = 3;
            let mut result = NNTrait::leaky_relu(@tensor, @alpha, threshold);

            let data = *result.data.at(0);
            assert(data == FixedTrait::new(33554432, false), 'result[0] == 33554432'); // 4 

            let data = *result.data.at(1);
            assert(data == FixedTrait::new(25165824, false), 'result[1] == 25165824'); // 3

            let data = *result.data.at(3);
            assert(data == FixedTrait::new(838861, false), 'result[3] == 838861'); // 0.1

            let data = *result.data.at(5);
            assert(data == FixedTrait::new(0, false), 'result[5] == 0');
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::signed_integer::{integer_trait::IntegerTrait};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;
        use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
        use orion::numbers::fixed_point::implementations::impl_16x16::{
            FP16x16Impl, FP16x16PartialEq
        };

        #[test]
        #[available_gas(2000000)]
        fn leaky_relu() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(6);

            let mut data = ArrayTrait::<u32>::new();
            let val_1 = 4;
            let val_2 = 3;
            let val_3 = 2;
            let val_4 = 1;
            let val_5 = 0;
            let val_6 = 0;

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);
            data.append(val_5);
            data.append(val_6);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let alpha = FixedTrait::new(6554, false); // 0.1
            let threshold = 3;
            let mut result = NNTrait::leaky_relu(@tensor, @alpha, threshold);

            let data = *result.data.at(0);
            assert(data == FixedTrait::new(262144, false), 'result[0] == 262144'); // 4 

            let data = *result.data.at(1);
            assert(data == FixedTrait::new(196608, false), 'result[1] == 196608'); // 3

            let data = *result.data.at(3);
            assert(data == FixedTrait::new(6554, false), 'result[3] == 6554'); // 0.1

            let data = *result.data.at(5);
            assert(data == FixedTrait::new(0, false), 'result[5] == 0');
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

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::signed_integer::{integer_trait::IntegerTrait};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;
        use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
        use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl, FP8x23PartialEq};

        #[test]
        #[available_gas(2000000)]
        fn leaky_relu() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(3);

            let mut data = ArrayTrait::<u32>::new();
            let val_1 = 4;
            let val_2 = 3;
            let val_3 = 2;
            let val_4 = 1;
            let val_5 = 0;
            let val_6 = 0;

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);
            data.append(val_5);
            data.append(val_6);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let alpha = FixedTrait::new(838861, false); // 0.1
            let threshold = 3;
            let mut result = NNTrait::leaky_relu(@tensor, @alpha, threshold);

            let data = *result.data.at(0);
            assert(data == FixedTrait::new(33554432, false), 'result[0] == 33554432'); // 4 

            let data = *result.data.at(1);
            assert(data == FixedTrait::new(25165824, false), 'result[1] == 25165824'); // 3

            let data = *result.data.at(3);
            assert(data == FixedTrait::new(838861, false), 'result[3] == 838861'); // 0.1

            let data = *result.data.at(5);
            assert(data == FixedTrait::new(0, false), 'result[5] == 0');
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::signed_integer::{integer_trait::IntegerTrait};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;
        use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
        use orion::numbers::fixed_point::implementations::impl_16x16::{
            FP16x16Impl, FP16x16PartialEq
        };

        #[test]
        #[available_gas(2000000)]
        fn leaky_relu() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(3);

            let mut data = ArrayTrait::<u32>::new();
            let val_1 = 4;
            let val_2 = 3;
            let val_3 = 2;
            let val_4 = 1;
            let val_5 = 0;
            let val_6 = 0;

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);
            data.append(val_5);
            data.append(val_6);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let alpha = FixedTrait::new(6554, false); // 0.1
            let threshold = 3;
            let mut result = NNTrait::leaky_relu(@tensor, @alpha, threshold);

            let data = *result.data.at(0);
            assert(data == FixedTrait::new(262144, false), 'result[0] == 262144'); // 4 

            let data = *result.data.at(1);
            assert(data == FixedTrait::new(196608, false), 'result[1] == 196608'); // 3

            let data = *result.data.at(3);
            assert(data == FixedTrait::new(6554, false), 'result[3] == 6554'); // 0.1

            let data = *result.data.at(5);
            assert(data == FixedTrait::new(0, false), 'result[5] == 0');
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

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::signed_integer::{integer_trait::IntegerTrait};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;
        use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
        use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl, FP8x23PartialEq};

        #[test]
        #[available_gas(2000000)]
        fn leaky_relu() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(2);
            shape.append(2);

            let mut data = ArrayTrait::<u32>::new();
            let val_1 = 4;
            let val_2 = 3;
            let val_3 = 2;
            let val_4 = 1;
            let val_5 = 0;
            let val_6 = 0;
            let val_7 = 0;
            let val_8 = 0;

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
            let threshold = 3;
            let mut result = NNTrait::leaky_relu(@tensor, @alpha, threshold);

            let data = *result.data.at(0);
            assert(data == FixedTrait::new(33554432, false), 'result[0] == 33554432'); // 4 

            let data = *result.data.at(1);
            assert(data == FixedTrait::new(25165824, false), 'result[1] == 25165824'); // 3

            let data = *result.data.at(3);
            assert(data == FixedTrait::new(838861, false), 'result[3] == 838861'); // 0.1

            let data = *result.data.at(5);
            assert(data == FixedTrait::new(0, false), 'result[5] == 0');
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::signed_integer::{integer_trait::IntegerTrait};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;
        use orion::numbers::fixed_point::core::{FixedType, FixedTrait, FixedImpl};
        use orion::numbers::fixed_point::implementations::impl_16x16::{
            FP16x16Impl, FP16x16PartialEq
        };

        #[test]
        #[available_gas(2000000)]
        fn leaky_relu() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(2);
            shape.append(2);

            let mut data = ArrayTrait::<u32>::new();
            let val_1 = 4;
            let val_2 = 3;
            let val_3 = 2;
            let val_4 = 1;
            let val_5 = 0;
            let val_6 = 0;
            let val_7 = 0;
            let val_8 = 0;

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
            let threshold = 3;
            let mut result = NNTrait::leaky_relu(@tensor, @alpha, threshold);

            let data = *result.data.at(0);
            assert(data == FixedTrait::new(262144, false), 'result[0] == 262144'); // 4 

            let data = *result.data.at(1);
            assert(data == FixedTrait::new(196608, false), 'result[1] == 196608'); // 3

            let data = *result.data.at(3);
            assert(data == FixedTrait::new(6554, false), 'result[3] == 6554'); // 0.1

            let data = *result.data.at(5);
            assert(data == FixedTrait::new(0, false), 'result[5] == 0');
        }
    }
}

