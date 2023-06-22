// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    #[cfg(test)]
    mod fp8x23 {
        use core::debug::PrintTrait;
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::signed_integer::integer_trait::IntegerTrait;
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};

        #[test]
        #[available_gas(5000000)]
        fn sigmoid() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(4);

            let mut data = ArrayTrait::<u32>::new();
            let val_1 = 0;
            let val_2 = 1;
            let val_3 = 2;
            let val_4 = 254;

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let mut result = NNTrait::sigmoid(@tensor);

            let data = *result.data.at(0).mag;
            assert(data == 4194304, 'result[0] == 4194304'); // 0.5

            let data = *result.data.at(1).mag;
            assert(data == 6132564, 'result[1] == 6132564'); // 0.7310...

            let data = *result.data.at(2).mag;
            assert(data == 7388661, 'result[2] == 7388661'); // 0.8807...

            let data = *result.data.at(3).mag;
            assert(data == 8388608, 'result[3] == 8388608'); // 1
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use core::debug::PrintTrait;
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::signed_integer::integer_trait::IntegerTrait;
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};

        #[test]
        #[available_gas(5000000)]
        fn sigmoid() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(4);

            let mut data = ArrayTrait::<u32>::new();
            let val_1 = 0;
            let val_2 = 1;
            let val_3 = 2;
            let val_4 = 254;

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let mut result = NNTrait::sigmoid(@tensor);

            let data = *result.data.at(0).mag;
            assert(data == 32768, 'result[0] == 32768'); // 0.5

            let data = *result.data.at(1).mag;
            assert(data == 47910, 'result[1] == 47910'); // 0.7310...

            let data = *result.data.at(2).mag;
            assert(data == 57724, 'result[2] == 57724'); // 0.8807...

            let data = *result.data.at(3).mag;
            assert(data == 65536, 'result[3] == 65536'); // 1
        }
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    #[cfg(test)]
    mod fp8x23 {
        use core::debug::PrintTrait;
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::signed_integer::integer_trait::IntegerTrait;
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};

        #[test]
        #[available_gas(5000000)]
        fn sigmoid() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(2);

            let mut data = ArrayTrait::<u32>::new();
            let val_1 = 0;
            let val_2 = 1;
            let val_3 = 2;
            let val_4 = 254;

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let mut result = NNTrait::sigmoid(@tensor);

            let data = *result.data.at(0).mag;
            assert(data == 4194304, 'result[0] == 4194304'); // 0.5

            let data = *result.data.at(1).mag;
            assert(data == 6132564, 'result[1] == 6132564'); // 0.7310...

            let data = *result.data.at(2).mag;
            assert(data == 7388661, 'result[2] == 7388661'); // 0.8807...

            let data = *result.data.at(3).mag;
            assert(data == 8388608, 'result[3] == 8388608'); // 1
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use core::debug::PrintTrait;
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::signed_integer::integer_trait::IntegerTrait;
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};

        #[test]
        #[available_gas(5000000)]
        fn sigmoid() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(2);

            let mut data = ArrayTrait::<u32>::new();
            let val_1 = 0;
            let val_2 = 1;
            let val_3 = 2;
            let val_4 = 254;

            data.append(val_1);
            data.append(val_2);
            data.append(val_3);
            data.append(val_4);

            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

            let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
            let mut result = NNTrait::sigmoid(@tensor);

            let data = *result.data.at(0).mag;
            assert(data == 32768, 'result[0] == 32768'); // 0.5

            let data = *result.data.at(1).mag;
            assert(data == 47910, 'result[1] == 47910'); // 0.7310...

            let data = *result.data.at(2).mag;
            assert(data == 57724, 'result[2] == 57724'); // 0.8807...

            let data = *result.data.at(3).mag;
            assert(data == 65536, 'result[3] == 65536'); // 1
        }
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    #[cfg(test)]
    mod fp8x23 {
        use core::debug::PrintTrait;
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::signed_integer::integer_trait::IntegerTrait;
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};

        #[test]
        #[available_gas(5000000)]
        fn sigmoid() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(2);
            shape.append(2);

            let mut data = ArrayTrait::<u32>::new();
            let val_1 = 0;
            let val_2 = 1;
            let val_3 = 2;
            let val_4 = 254;
            let val_5 = 0;
            let val_6 = 1;
            let val_7 = 2;
            let val_8 = 254;

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
            let mut result = NNTrait::sigmoid(@tensor);

            let data = *result.data.at(0).mag;
            assert(data == 4194304, 'result[0] == 4194304'); // 0.5

            let data = *result.data.at(1).mag;
            assert(data == 6132564, 'result[1] == 6132564'); // 0.7310...

            let data = *result.data.at(2).mag;
            assert(data == 7388661, 'result[2] == 7388661'); // 0.8807...

            let data = *result.data.at(3).mag;
            assert(data == 8388608, 'result[3] == 8388608'); // 1

            let data = *result.data.at(4).mag;
            assert(data == 4194304, 'result[4] == 4194304'); // 0.5

            let data = *result.data.at(5).mag;
            assert(data == 6132564, 'result[5] == 6132564'); // 0.7310...

            let data = *result.data.at(6).mag;
            assert(data == 7388661, 'result[6] == 7388661'); // 0.8807...

            let data = *result.data.at(7).mag;
            assert(data == 8388608, 'result[7] == 8388608'); // 1
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use core::debug::PrintTrait;
        use array::ArrayTrait;
        use array::SpanTrait;

        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::signed_integer::integer_trait::IntegerTrait;
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;
        use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};

        #[test]
        #[available_gas(5000000)]
        fn sigmoid() {
            let mut shape = ArrayTrait::<usize>::new();
            shape.append(2);
            shape.append(2);
            shape.append(2);

            let mut data = ArrayTrait::<u32>::new();
            let val_1 = 0;
            let val_2 = 1;
            let val_3 = 2;
            let val_4 = 254;
            let val_5 = 0;
            let val_6 = 1;
            let val_7 = 2;
            let val_8 = 254;

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
            let mut result = NNTrait::sigmoid(@tensor);

            let data = *result.data.at(0).mag;
            assert(data == 32768, 'result[0] == 32768'); // 0.5

            let data = *result.data.at(1).mag;
            assert(data == 47910, 'result[1] == 47910'); // 0.7310...

            let data = *result.data.at(2).mag;
            assert(data == 57724, 'result[2] == 57724'); // 0.8807...

            let data = *result.data.at(3).mag;
            assert(data == 65536, 'result[3] == 65536'); // 1

            let data = *result.data.at(4).mag;
            assert(data == 32768, 'result[4] == 32768'); // 0.5

            let data = *result.data.at(5).mag;
            assert(data == 47910, 'result[5] == 47910'); // 0.7310...

            let data = *result.data.at(6).mag;
            assert(data == 57724, 'result[6] == 57724'); // 0.8807...

            let data = *result.data.at(7).mag;
            assert(data == 65536, 'result[7] == 65536'); // 1
        }
    }
}

