// ===== 1D ===== //

#[cfg(test)]
mod tensor_1D {
    #[cfg(test)]
    mod fp8x23 {
        use array::SpanTrait;
        use traits::Into;

        use orion::tests::helpers::tensor::u32::u32_tensor_1x3_helper;
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl, FP8x23Into};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;

        #[test]
        #[available_gas(20000000)]
        fn logsoftmax() {
            let tensor = u32_tensor_1x3_helper();
            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
            let tensor = TensorTrait::new(tensor.shape, tensor.data, Option::Some(extra));

            let mut result = NNTrait::logsoftmax(@tensor, 0).data;

            assert((*result[0]).into() == -20196463, 'result[0] = -2.40760596');
            assert((*result[1]).into() == -11807856, 'result[1] = -1.40760596');
            assert((*result[2]).into() == -3419249, 'result[2] = -0.40760596');
        }

        #[test]
        #[should_panic(expected: ('axis out of dimensions', ))]
        #[available_gas(20000000)]
        fn wrong_axis() {
            let tensor = u32_tensor_1x3_helper();
            let mut result = NNTrait::logsoftmax(@tensor, 1);
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use array::SpanTrait;
        use traits::Into;

        use orion::tests::helpers::tensor::u32::u32_tensor_1x3_helper;
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16Into};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;

        #[test]
        #[available_gas(20000000)]
        fn logsoftmax() {
            let tensor = u32_tensor_1x3_helper();
            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
            let tensor = TensorTrait::new(tensor.shape, tensor.data, Option::Some(extra));

            let mut result = NNTrait::logsoftmax(@tensor, 0).data;

            assert((*result[0]).into() == -157788, 'result[0] = -2.40760596');
            assert((*result[1]).into() == -92250, 'result[1] = -1.40760596');
            assert((*result[2]).into() == -26710, 'result[2] = -0.40760596');
        }

        #[test]
        #[should_panic(expected: ('axis out of dimensions', ))]
        #[available_gas(20000000)]
        fn wrong_axis() {
            let tensor = u32_tensor_1x3_helper();
            let mut result = NNTrait::logsoftmax(@tensor, 1);
        }
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod tensor_2D {
    #[cfg(test)]
    mod fp8x23 {
        use array::SpanTrait;
        use traits::Into;

        use orion::tests::helpers::tensor::u32::u32_tensor_2x2_helper;
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl, FP8x23Into};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;

        #[test]
        #[available_gas(20000000)]
        fn logsoftmax() {
            let tensor = u32_tensor_2x2_helper();
            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
            let tensor = TensorTrait::new(tensor.shape, tensor.data, Option::Some(extra));

            let mut result = NNTrait::logsoftmax(@tensor, 0).data;

            assert((*result[0]).into() == -17841970, 'result[0] = -2.12693');
            assert((*result[1]).into() == -17841970, 'result[1] = -2.12695');
            assert((*result[2]).into() == -1064751, 'result[2] = -0.12692');
            assert((*result[3]).into() == -1064751, 'result[3] = -0.12692');

            let mut result = NNTrait::logsoftmax(@tensor, 1).data;

            assert((*result.at(0)).into() == -11016451, 'result[0] = -1.3134');
            assert((*result.at(1)).into() == -2627827, 'result[1] = -0.3132');
            assert((*result.at(2)).into() == -11016460, 'result[2] = -1.3134');
            assert((*result.at(3)).into() == -2627829, 'result[3] = -0.3132');
        }

        #[test]
        #[should_panic(expected: ('axis out of dimensions', ))]
        #[available_gas(20000000)]
        fn wrong_axis() {
            let tensor = u32_tensor_2x2_helper();
            let mut result = NNTrait::logsoftmax(@tensor, 2);
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use array::SpanTrait;
        use traits::Into;

        use orion::tests::helpers::tensor::u32::u32_tensor_2x2_helper;
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16Into};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;

        #[test]
        #[available_gas(20000000)]
        fn logsoftmax() {
            let tensor = u32_tensor_2x2_helper();
            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
            let tensor = TensorTrait::new(tensor.shape, tensor.data, Option::Some(extra));

            let mut result = NNTrait::logsoftmax(@tensor, 0).data;

            assert((*result[0]).into() == -139390, 'result[0] = -2.12693');
            assert((*result[1]).into() == -139390, 'result[1] = -2.12695');
            assert((*result[2]).into() == -8317, 'result[2] = -0.12692');
            assert((*result[3]).into() == -8317, 'result[3] = -0.12692');

            let mut result = NNTrait::logsoftmax(@tensor, 1).data;

            assert((*result.at(0)).into() == -86024, 'result[0] = -1.3134');
            assert((*result.at(1)).into() == -20523, 'result[1] = -0.3132');
            assert((*result.at(2)).into() == -86024, 'result[2] = -1.3134');
            assert((*result.at(3)).into() == -20523, 'result[3] = -0.3132');
        }

        #[test]
        #[should_panic(expected: ('axis out of dimensions', ))]
        #[available_gas(20000000)]
        fn wrong_axis() {
            let tensor = u32_tensor_2x2_helper();
            let mut result = NNTrait::logsoftmax(@tensor, 2);
        }
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    #[cfg(test)]
    mod fp8x23 {
        use array::SpanTrait;
        use traits::Into;

        use orion::tests::helpers::tensor::u32::u32_tensor_2x2x2_helper;
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl, FP8x23Into};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;

        #[test]
        #[available_gas(200000000)]
        fn logsoftmax() {
            let tensor = u32_tensor_2x2x2_helper();
            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
            let tensor = TensorTrait::new(tensor.shape, tensor.data, Option::Some(extra));

            let mut result = NNTrait::logsoftmax(@tensor, 0).data;

            assert((*result[0]).into() == -33706687, 'result[0] = -4.01814993');
            assert((*result[1]).into() == -33706687, 'result[1] = -4.01814993');
            assert((*result[2]).into() == -33706687, 'result[2] = -4.01814993');
            assert((*result[3]).into() == -33706687, 'result[3] = -4.01814993');
            assert((*result[4]).into() == -152253, 'result[4] = -0.01814993');
            assert((*result[5]).into() == -152253, 'result[5] = -0.01814993');
            assert((*result[6]).into() == -152253, 'result[6] = -0.01814993');
            assert((*result[7]).into() == -152253, 'result[7] = -0.01814993');

            let mut result = NNTrait::logsoftmax(@tensor, 1).data;

            assert((*result[0]).into() == -17841970, 'result[0] = -2.12692801');
            assert((*result[1]).into() == -17841970, 'result[1] = -2.12692801');
            assert((*result[2]).into() == -1064751, 'result[2] = -0.12692801');
            assert((*result[3]).into() == -1064751, 'result[3] = -0.12692801');
            assert((*result[4]).into() == -17841970, 'result[4] = -2.12692801');
            assert((*result[5]).into() == -17841970, 'result[5] = -2.12692801');
            assert((*result[6]).into() == -1064751, 'result[6] = -0.12692801');
            assert((*result[7]).into() == -1064751, 'result[7] = -0.12692801');

            let mut result = NNTrait::logsoftmax(@tensor, 2).data;

            assert((*result[0]).into() == -11016451, 'result[0] = -1.31326169');
            assert((*result[1]).into() == -2627827, 'result[1] = -0.31326169');
            assert((*result[2]).into() == -11016460, 'result[2] = -1.31326169');
            assert((*result[3]).into() == -2627829, 'result[3] = -0.31326169');
            assert((*result[4]).into() == -11016451, 'result[4] = -1.31326169');
            assert((*result[5]).into() == -2627827, 'result[5] = -0.31326169');
            assert((*result[6]).into() == -11016451, 'result[6] = -1.31326169');
            assert((*result[7]).into() == -2627827, 'result[7] = -0.31326169');
        }

        #[test]
        #[should_panic(expected: ('axis out of dimensions', ))]
        #[available_gas(20000000)]
        fn wrong_axis() {
            let tensor = u32_tensor_2x2x2_helper();
            let mut result = NNTrait::logsoftmax(@tensor, 3);
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use array::SpanTrait;
        use traits::Into;

        use orion::tests::helpers::tensor::u32::u32_tensor_2x2x2_helper;
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16Into};
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;

        #[test]
        #[available_gas(200000000)]
        fn logsoftmax() {
            let tensor = u32_tensor_2x2x2_helper();
            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
            let tensor = TensorTrait::new(tensor.shape, tensor.data, Option::Some(extra));

            let mut result = NNTrait::logsoftmax(@tensor, 0).data;

            assert((*result[0]).into() == -263347, 'result[0] = -4.01814993');
            assert((*result[1]).into() == -263347, 'result[1] = -4.01814993');
            assert((*result[2]).into() == -263347, 'result[2] = -4.01814993');
            assert((*result[3]).into() == -263347, 'result[3] = -4.01814993');
            assert((*result[4]).into() == -1188, 'result[4] = -0.01814993');
            assert((*result[5]).into() == -1188, 'result[5] = -0.01814993');
            assert((*result[6]).into() == -1188, 'result[6] = -0.01814993');
            assert((*result[7]).into() == -1188, 'result[7] = -0.01814993');

            let mut result = NNTrait::logsoftmax(@tensor, 1).data;

            assert((*result[0]).into() == -139390, 'result[0] = -2.12692801');
            assert((*result[1]).into() == -139390, 'result[1] = -2.12692801');
            assert((*result[2]).into() == -8317, 'result[2] = -0.12692801');
            assert((*result[3]).into() == -8317, 'result[3] = -0.12692801');
            assert((*result[4]).into() == -139390, 'result[4] = -2.12692801');
            assert((*result[5]).into() == -139390, 'result[5] = -2.12692801');
            assert((*result[6]).into() == -8317, 'result[6] = -0.12692801');
            assert((*result[7]).into() == -8317, 'result[7] = -0.12692801');

            let mut result = NNTrait::logsoftmax(@tensor, 2).data;

            assert((*result[0]).into() == -86024, 'result[0] = -1.31326169');
            assert((*result[1]).into() == -20523, 'result[1] = -0.31326169');
            assert((*result[2]).into() == -86024, 'result[2] = -1.31326169');
            assert((*result[3]).into() == -20523, 'result[3] = -0.31326169');
            assert((*result[4]).into() == -86024, 'result[4] = -1.31326169');
            assert((*result[5]).into() == -20523, 'result[5] = -0.31326169');
            assert((*result[6]).into() == -86024, 'result[6] = -1.31326169');
            assert((*result[7]).into() == -20523, 'result[7] = -0.31326169');
        }

        #[test]
        #[should_panic(expected: ('axis out of dimensions', ))]
        #[available_gas(20000000)]
        fn wrong_axis() {
            let tensor = u32_tensor_2x2x2_helper();
            let mut result = NNTrait::logsoftmax(@tensor, 3);
        }
    }
}
