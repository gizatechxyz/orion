// ===== 1D ===== //

#[cfg(test)]
mod input_1D {
    #[cfg(test)]
    mod fp8x23 {
        use array::SpanTrait;
        use traits::Into;
        use debug::PrintTrait;

        use orion::tests::helpers::tensor::u32::u32_tensor_1x3_helper;
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_8x23::{
            FP8x23Impl, FP8x23Into, FP8x23Print
        };
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;

        #[test]
        #[available_gas(20000000)]
        fn softmax() {
            let tensor = u32_tensor_1x3_helper();
            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
            let tensor = TensorTrait::new(tensor.shape, tensor.data, Option::Some(extra));

            let mut result = NNTrait::softmax(@tensor, 0).data;

            assert((*result.at(0)).into() == 755231, 'result[0] = 0.09003057');
            assert((*result.at(1)).into() == 2052931, 'result[1] = 0.24472847');
            assert((*result.at(2)).into() == 5580445, 'result[2] = 0.66524096');
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use array::SpanTrait;
        use traits::Into;
        use debug::PrintTrait;

        use orion::tests::helpers::tensor::u32::u32_tensor_1x3_helper;
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_16x16::{
            FP16x16Impl, FP16x16Into, FP16x16Print
        };
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;

        #[test]
        #[available_gas(20000000)]
        fn softmax() {
            let tensor = u32_tensor_1x3_helper();
            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
            let tensor = TensorTrait::new(tensor.shape, tensor.data, Option::Some(extra));

            let mut result = NNTrait::softmax(@tensor, 0).data;

            assert((*result.at(0)).into() == 5900, 'result[0] = 0.09003057');
            assert((*result.at(1)).into() == 16038, 'result[1] = 0.24472847');
            assert((*result.at(2)).into() == 43596, 'result[2] = 0.66524096');
        }
    }
}

// ===== 2D ===== //

#[cfg(test)]
mod input_2D {
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
        fn softmax() {
            let tensor = u32_tensor_2x2_helper();
            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
            let tensor = TensorTrait::new(tensor.shape, tensor.data, Option::Some(extra));

            let mut result = NNTrait::softmax(@tensor, 0).data;

            assert((*result.at(0)).into() == 999946, 'result[0] = 0.1192');
            assert((*result.at(1)).into() == 999946, 'result[1] = 0.1192');
            assert((*result.at(2)).into() == 7388661, 'result[2] = 0.8808');
            assert((*result.at(3)).into() == 7388661, 'result[3] = 0.8808');

            let mut result = NNTrait::softmax(@tensor, 1).data;

            assert((*result.at(0)).into() == 2256044, 'result[0] = 0.2689');
            assert((*result.at(1)).into() == 6132563, 'result[1] = 0.7311');
            assert((*result.at(2)).into() == 2256043, 'result[2] = 0.2689');
            assert((*result.at(3)).into() == 6132564, 'result[4] = 0.7311');
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use array::SpanTrait;
        use traits::Into;
        use debug::PrintTrait;

        use orion::tests::helpers::tensor::u32::u32_tensor_2x2_helper;
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_16x16::{
            FP16x16Impl, FP16x16Into, FP16x16Print
        };
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;

        #[test]
        #[available_gas(20000000)]
        fn softmax() {
            let tensor = u32_tensor_2x2_helper();
            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
            let tensor = TensorTrait::new(tensor.shape, tensor.data, Option::Some(extra));

            let mut result = NNTrait::softmax(@tensor, 0).data;

            assert((*result.at(0)).into() == 7812, 'result[0] = 0.1192');
            assert((*result.at(1)).into() == 7812, 'result[1] = 0.1192');
            assert((*result.at(2)).into() == 57723, 'result[2] = 0.8808');
            assert((*result.at(3)).into() == 57723, 'result[3] = 0.8808');

            let mut result = NNTrait::softmax(@tensor, 1).data;

            assert((*result.at(0)).into() == 17625, 'result[0] = 0.2689');
            assert((*result.at(1)).into() == 47910, 'result[1] = 0.7311');
            assert((*result.at(2)).into() == 17625, 'result[2] = 0.2689');
            assert((*result.at(3)).into() == 47910, 'result[4] = 0.7311');
        }
    }
}

// ===== 3D ===== //

#[cfg(test)]
mod input_3D {
    #[cfg(test)]
    mod fp8x23 {
        use array::SpanTrait;
        use traits::Into;
        use debug::PrintTrait;

        use orion::tests::helpers::tensor::u32::u32_tensor_2x2x2_helper;
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_8x23::{
            FP8x23Impl, FP8x23Into, FP8x23Print
        };
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;

        #[test]
        #[available_gas(200000000)]
        fn softmax() {
            let tensor = u32_tensor_2x2x2_helper();
            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };
            let tensor = TensorTrait::new(tensor.shape, tensor.data, Option::Some(extra));

            let mut result = NNTrait::softmax(@tensor, 0).data;

            assert((*result.at(0)).into() == 150879, 'result[0] = 0.01798621');
            assert((*result.at(1)).into() == 150879, 'result[1] = 0.01798621');
            assert((*result.at(2)).into() == 150879, 'result[2] = 0.01798621');
            assert((*result.at(3)).into() == 150879, 'result[3] = 0.01798621');
            assert((*result.at(4)).into() == 8237728, 'result[4] = 0.01798621');
            assert((*result.at(5)).into() == 8237728, 'result[5] = 0.01798621');
            assert((*result.at(6)).into() == 8237728, 'result[6] = 0.01798621');
            assert((*result.at(7)).into() == 8237728, 'result[7] = 0.01798621');

            let mut result = NNTrait::softmax(@tensor, 1).data;

            assert((*result.at(0)).into() == 999946, 'result[0] = 0.11920292');
            assert((*result.at(1)).into() == 999946, 'result[1] = 0.11920292');
            assert((*result.at(2)).into() == 7388661, 'result[2] = 0.88079708');
            assert((*result.at(3)).into() == 7388661, 'result[3] = 0.88079708');
            assert((*result.at(4)).into() == 999946, 'result[4] = 0.11920292');
            assert((*result.at(5)).into() == 999946, 'result[5] = 0.11920292');
            assert((*result.at(6)).into() == 7388661, 'result[6] = 0.88079708');
            assert((*result.at(7)).into() == 7388661, 'result[7] = 0.88079708');

            let mut result = NNTrait::softmax(@tensor, 2).data;

            assert((*result.at(0)).into() == 2256044, 'result[0] = 0.26894142');
            assert((*result.at(1)).into() == 6132563, 'result[1] = 0.73105858');
            assert((*result.at(2)).into() == 2256043, 'result[2] = 0.26894142');
            assert((*result.at(3)).into() == 6132564, 'result[3] = 0.73105858');
            assert((*result.at(4)).into() == 2256044, 'result[4] = 0.26894142');
            assert((*result.at(5)).into() == 6132563, 'result[5] = 0.73105858');
            assert((*result.at(6)).into() == 2256044, 'result[6] = 0.26894142');
            assert((*result.at(7)).into() == 6132563, 'result[7] = 0.73105858');
        }
    }

    #[cfg(test)]
    mod fp16x16 {
        use array::SpanTrait;
        use traits::Into;
        use debug::PrintTrait;

        use orion::tests::helpers::tensor::u32::u32_tensor_2x2x2_helper;
        use orion::operators::tensor::core::{TensorTrait, ExtraParams};
        use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
        use orion::numbers::fixed_point::core::FixedImpl;
        use orion::numbers::fixed_point::implementations::impl_16x16::{
            FP16x16Impl, FP16x16Into, FP16x16Print
        };
        use orion::operators::nn::core::NNTrait;
        use orion::operators::nn::implementations::impl_nn_u32::NN_u32;

        #[test]
        #[available_gas(200000000)]
        fn softmax() {
            let tensor = u32_tensor_2x2x2_helper();
            let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
            let tensor = TensorTrait::new(tensor.shape, tensor.data, Option::Some(extra));

            let mut result = NNTrait::softmax(@tensor, 0).data;

            assert((*result.at(0)).into() == 1178, 'result[0] = 0.01798621');
            assert((*result.at(1)).into() == 1178, 'result[1] = 0.01798621');
            assert((*result.at(2)).into() == 1178, 'result[2] = 0.01798621');
            assert((*result.at(3)).into() == 1178, 'result[3] = 0.01798621');
            assert((*result.at(4)).into() == 64357, 'result[4] = 0.01798621');
            assert((*result.at(5)).into() == 64357, 'result[5] = 0.01798621');
            assert((*result.at(6)).into() == 64357, 'result[6] = 0.01798621');
            assert((*result.at(7)).into() == 64357, 'result[7] = 0.01798621');

            let mut result = NNTrait::softmax(@tensor, 1).data;

            assert((*result.at(0)).into() == 7812, 'result[0] = 0.11920292');
            assert((*result.at(1)).into() == 7812, 'result[1] = 0.11920292');
            assert((*result.at(2)).into() == 57723, 'result[2] = 0.88079708');
            assert((*result.at(3)).into() == 57723, 'result[3] = 0.88079708');
            assert((*result.at(4)).into() == 7812, 'result[4] = 0.11920292');
            assert((*result.at(5)).into() == 7812, 'result[5] = 0.11920292');
            assert((*result.at(6)).into() == 57723, 'result[6] = 0.88079708');
            assert((*result.at(7)).into() == 57723, 'result[7] = 0.88079708');

            let mut result = NNTrait::softmax(@tensor, 2).data;

            assert((*result.at(0)).into() == 17625, 'result[0] = 0.26894142');
            assert((*result.at(1)).into() == 47910, 'result[1] = 0.73105858');
            assert((*result.at(2)).into() == 17625, 'result[2] = 0.26894142');
            assert((*result.at(3)).into() == 47910, 'result[3] = 0.73105858');
            assert((*result.at(4)).into() == 17625, 'result[4] = 0.26894142');
            assert((*result.at(5)).into() == 47910, 'result[5] = 0.73105858');
            assert((*result.at(6)).into() == 17625, 'result[6] = 0.26894142');
            assert((*result.at(7)).into() == 47910, 'result[7] = 0.73105858');
        }
    }
}

