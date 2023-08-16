use core::traits::Into;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::fixed_point::implementations::fp8x23::core::{
    FP8x23Impl, FP8x23Add, FP8x23Mul, FP8x23Div
};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};


/// Cf: NNTrait::sigmoid docstring
fn sigmoid_fp(z: @Tensor<FixedType>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::<FixedType>::new();
    let mut data = *z.data;
    let fp_one = FixedTrait::new_unscaled(1, false);
    loop {
        if data.len() == 0 {
            break ();
        };

        let result = fp_one
            / (fp_one
                + (*data.pop_front().unwrap() * FixedType { mag: 8388608, sign: true }).exp());
        data_result.append(result);
    };
    return TensorTrait::<FixedType>::new(*z.shape, data_result.span(), *z.extra);
}

