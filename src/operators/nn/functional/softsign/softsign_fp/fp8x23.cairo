use core::traits::Into;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp8x23::core::{FP8x23Impl, FP8x23Add, FP8x23Div};


/// Cf: NNTrait::softsign docstring
fn softsign(mut z: Tensor<FixedType>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::new();

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                let result = *item / (FixedTrait::ONE() + (*item).abs());
                data_result.append(result);
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(z.shape, data_result.span(), z.extra);
}

