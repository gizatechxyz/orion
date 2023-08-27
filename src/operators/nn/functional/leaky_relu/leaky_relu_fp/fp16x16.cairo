use core::traits::Into;
use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;

use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::fp16x16::core::{
    FP16x16Impl, ONE, FP16x16Mul, FP16x16PartialOrd
};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::operators::tensor::core::{Tensor, TensorTrait};


/// Cf: NNTrait::leaky_relu docstring
fn leaky_relu(mut z: Tensor<FixedType>, alpha: @FixedType) -> Tensor<FixedType> {
    assert(*alpha.mag < ONE, 'alpha must be less than 1_fp');

    let mut data_result = ArrayTrait::<FixedType>::new();

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                if (*item >= FixedType { mag: 0, sign: false }) {
                    data_result.append(*item);
                } else {
                    data_result.append(*item * *alpha);
                };
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::new(z.shape, data_result.span(), z.extra);
}
