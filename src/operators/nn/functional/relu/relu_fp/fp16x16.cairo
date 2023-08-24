use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::numbers::fixed_point::core::FixedType;
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16PartialOrd;

/// Cf: NNTrait::relu docstring
fn relu(mut z: Tensor<FixedType>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::<FixedType>::new();

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                if (*item < FixedType { mag: 0, sign: false }) {
                    data_result.append(FixedType { mag: 0, sign: false });
                } else {
                    data_result.append(*item);
                };
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::<FixedType>::new(z.shape, data_result.span(), z.extra);
}

