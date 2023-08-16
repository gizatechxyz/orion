use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::numbers::fixed_point::core::FixedType;
use orion::numbers::fixed_point::implementations::fp16x16::core::FP16x16PartialOrd;

/// Cf: NNTrait::relu docstring
fn relu(z: @Tensor<FixedType>) -> Tensor<FixedType> {
    let mut data_result = ArrayTrait::<FixedType>::new();
    let mut data = *z.data;

    loop {
        if data.len() == 0 {
            break ();
        };

        let current_index = *data.pop_front().unwrap();
        if (current_index < FixedType { mag: 0, sign: false }) {
            data_result.append(FixedType { mag: 0, sign: false });
        } else {
            data_result.append(current_index);
        };
    };

    return TensorTrait::<FixedType>::new(*z.shape, data_result.span(), *z.extra);
}
