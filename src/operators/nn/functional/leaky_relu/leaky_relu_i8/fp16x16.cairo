use core::traits::Into;
use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;

use orion::numbers::fixed_point::core::{FixedType, FixedTrait};
use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, ONE, FP16x16Mul};

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::operators::tensor::core::{Tensor, TensorTrait};


/// Cf: NNTrait::leaky_relu docstring
fn leaky_relu(z: @Tensor<i8>, alpha: @FixedType, threshold: i8) -> Tensor<FixedType> {
    assert(*alpha.mag < ONE, 'alpha must be less than 1_fp');

    let mut data_result = ArrayTrait::<FixedType>::new();
    let mut data = *z.data;
    loop {
        if data.len() == 0 {
            break ();
        };

        let current_index = *data.pop_front().unwrap();
        let fp_current_index = FixedTrait::new_unscaled(
            current_index.mag.into(), current_index.sign
        );
        if current_index >= threshold {
            data_result.append(fp_current_index);
        } else {
            data_result.append(fp_current_index * *alpha);
        };
    };

    return TensorTrait::new(*z.shape, data_result.span(), *z.extra);
}

