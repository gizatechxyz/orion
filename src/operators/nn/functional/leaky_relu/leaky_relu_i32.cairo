use core::traits::Into;
use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;
use onnx_cairo::numbers::fixed_point::types::{FixedType, Fixed, ONE_u128};
use onnx_cairo::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use onnx_cairo::operators::tensor::implementations::impl_tensor_fp;
use onnx_cairo::operators::tensor::core::{Tensor, TensorTrait};
use onnx_cairo::utils::check_gas;

/// Cf: NNTrait::leaky_relu docstring
fn leaky_relu_i32(z: @Tensor<i32>, alpha: @FixedType, threshold: i32) -> Tensor<FixedType> {
    assert(*alpha.mag < ONE_u128, 'alpha must be less than 1_fp');

    let mut data_result = ArrayTrait::<FixedType>::new();
    let mut data = *z.data;
    loop {
        check_gas();

        if data.len() == 0 {
            break ();
        };

        let current_index = *data.pop_front().unwrap();
        let fp_current_index = Fixed::new_unscaled(current_index.mag.into(), current_index.sign);
        if current_index >= threshold {
            data_result.append(fp_current_index);
        } else {
            data_result.append(fp_current_index * *alpha);
        };
    };

    return TensorTrait::<FixedType>::new(*z.shape, data_result.span());
}

