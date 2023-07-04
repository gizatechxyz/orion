use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;


/// Cf: NNTrait::relu docstring
fn relu_i32(z: @Tensor<i32>, threshold: i32) -> Tensor<i32> {
    let mut data_result = ArrayTrait::<i32>::new();
    let mut data = *z.data;

    let zero = IntegerTrait::new(0, false);
    loop {


        if data.len() == 0 {
            break ();
        };

        let current_index = *data.pop_front().unwrap();
        if current_index < threshold {
            data_result.append(zero);
        } else {
            data_result.append(current_index);
        };
    };

    return TensorTrait::<i32>::new(*z.shape, data_result.span(), *z.extra);
}
