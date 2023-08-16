use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_i8::Tensor_i8;


/// Cf: NNTrait::relu docstring
fn relu_i8(z: @Tensor<i8>) -> Tensor<i8> {
    let mut data_result = ArrayTrait::<i8>::new();
    let mut data = *z.data;

    loop {
        if data.len() == 0 {
            break ();
        };

        let current_index = *data.pop_front().unwrap();
        if current_index < IntegerTrait::new(0, false) {
            data_result.append(IntegerTrait::new(0, false));
        } else {
            data_result.append(current_index);
        };
    };

    return TensorTrait::<i8>::new(*z.shape, data_result.span(), *z.extra);
}
