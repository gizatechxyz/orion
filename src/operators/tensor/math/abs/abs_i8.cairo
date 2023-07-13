use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};
use orion::operators::tensor::implementations::impl_tensor_i8::Tensor_i8;
use orion::operators::tensor::core::{Tensor, TensorTrait};


/// Cf: TensorTrait::abs docstring
fn abs(z: @Tensor<i8>) -> Tensor<i8> {
    let mut data_result = ArrayTrait::<i8>::new();
    let mut data = *z.data;
    loop {
        if data.len() == 0 {
            break ();
        };

        let current_index = *data.pop_front().unwrap();
        data_result.append(current_index.abs());
    };

    return TensorTrait::<i8>::new(*z.shape, data_result.span(), *z.extra);
}
