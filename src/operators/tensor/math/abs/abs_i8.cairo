use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};
use orion::operators::tensor::implementations::impl_tensor_i8::Tensor_i8;
use orion::operators::tensor::core::{Tensor, TensorTrait};


/// Cf: TensorTrait::abs docstring
fn abs(mut z: Tensor<i8>) -> Tensor<i8> {
    let mut data_result = ArrayTrait::<i8>::new();
    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                data_result.append((*item).abs());
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::<i8>::new(z.shape, data_result.span(), z.extra);
}
