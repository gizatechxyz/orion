use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::operators::tensor::core::{Tensor, TensorTrait};


/// Cf: TensorTrait::abs docstring
fn abs(mut z: Tensor<i32>) -> Tensor<i32> {
    let mut data_result = ArrayTrait::<i32>::new();
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

    return TensorTrait::<i32>::new(z.shape, data_result.span(), z.extra);
}
