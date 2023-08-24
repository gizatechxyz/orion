use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;


/// Cf: NNTrait::relu docstring
fn relu_i32(mut z: Tensor<i32>) -> Tensor<i32> {
    let mut data_result = ArrayTrait::<i32>::new();

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                if (*item) < IntegerTrait::new(0, false) {
                    data_result.append(IntegerTrait::new(0, false));
                } else {
                    data_result.append(*item);
                };
            },
            Option::None(_) => {
                break;
            }
        };
    };

    return TensorTrait::<i32>::new(z.shape, data_result.span(), z.extra);
}
