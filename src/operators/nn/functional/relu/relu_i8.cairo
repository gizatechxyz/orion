use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_i8::Tensor_i8;


/// Cf: NNTrait::relu docstring
fn relu_i8(mut z: Tensor<i8>) -> Tensor<i8> {
    let mut data_result = ArrayTrait::<i8>::new();

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

    return TensorTrait::<i8>::new(z.shape, data_result.span(), z.extra);
}

