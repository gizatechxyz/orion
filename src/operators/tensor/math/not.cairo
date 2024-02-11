use core::array::ArrayTrait;
use core::array::SpanTrait;
use core::option::OptionTrait;

use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::{tensor_bool::BoolTensor};


// Cf TensorTrait::not docstring
fn not(mut z: Tensor<bool>) -> Tensor<bool> {
    let mut data_result = ArrayTrait::<bool>::new();

    loop {
        match z.data.pop_front() {
            Option::Some(item) => { data_result.append((!*item)); },
            Option::None => { break; }
        };
    };

    return TensorTrait::new(z.shape, data_result.span());
}
