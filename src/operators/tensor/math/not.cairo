use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::{tensor_bool::BoolTensor};

// Cf TensorTrait::not docstring
fn not(mut z: Tensor<bool>) -> Tensor<bool> {
    let mut data_result: Array<bool> = array![];

    loop {
        match z.data.pop_front() {
            Option::Some(item) => { data_result.append((!*item)); },
            Option::None => { break; }
        };
    };

    TensorTrait::new(z.shape, data_result.span())
}
