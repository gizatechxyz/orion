use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

use orion::operators::tensor::I8Tensor;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 120, sign: false });
    data.append(i8 { mag: 116, sign: false });
    data.append(i8 { mag: 53, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}