use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

use orion::operators::tensor::I8Tensor;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 2, sign: true });
    data.append(i8 { mag: 95, sign: true });
    data.append(i8 { mag: 17, sign: false });
    data.append(i8 { mag: 118, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}