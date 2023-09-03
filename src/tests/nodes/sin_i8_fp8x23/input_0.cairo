use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

use orion::operators::tensor::I8Tensor;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 4, sign: true });
    data.append(i8 { mag: 6, sign: true });
    data.append(i8 { mag: 2, sign: false });
    data.append(i8 { mag: 3, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}