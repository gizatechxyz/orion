use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

use orion::operators::tensor::I8Tensor;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 110, sign: true });
    data.append(i8 { mag: 46, sign: false });
    data.append(i8 { mag: 50, sign: false });
    data.append(i8 { mag: 96, sign: false });
    data.append(i8 { mag: 43, sign: false });
    data.append(i8 { mag: 47, sign: true });
    data.append(i8 { mag: 74, sign: false });
    data.append(i8 { mag: 73, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}