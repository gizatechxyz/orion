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
    data.append(i8 { mag: 118, sign: false });
    data.append(i8 { mag: 72, sign: false });
    data.append(i8 { mag: 115, sign: false });
    data.append(i8 { mag: 60, sign: false });
    data.append(i8 { mag: 48, sign: false });
    data.append(i8 { mag: 51, sign: false });
    data.append(i8 { mag: 112, sign: true });
    data.append(i8 { mag: 72, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}