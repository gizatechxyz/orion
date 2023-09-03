use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

use orion::operators::tensor::I8Tensor;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 8, sign: false });
    data.append(i8 { mag: 117, sign: false });
    data.append(i8 { mag: 37, sign: true });
    data.append(i8 { mag: 57, sign: true });
    data.append(i8 { mag: 30, sign: true });
    data.append(i8 { mag: 92, sign: false });
    data.append(i8 { mag: 124, sign: false });
    data.append(i8 { mag: 28, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}