use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor};

use orion::operators::tensor::implementations::tensor_i8_fp16x16::Tensor_i8_fp16x16;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 29, sign: false });
    data.append(i8 { mag: 35, sign: true });
    data.append(i8 { mag: 76, sign: false });
    data.append(i8 { mag: 117, sign: true });
    data.append(i8 { mag: 3, sign: true });
    data.append(i8 { mag: 104, sign: true });
    data.append(i8 { mag: 57, sign: true });
    data.append(i8 { mag: 61, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}