use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor};

use orion::operators::tensor::implementations::tensor_i8_fp16x16::Tensor_i8_fp16x16;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 2, sign: false });
    data.append(i8 { mag: 1, sign: false });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 2, sign: false });
    data.append(i8 { mag: 4, sign: true });
    data.append(i8 { mag: 6, sign: true });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 4, sign: false });
    data.append(i8 { mag: 4, sign: false });
    data.append(i8 { mag: 6, sign: false });
    data.append(i8 { mag: 2, sign: true });
    data.append(i8 { mag: 1, sign: true });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 9, sign: false });
    data.append(i8 { mag: 6, sign: false });
    data.append(i8 { mag: 4, sign: false });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 4, sign: false });
    data.append(i8 { mag: 3, sign: false });
    data.append(i8 { mag: 4, sign: false });
    data.append(i8 { mag: 6, sign: true });
    data.append(i8 { mag: 1, sign: true });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 2, sign: false });
    data.append(i8 { mag: 2, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}