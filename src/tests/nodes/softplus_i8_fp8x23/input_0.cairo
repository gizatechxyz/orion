use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor};

use orion::operators::tensor::implementations::tensor_i8_fp8x23::Tensor_i8_fp8x23;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 2, sign: false });
    data.append(i8 { mag: 3, sign: true });
    data.append(i8 { mag: 1, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}