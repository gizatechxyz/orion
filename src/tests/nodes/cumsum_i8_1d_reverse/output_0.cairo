use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

use orion::operators::tensor::I8Tensor;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 15, sign: false });
    data.append(i8 { mag: 14, sign: false });
    data.append(i8 { mag: 12, sign: false });
    data.append(i8 { mag: 9, sign: false });
    data.append(i8 { mag: 5, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}