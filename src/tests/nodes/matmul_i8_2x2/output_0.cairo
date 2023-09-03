use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor};

use orion::operators::tensor::I8Tensor;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 12, sign: false });
    data.append(i8 { mag: 18, sign: false });
    data.append(i8 { mag: 14, sign: true });
    data.append(i8 { mag: 20, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}