use array::{ArrayTrait,SpanTrait};

use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::tests::operators::tensor::helpers::{i32_tensor_2x2_helper, i32_tensor_2x2x2_helper};


#[test]
#[available_gas(20000000)]
fn tensor1x3_argmax() {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0_u32, false));
    data.append(IntegerTrait::new(1_u32, false));
    data.append(IntegerTrait::new(2_u32, false));
    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);
    
    let result = tensor.argmax(0);
    assert(*result.data.at(0) == 2, 'result[0] = 2');
    assert(result.data.len() == 1, 'length == 1');
    assert(result.shape.len() == 1, 'result.shape.len() == 1');


    let mut data = ArrayTrait::new();
    data.append(IntegerTrait::new(0_u32, false));
    data.append(IntegerTrait::new(1_u32, true));
    data.append(IntegerTrait::new(2_u32, true));
    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<i32>::new(sizes.span(), data.span(), extra);
    
    let result = tensor.argmax(0);
    assert(*result.data.at(0) == 0, 'result[0] = 0');
    assert(result.data.len() == 1, 'length == 1');
    assert(result.shape.len() == 1, 'result.shape.len() == 1');
}


#[test]
#[available_gas(20000000)]
fn tensor_argmax() {
    let tensor = i32_tensor_2x2_helper();

    let result = tensor.argmax(0);
    assert(*result.data.at(0) == 1, 'result[0] = 1');
    assert(*result.data.at(1) == 1, 'result[1] = 1');
    assert(result.data.len() == 2, 'length == 2');

    let result = tensor.argmax(1);

    assert(*result.data.at(0) == 1, 'result[0] = 1');
    assert(*result.data.at(1) == 1, 'result[1] = 1');
    assert(result.data.len() == 2, 'length == 2');

    let tensor = i32_tensor_2x2x2_helper();

    let result = tensor.argmax(0).data;

    assert(*result.at(0) == 1, 'result[0] = 1');
    assert(*result.at(1) == 1, 'result[1] = 1');
    assert(*result.at(2) == 1, 'result[2] = 1');
    assert(*result.at(3) == 1, 'result[3] = 1');
    assert(result.len() == 4, 'length == 4');

    let result = tensor.argmax(1).data;

    assert(*result.at(0) == 1, 'result[0] = 1');
    assert(*result.at(1) == 1, 'result[1] = 1');
    assert(*result.at(2) == 1, 'result[2] = 1');
    assert(*result.at(3) == 1, 'result[3] = 1');
    assert(result.len() == 4, 'length == 4');

    let result = tensor.argmax(2).data;

    assert(*result.at(0) == 1, 'result[0] = 1');
    assert(*result.at(1) == 1, 'result[1] = 1');
    assert(*result.at(2) == 1, 'result[2] = 1');
    assert(*result.at(3) == 1, 'result[3] = 1');
    assert(result.len() == 4, 'length == 4');
}
