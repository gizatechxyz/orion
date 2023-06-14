use array::{ArrayTrait,SpanTrait};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::operators::tensor::implementations::impl_tensor_fp;
use orion::numbers::fixed_point::implementations::impl_8x23;

use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};


#[test]
#[available_gas(20000000)]
fn tensor1x3_argmin_fp() {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(1, false));
    data.append(FixedTrait::new(2, false));
    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);
    
    let result = tensor.argmin(0,Option::None(()),Option::None(()));
    assert(*result.data.at(0) == 0, 'result[0] = 0');
    assert(result.data.len() == 1, 'length == 1');
    assert(result.shape.len() == 1, 'result.shape.len() == 1');


    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(1, false));
    data.append(FixedTrait::new(2, true));
    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);
    
    let result = tensor.argmin(0,Option::None(()),Option::None(()));
    assert(*result.data.at(0) == 2, 'result[0] = 2');
    assert(result.data.len() == 1, 'length == 1');
    assert(result.shape.len() == 1, 'result.shape.len() == 1');
}


#[test]
#[available_gas(20000000)]
fn tensor2x2_argmin_fp() {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::<FixedType>::new();
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(1, false));
    data.append(FixedTrait::new(2, false));
    data.append(FixedTrait::new(3, false));
    let extra = Option::<ExtraParams>::None(());
    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);
    

    let result = tensor.argmin(0,Option::None(()),Option::None(()));
    assert(*result.data.at(0) == 0, 'result[0] = 0');
    assert(*result.data.at(1) == 0, 'result[1] = 0');
    assert(result.data.len() == 2, 'length == 2');

    let result = tensor.argmin(1,Option::None(()),Option::None(()));
    assert(*result.data.at(0) == 0, 'result[0] = 0');
    assert(*result.data.at(1) == 0, 'result[1] = 0');
    assert(result.data.len() == 2, 'length == 2');

    let result = tensor.argmin(1,Option::Some(false),Option::Some(false));
    assert(*result.data.at(0) == 0, 'result[0] = 0');
    assert(*result.data.at(1) == 0, 'result[1] = 0');
    assert(result.data.len() == 2, 'length == 2');
}


#[test]
#[available_gas(20000000)]
fn tensor2x2x2_argmin_fp() {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::<FixedType>::new();
    data.append(FixedTrait::new(0, false));
    data.append(FixedTrait::new(1, false));
    data.append(FixedTrait::new(2, false));
    data.append(FixedTrait::new(3, false));
    data.append(FixedTrait::new(4, false));
    data.append(FixedTrait::new(5, false));
    data.append(FixedTrait::new(6, false));
    data.append(FixedTrait::new(7, false));
    let extra = Option::<ExtraParams>::None(());
    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);
    

    let result = tensor.argmin(0,Option::None(()),Option::None(())).data;

    assert(*result.at(0) == 0, 'result[0] = 0');
    assert(*result.at(1) == 0, 'result[1] = 0');
    assert(*result.at(2) == 0, 'result[2] = 0');
    assert(*result.at(3) == 0, 'result[3] = 0');
    assert(result.len() == 4, 'length == 4');

    let result = tensor.argmin(1,Option::None(()),Option::None(())).data;

    assert(*result.at(0) == 0, 'result[0] = 0');
    assert(*result.at(1) == 0, 'result[1] = 0');
    assert(*result.at(2) == 0, 'result[2] = 0');
    assert(*result.at(3) == 0, 'result[3] = 0');
    assert(result.len() == 4, 'length == 4');

    let result = tensor.argmin(2,Option::None(()),Option::None(())).data;

    assert(*result.at(0) == 0, 'result[0] = 0');
    assert(*result.at(1) == 0, 'result[1] = 0');
    assert(*result.at(2) == 0, 'result[2] = 0');
    assert(*result.at(3) == 0, 'result[3] = 0');
    assert(result.len() == 4, 'length == 4');

}

#[test]
#[available_gas(20000000)]
fn tensor_argmin_select_last_index_fp(){
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new(2, false));
    data.append(FixedTrait::new(2, false));
    data.append(FixedTrait::new(3, false));
    data.append(FixedTrait::new(4, false));
    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);
    
    let result = tensor.argmin(1,Option::None(()),Option::None(()));
    assert(*result.data.at(0) == 0, 'result[0] = 0');
    assert(*result.data.at(1) == 0, 'result[1] = 0');
    assert(result.data.len() == 2, 'length == 2');

    let result = tensor.argmin(1,Option::Some(false),Option::Some(true));
    assert(*result.data.at(0) == 1, 'result[0] = 1');
    assert(*result.data.at(1) == 0, 'result[1] = 0');
    assert(result.data.len() == 2, 'length == 2');

}

#[test]
#[available_gas(20000000)]
fn tensor_argmin_keepdims_fp(){
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedTrait::new(2, false));
    data.append(FixedTrait::new(2, false));
    data.append(FixedTrait::new(3, false));
    data.append(FixedTrait::new(4, false));
    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), data.span(), extra);
    
    let result = tensor.argmin(1,Option::Some(false),Option::None(()));
    assert(*result.data.at(0) == 0, 'result[0] = 0');
    assert(*result.data.at(1) == 0, 'result[1] = 0');
    assert(result.data.len() == 2, 'length == 2');
    assert(result.shape.len() == 1, 'result.shape.len() == 1');


    let result = tensor.argmin(1,Option::Some(true), Option::Some(true));
    assert(*result.data.at(0) == 1, 'result[0] = 1');
    assert(*result.data.at(1) == 0, 'result[1] = 0');
    assert(result.data.len() == 2, 'length == 2');
    assert(result.shape.len() == 2, 'result.shape.len() == 2');

    let result = tensor.argmin(1,Option::Some(true), Option::Some(false));
    assert(*result.data.at(0) == 0, 'result[0] = 1');
    assert(*result.data.at(1) == 0, 'result[1] = 0');
    assert(result.data.len() == 2, 'length == 2');
    assert(result.shape.len() == 2, 'result.shape.len() == 2');
}
