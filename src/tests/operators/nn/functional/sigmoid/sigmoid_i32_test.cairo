use array::ArrayTrait;
use core::debug::PrintTrait;
use array::SpanTrait;
use core::Into;

use orion::operators::tensor::core::{TensorTrait, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::implementations::impl_nn_i32::NN_i32;
use orion::numbers::fixed_point::core::{FixedTrait, FixedImpl};


#[test]
#[available_gas(5000000)]
fn sigmoid_i32_test() {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::<i32>::new();
    let val_1 = IntegerTrait::new(0, false);
    let val_2 = IntegerTrait::new(1, false);
    let val_3 = IntegerTrait::new(2, true);
    let val_4 = IntegerTrait::new(254, false);

    data.append(val_1);
    data.append(val_2);
    data.append(val_3);
    data.append(val_4);

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };

    let mut tensor = TensorTrait::new(shape.span(), data.span(), Option::Some(extra));
    let mut result = NNTrait::sigmoid(@tensor);

    let data = *result.data.at(0).mag;
    assert(data == 32768, 'result[0] == 4194304'); // 0.5

    let data = *result.data.at(1).mag;
    assert(data == 47910, 'result[1] == 47910'); // 0.7310...

    let data = *result.data.at(2).mag;
    assert(data == 7812, 'result[2] == 7812'); // 0.11920...

    let data = *result.data.at(3).mag;
    assert(data == 65536, 'result[3] == 65536'); // 1
}

