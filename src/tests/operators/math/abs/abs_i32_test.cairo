use array::SpanTrait;
use array::ArrayTrait;
use option::OptionTrait;

use orion::operators::tensor::implementations::impl_tensor_i32;
use orion::operators::tensor::core::{TensorTrait, ExtraParams};
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};


#[test]
#[available_gas(2000000)]
fn tensor_abs_i32() {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);

    let mut arr = ArrayTrait::<i32>::new();
    arr.append(IntegerTrait::new(0, true));
    arr.append(IntegerTrait::new(1, true));
    arr.append(IntegerTrait::new(2, false));
    arr.append(IntegerTrait::new(3, false));
    arr.append(IntegerTrait::new(4, true));
    arr.append(IntegerTrait::new(5, false));
    arr.append(IntegerTrait::new(6, false));
    arr.append(IntegerTrait::new(7, true));
    arr.append(IntegerTrait::new(8, false));

    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<i32>::new(sizes.span(), arr.span(), extra);

    let result = tensor.abs();
    assert(*result.data.at(0) == IntegerTrait::new(0, false), 'result[0] = 0');
    assert(*result.data.at(1) == IntegerTrait::new(1, false), 'result[1] = 1');
    assert(*result.data.at(2) == IntegerTrait::new(2, false), 'result[2] = 2');
    assert(*result.data.at(3) == IntegerTrait::new(3, false), 'result[3] = 3');
    assert(*result.data.at(4) == IntegerTrait::new(4, false), 'result[4] = 4');
    assert(*result.data.at(5) == IntegerTrait::new(5, false), 'result[5] = 5');
    assert(*result.data.at(6) == IntegerTrait::new(6, false), 'result[6] = 6');
    assert(*result.data.at(7) == IntegerTrait::new(7, false), 'result[7] = 7');
    assert(*result.data.at(8) == IntegerTrait::new(8, false), 'result[8] = 8');

    assert(result.data.len() == tensor.data.len(), 'tensor length mismatch');
}

