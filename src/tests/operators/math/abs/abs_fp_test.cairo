use array::SpanTrait;
use array::ArrayTrait;
use option::OptionTrait;

use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::operators::tensor::core::{TensorTrait, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::numbers::fixed_point::implementations::impl_8x23::{FP8x23Impl, FP8x23PartialEq};


#[test]
#[available_gas(2000000)]
fn tensor_abs_fp() {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);

    let mut arr = ArrayTrait::<FixedType>::new();
    arr.append(FixedTrait::new(0, true));
    arr.append(FixedTrait::new(1, true));
    arr.append(FixedTrait::new(2, false));
    arr.append(FixedTrait::new(3, false));
    arr.append(FixedTrait::new(4, true));
    arr.append(FixedTrait::new(5, false));
    arr.append(FixedTrait::new(6, false));
    arr.append(FixedTrait::new(7, true));
    arr.append(FixedTrait::new(8, false));

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), arr.span(), Option::Some(extra));

    let result = tensor.abs();
    assert(*result.data.at(0) == FixedTrait::new(0, false), 'result[0] = 0');
    assert(*result.data.at(1) == FixedTrait::new(1, false), 'result[1] = 1');
    assert(*result.data.at(2) == FixedTrait::new(2, false), 'result[2] = 2');
    assert(*result.data.at(3) == FixedTrait::new(3, false), 'result[3] = 3');
    assert(*result.data.at(4) == FixedTrait::new(4, false), 'result[4] = 4');
    assert(*result.data.at(5) == FixedTrait::new(5, false), 'result[5] = 5');
    assert(*result.data.at(6) == FixedTrait::new(6, false), 'result[6] = 6');
    assert(*result.data.at(7) == FixedTrait::new(7, false), 'result[7] = 7');
    assert(*result.data.at(8) == FixedTrait::new(8, false), 'result[8] = 8');

    assert(result.data.len() == tensor.data.len(), 'tensor length mismatch');
}

