use array::SpanTrait;
use array::ArrayTrait;
use option::OptionTrait;
use orion::operators::tensor::implementations::impl_tensor_fp;
use orion::operators::tensor::core::{TensorTrait, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::numbers::fixed_point::implementations::impl_8x23;


#[test]
#[available_gas(2000000)]
fn tensor_ceil_fp() {
    let mut sizes = ArrayTrait::new();
    sizes.append(1);
    sizes.append(4);

    let mut arr = ArrayTrait::<FixedType>::new();
    arr.append(FixedTrait::new(0, false));
    arr.append(FixedTrait::new(30000, false)); // 0.00357627868
    arr.append(FixedTrait::new(100663252, false)); // 11.9999947548
    arr.append(FixedTrait::new(100663252, true)); // - 11.9999947548

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(
        sizes.span(), arr.span(), Option::Some(extra)
    );

    let result = tensor.ceil();
    assert(*result.data.at(0) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
    assert(*result.data.at(1) == FixedTrait::new_unscaled(1, false), 'result[1] = 8388608'); // 1
    assert(*result.data.at(2) == FixedTrait::new_unscaled(12, false),'result[2] = 100663296'); // 12 
    assert(*result.data.at(3) == FixedTrait::new_unscaled(11, true), 'result[3] = -92274688'); // -11 
    assert(result.data.len() == tensor.data.len(), 'tensor length mismatch');
}

