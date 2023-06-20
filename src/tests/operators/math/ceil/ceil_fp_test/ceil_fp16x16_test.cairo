use array::SpanTrait;
use array::ArrayTrait;
use option::OptionTrait;
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::operators::tensor::core::{TensorTrait, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::numbers::fixed_point::implementations::impl_16x16::{FP16x16Impl, FP16x16PartialEq};

// ===== 1D ===== //

#[test]
#[available_gas(2000000)]
fn tensor_ceil_1D() {
    let mut sizes = ArrayTrait::new();
    sizes.append(4);
    let mut arr = ArrayTrait::<FixedType>::new();
    arr.append(FixedTrait::new(0, false));
    arr.append(FixedTrait::new(234, false)); // 0.00357627868
    arr.append(FixedTrait::new(786431, false)); // 11.9999947548
    arr.append(FixedTrait::new(786431, true)); // - 11.9999947548
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
    let tensor = TensorTrait::<FixedType>::new(sizes.span(), arr.span(), Option::Some(extra));

    let result = tensor.ceil();
    assert(*result.data.at(0) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
    assert(*result.data.at(1) == FixedTrait::new_unscaled(1, false), 'result[1] = 1'); // 1
    assert(*result.data.at(2) == FixedTrait::new_unscaled(12, false), 'result[2] = 12'); // 12 
    assert(*result.data.at(3) == FixedTrait::new_unscaled(11, true), 'result[3] = -11'); // -11 
    assert(result.data.len() == tensor.data.len(), 'tensor length mismatch');
}

// ===== 2D ===== //

#[test]
#[available_gas(2000000)]
fn tensor_ceil_2D() {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);
    let mut arr = ArrayTrait::<FixedType>::new();
    arr.append(FixedTrait::new(0, false));
    arr.append(FixedTrait::new(234, false)); // 0.00357627868
    arr.append(FixedTrait::new(786431, false)); // 11.9999947548
    arr.append(FixedTrait::new(786431, true)); // - 11.9999947548
    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
    let tensor = TensorTrait::<FixedType>::new(sizes.span(), arr.span(), Option::Some(extra));

    let result = tensor.ceil();
    assert(*result.data.at(0) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
    assert(*result.data.at(1) == FixedTrait::new_unscaled(1, false), 'result[1] = 1'); // 1
    assert(*result.data.at(2) == FixedTrait::new_unscaled(12, false), 'result[2] = 12'); // 12 
    assert(*result.data.at(3) == FixedTrait::new_unscaled(11, true), 'result[3] = -11'); // -11 
    assert(result.data.len() == tensor.data.len(), 'tensor length mismatch');
}

// ===== 3D ===== //

#[test]
#[available_gas(2000000)]
fn tensor_ceil_3D() {
    let mut sizes = ArrayTrait::new();
    sizes.append(2);
    sizes.append(2);
    sizes.append(2);

    let mut arr = ArrayTrait::<FixedType>::new();
    arr.append(FixedTrait::new(0, false));
    arr.append(FixedTrait::new(234, false)); // 0.00357627868
    arr.append(FixedTrait::new(786431, false)); // 11.9999947548
    arr.append(FixedTrait::new(786431, true)); // - 11.9999947548
    arr.append(FixedTrait::new(32768, false)); // 0.5
    arr.append(FixedTrait::new(32768, true)); // - 0.5
    arr.append(FixedTrait::new(98304, false)); //  1.5
    arr.append(FixedTrait::new(98304, true)); // - 1.5

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16(())) };
    let tensor = TensorTrait::<FixedType>::new(sizes.span(), arr.span(), Option::Some(extra));

    let result = tensor.ceil();
    assert(*result.data.at(0) == FixedTrait::new_unscaled(0, false), 'result[0] = 0');
    assert(*result.data.at(1) == FixedTrait::new_unscaled(1, false), 'result[1] = 1');
    assert(*result.data.at(2) == FixedTrait::new_unscaled(12, false), 'result[2] = 12');
    assert(*result.data.at(3) == FixedTrait::new_unscaled(11, true), 'result[3] = -11');
    assert(*result.data.at(4) == FixedTrait::new_unscaled(1, false), 'result[4] = 1');
    assert(*result.data.at(5) == FixedTrait::new_unscaled(0, false), 'result[5] = 0');
    assert(*result.data.at(6) == FixedTrait::new_unscaled(2, false), 'result[6] = 2');
    assert(*result.data.at(7) == FixedTrait::new_unscaled(1, true), 'result[7] = -1');
    assert(result.data.len() == tensor.data.len(), 'tensor length mismatch');
}

