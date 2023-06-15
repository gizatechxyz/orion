use array::SpanTrait;
use array::{ArrayTrait};
use orion::operators::tensor::implementations::impl_tensor_fp;
use orion::operators::tensor::core::{TensorTrait, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::numbers::fixed_point::implementations::impl_8x23;

#[test]
#[available_gas(2000000000)]
fn asinh_fp_test() {
    let mut sizes = ArrayTrait::new();
    sizes.append(5);

    let mut arr = ArrayTrait::<FixedType>::new();
    arr.append(FixedTrait::new_unscaled(2, true));
    arr.append(FixedTrait::new_unscaled(1, true));
    arr.append(FixedTrait::new_unscaled(0, false));
    arr.append(FixedTrait::new_unscaled(1, false));
    arr.append(FixedTrait::new_unscaled(2, false));

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23(())) };

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), arr.span(), Option::Some(extra));

    let result = tensor.asinh().data;

    assert((*result.at(0).mag) == 12105834, 'result[0] = -1.4436');
    assert(*result.at(0).sign == true, 'result[0] = true');
    assert((*result.at(1).mag) == 7390442, 'result[1] = -0.8814');
    assert(*result.at(1).sign == true, 'result[1] = true');
    assert((*result.at(2).mag) == 0, 'result[2] = 0');
    assert(*result.at(3).sign == false, 'result[3] = false');
    assert((*result.at(3).mag) == 7394022, 'result[3] = 0.8814');
    assert(*result.at(4).sign == false, 'result[4] = false');
    assert((*result.at(4).mag) == 12110330, 'result[4] = 1.4436');

    //Note the discrepancy between asinh(-2) and asin(2) -- 12105834 and 12110330.  
    //The magnitude of asin(2) is more accurate.  The values of the magnitude should be identical...right now asinh(-2)/asin(2) ~ 99.96% 

}