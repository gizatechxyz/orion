use array::SpanTrait;
use array::{ArrayTrait};
use orion::operators::tensor::implementations::impl_tensor_fp;
use orion::operators::tensor::core::{TensorTrait, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::numbers::fixed_point::implementations::impl_8x23;

#[test]
#[available_gas(2000000000)]
fn sinh_fp_test() {
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

    let result = tensor.sinh().data;

    assert((*result.at(0).mag) == 30424303, 'result[0] = 3.6269');
    assert(*result.at(0).sign == true, 'result[0] = true');
    assert((*result.at(1).mag) == 9858301, 'result[1] = 1.1752');
    assert(*result.at(1).sign == true, 'result[1] = true');
    assert((*result.at(2).mag) == 0, 'result[2] = 0');
    assert(*result.at(3).sign == false, 'result[3] = false');
    assert((*result.at(3).mag) == 9858301, 'result[3] = 1.1752');
    assert(*result.at(4).sign == false, 'result[4] = false');
    assert((*result.at(4).mag) == 30424303, 'result[4] = 3.6269');

}
