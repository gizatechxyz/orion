use array::SpanTrait;
use array::{ArrayTrait};
use orion::operators::tensor::implementations::impl_tensor_fp;
use orion::operators::tensor::core::{TensorTrait, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::numbers::fixed_point::implementations::impl_8x23;
use debug::PrintTrait;

#[test]
#[available_gas(2000000000)]
fn cosh_fp_test() {
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

    let result = tensor.cosh().data;

    // let result1 = *result.at(0);
    // result1.print();
    // let result2 = *result.at(1);
    // result2.print();
    // let result3 = *result.at(2);
    // result3.print();
    // let result4 = *result.at(3);
    // result4.print();
    // let result5 = *result.at(4);
    // result5.print();

    assert((*result.at(0).mag) == 31559577, 'result[0] = 3.7622');
    assert(*result.at(0).sign == false, 'result[0] = false');
    assert((*result.at(1).mag) == 12944297, 'result[1] = 1.5431');
    assert(*result.at(1).sign == false, 'result[1] = false');
    assert((*result.at(2).mag) == 8388608, 'result[2] = 1');
    assert(*result.at(3).sign == false, 'result[3] = false');
    assert((*result.at(3).mag) == 12944297, 'result[3] = 1.5431');
    assert(*result.at(4).sign == false, 'result[4] = false');
    assert((*result.at(4).mag) == 31559577, 'result[4] = 3.7622');

}
