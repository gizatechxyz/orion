use array::SpanTrait;
use array::{ArrayTrait};
use orion::operators::tensor::implementations::impl_tensor_fp8x23;
use orion::operators::tensor::core::TensorTrait;
use orion::numbers::fixed_point::core::{FixedTrait,FixedType};
use orion::numbers::fixed_point::implementations::impl_8x23;

#[test]
#[available_gas(2000000)]
fn tensor_eq_fp() {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);

    let mut arr_1 = ArrayTrait::<FixedType>::new();
    arr_1.append(FixedTrait::new(0,false));
    arr_1.append(FixedTrait::new(1,false));
    arr_1.append(FixedTrait::new(2,false));
    arr_1.append(FixedTrait::new(3,false));
    arr_1.append(FixedTrait::new(4,false));
    arr_1.append(FixedTrait::new(5,false));
    arr_1.append(FixedTrait::new(6,false));
    arr_1.append(FixedTrait::new(7,false));
    arr_1.append(FixedTrait::new(8,false));

    
    let mut arr_2 = ArrayTrait::<FixedType>::new();
    arr_2.append(FixedTrait::new(10,false));
    arr_2.append(FixedTrait::new(11,false));
    arr_2.append(FixedTrait::new(2,true));
    arr_2.append(FixedTrait::new(3,true));
    arr_2.append(FixedTrait::new(4,false));
    arr_2.append(FixedTrait::new(5,false));
    arr_2.append(FixedTrait::new(16,false));
    arr_2.append(FixedTrait::new(17,false));
    arr_2.append(FixedTrait::new(18,false));


    let tensor_a = TensorTrait::<FixedType>::new(sizes.span(), arr_1.span());
    let tensor_b = TensorTrait::<FixedType>::new(sizes.span(), arr_2.span());

    let result = tensor_a.eq(@tensor_b);
    assert(*result.data.at(0) == 0, 'result[0] = 0');
    assert(*result.data.at(1) == 0, 'result[1] = 0');
    assert(*result.data.at(2) == 0, 'result[2] = 0');
    assert(*result.data.at(3) == 0, 'result[3] = 0');
    assert(*result.data.at(4) == 1, 'result[4] = 1');
    assert(*result.data.at(5) == 1, 'result[5] = 1');
    assert(*result.data.at(6) == 0, 'result[6] = 0');
    assert(*result.data.at(7) == 0, 'result[7] = 0');
    assert(*result.data.at(8) == 0, 'result[8] = 0');

    assert(result.data.len() == tensor_a.data.len(), 'tensor length mismatch');
}

#[test]
#[available_gas(2000000)]
fn tensor_eq_fp_broadcast() {
    let mut sizes_1 = ArrayTrait::new();
    sizes_1.append(4);
    sizes_1.append(3);

    let mut sizes_2 = ArrayTrait::new();
    sizes_2.append(1);
    sizes_2.append(3);

    let mut arr_1 = ArrayTrait::<FixedType>::new();
    arr_1.append(FixedTrait::new(0,false));
    arr_1.append(FixedTrait::new(1,false));
    arr_1.append(FixedTrait::new(2,false));
    arr_1.append(FixedTrait::new(3,false));
    arr_1.append(FixedTrait::new(4,false));
    arr_1.append(FixedTrait::new(5,false));
    arr_1.append(FixedTrait::new(6,false));
    arr_1.append(FixedTrait::new(7,false));
    arr_1.append(FixedTrait::new(8,false));
    arr_1.append(FixedTrait::new(9,false));
    arr_1.append(FixedTrait::new(10,false));
    arr_1.append(FixedTrait::new(11,false));

    let mut arr_2 = ArrayTrait::<FixedType>::new();
    arr_2.append(FixedTrait::new(0,false));
    arr_2.append(FixedTrait::new(1,false));
    arr_2.append(FixedTrait::new(2,false));
    
    let tensor_a = TensorTrait::<FixedType>::new(sizes_1.span(), arr_1.span());
    let tensor_b = TensorTrait::<FixedType>::new(sizes_2.span(), arr_2.span());

    let result_a = tensor_b.eq(@tensor_a);
    assert(*result_a.data.at(0) == 1, 'result[0] = 1');
    assert(*result_a.data.at(1) == 1, 'result[1] = 1');
    assert(*result_a.data.at(2) == 1, 'result[2] = 1');
    assert(*result_a.data.at(3) == 0, 'result[3] = 0');
    assert(*result_a.data.at(4) == 0, 'result[4] = 0');
    assert(*result_a.data.at(5) == 0, 'result[5] = 0');
    assert(*result_a.data.at(6) == 0, 'result[6] = 0');
    assert(*result_a.data.at(7) == 0, 'result[7] = 0');
    assert(*result_a.data.at(8) == 0, 'result[8] = 0');
    assert(*result_a.data.at(9) == 0, 'result[9] = 0');
    assert(*result_a.data.at(10) == 0, 'result[10] = 0');
    assert(*result_a.data.at(11) == 0, 'result[11] = 0');
    assert(result_a.data.len() == tensor_a.data.len(), 'tensor length mismatch');

    let result_b = tensor_a.eq(@tensor_b);
    assert(*result_b.data.at(0) == 1, 'result[0] = 1');
    assert(*result_b.data.at(1) == 1, 'result[1] = 1');
    assert(*result_b.data.at(2) == 1, 'result[2] = 1');
    assert(*result_b.data.at(3) == 0, 'result[3] = 0');
    assert(*result_b.data.at(4) == 0, 'result[4] = 0');
    assert(*result_b.data.at(5) == 0, 'result[5] = 0');
    assert(*result_b.data.at(6) == 0, 'result[6] = 0');
    assert(*result_b.data.at(7) == 0, 'result[7] = 0');
    assert(*result_b.data.at(8) == 0, 'result[8] = 0');
    assert(*result_b.data.at(9) == 0, 'result[9] = 0');
    assert(*result_b.data.at(10) == 0, 'result[10] = 0');
    assert(*result_b.data.at(11) == 0, 'result[11] = 0');
    assert(result_b.data.len() == tensor_a.data.len(), 'tensor length mismatch');
}
