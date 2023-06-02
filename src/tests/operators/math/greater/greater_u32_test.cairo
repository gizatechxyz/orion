use array::SpanTrait;
use array::{ArrayTrait};
use orion::operators::tensor::implementations::impl_tensor_u32;
use orion::operators::tensor::core::TensorTrait;

#[test]
#[available_gas(2000000)]
fn tensor_greater_u32() {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);

    let mut arr_1 = ArrayTrait::<u32>::new();
    arr_1.append(0_u32);
    arr_1.append(1_u32);
    arr_1.append(2_u32);
    arr_1.append(3_u32);
    arr_1.append(4_u32);
    arr_1.append(7_u32);
    arr_1.append(6_u32);
    arr_1.append(7_u32);
    arr_1.append(8_u32);
    

    let mut arr_2 = ArrayTrait::<u32>::new();
    arr_2.append(10_u32);
    arr_2.append(11_u32);
    arr_2.append(12_u32);
    arr_2.append(13_u32);
    arr_2.append(4_u32);
    arr_2.append(5_u32);
    arr_2.append(16_u32);
    arr_2.append(17_u32);
    arr_2.append(18_u32);

    
    
    let tensor_a = TensorTrait::<u32>::new(sizes.span(), arr_1.span());
    let tensor_b = TensorTrait::<u32>::new(sizes.span(), arr_2.span());

    let result_a = tensor_a.greater(@tensor_b);
    assert(*result_a.data.at(0) == 0, 'result[0] = 0');
    assert(*result_a.data.at(1) == 0, 'result[1] = 0');
    assert(*result_a.data.at(2) == 0, 'result[2] = 0');
    assert(*result_a.data.at(3) == 0, 'result[3] = 0');
    assert(*result_a.data.at(4) == 0, 'result[4] = 0');
    assert(*result_a.data.at(5) == 1, 'result[5] = 1');
    assert(*result_a.data.at(6) == 0, 'result[6] = 0');
    assert(*result_a.data.at(7) == 0, 'result[7] = 0');
    assert(*result_a.data.at(8) == 0, 'result[8] = 0');

    assert(result_a.data.len() == tensor_a.data.len(), 'tensor length mismatch');

    let result_b = tensor_b.greater(@tensor_a);
    assert(*result_b.data.at(0) == 1, 'result[0] = 1');
    assert(*result_b.data.at(1) == 1, 'result[1] = 1');
    assert(*result_b.data.at(2) == 1, 'result[2] = 1');
    assert(*result_b.data.at(3) == 1, 'result[3] = 1');
    assert(*result_b.data.at(4) == 0, 'result[4] = 0');
    assert(*result_b.data.at(5) == 0, 'result[5] = 0');
    assert(*result_b.data.at(6) == 1, 'result[6] = 1');
    assert(*result_b.data.at(7) == 1, 'result[7] = 1');
    assert(*result_b.data.at(8) == 1, 'result[8] = 1');

    assert(result_b.data.len() == tensor_a.data.len(), 'tensor length mismatch');
}

#[test]
#[available_gas(2000000)]
fn tensor_greater_u32_broadcast() {
    let mut sizes_1 = ArrayTrait::new();
    sizes_1.append(4);
    sizes_1.append(3);

    let mut sizes_2 = ArrayTrait::new();
    sizes_2.append(1);
    sizes_2.append(3);

    let mut arr_1 = ArrayTrait::<u32>::new();
    arr_1.append(0_u32);
    arr_1.append(1_u32);
    arr_1.append(2_u32);
    arr_1.append(3_u32);
    arr_1.append(4_u32);
    arr_1.append(5_u32);
    arr_1.append(6_u32);
    arr_1.append(7_u32);
    arr_1.append(8_u32);
    arr_1.append(9_u32);
    arr_1.append(10_u32);
    arr_1.append(11_u32);

    let mut arr_2 = ArrayTrait::<u32>::new();
    arr_2.append(0_u32);
    arr_2.append(1_u32);
    arr_2.append(2_u32);
    
    
    let tensor_a = TensorTrait::<u32>::new(sizes_1.span(), arr_1.span());
    let tensor_b = TensorTrait::<u32>::new(sizes_2.span(), arr_2.span());

    let result_a = tensor_b.greater(@tensor_a);
    assert(*result_a.data.at(0) == 0, 'result[0] = 0');
    assert(*result_a.data.at(1) == 0, 'result[1] = 0');
    assert(*result_a.data.at(2) == 0, 'result[2] = 0');
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

    let result_b = tensor_a.greater(@tensor_b);
    assert(*result_b.data.at(0) == 0, 'result[0] = 0');
    assert(*result_b.data.at(1) == 0, 'result[1] = 0');
    assert(*result_b.data.at(2) == 0, 'result[2] = 0');
    assert(*result_b.data.at(3) == 1, 'result[3] = 1');
    assert(*result_b.data.at(4) == 1, 'result[4] = 1');
    assert(*result_b.data.at(5) == 1, 'result[5] = 1');
    assert(*result_b.data.at(6) == 1, 'result[6] = 1');
    assert(*result_b.data.at(7) == 1, 'result[7] = 1');
    assert(*result_b.data.at(8) == 1, 'result[8] = 1');
    assert(*result_b.data.at(9) == 1, 'result[9] = 1');
    assert(*result_b.data.at(10) == 1, 'result[10] = 1');
    assert(*result_b.data.at(11) == 1, 'result[11] = 1');

    assert(result_b.data.len() == tensor_a.data.len(), 'tensor length mismatch');
}
