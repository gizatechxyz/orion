use array::SpanTrait;
use array::ArrayTrait;
use orion::operators::tensor::implementations::impl_tensor_u32;
use orion::operators::tensor::core::{TensorTrait, ExtraParams};



#[test]
#[available_gas(2000000)]
fn tensor_ceil_u32() {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);

    let mut arr = ArrayTrait::<u32>::new();
    arr.append(0);
    arr.append(1);
    arr.append(2);
    arr.append(3);
    arr.append(4);
    arr.append(5);
    arr.append(6);
    arr.append(7);
    arr.append(8);

    let extra = Option::<ExtraParams>::None(());

    let tensor = TensorTrait::<u32>::new(sizes.span(), arr.span(), extra);

    let result = tensor.ceil();
    assert(*result.data.at(0) == 0, 'result[0] = 0');
    assert(*result.data.at(1) == 1, 'result[1] = 1');
    assert(*result.data.at(2) == 2, 'result[2] = 2');
    assert(*result.data.at(3) == 3, 'result[3] = 3');
    assert(*result.data.at(4) == 4, 'result[4] = 4');
    assert(*result.data.at(5) == 5, 'result[5] = 5');
    assert(*result.data.at(6) == 6, 'result[6] = 6');
    assert(*result.data.at(7) == 7, 'result[7] = 7');
    assert(*result.data.at(8) == 8, 'result[8] = 8');
    
    assert(result.data.len() == tensor.data.len(), 'tensor length mismatch');
}

