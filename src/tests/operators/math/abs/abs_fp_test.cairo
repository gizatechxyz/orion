use array::SpanTrait;
use array::ArrayTrait;
use orion::operators::tensor::implementations::impl_tensor_fp;
use orion::operators::tensor::core::TensorTrait;
use orion::numbers::fixed_point::types::{Fixed,FixedType};



#[test]
#[available_gas(2000000)]
fn tensor_abs_fp() {
    let mut sizes = ArrayTrait::new();
    sizes.append(3);
    sizes.append(3);

    let mut arr = ArrayTrait::<FixedType>::new();
    arr.append(Fixed::new(0,true));
    arr.append(Fixed::new(1,true));
    arr.append(Fixed::new(2,false));
    arr.append(Fixed::new(3,false));
    arr.append(Fixed::new(4,true));
    arr.append(Fixed::new(5,false));
    arr.append(Fixed::new(6,false));
    arr.append(Fixed::new(7,true));
    arr.append(Fixed::new(8,false));

    let tensor = TensorTrait::<FixedType>::new(sizes.span(), arr.span());

    let result = tensor.abs();
    assert(*result.data.at(0) == Fixed::new(0,false), 'result[0] = 0');
    assert(*result.data.at(1) == Fixed::new(1,false), 'result[1] = 1');
    assert(*result.data.at(2) == Fixed::new(2,false), 'result[2] = 2');
    assert(*result.data.at(3) == Fixed::new(3,false), 'result[3] = 3');
    assert(*result.data.at(4) == Fixed::new(4,false), 'result[4] = 4');
    assert(*result.data.at(5) == Fixed::new(5,false), 'result[5] = 5');
    assert(*result.data.at(6) == Fixed::new(6,false), 'result[6] = 6');
    assert(*result.data.at(7) == Fixed::new(7,false), 'result[7] = 7');
    assert(*result.data.at(8) == Fixed::new(8,false), 'result[8] = 8');
    

    assert(result.data.len() == tensor.data.len(), 'tensor length mismatch');
}

