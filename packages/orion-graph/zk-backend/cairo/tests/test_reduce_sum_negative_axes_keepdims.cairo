use core::array::{ArrayTrait, SpanTrait};
use orion_cairo::tensors::Tensor;
use orion_cairo::primops::PrimopsTrait;
use orion_cairo::numbers::f16x16::f16x16;


#[test]
#[available_gas(2000000000)]
fn test_reduce_sum_default_axes_keepdims() {
    let input_0 = input_0();
    let z = output_0();

    let y = input_0.reduce_sum(Option::Some(array![-2].span()), Option::Some(true), Option::None);

    assert(z == y, 'should be equal');
}


fn input_0() -> Tensor {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(65536);
    data.append(131072);
    data.append(196608);
    data.append(262144);
    data.append(327680);
    data.append(393216);
    data.append(458752);
    data.append(524288);
    data.append(589824);
    data.append(655360);
    data.append(720896);
    data.append(786432);
    Tensor { shape: shape.span(), data: data.span() }
}

fn output_0() -> Tensor {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(262144);
    data.append(393216);
    data.append(786432);
    data.append(917504);
    data.append(1310720);
    data.append(1441792);
    Tensor { shape: shape.span(), data: data.span() }
}
