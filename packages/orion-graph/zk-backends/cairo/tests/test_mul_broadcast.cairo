use core::array::{ArrayTrait, SpanTrait};
use orion_cairo::tensors::Tensor;
use orion_cairo::primops::PrimopsTrait;
use orion_cairo::numbers::f16x16::f16x16;
use orion_cairo::helpers::broadcast_shape;

#[test]
#[available_gas(2000000000)]
fn test_mul_fp16x16_broadcast() {
    let input_0 = input_0();
    let input_1 = input_1();

    let broadcasted_shape = broadcast_shape(input_0.shape, input_1.shape);

    let z = output_0();
    let y = input_0.mul(@input_1, broadcasted_shape);

    assert(z == y, 'should be equal');
}


fn input_0() -> Tensor {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(-196608);
    data.append(-196608);
    data.append(-131072);
    data.append(65536);
    data.append(-196608);
    data.append(65536);
    data.append(131072);
    data.append(0);
    data.append(65536);
    data.append(65536);
    data.append(0);
    data.append(65536);
    data.append(65536);
    data.append(-196608);
    data.append(-131072);
    data.append(-196608);
    data.append(-196608);
    data.append(-196608);
    data.append(-65536);
    data.append(-65536);
    data.append(-131072);
    data.append(-131072);
    data.append(-65536);
    data.append(131072);
    data.append(-131072);
    data.append(0);
    data.append(-131072);
    Tensor { shape: shape.span(), data: data.span() }
}

fn input_1() -> Tensor {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(-131072);
    data.append(65536);
    data.append(-196608);
    Tensor { shape: shape.span(), data: data.span() }
}

fn output_0() -> Tensor {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(393216);
    data.append(393216);
    data.append(262144);
    data.append(65536);
    data.append(-196608);
    data.append(65536);
    data.append(-393216);
    data.append(0);
    data.append(-196608);
    data.append(-131072);
    data.append(0);
    data.append(-131072);
    data.append(65536);
    data.append(-196608);
    data.append(-131072);
    data.append(589824);
    data.append(589824);
    data.append(589824);
    data.append(131072);
    data.append(131072);
    data.append(262144);
    data.append(-131072);
    data.append(-65536);
    data.append(131072);
    data.append(393216);
    data.append(0);
    data.append(393216);
    Tensor { shape: shape.span(), data: data.span() }
}
