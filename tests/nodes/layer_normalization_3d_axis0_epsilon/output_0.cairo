use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 142574, sign: false });
    data.append(FP16x16 { mag: 81046, sign: false });
    data.append(FP16x16 { mag: 23127, sign: true });
    data.append(FP16x16 { mag: 168203, sign: true });
    data.append(FP16x16 { mag: 63557, sign: true });
    data.append(FP16x16 { mag: 59277, sign: false });
    data.append(FP16x16 { mag: 42308, sign: true });
    data.append(FP16x16 { mag: 44993, sign: true });
    data.append(FP16x16 { mag: 55077, sign: false });
    data.append(FP16x16 { mag: 76461, sign: true });
    data.append(FP16x16 { mag: 108813, sign: false });
    data.append(FP16x16 { mag: 50883, sign: false });
    data.append(FP16x16 { mag: 27205, sign: false });
    data.append(FP16x16 { mag: 7326, sign: false });
    data.append(FP16x16 { mag: 23613, sign: false });
    data.append(FP16x16 { mag: 27157, sign: false });
    data.append(FP16x16 { mag: 5637, sign: true });
    data.append(FP16x16 { mag: 90697, sign: false });
    data.append(FP16x16 { mag: 112821, sign: false });
    data.append(FP16x16 { mag: 91044, sign: true });
    data.append(FP16x16 { mag: 123095, sign: true });
    data.append(FP16x16 { mag: 204883, sign: false });
    data.append(FP16x16 { mag: 9454, sign: false });
    data.append(FP16x16 { mag: 55780, sign: true });
    data.append(FP16x16 { mag: 159695, sign: false });
    data.append(FP16x16 { mag: 6840, sign: true });
    data.append(FP16x16 { mag: 119261, sign: false });
    data.append(FP16x16 { mag: 139557, sign: true });
    data.append(FP16x16 { mag: 38352, sign: false });
    data.append(FP16x16 { mag: 8088, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
