use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 10716, sign: true });
    data.append(FP16x16 { mag: 32353, sign: true });
    data.append(FP16x16 { mag: 42375, sign: false });
    data.append(FP16x16 { mag: 138711, sign: false });
    data.append(FP16x16 { mag: 158441, sign: true });
    data.append(FP16x16 { mag: 57718, sign: true });
    data.append(FP16x16 { mag: 28750, sign: true });
    data.append(FP16x16 { mag: 110550, sign: false });
    data.append(FP16x16 { mag: 247834, sign: true });
    data.append(FP16x16 { mag: 56680, sign: true });
    data.append(FP16x16 { mag: 37859, sign: false });
    data.append(FP16x16 { mag: 63446, sign: false });
    data.append(FP16x16 { mag: 13827, sign: false });
    data.append(FP16x16 { mag: 80402, sign: false });
    data.append(FP16x16 { mag: 74600, sign: false });
    data.append(FP16x16 { mag: 109615, sign: false });
    data.append(FP16x16 { mag: 20165, sign: false });
    data.append(FP16x16 { mag: 21935, sign: true });
    data.append(FP16x16 { mag: 42407, sign: false });
    data.append(FP16x16 { mag: 151122, sign: true });
    data.append(FP16x16 { mag: 17941, sign: true });
    data.append(FP16x16 { mag: 33712, sign: true });
    data.append(FP16x16 { mag: 31466, sign: true });
    data.append(FP16x16 { mag: 92787, sign: true });
    data.append(FP16x16 { mag: 29459, sign: true });
    data.append(FP16x16 { mag: 15272, sign: false });
    data.append(FP16x16 { mag: 190594, sign: true });
    data.append(FP16x16 { mag: 21712, sign: true });
    data.append(FP16x16 { mag: 199251, sign: false });
    data.append(FP16x16 { mag: 182464, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
