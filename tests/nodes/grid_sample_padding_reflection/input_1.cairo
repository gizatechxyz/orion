use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 655360, sign: true });
    data.append(FP16x16 { mag: 655360, sign: true });
    data.append(FP16x16 { mag: 327680, sign: true });
    data.append(FP16x16 { mag: 327680, sign: true });
    data.append(FP16x16 { mag: 13107, sign: true });
    data.append(FP16x16 { mag: 13107, sign: true });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 13107, sign: true });
    data.append(FP16x16 { mag: 13107, sign: true });
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
