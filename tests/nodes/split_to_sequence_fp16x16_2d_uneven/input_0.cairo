use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(8);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 7995392, sign: true });
    data.append(FP16x16 { mag: 5898240, sign: true });
    data.append(FP16x16 { mag: 5767168, sign: true });
    data.append(FP16x16 { mag: 1572864, sign: false });
    data.append(FP16x16 { mag: 3932160, sign: false });
    data.append(FP16x16 { mag: 1572864, sign: true });
    data.append(FP16x16 { mag: 589824, sign: true });
    data.append(FP16x16 { mag: 5373952, sign: false });
    data.append(FP16x16 { mag: 786432, sign: true });
    data.append(FP16x16 { mag: 7536640, sign: true });
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 5111808, sign: false });
    data.append(FP16x16 { mag: 5898240, sign: true });
    data.append(FP16x16 { mag: 6160384, sign: true });
    data.append(FP16x16 { mag: 2490368, sign: true });
    data.append(FP16x16 { mag: 7208960, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
