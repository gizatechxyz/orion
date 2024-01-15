use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 22549, sign: false });
    data.append(FP16x16 { mag: 29927, sign: false });
    data.append(FP16x16 { mag: 5624, sign: false });
    data.append(FP16x16 { mag: 43955, sign: false });
    data.append(FP16x16 { mag: 38785, sign: false });
    data.append(FP16x16 { mag: 14854, sign: false });
    data.append(FP16x16 { mag: 4727, sign: false });
    data.append(FP16x16 { mag: 24506, sign: false });
    data.append(FP16x16 { mag: 29042, sign: false });
    data.append(FP16x16 { mag: 35461, sign: false });
    data.append(FP16x16 { mag: 53031, sign: false });
    data.append(FP16x16 { mag: 2059, sign: false });
    data.append(FP16x16 { mag: 45485, sign: false });
    data.append(FP16x16 { mag: 54450, sign: false });
    data.append(FP16x16 { mag: 13645, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
