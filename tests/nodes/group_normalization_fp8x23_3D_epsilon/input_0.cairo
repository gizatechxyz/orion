use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 10614729, sign: false });
    data.append(FP8x23 { mag: 10324723, sign: true });
    data.append(FP8x23 { mag: 17253290, sign: false });
    data.append(FP8x23 { mag: 1202276, sign: false });
    data.append(FP8x23 { mag: 2141559, sign: true });
    data.append(FP8x23 { mag: 2238481, sign: true });
    data.append(FP8x23 { mag: 4732881, sign: false });
    data.append(FP8x23 { mag: 2781880, sign: false });
    data.append(FP8x23 { mag: 3924081, sign: true });
    data.append(FP8x23 { mag: 3475410, sign: false });
    data.append(FP8x23 { mag: 18367708, sign: false });
    data.append(FP8x23 { mag: 2550018, sign: true });
    data.append(FP8x23 { mag: 4153469, sign: true });
    data.append(FP8x23 { mag: 6400054, sign: false });
    data.append(FP8x23 { mag: 1315133, sign: true });
    data.append(FP8x23 { mag: 11185587, sign: false });
    data.append(FP8x23 { mag: 11006869, sign: true });
    data.append(FP8x23 { mag: 7368636, sign: false });
    data.append(FP8x23 { mag: 1108425, sign: true });
    data.append(FP8x23 { mag: 10114867, sign: false });
    data.append(FP8x23 { mag: 6575382, sign: true });
    data.append(FP8x23 { mag: 2328343, sign: false });
    data.append(FP8x23 { mag: 113773, sign: false });
    data.append(FP8x23 { mag: 502797, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
