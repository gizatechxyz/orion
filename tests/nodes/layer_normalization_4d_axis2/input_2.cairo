use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_2() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1759247, sign: false });
    data.append(FP8x23 { mag: 3714086, sign: true });
    data.append(FP8x23 { mag: 1622474, sign: true });
    data.append(FP8x23 { mag: 8081111, sign: false });
    data.append(FP8x23 { mag: 15897326, sign: true });
    data.append(FP8x23 { mag: 758940, sign: true });
    data.append(FP8x23 { mag: 12821375, sign: true });
    data.append(FP8x23 { mag: 5124450, sign: true });
    data.append(FP8x23 { mag: 23434418, sign: false });
    data.append(FP8x23 { mag: 3803789, sign: true });
    data.append(FP8x23 { mag: 1614523, sign: true });
    data.append(FP8x23 { mag: 5084428, sign: true });
    data.append(FP8x23 { mag: 1234480, sign: true });
    data.append(FP8x23 { mag: 16545990, sign: false });
    data.append(FP8x23 { mag: 281637, sign: true });
    data.append(FP8x23 { mag: 2667010, sign: false });
    data.append(FP8x23 { mag: 10535491, sign: false });
    data.append(FP8x23 { mag: 4933426, sign: true });
    data.append(FP8x23 { mag: 327930, sign: true });
    data.append(FP8x23 { mag: 4062404, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
