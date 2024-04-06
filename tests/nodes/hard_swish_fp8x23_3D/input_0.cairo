use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 6427647, sign: false });
    data.append(FP8x23 { mag: 12552493, sign: true });
    data.append(FP8x23 { mag: 22592528, sign: false });
    data.append(FP8x23 { mag: 4899970, sign: true });
    data.append(FP8x23 { mag: 7805612, sign: false });
    data.append(FP8x23 { mag: 21483894, sign: false });
    data.append(FP8x23 { mag: 2957801, sign: true });
    data.append(FP8x23 { mag: 1598294, sign: false });
    data.append(FP8x23 { mag: 9891744, sign: false });
    data.append(FP8x23 { mag: 6470103, sign: false });
    data.append(FP8x23 { mag: 20011662, sign: true });
    data.append(FP8x23 { mag: 17098128, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
