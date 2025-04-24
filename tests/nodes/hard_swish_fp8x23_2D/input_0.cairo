use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 10429047, sign: true });
    data.append(FP8x23 { mag: 8952725, sign: true });
    data.append(FP8x23 { mag: 4984775, sign: true });
    data.append(FP8x23 { mag: 21069174, sign: false });
    data.append(FP8x23 { mag: 25044698, sign: true });
    data.append(FP8x23 { mag: 8979134, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
