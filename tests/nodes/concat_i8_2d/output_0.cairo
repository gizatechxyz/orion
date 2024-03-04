use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 1, sign: false });
    data.append(FP8x23 { mag: 2, sign: false });
    data.append(FP8x23 { mag: 3, sign: false });
    data.append(FP8x23 { mag: 4, sign: false });
    data.append(FP8x23 { mag: 5, sign: false });
    data.append(FP8x23 { mag: 6, sign: false });
    data.append(FP8x23 { mag: 7, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
