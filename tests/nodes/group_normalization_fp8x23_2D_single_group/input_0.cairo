use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2081578, sign: false });
    data.append(FP8x23 { mag: 5370928, sign: true });
    data.append(FP8x23 { mag: 2408851, sign: false });
    data.append(FP8x23 { mag: 1358010, sign: true });
    data.append(FP8x23 { mag: 4838638, sign: true });
    data.append(FP8x23 { mag: 18546742, sign: false });
    data.append(FP8x23 { mag: 2664093, sign: true });
    data.append(FP8x23 { mag: 7069699, sign: true });
    data.append(FP8x23 { mag: 4005589, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
