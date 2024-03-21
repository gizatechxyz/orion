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
    data.append(FP8x23 { mag: 3077273, sign: false });
    data.append(FP8x23 { mag: 4852634, sign: false });
    data.append(FP8x23 { mag: 4419777, sign: false });
    data.append(FP8x23 { mag: 6740174, sign: false });
    data.append(FP8x23 { mag: 7837717, sign: false });
    data.append(FP8x23 { mag: 4927123, sign: false });
    data.append(FP8x23 { mag: 7864395, sign: false });
    data.append(FP8x23 { mag: 5638444, sign: true });
    data.append(FP8x23 { mag: 1324249, sign: true });
    data.append(FP8x23 { mag: 10167902, sign: true });
    data.append(FP8x23 { mag: 320398, sign: true });
    data.append(FP8x23 { mag: 1421857, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
