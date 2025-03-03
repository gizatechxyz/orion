use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 12787712, sign: true });
    data.append(FP8x23 { mag: 2215503, sign: true });
    data.append(FP8x23 { mag: 2956106, sign: false });
    data.append(FP8x23 { mag: 17330078, sign: true });
    data.append(FP8x23 { mag: 5947315, sign: true });
    data.append(FP8x23 { mag: 3395557, sign: true });
    data.append(FP8x23 { mag: 5349690, sign: true });
    data.append(FP8x23 { mag: 2202011, sign: true });
    data.append(FP8x23 { mag: 2904789, sign: true });
    data.append(FP8x23 { mag: 558183, sign: false });
    data.append(FP8x23 { mag: 15797489, sign: false });
    data.append(FP8x23 { mag: 8109657, sign: false });
    data.append(FP8x23 { mag: 3956818, sign: true });
    data.append(FP8x23 { mag: 13018541, sign: false });
    data.append(FP8x23 { mag: 7774233, sign: true });
    data.append(FP8x23 { mag: 7510892, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
