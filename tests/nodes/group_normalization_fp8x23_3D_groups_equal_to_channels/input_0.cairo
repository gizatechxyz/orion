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
    data.append(FP8x23 { mag: 6738477, sign: false });
    data.append(FP8x23 { mag: 2684814, sign: true });
    data.append(FP8x23 { mag: 19007338, sign: false });
    data.append(FP8x23 { mag: 7673336, sign: false });
    data.append(FP8x23 { mag: 16506052, sign: false });
    data.append(FP8x23 { mag: 11748971, sign: false });
    data.append(FP8x23 { mag: 772563, sign: true });
    data.append(FP8x23 { mag: 14811387, sign: false });
    data.append(FP8x23 { mag: 4826588, sign: true });
    data.append(FP8x23 { mag: 2386459, sign: true });
    data.append(FP8x23 { mag: 2652093, sign: true });
    data.append(FP8x23 { mag: 6657591, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
