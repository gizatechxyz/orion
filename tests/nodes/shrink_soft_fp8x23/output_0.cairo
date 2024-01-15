use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 6897646, sign: false });
    data.append(FP8x23 { mag: 269503, sign: false });
    data.append(FP8x23 { mag: 12174435, sign: true });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 9502491, sign: false });
    data.append(FP8x23 { mag: 4197091, sign: true });
    data.append(FP8x23 { mag: 7595341, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 7936735, sign: false });
    data.append(FP8x23 { mag: 449393, sign: false });
    data.append(FP8x23 { mag: 4554529, sign: true });
    data.append(FP8x23 { mag: 2705114, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 11761273, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 14781474, sign: true });
    data.append(FP8x23 { mag: 12910068, sign: false });
    data.append(FP8x23 { mag: 16109905, sign: true });
    data.append(FP8x23 { mag: 1830919, sign: true });
    data.append(FP8x23 { mag: 5523568, sign: false });
    data.append(FP8x23 { mag: 2820041, sign: true });
    data.append(FP8x23 { mag: 9064207, sign: false });
    data.append(FP8x23 { mag: 15855213, sign: true });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 2825334, sign: false });
    data.append(FP8x23 { mag: 13181159, sign: false });
    data.append(FP8x23 { mag: 1689578, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
