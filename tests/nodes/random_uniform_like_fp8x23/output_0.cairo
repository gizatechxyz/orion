use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 43543406, sign: false });
    data.append(FP8x23 { mag: 82865885, sign: false });
    data.append(FP8x23 { mag: 16954874, sign: false });
    data.append(FP8x23 { mag: 19949400, sign: false });
    data.append(FP8x23 { mag: 61973159, sign: false });
    data.append(FP8x23 { mag: 46269033, sign: false });
    data.append(FP8x23 { mag: 21503499, sign: false });
    data.append(FP8x23 { mag: 19010224, sign: false });
    data.append(FP8x23 { mag: 51919405, sign: false });
    data.append(FP8x23 { mag: 53133236, sign: false });
    data.append(FP8x23 { mag: 62431439, sign: false });
    data.append(FP8x23 { mag: 22875863, sign: false });
    data.append(FP8x23 { mag: 65788925, sign: false });
    data.append(FP8x23 { mag: 21059738, sign: false });
    data.append(FP8x23 { mag: 81958342, sign: false });
    data.append(FP8x23 { mag: 76995797, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
