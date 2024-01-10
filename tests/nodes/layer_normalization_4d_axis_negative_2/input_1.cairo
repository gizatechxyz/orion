use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1860902, sign: true });
    data.append(FP8x23 { mag: 7189990, sign: false });
    data.append(FP8x23 { mag: 5594953, sign: false });
    data.append(FP8x23 { mag: 14949612, sign: false });
    data.append(FP8x23 { mag: 1598676, sign: true });
    data.append(FP8x23 { mag: 19332304, sign: true });
    data.append(FP8x23 { mag: 13237330, sign: true });
    data.append(FP8x23 { mag: 13876161, sign: true });
    data.append(FP8x23 { mag: 2710915, sign: false });
    data.append(FP8x23 { mag: 1998193, sign: false });
    data.append(FP8x23 { mag: 10029104, sign: true });
    data.append(FP8x23 { mag: 5128877, sign: true });
    data.append(FP8x23 { mag: 12692706, sign: false });
    data.append(FP8x23 { mag: 7217481, sign: false });
    data.append(FP8x23 { mag: 2729123, sign: true });
    data.append(FP8x23 { mag: 12888666, sign: false });
    data.append(FP8x23 { mag: 4258854, sign: false });
    data.append(FP8x23 { mag: 1006706, sign: true });
    data.append(FP8x23 { mag: 3116978, sign: false });
    data.append(FP8x23 { mag: 10767356, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
