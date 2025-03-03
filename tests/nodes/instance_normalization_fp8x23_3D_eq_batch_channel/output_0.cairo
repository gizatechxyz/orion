use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 9701290, sign: false });
    data.append(FP8x23 { mag: 4530453, sign: true });
    data.append(FP8x23 { mag: 11492202, sign: true });
    data.append(FP8x23 { mag: 15815982, sign: false });
    data.append(FP8x23 { mag: 13668626, sign: true });
    data.append(FP8x23 { mag: 10911318, sign: true });
    data.append(FP8x23 { mag: 13022861, sign: true });
    data.append(FP8x23 { mag: 9621627, sign: true });
    data.append(FP8x23 { mag: 14749122, sign: false });
    data.append(FP8x23 { mag: 9582600, sign: false });
    data.append(FP8x23 { mag: 13153414, sign: true });
    data.append(FP8x23 { mag: 1683689, sign: true });
    data.append(FP8x23 { mag: 14106367, sign: true });
    data.append(FP8x23 { mag: 7763758, sign: true });
    data.append(FP8x23 { mag: 15532692, sign: true });
    data.append(FP8x23 { mag: 9821615, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
