use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 83886080, sign: false });
    data.append(FP8x23 { mag: 729808896, sign: true });
    data.append(FP8x23 { mag: 536870912, sign: true });
    data.append(FP8x23 { mag: 905969664, sign: false });
    data.append(FP8x23 { mag: 75497472, sign: true });
    data.append(FP8x23 { mag: 377487360, sign: true });
    data.append(FP8x23 { mag: 92274688, sign: true });
    data.append(FP8x23 { mag: 209715200, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
