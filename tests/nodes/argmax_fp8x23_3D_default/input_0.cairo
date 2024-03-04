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
    data.append(FP8x23 { mag: 612368384, sign: false });
    data.append(FP8x23 { mag: 578813952, sign: false });
    data.append(FP8x23 { mag: 947912704, sign: false });
    data.append(FP8x23 { mag: 201326592, sign: true });
    data.append(FP8x23 { mag: 1031798784, sign: true });
    data.append(FP8x23 { mag: 729808896, sign: false });
    data.append(FP8x23 { mag: 922746880, sign: false });
    data.append(FP8x23 { mag: 33554432, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
