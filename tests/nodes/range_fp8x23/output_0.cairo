use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(14);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 8388608, sign: false });
    data.append(FP8x23 { mag: 10905190, sign: false });
    data.append(FP8x23 { mag: 13421772, sign: false });
    data.append(FP8x23 { mag: 15938355, sign: false });
    data.append(FP8x23 { mag: 18454937, sign: false });
    data.append(FP8x23 { mag: 20971520, sign: false });
    data.append(FP8x23 { mag: 23488102, sign: false });
    data.append(FP8x23 { mag: 26004684, sign: false });
    data.append(FP8x23 { mag: 28521267, sign: false });
    data.append(FP8x23 { mag: 31037849, sign: false });
    data.append(FP8x23 { mag: 33554432, sign: false });
    data.append(FP8x23 { mag: 36071014, sign: false });
    data.append(FP8x23 { mag: 38587596, sign: false });
    data.append(FP8x23 { mag: 41104179, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
