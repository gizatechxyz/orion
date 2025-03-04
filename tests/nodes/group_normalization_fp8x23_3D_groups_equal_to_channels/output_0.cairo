use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 12588730, sign: true });
    data.append(FP8x23 { mag: 1068370, sign: true });
    data.append(FP8x23 { mag: 17095322, sign: false });
    data.append(FP8x23 { mag: 3005038, sign: true });
    data.append(FP8x23 { mag: 11241526, sign: true });
    data.append(FP8x23 { mag: 2415572, sign: true });
    data.append(FP8x23 { mag: 3459556, sign: true });
    data.append(FP8x23 { mag: 17549840, sign: false });
    data.append(FP8x23 { mag: 4066379, sign: true });
    data.append(FP8x23 { mag: 9590720, sign: true });
    data.append(FP8x23 { mag: 13731403, sign: false });
    data.append(FP8x23 { mag: 358881, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
