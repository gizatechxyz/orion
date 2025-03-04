use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 4769910, sign: true });
    data.append(FP8x23 { mag: 19188644, sign: false });
    data.append(FP8x23 { mag: 184272, sign: false });
    data.append(FP8x23 { mag: 153973, sign: false });
    data.append(FP8x23 { mag: 7296125, sign: true });
    data.append(FP8x23 { mag: 7352361, sign: true });
    data.append(FP8x23 { mag: 11356667, sign: true });
    data.append(FP8x23 { mag: 9151210, sign: true });
    data.append(FP8x23 { mag: 12327655, sign: false });
    data.append(FP8x23 { mag: 2500393, sign: false });
    data.append(FP8x23 { mag: 192377, sign: false });
    data.append(FP8x23 { mag: 146544, sign: false });
    data.append(FP8x23 { mag: 8136229, sign: true });
    data.append(FP8x23 { mag: 4408079, sign: true });
    data.append(FP8x23 { mag: 3902368, sign: true });
    data.append(FP8x23 { mag: 12505990, sign: true });
    data.append(FP8x23 { mag: 19486488, sign: false });
    data.append(FP8x23 { mag: 6514221, sign: true });
    data.append(FP8x23 { mag: 154830, sign: false });
    data.append(FP8x23 { mag: 181029, sign: false });
    data.append(FP8x23 { mag: 8695745, sign: true });
    data.append(FP8x23 { mag: 3830705, sign: true });
    data.append(FP8x23 { mag: 7979715, sign: true });
    data.append(FP8x23 { mag: 8393850, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
