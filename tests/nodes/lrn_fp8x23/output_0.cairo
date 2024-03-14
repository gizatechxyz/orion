use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 111256816, sign: false });
    data.append(FP8x23 { mag: 46580759, sign: false });
    data.append(FP8x23 { mag: 128277313, sign: true });
    data.append(FP8x23 { mag: 83611630, sign: false });
    data.append(FP8x23 { mag: 126017564, sign: true });
    data.append(FP8x23 { mag: 1543992, sign: false });
    data.append(FP8x23 { mag: 72682307, sign: true });
    data.append(FP8x23 { mag: 155433087, sign: false });
    data.append(FP8x23 { mag: 157125508, sign: true });
    data.append(FP8x23 { mag: 132534076, sign: false });
    data.append(FP8x23 { mag: 56185980, sign: false });
    data.append(FP8x23 { mag: 163415641, sign: true });
    data.append(FP8x23 { mag: 16276062, sign: false });
    data.append(FP8x23 { mag: 38837883, sign: false });
    data.append(FP8x23 { mag: 172269841, sign: false });
    data.append(FP8x23 { mag: 125158185, sign: true });
    data.append(FP8x23 { mag: 49451868, sign: true });
    data.append(FP8x23 { mag: 75700225, sign: true });
    data.append(FP8x23 { mag: 116359950, sign: true });
    data.append(FP8x23 { mag: 35514562, sign: true });
    data.append(FP8x23 { mag: 146666826, sign: true });
    data.append(FP8x23 { mag: 123872486, sign: true });
    data.append(FP8x23 { mag: 83903259, sign: false });
    data.append(FP8x23 { mag: 39659332, sign: true });
    data.append(FP8x23 { mag: 57700200, sign: true });
    data.append(FP8x23 { mag: 82042811, sign: false });
    data.append(FP8x23 { mag: 121547869, sign: true });
    data.append(FP8x23 { mag: 124950489, sign: true });
    data.append(FP8x23 { mag: 11480852, sign: false });
    data.append(FP8x23 { mag: 102419757, sign: false });
    data.append(FP8x23 { mag: 99907452, sign: true });
    data.append(FP8x23 { mag: 133080503, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
