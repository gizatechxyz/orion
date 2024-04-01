use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 147540, sign: false });
    data.append(FP16x16 { mag: 126989, sign: true });
    data.append(FP16x16 { mag: 30444, sign: false });
    data.append(FP16x16 { mag: 51163, sign: false });
    data.append(FP16x16 { mag: 57418, sign: true });
    data.append(FP16x16 { mag: 6674, sign: true });
    data.append(FP16x16 { mag: 5225, sign: false });
    data.append(FP16x16 { mag: 17319, sign: true });
    data.append(FP16x16 { mag: 100931, sign: true });
    data.append(FP16x16 { mag: 148297, sign: true });
    data.append(FP16x16 { mag: 8755, sign: false });
    data.append(FP16x16 { mag: 103665, sign: false });
    data.append(FP16x16 { mag: 8492, sign: true });
    data.append(FP16x16 { mag: 163528, sign: true });
    data.append(FP16x16 { mag: 51308, sign: true });
    data.append(FP16x16 { mag: 7822, sign: true });
    data.append(FP16x16 { mag: 8571, sign: true });
    data.append(FP16x16 { mag: 7942, sign: true });
    data.append(FP16x16 { mag: 21883, sign: false });
    data.append(FP16x16 { mag: 8928, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
