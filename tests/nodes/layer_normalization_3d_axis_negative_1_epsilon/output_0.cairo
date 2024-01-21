use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 203617, sign: false });
    data.append(FP16x16 { mag: 190225, sign: false });
    data.append(FP16x16 { mag: 41182, sign: true });
    data.append(FP16x16 { mag: 7721, sign: false });
    data.append(FP16x16 { mag: 83051, sign: false });
    data.append(FP16x16 { mag: 145985, sign: false });
    data.append(FP16x16 { mag: 86537, sign: false });
    data.append(FP16x16 { mag: 65930, sign: false });
    data.append(FP16x16 { mag: 1205, sign: false });
    data.append(FP16x16 { mag: 140868, sign: false });
    data.append(FP16x16 { mag: 126973, sign: false });
    data.append(FP16x16 { mag: 144796, sign: false });
    data.append(FP16x16 { mag: 89906, sign: false });
    data.append(FP16x16 { mag: 4652, sign: true });
    data.append(FP16x16 { mag: 64288, sign: false });
    data.append(FP16x16 { mag: 106407, sign: false });
    data.append(FP16x16 { mag: 123409, sign: false });
    data.append(FP16x16 { mag: 141950, sign: true });
    data.append(FP16x16 { mag: 5939, sign: false });
    data.append(FP16x16 { mag: 178478, sign: false });
    data.append(FP16x16 { mag: 55197, sign: false });
    data.append(FP16x16 { mag: 199890, sign: false });
    data.append(FP16x16 { mag: 69050, sign: true });
    data.append(FP16x16 { mag: 4518, sign: false });
    data.append(FP16x16 { mag: 9257, sign: false });
    data.append(FP16x16 { mag: 121631, sign: false });
    data.append(FP16x16 { mag: 202403, sign: false });
    data.append(FP16x16 { mag: 35816, sign: true });
    data.append(FP16x16 { mag: 2586, sign: true });
    data.append(FP16x16 { mag: 56755, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
