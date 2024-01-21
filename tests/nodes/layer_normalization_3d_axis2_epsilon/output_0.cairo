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
    data.append(FP16x16 { mag: 53879, sign: false });
    data.append(FP16x16 { mag: 15631, sign: false });
    data.append(FP16x16 { mag: 16210, sign: true });
    data.append(FP16x16 { mag: 31819, sign: true });
    data.append(FP16x16 { mag: 1365, sign: false });
    data.append(FP16x16 { mag: 30042, sign: false });
    data.append(FP16x16 { mag: 184179, sign: true });
    data.append(FP16x16 { mag: 13663, sign: false });
    data.append(FP16x16 { mag: 67432, sign: true });
    data.append(FP16x16 { mag: 86296, sign: true });
    data.append(FP16x16 { mag: 22179, sign: false });
    data.append(FP16x16 { mag: 71910, sign: true });
    data.append(FP16x16 { mag: 20350, sign: false });
    data.append(FP16x16 { mag: 99451, sign: true });
    data.append(FP16x16 { mag: 29814, sign: false });
    data.append(FP16x16 { mag: 13673, sign: false });
    data.append(FP16x16 { mag: 113125, sign: true });
    data.append(FP16x16 { mag: 44314, sign: true });
    data.append(FP16x16 { mag: 41557, sign: true });
    data.append(FP16x16 { mag: 143319, sign: true });
    data.append(FP16x16 { mag: 7507, sign: true });
    data.append(FP16x16 { mag: 34257, sign: false });
    data.append(FP16x16 { mag: 23982, sign: true });
    data.append(FP16x16 { mag: 49325, sign: true });
    data.append(FP16x16 { mag: 113251, sign: false });
    data.append(FP16x16 { mag: 19969, sign: false });
    data.append(FP16x16 { mag: 39099, sign: false });
    data.append(FP16x16 { mag: 14713, sign: true });
    data.append(FP16x16 { mag: 88159, sign: true });
    data.append(FP16x16 { mag: 41236, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
