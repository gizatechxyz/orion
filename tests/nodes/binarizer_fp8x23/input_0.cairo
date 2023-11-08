use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 7205493, sign: true });
    data.append(FP8x23 { mag: 14503128, sign: false });
    data.append(FP8x23 { mag: 9682839, sign: false });
    data.append(FP8x23 { mag: 25030485, sign: false });
    data.append(FP8x23 { mag: 3669115, sign: true });
    data.append(FP8x23 { mag: 16376632, sign: true });
    data.append(FP8x23 { mag: 4670619, sign: false });
    data.append(FP8x23 { mag: 24976405, sign: true });
    data.append(FP8x23 { mag: 19811890, sign: false });
    data.append(FP8x23 { mag: 10781071, sign: true });
    data.append(FP8x23 { mag: 10216945, sign: false });
    data.append(FP8x23 { mag: 25004285, sign: false });
    data.append(FP8x23 { mag: 7482592, sign: false });
    data.append(FP8x23 { mag: 1360, sign: true });
    data.append(FP8x23 { mag: 10632602, sign: true });
    data.append(FP8x23 { mag: 275175, sign: true });
    data.append(FP8x23 { mag: 21731586, sign: true });
    data.append(FP8x23 { mag: 5638921, sign: true });
    data.append(FP8x23 { mag: 6096613, sign: false });
    data.append(FP8x23 { mag: 22105612, sign: false });
    data.append(FP8x23 { mag: 17084000, sign: true });
    data.append(FP8x23 { mag: 12627954, sign: true });
    data.append(FP8x23 { mag: 23194838, sign: false });
    data.append(FP8x23 { mag: 9012947, sign: false });
    data.append(FP8x23 { mag: 5163417, sign: true });
    data.append(FP8x23 { mag: 13580853, sign: false });
    data.append(FP8x23 { mag: 5624227, sign: false });
    TensorTrait::new(shape.span(), data.span())
}