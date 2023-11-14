use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 13404706, sign: true });
    data.append(FP8x23 { mag: 1642563, sign: true });
    data.append(FP8x23 { mag: 16689459, sign: false });
    data.append(FP8x23 { mag: 2347829, sign: false });
    data.append(FP8x23 { mag: 3467112, sign: false });
    data.append(FP8x23 { mag: 592703, sign: true });
    data.append(FP8x23 { mag: 1941576, sign: false });
    data.append(FP8x23 { mag: 4972, sign: true });
    data.append(FP8x23 { mag: 21738998, sign: false });
    data.append(FP8x23 { mag: 11513586, sign: true });
    data.append(FP8x23 { mag: 23646070, sign: false });
    data.append(FP8x23 { mag: 13206001, sign: true });
    data.append(FP8x23 { mag: 10769444, sign: false });
    data.append(FP8x23 { mag: 10345798, sign: false });
    data.append(FP8x23 { mag: 1337911, sign: true });
    data.append(FP8x23 { mag: 23240711, sign: false });
    data.append(FP8x23 { mag: 3398664, sign: false });
    data.append(FP8x23 { mag: 13332160, sign: true });
    data.append(FP8x23 { mag: 2545195, sign: false });
    data.append(FP8x23 { mag: 21250720, sign: true });
    data.append(FP8x23 { mag: 16789138, sign: true });
    data.append(FP8x23 { mag: 23029037, sign: true });
    data.append(FP8x23 { mag: 19022158, sign: false });
    data.append(FP8x23 { mag: 20748602, sign: false });
    data.append(FP8x23 { mag: 21590667, sign: false });
    data.append(FP8x23 { mag: 1180061, sign: false });
    data.append(FP8x23 { mag: 8043052, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
