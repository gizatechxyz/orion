use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2055097, sign: false });
    data.append(FP8x23 { mag: 4681757, sign: true });
    data.append(FP8x23 { mag: 9467429, sign: true });
    data.append(FP8x23 { mag: 10714904, sign: true });
    data.append(FP8x23 { mag: 4545937, sign: true });
    data.append(FP8x23 { mag: 6802699, sign: true });
    data.append(FP8x23 { mag: 8448431, sign: false });
    data.append(FP8x23 { mag: 1309417, sign: false });
    data.append(FP8x23 { mag: 1427043, sign: true });
    data.append(FP8x23 { mag: 16194403, sign: true });
    data.append(FP8x23 { mag: 10729787, sign: false });
    data.append(FP8x23 { mag: 11312058, sign: false });
    data.append(FP8x23 { mag: 4344780, sign: true });
    data.append(FP8x23 { mag: 2117222, sign: true });
    data.append(FP8x23 { mag: 4305543, sign: true });
    data.append(FP8x23 { mag: 19256026, sign: false });
    data.append(FP8x23 { mag: 6015612, sign: true });
    data.append(FP8x23 { mag: 3912062, sign: true });
    data.append(FP8x23 { mag: 6010057, sign: true });
    data.append(FP8x23 { mag: 5751897, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
