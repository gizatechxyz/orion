use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 377487360, sign: false });
    data.append(FP8x23 { mag: 276824064, sign: true });
    data.append(FP8x23 { mag: 620756992, sign: true });
    data.append(FP8x23 { mag: 989855744, sign: true });
    data.append(FP8x23 { mag: 301989888, sign: false });
    data.append(FP8x23 { mag: 645922816, sign: false });
    data.append(FP8x23 { mag: 343932928, sign: false });
    data.append(FP8x23 { mag: 713031680, sign: true });
    data.append(FP8x23 { mag: 310378496, sign: false });
    data.append(FP8x23 { mag: 603979776, sign: false });
    data.append(FP8x23 { mag: 511705088, sign: false });
    data.append(FP8x23 { mag: 377487360, sign: false });
    data.append(FP8x23 { mag: 276824064, sign: false });
    data.append(FP8x23 { mag: 553648128, sign: false });
    data.append(FP8x23 { mag: 788529152, sign: false });
    data.append(FP8x23 { mag: 838860800, sign: true });
    data.append(FP8x23 { mag: 562036736, sign: false });
    data.append(FP8x23 { mag: 989855744, sign: false });
    data.append(FP8x23 { mag: 394264576, sign: true });
    data.append(FP8x23 { mag: 503316480, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
