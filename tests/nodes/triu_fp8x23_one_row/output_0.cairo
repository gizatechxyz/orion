use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 75497472, sign: false });
    data.append(FP8x23 { mag: 192937984, sign: false });
    data.append(FP8x23 { mag: 838860800, sign: false });
    data.append(FP8x23 { mag: 637534208, sign: false });
    data.append(FP8x23 { mag: 209715200, sign: false });
    data.append(FP8x23 { mag: 486539264, sign: true });
    data.append(FP8x23 { mag: 310378496, sign: false });
    data.append(FP8x23 { mag: 603979776, sign: false });
    data.append(FP8x23 { mag: 50331648, sign: true });
    data.append(FP8x23 { mag: 394264576, sign: true });
    data.append(FP8x23 { mag: 939524096, sign: true });
    data.append(FP8x23 { mag: 931135488, sign: true });
    data.append(FP8x23 { mag: 134217728, sign: false });
    data.append(FP8x23 { mag: 545259520, sign: true });
    data.append(FP8x23 { mag: 587202560, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
