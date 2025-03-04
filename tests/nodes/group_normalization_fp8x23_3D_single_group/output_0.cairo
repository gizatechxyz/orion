use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2147314, sign: false });
    data.append(FP8x23 { mag: 3433512, sign: false });
    data.append(FP8x23 { mag: 18690976, sign: false });
    data.append(FP8x23 { mag: 28525212, sign: false });
    data.append(FP8x23 { mag: 11896850, sign: false });
    data.append(FP8x23 { mag: 15742165, sign: false });
    data.append(FP8x23 { mag: 6504215, sign: false });
    data.append(FP8x23 { mag: 6031613, sign: false });
    data.append(FP8x23 { mag: 32893548, sign: false });
    data.append(FP8x23 { mag: 12621361, sign: false });
    data.append(FP8x23 { mag: 18265496, sign: false });
    data.append(FP8x23 { mag: 19834080, sign: false });
    data.append(FP8x23 { mag: 8916694, sign: false });
    data.append(FP8x23 { mag: 5103593, sign: false });
    data.append(FP8x23 { mag: 744094, sign: false });
    data.append(FP8x23 { mag: 15329768, sign: false });
    data.append(FP8x23 { mag: 9963178, sign: false });
    data.append(FP8x23 { mag: 19606840, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
