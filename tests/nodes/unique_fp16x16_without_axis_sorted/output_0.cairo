use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(27);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 16416, sign: false });
    data.append(FP16x16 { mag: 16688, sign: false });
    data.append(FP16x16 { mag: 18912, sign: false });
    data.append(FP16x16 { mag: 28224, sign: false });
    data.append(FP16x16 { mag: 31632, sign: false });
    data.append(FP16x16 { mag: 35008, sign: false });
    data.append(FP16x16 { mag: 50304, sign: false });
    data.append(FP16x16 { mag: 65792, sign: false });
    data.append(FP16x16 { mag: 66880, sign: false });
    data.append(FP16x16 { mag: 67072, sign: false });
    data.append(FP16x16 { mag: 71040, sign: false });
    data.append(FP16x16 { mag: 73728, sign: false });
    data.append(FP16x16 { mag: 79104, sign: false });
    data.append(FP16x16 { mag: 80960, sign: false });
    data.append(FP16x16 { mag: 88576, sign: false });
    data.append(FP16x16 { mag: 94144, sign: false });
    data.append(FP16x16 { mag: 102208, sign: false });
    data.append(FP16x16 { mag: 106688, sign: false });
    data.append(FP16x16 { mag: 113408, sign: false });
    data.append(FP16x16 { mag: 141568, sign: false });
    data.append(FP16x16 { mag: 150528, sign: false });
    data.append(FP16x16 { mag: 152576, sign: false });
    data.append(FP16x16 { mag: 185472, sign: false });
    data.append(FP16x16 { mag: 192896, sign: false });
    data.append(FP16x16 { mag: 193408, sign: false });
    data.append(FP16x16 { mag: 194816, sign: false });
    data.append(FP16x16 { mag: 196096, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
