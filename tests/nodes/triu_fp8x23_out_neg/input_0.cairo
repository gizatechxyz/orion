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
    data.append(FP8x23 { mag: 260046848, sign: false });
    data.append(FP8x23 { mag: 780140544, sign: true });
    data.append(FP8x23 { mag: 713031680, sign: true });
    data.append(FP8x23 { mag: 1040187392, sign: true });
    data.append(FP8x23 { mag: 520093696, sign: true });
    data.append(FP8x23 { mag: 553648128, sign: true });
    data.append(FP8x23 { mag: 981467136, sign: true });
    data.append(FP8x23 { mag: 402653184, sign: false });
    data.append(FP8x23 { mag: 796917760, sign: false });
    data.append(FP8x23 { mag: 142606336, sign: false });
    data.append(FP8x23 { mag: 394264576, sign: false });
    data.append(FP8x23 { mag: 117440512, sign: false });
    data.append(FP8x23 { mag: 494927872, sign: true });
    data.append(FP8x23 { mag: 578813952, sign: false });
    data.append(FP8x23 { mag: 41943040, sign: true });
    data.append(FP8x23 { mag: 1056964608, sign: true });
    data.append(FP8x23 { mag: 830472192, sign: false });
    data.append(FP8x23 { mag: 813694976, sign: false });
    data.append(FP8x23 { mag: 763363328, sign: true });
    data.append(FP8x23 { mag: 318767104, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
