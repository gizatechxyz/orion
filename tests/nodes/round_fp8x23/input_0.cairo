use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(15);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 838860, sign: false });
    data.append(FP8x23 { mag: 4194304, sign: false });
    data.append(FP8x23 { mag: 7549747, sign: false });
    data.append(FP8x23 { mag: 10066329, sign: false });
    data.append(FP8x23 { mag: 12582912, sign: false });
    data.append(FP8x23 { mag: 15099494, sign: false });
    data.append(FP8x23 { mag: 19293798, sign: false });
    data.append(FP8x23 { mag: 20971520, sign: false });
    data.append(FP8x23 { mag: 22649241, sign: false });
    data.append(FP8x23 { mag: 9227468, sign: true });
    data.append(FP8x23 { mag: 12582912, sign: true });
    data.append(FP8x23 { mag: 15938355, sign: true });
    data.append(FP8x23 { mag: 18454937, sign: true });
    data.append(FP8x23 { mag: 20971520, sign: true });
    data.append(FP8x23 { mag: 23488102, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
