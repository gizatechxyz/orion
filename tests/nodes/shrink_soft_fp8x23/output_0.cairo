use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 5016098, sign: true });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 8300851, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 13350390, sign: false });
    data.append(FP8x23 { mag: 3124978, sign: true });
    data.append(FP8x23 { mag: 15257462, sign: false });
    data.append(FP8x23 { mag: 4817393, sign: true });
    data.append(FP8x23 { mag: 2380836, sign: false });
    data.append(FP8x23 { mag: 1957190, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 14852103, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 4943552, sign: true });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 12862112, sign: true });
    data.append(FP8x23 { mag: 8400530, sign: true });
    data.append(FP8x23 { mag: 14640429, sign: true });
    data.append(FP8x23 { mag: 10633550, sign: false });
    data.append(FP8x23 { mag: 12359994, sign: false });
    data.append(FP8x23 { mag: 13202059, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
