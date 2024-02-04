use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 21264, sign: false });
    data.append(FP16x16 { mag: 40785, sign: false });
    data.append(FP16x16 { mag: 3120, sign: false });
    data.append(FP16x16 { mag: 5275, sign: false });
    data.append(FP16x16 { mag: 38611, sign: false });
    data.append(FP16x16 { mag: 30792, sign: false });
    data.append(FP16x16 { mag: 9186, sign: false });
    data.append(FP16x16 { mag: 7839, sign: false });
    data.append(FP16x16 { mag: 61914, sign: false });
    data.append(FP16x16 { mag: 53606, sign: false });
    data.append(FP16x16 { mag: 5497, sign: false });
    data.append(FP16x16 { mag: 49410, sign: false });
    data.append(FP16x16 { mag: 33114, sign: false });
    data.append(FP16x16 { mag: 20996, sign: false });
    data.append(FP16x16 { mag: 11300, sign: false });
    data.append(FP16x16 { mag: 19630, sign: false });
    data.append(FP16x16 { mag: 14015, sign: false });
    data.append(FP16x16 { mag: 25247, sign: false });
    data.append(FP16x16 { mag: 1692, sign: false });
    data.append(FP16x16 { mag: 43693, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
