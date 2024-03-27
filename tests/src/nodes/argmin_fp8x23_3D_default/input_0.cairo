use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 922746880, sign: false });
    data.append(FP8x23 { mag: 343932928, sign: false });
    data.append(FP8x23 { mag: 754974720, sign: false });
    data.append(FP8x23 { mag: 293601280, sign: false });
    data.append(FP8x23 { mag: 947912704, sign: true });
    data.append(FP8x23 { mag: 931135488, sign: true });
    data.append(FP8x23 { mag: 251658240, sign: false });
    data.append(FP8x23 { mag: 528482304, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
