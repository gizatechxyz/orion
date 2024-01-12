use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 16045555, sign: false });
    data.append(FP8x23 { mag: 6797189, sign: true });
    data.append(FP8x23 { mag: 338571, sign: true });
    data.append(FP8x23 { mag: 14826208, sign: true });
    data.append(FP8x23 { mag: 6612261, sign: false });
    data.append(FP8x23 { mag: 2255963, sign: false });
    data.append(FP8x23 { mag: 7694826, sign: false });
    data.append(FP8x23 { mag: 8157877, sign: true });
    data.append(FP8x23 { mag: 10027904, sign: true });
    data.append(FP8x23 { mag: 4144258, sign: false });
    data.append(FP8x23 { mag: 12368555, sign: true });
    data.append(FP8x23 { mag: 1431810, sign: false });
    data.append(FP8x23 { mag: 993247, sign: true });
    data.append(FP8x23 { mag: 10015980, sign: true });
    data.append(FP8x23 { mag: 11250731, sign: false });
    data.append(FP8x23 { mag: 12224184, sign: true });
    data.append(FP8x23 { mag: 14407597, sign: true });
    data.append(FP8x23 { mag: 1255469, sign: true });
    data.append(FP8x23 { mag: 48578, sign: true });
    data.append(FP8x23 { mag: 14580561, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
