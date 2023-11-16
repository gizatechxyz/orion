use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 3604480, sign: true });
    data.append(FP16x16 { mag: 2621440, sign: false });
    data.append(FP16x16 { mag: 262144, sign: true });
    data.append(FP16x16 { mag: 458752, sign: false });
    data.append(FP16x16 { mag: 2490368, sign: true });
    data.append(FP16x16 { mag: 2555904, sign: false });
    data.append(FP16x16 { mag: 393216, sign: true });
    data.append(FP16x16 { mag: 4849664, sign: false });
    data.append(FP16x16 { mag: 5636096, sign: true });
    data.append(FP16x16 { mag: 2949120, sign: false });
    data.append(FP16x16 { mag: 6291456, sign: false });
    data.append(FP16x16 { mag: 3014656, sign: true });
    data.append(FP16x16 { mag: 3342336, sign: true });
    data.append(FP16x16 { mag: 6750208, sign: true });
    data.append(FP16x16 { mag: 7798784, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
