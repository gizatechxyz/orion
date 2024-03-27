use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(2);
    shape.append(4);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 5046272, sign: true });
    data.append(FP16x16 { mag: 3801088, sign: true });
    data.append(FP16x16 { mag: 3014656, sign: true });
    data.append(FP16x16 { mag: 7143424, sign: false });
    data.append(FP16x16 { mag: 4718592, sign: true });
    data.append(FP16x16 { mag: 7274496, sign: false });
    data.append(FP16x16 { mag: 2752512, sign: false });
    data.append(FP16x16 { mag: 1441792, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
