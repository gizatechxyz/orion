use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 2228224, sign: true });
    data.append(FP16x16 { mag: 7471104, sign: false });
    data.append(FP16x16 { mag: 1441792, sign: false });
    data.append(FP16x16 { mag: 1638400, sign: true });
    data.append(FP16x16 { mag: 786432, sign: false });
    data.append(FP16x16 { mag: 1310720, sign: false });
    data.append(FP16x16 { mag: 7602176, sign: true });
    data.append(FP16x16 { mag: 8060928, sign: false });
    data.append(FP16x16 { mag: 4194304, sign: false });
    data.append(FP16x16 { mag: 2818048, sign: false });
    data.append(FP16x16 { mag: 458752, sign: false });
    data.append(FP16x16 { mag: 4718592, sign: true });
    data.append(FP16x16 { mag: 4784128, sign: false });
    data.append(FP16x16 { mag: 6094848, sign: false });
    data.append(FP16x16 { mag: 5636096, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
