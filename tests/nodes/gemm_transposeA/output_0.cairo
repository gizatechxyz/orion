use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 87420, sign: false });
    data.append(FP16x16 { mag: 56805, sign: false });
    data.append(FP16x16 { mag: 118934, sign: false });
    data.append(FP16x16 { mag: 106965, sign: false });
    data.append(FP16x16 { mag: 88852, sign: false });
    data.append(FP16x16 { mag: 31983, sign: false });
    data.append(FP16x16 { mag: 137135, sign: false });
    data.append(FP16x16 { mag: 103030, sign: false });
    data.append(FP16x16 { mag: 81868, sign: false });
    data.append(FP16x16 { mag: 49808, sign: false });
    data.append(FP16x16 { mag: 100563, sign: false });
    data.append(FP16x16 { mag: 81533, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
