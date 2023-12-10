use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 1900544, sign: false });
    data.append(FP16x16 { mag: 5570560, sign: false });
    data.append(FP16x16 { mag: 3735552, sign: false });
    data.append(FP16x16 { mag: 6356992, sign: true });
    data.append(FP16x16 { mag: 2359296, sign: false });
    data.append(FP16x16 { mag: 5308416, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 7208960, sign: true });
    data.append(FP16x16 { mag: 5439488, sign: true });
    data.append(FP16x16 { mag: 5177344, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 4718592, sign: true });
    data.append(FP16x16 { mag: 6619136, sign: false });
    data.append(FP16x16 { mag: 5898240, sign: true });
    data.append(FP16x16 { mag: 2490368, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 3407872, sign: false });
    data.append(FP16x16 { mag: 2031616, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
