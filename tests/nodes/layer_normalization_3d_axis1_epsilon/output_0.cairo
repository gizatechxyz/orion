use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 59799, sign: false });
    data.append(FP16x16 { mag: 29975, sign: false });
    data.append(FP16x16 { mag: 187208, sign: false });
    data.append(FP16x16 { mag: 35149, sign: true });
    data.append(FP16x16 { mag: 54676, sign: false });
    data.append(FP16x16 { mag: 26994, sign: false });
    data.append(FP16x16 { mag: 162286, sign: false });
    data.append(FP16x16 { mag: 86683, sign: true });
    data.append(FP16x16 { mag: 21676, sign: true });
    data.append(FP16x16 { mag: 39399, sign: false });
    data.append(FP16x16 { mag: 123115, sign: true });
    data.append(FP16x16 { mag: 92377, sign: false });
    data.append(FP16x16 { mag: 13773, sign: false });
    data.append(FP16x16 { mag: 26238, sign: false });
    data.append(FP16x16 { mag: 19656, sign: true });
    data.append(FP16x16 { mag: 51756, sign: true });
    data.append(FP16x16 { mag: 24121, sign: false });
    data.append(FP16x16 { mag: 60347, sign: false });
    data.append(FP16x16 { mag: 34230, sign: true });
    data.append(FP16x16 { mag: 54552, sign: false });
    data.append(FP16x16 { mag: 31465, sign: false });
    data.append(FP16x16 { mag: 184284, sign: false });
    data.append(FP16x16 { mag: 96496, sign: true });
    data.append(FP16x16 { mag: 47101, sign: true });
    data.append(FP16x16 { mag: 6198, sign: false });
    data.append(FP16x16 { mag: 64574, sign: true });
    data.append(FP16x16 { mag: 201408, sign: true });
    data.append(FP16x16 { mag: 8115, sign: false });
    data.append(FP16x16 { mag: 133313, sign: true });
    data.append(FP16x16 { mag: 1471, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
