use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 7471104, sign: false });
    data.append(FP16x16 { mag: 7864320, sign: true });
    data.append(FP16x16 { mag: 2555904, sign: false });
    data.append(FP16x16 { mag: 3407872, sign: false });
    data.append(FP16x16 { mag: 6422528, sign: false });
    data.append(FP16x16 { mag: 6029312, sign: false });
    data.append(FP16x16 { mag: 3997696, sign: true });
    data.append(FP16x16 { mag: 5701632, sign: true });
    data.append(FP16x16 { mag: 196608, sign: false });
    data.append(FP16x16 { mag: 5701632, sign: true });
    data.append(FP16x16 { mag: 5570560, sign: false });
    data.append(FP16x16 { mag: 5308416, sign: true });
    data.append(FP16x16 { mag: 1179648, sign: true });
    data.append(FP16x16 { mag: 7077888, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 5242880, sign: true });
    data.append(FP16x16 { mag: 2818048, sign: true });
    data.append(FP16x16 { mag: 4063232, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 262144, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
