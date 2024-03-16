use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 10768, sign: true });
    data.append(FP16x16 { mag: 126592, sign: false });
    data.append(FP16x16 { mag: 17984, sign: true });
    data.append(FP16x16 { mag: 87296, sign: false });
    data.append(FP16x16 { mag: 144384, sign: false });
    data.append(FP16x16 { mag: 19360, sign: true });
    data.append(FP16x16 { mag: 77056, sign: false });
    data.append(FP16x16 { mag: 66176, sign: false });
    data.append(FP16x16 { mag: 63104, sign: false });
    data.append(FP16x16 { mag: 61152, sign: false });
    data.append(FP16x16 { mag: 107776, sign: false });
    data.append(FP16x16 { mag: 43712, sign: false });
    data.append(FP16x16 { mag: 11640, sign: true });
    data.append(FP16x16 { mag: 3926, sign: true });
    data.append(FP16x16 { mag: 9824, sign: true });
    data.append(FP16x16 { mag: 30864, sign: false });
    data.append(FP16x16 { mag: 12616, sign: true });
    data.append(FP16x16 { mag: 178816, sign: false });
    data.append(FP16x16 { mag: 16560, sign: true });
    data.append(FP16x16 { mag: 100096, sign: false });
    data.append(FP16x16 { mag: 18464, sign: true });
    data.append(FP16x16 { mag: 17472, sign: true });
    data.append(FP16x16 { mag: 12312, sign: true });
    data.append(FP16x16 { mag: 14216, sign: true });
    data.append(FP16x16 { mag: 19968, sign: true });
    data.append(FP16x16 { mag: 5412, sign: true });
    data.append(FP16x16 { mag: 19520, sign: true });
    data.append(FP16x16 { mag: 11216, sign: true });
    data.append(FP16x16 { mag: 117696, sign: false });
    data.append(FP16x16 { mag: 13784, sign: false });
    data.append(FP16x16 { mag: 16088, sign: true });
    data.append(FP16x16 { mag: 19904, sign: true });
    data.append(FP16x16 { mag: 185984, sign: false });
    data.append(FP16x16 { mag: 189312, sign: false });
    data.append(FP16x16 { mag: 17728, sign: true });
    data.append(FP16x16 { mag: 16784, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
