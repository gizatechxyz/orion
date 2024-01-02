use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 7143424, sign: false });
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 4456448, sign: true });
    data.append(FP16x16 { mag: 7340032, sign: false });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 5963776, sign: true });
    data.append(FP16x16 { mag: 3145728, sign: true });
    data.append(FP16x16 { mag: 983040, sign: false });
    data.append(FP16x16 { mag: 917504, sign: true });
    data.append(FP16x16 { mag: 5177344, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 2162688, sign: false });
    data.append(FP16x16 { mag: 7995392, sign: false });
    data.append(FP16x16 { mag: 3801088, sign: true });
    data.append(FP16x16 { mag: 1114112, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 8257536, sign: false });
    data.append(FP16x16 { mag: 3866624, sign: false });
    data.append(FP16x16 { mag: 7798784, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
