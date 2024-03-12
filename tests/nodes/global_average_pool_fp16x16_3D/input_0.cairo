use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 1927168, sign: false });
    data.append(FP16x16 { mag: 736768, sign: false });
    data.append(FP16x16 { mag: 1823744, sign: false });
    data.append(FP16x16 { mag: 831488, sign: true });
    data.append(FP16x16 { mag: 768512, sign: false });
    data.append(FP16x16 { mag: 1181696, sign: true });
    data.append(FP16x16 { mag: 1544192, sign: true });
    data.append(FP16x16 { mag: 1265664, sign: false });
    data.append(FP16x16 { mag: 1175552, sign: true });
    data.append(FP16x16 { mag: 261376, sign: true });
    data.append(FP16x16 { mag: 1157120, sign: false });
    data.append(FP16x16 { mag: 812032, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
