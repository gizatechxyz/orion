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
    data.append(FP16x16 { mag: 53283, sign: true });
    data.append(FP16x16 { mag: 57948, sign: false });
    data.append(FP16x16 { mag: 182112, sign: false });
    data.append(FP16x16 { mag: 92064, sign: false });
    data.append(FP16x16 { mag: 79866, sign: false });
    data.append(FP16x16 { mag: 12556, sign: true });
    data.append(FP16x16 { mag: 57172, sign: false });
    data.append(FP16x16 { mag: 87126, sign: false });
    data.append(FP16x16 { mag: 39713, sign: false });
    data.append(FP16x16 { mag: 22679, sign: false });
    data.append(FP16x16 { mag: 165649, sign: true });
    data.append(FP16x16 { mag: 13773, sign: true });
    data.append(FP16x16 { mag: 96744, sign: false });
    data.append(FP16x16 { mag: 48255, sign: false });
    data.append(FP16x16 { mag: 78994, sign: false });
    data.append(FP16x16 { mag: 137324, sign: true });
    data.append(FP16x16 { mag: 81318, sign: false });
    data.append(FP16x16 { mag: 37094, sign: false });
    data.append(FP16x16 { mag: 64052, sign: false });
    data.append(FP16x16 { mag: 102111, sign: false });
    data.append(FP16x16 { mag: 51160, sign: true });
    data.append(FP16x16 { mag: 98975, sign: false });
    data.append(FP16x16 { mag: 27582, sign: false });
    data.append(FP16x16 { mag: 68415, sign: false });
    data.append(FP16x16 { mag: 32080, sign: false });
    data.append(FP16x16 { mag: 76560, sign: true });
    data.append(FP16x16 { mag: 70073, sign: false });
    data.append(FP16x16 { mag: 90063, sign: false });
    data.append(FP16x16 { mag: 13883, sign: false });
    data.append(FP16x16 { mag: 124780, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
