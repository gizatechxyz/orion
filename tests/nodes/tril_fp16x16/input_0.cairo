use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 3801088, sign: false });
    data.append(FP16x16 { mag: 5701632, sign: true });
    data.append(FP16x16 { mag: 8126464, sign: true });
    data.append(FP16x16 { mag: 6422528, sign: false });
    data.append(FP16x16 { mag: 4784128, sign: true });
    data.append(FP16x16 { mag: 4456448, sign: false });
    data.append(FP16x16 { mag: 2293760, sign: true });
    data.append(FP16x16 { mag: 1048576, sign: false });
    data.append(FP16x16 { mag: 7208960, sign: false });
    data.append(FP16x16 { mag: 5963776, sign: true });
    data.append(FP16x16 { mag: 4653056, sign: false });
    data.append(FP16x16 { mag: 7733248, sign: false });
    data.append(FP16x16 { mag: 3866624, sign: true });
    data.append(FP16x16 { mag: 6684672, sign: false });
    data.append(FP16x16 { mag: 5177344, sign: true });
    data.append(FP16x16 { mag: 4194304, sign: false });
    data.append(FP16x16 { mag: 393216, sign: false });
    data.append(FP16x16 { mag: 5177344, sign: true });
    data.append(FP16x16 { mag: 8126464, sign: false });
    data.append(FP16x16 { mag: 3276800, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
