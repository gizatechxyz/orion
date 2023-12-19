use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 131072, sign: false });
    data.append(FP16x16 { mag: 393216, sign: false });
    data.append(FP16x16 { mag: 851968, sign: false });
    data.append(FP16x16 { mag: 1245184, sign: false });
    data.append(FP16x16 { mag: 1638400, sign: false });
    data.append(FP16x16 { mag: 2162688, sign: false });
    data.append(FP16x16 { mag: 2490368, sign: false });
    data.append(FP16x16 { mag: 2949120, sign: false });
    data.append(FP16x16 { mag: 3342336, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
