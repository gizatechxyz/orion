use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);
    shape.append(2);
    shape.append(1);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 29054, sign: true });
    data.append(FP16x16 { mag: 16853, sign: false });
    data.append(FP16x16 { mag: 55842, sign: true });
    data.append(FP16x16 { mag: 50216, sign: true });
    data.append(FP16x16 { mag: 77247, sign: false });
    data.append(FP16x16 { mag: 12027, sign: true });
    data.append(FP16x16 { mag: 97654, sign: false });
    data.append(FP16x16 { mag: 194489, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
