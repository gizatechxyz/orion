use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 60552, sign: false });
    data.append(FP16x16 { mag: 32131, sign: true });
    data.append(FP16x16 { mag: 17263, sign: false });
    data.append(FP16x16 { mag: 17309, sign: false });
    data.append(FP16x16 { mag: 75079, sign: true });
    data.append(FP16x16 { mag: 132412, sign: true });
    data.append(FP16x16 { mag: 72947, sign: false });
    data.append(FP16x16 { mag: 89468, sign: false });
    data.append(FP16x16 { mag: 9763, sign: false });
    data.append(FP16x16 { mag: 43535, sign: true });
    data.append(FP16x16 { mag: 137407, sign: false });
    data.append(FP16x16 { mag: 45433, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
