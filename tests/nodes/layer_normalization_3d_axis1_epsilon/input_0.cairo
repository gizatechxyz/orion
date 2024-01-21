use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 176084, sign: true });
    data.append(FP16x16 { mag: 134870, sign: false });
    data.append(FP16x16 { mag: 11194, sign: true });
    data.append(FP16x16 { mag: 34834, sign: true });
    data.append(FP16x16 { mag: 6625, sign: true });
    data.append(FP16x16 { mag: 99311, sign: true });
    data.append(FP16x16 { mag: 84659, sign: true });
    data.append(FP16x16 { mag: 37723, sign: false });
    data.append(FP16x16 { mag: 78508, sign: true });
    data.append(FP16x16 { mag: 31024, sign: false });
    data.append(FP16x16 { mag: 72988, sign: true });
    data.append(FP16x16 { mag: 9818, sign: true });
    data.append(FP16x16 { mag: 34996, sign: false });
    data.append(FP16x16 { mag: 87265, sign: true });
    data.append(FP16x16 { mag: 45795, sign: true });
    data.append(FP16x16 { mag: 45583, sign: true });
    data.append(FP16x16 { mag: 10423, sign: true });
    data.append(FP16x16 { mag: 71376, sign: true });
    data.append(FP16x16 { mag: 31238, sign: true });
    data.append(FP16x16 { mag: 84702, sign: false });
    data.append(FP16x16 { mag: 18617, sign: true });
    data.append(FP16x16 { mag: 90788, sign: false });
    data.append(FP16x16 { mag: 32546, sign: true });
    data.append(FP16x16 { mag: 43302, sign: true });
    data.append(FP16x16 { mag: 90009, sign: true });
    data.append(FP16x16 { mag: 19958, sign: true });
    data.append(FP16x16 { mag: 161585, sign: false });
    data.append(FP16x16 { mag: 40607, sign: false });
    data.append(FP16x16 { mag: 34601, sign: true });
    data.append(FP16x16 { mag: 7286, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
