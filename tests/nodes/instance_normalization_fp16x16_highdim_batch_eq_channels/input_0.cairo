use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);
    shape.append(1);
    shape.append(1);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 28963, sign: true });
    data.append(FP16x16 { mag: 40315, sign: true });
    data.append(FP16x16 { mag: 98072, sign: false });
    data.append(FP16x16 { mag: 15541, sign: false });
    data.append(FP16x16 { mag: 23259, sign: false });
    data.append(FP16x16 { mag: 104825, sign: false });
    data.append(FP16x16 { mag: 4156, sign: false });
    data.append(FP16x16 { mag: 10319, sign: false });
    data.append(FP16x16 { mag: 102523, sign: true });
    data.append(FP16x16 { mag: 85995, sign: true });
    data.append(FP16x16 { mag: 64059, sign: false });
    data.append(FP16x16 { mag: 15066, sign: true });
    data.append(FP16x16 { mag: 37326, sign: false });
    data.append(FP16x16 { mag: 2589, sign: false });
    data.append(FP16x16 { mag: 38698, sign: true });
    data.append(FP16x16 { mag: 191669, sign: false });
    data.append(FP16x16 { mag: 79835, sign: true });
    data.append(FP16x16 { mag: 81296, sign: false });
    data.append(FP16x16 { mag: 38760, sign: false });
    data.append(FP16x16 { mag: 2552, sign: false });
    data.append(FP16x16 { mag: 98397, sign: true });
    data.append(FP16x16 { mag: 36264, sign: false });
    data.append(FP16x16 { mag: 28751, sign: true });
    data.append(FP16x16 { mag: 4126, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
