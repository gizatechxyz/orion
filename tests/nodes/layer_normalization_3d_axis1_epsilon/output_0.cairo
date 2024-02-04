use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 251, sign: true });
    data.append(FP16x16 { mag: 32754, sign: true });
    data.append(FP16x16 { mag: 33505, sign: false });
    data.append(FP16x16 { mag: 47029, sign: false });
    data.append(FP16x16 { mag: 52651, sign: true });
    data.append(FP16x16 { mag: 95360, sign: true });
    data.append(FP16x16 { mag: 4402, sign: true });
    data.append(FP16x16 { mag: 116033, sign: true });
    data.append(FP16x16 { mag: 7323, sign: false });
    data.append(FP16x16 { mag: 39982, sign: false });
    data.append(FP16x16 { mag: 120917, sign: true });
    data.append(FP16x16 { mag: 29534, sign: false });
    data.append(FP16x16 { mag: 139597, sign: false });
    data.append(FP16x16 { mag: 48872, sign: false });
    data.append(FP16x16 { mag: 44543, sign: false });
    data.append(FP16x16 { mag: 25225, sign: true });
    data.append(FP16x16 { mag: 18016, sign: true });
    data.append(FP16x16 { mag: 78919, sign: false });
    data.append(FP16x16 { mag: 45917, sign: false });
    data.append(FP16x16 { mag: 21307, sign: true });
    data.append(FP16x16 { mag: 90805, sign: true });
    data.append(FP16x16 { mag: 111010, sign: true });
    data.append(FP16x16 { mag: 45210, sign: true });
    data.append(FP16x16 { mag: 7624, sign: false });
    data.append(FP16x16 { mag: 76340, sign: true });
    data.append(FP16x16 { mag: 111388, sign: true });
    data.append(FP16x16 { mag: 143559, sign: false });
    data.append(FP16x16 { mag: 112995, sign: false });
    data.append(FP16x16 { mag: 39651, sign: false });
    data.append(FP16x16 { mag: 74850, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
