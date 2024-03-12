use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 1332224, sign: false });
    data.append(FP16x16 { mag: 496128, sign: false });
    data.append(FP16x16 { mag: 206592, sign: true });
    data.append(FP16x16 { mag: 139264, sign: true });
    data.append(FP16x16 { mag: 718336, sign: true });
    data.append(FP16x16 { mag: 984576, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
