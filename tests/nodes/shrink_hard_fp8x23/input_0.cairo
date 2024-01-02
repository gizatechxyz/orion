use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 24359422, sign: false });
    data.append(FP8x23 { mag: 1702352, sign: true });
    data.append(FP8x23 { mag: 24476144, sign: false });
    data.append(FP8x23 { mag: 5048882, sign: false });
    data.append(FP8x23 { mag: 15295379, sign: false });
    data.append(FP8x23 { mag: 20023509, sign: false });
    data.append(FP8x23 { mag: 22548090, sign: true });
    data.append(FP8x23 { mag: 2168923, sign: true });
    data.append(FP8x23 { mag: 1169120, sign: false });
    data.append(FP8x23 { mag: 21744823, sign: false });
    data.append(FP8x23 { mag: 3101366, sign: false });
    data.append(FP8x23 { mag: 22241666, sign: true });
    data.append(FP8x23 { mag: 18004240, sign: true });
    data.append(FP8x23 { mag: 5000783, sign: false });
    data.append(FP8x23 { mag: 16034040, sign: false });
    data.append(FP8x23 { mag: 16596277, sign: false });
    data.append(FP8x23 { mag: 2746457, sign: true });
    data.append(FP8x23 { mag: 3600474, sign: true });
    data.append(FP8x23 { mag: 250279, sign: true });
    data.append(FP8x23 { mag: 12351379, sign: false });
    data.append(FP8x23 { mag: 16444516, sign: false });
    data.append(FP8x23 { mag: 5440394, sign: false });
    data.append(FP8x23 { mag: 1795224, sign: true });
    data.append(FP8x23 { mag: 15465976, sign: true });
    data.append(FP8x23 { mag: 784106, sign: false });
    data.append(FP8x23 { mag: 10988354, sign: true });
    data.append(FP8x23 { mag: 10264893, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
