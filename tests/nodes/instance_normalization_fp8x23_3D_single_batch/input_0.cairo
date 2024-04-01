use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1551343, sign: false });
    data.append(FP8x23 { mag: 3645780, sign: true });
    data.append(FP8x23 { mag: 19168982, sign: false });
    data.append(FP8x23 { mag: 17625466, sign: false });
    data.append(FP8x23 { mag: 7313506, sign: false });
    data.append(FP8x23 { mag: 340718, sign: true });
    data.append(FP8x23 { mag: 17950070, sign: false });
    data.append(FP8x23 { mag: 3596672, sign: true });
    data.append(FP8x23 { mag: 3351185, sign: true });
    data.append(FP8x23 { mag: 14032929, sign: true });
    data.append(FP8x23 { mag: 9837649, sign: true });
    data.append(FP8x23 { mag: 14232717, sign: false });
    data.append(FP8x23 { mag: 603129, sign: false });
    data.append(FP8x23 { mag: 12666451, sign: false });
    data.append(FP8x23 { mag: 847016, sign: false });
    data.append(FP8x23 { mag: 4107616, sign: false });
    data.append(FP8x23 { mag: 2414115, sign: false });
    data.append(FP8x23 { mag: 12068298, sign: true });
    data.append(FP8x23 { mag: 5929889, sign: false });
    data.append(FP8x23 { mag: 6687957, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
