use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 7479190, sign: false });
    data.append(FP8x23 { mag: 16922838, sign: true });
    data.append(FP8x23 { mag: 473178, sign: true });
    data.append(FP8x23 { mag: 3721511, sign: true });
    data.append(FP8x23 { mag: 3774515, sign: true });
    data.append(FP8x23 { mag: 6113407, sign: false });
    data.append(FP8x23 { mag: 8620234, sign: false });
    data.append(FP8x23 { mag: 5367104, sign: true });
    data.append(FP8x23 { mag: 3900209, sign: true });
    data.append(FP8x23 { mag: 5772277, sign: false });
    data.append(FP8x23 { mag: 8285089, sign: true });
    data.append(FP8x23 { mag: 6523560, sign: false });
    data.append(FP8x23 { mag: 8672672, sign: false });
    data.append(FP8x23 { mag: 4406916, sign: false });
    data.append(FP8x23 { mag: 43023, sign: false });
    data.append(FP8x23 { mag: 11408292, sign: true });
    data.append(FP8x23 { mag: 3561506, sign: false });
    data.append(FP8x23 { mag: 9769441, sign: true });
    data.append(FP8x23 { mag: 12925958, sign: true });
    data.append(FP8x23 { mag: 8378401, sign: true });
    data.append(FP8x23 { mag: 8723198, sign: false });
    data.append(FP8x23 { mag: 4165058, sign: false });
    data.append(FP8x23 { mag: 14718410, sign: false });
    data.append(FP8x23 { mag: 3915621, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
