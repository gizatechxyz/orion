use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 85392644, sign: false });
    data.append(FP8x23 { mag: 61594092, sign: false });
    data.append(FP8x23 { mag: 163676643, sign: true });
    data.append(FP8x23 { mag: 180530738, sign: false });
    data.append(FP8x23 { mag: 168048412, sign: true });
    data.append(FP8x23 { mag: 5915510, sign: false });
    data.append(FP8x23 { mag: 9047009, sign: true });
    data.append(FP8x23 { mag: 46030420, sign: false });
    data.append(FP8x23 { mag: 184797857, sign: false });
    data.append(FP8x23 { mag: 129370611, sign: false });
    data.append(FP8x23 { mag: 174006060, sign: true });
    data.append(FP8x23 { mag: 162252480, sign: false });
    data.append(FP8x23 { mag: 139240444, sign: true });
    data.append(FP8x23 { mag: 168836878, sign: true });
    data.append(FP8x23 { mag: 246913333, sign: true });
    data.append(FP8x23 { mag: 1047194, sign: true });
    data.append(FP8x23 { mag: 238599466, sign: true });
    data.append(FP8x23 { mag: 216763643, sign: true });
    data.append(FP8x23 { mag: 40581779, sign: true });
    data.append(FP8x23 { mag: 209811161, sign: true });
    data.append(FP8x23 { mag: 250078311, sign: false });
    data.append(FP8x23 { mag: 31811183, sign: true });
    data.append(FP8x23 { mag: 36411415, sign: true });
    data.append(FP8x23 { mag: 107986324, sign: false });
    data.append(FP8x23 { mag: 69727339, sign: false });
    data.append(FP8x23 { mag: 223159880, sign: true });
    data.append(FP8x23 { mag: 184932087, sign: true });
    data.append(FP8x23 { mag: 118617436, sign: false });
    data.append(FP8x23 { mag: 134825391, sign: true });
    data.append(FP8x23 { mag: 217861279, sign: false });
    data.append(FP8x23 { mag: 199069387, sign: false });
    data.append(FP8x23 { mag: 192925915, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
