use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(1);
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 3868544, sign: false });
    data.append(FP8x23 { mag: 4293062, sign: true });
    data.append(FP8x23 { mag: 13295421, sign: false });
    data.append(FP8x23 { mag: 3500332, sign: false });
    data.append(FP8x23 { mag: 14828980, sign: true });
    data.append(FP8x23 { mag: 6296813, sign: true });
    data.append(FP8x23 { mag: 6553457, sign: false });
    data.append(FP8x23 { mag: 11025182, sign: false });
    data.append(FP8x23 { mag: 34370624, sign: false });
    data.append(FP8x23 { mag: 23510534, sign: false });
    data.append(FP8x23 { mag: 5712811, sign: false });
    data.append(FP8x23 { mag: 10839106, sign: false });
    data.append(FP8x23 { mag: 7711501, sign: true });
    data.append(FP8x23 { mag: 8035198, sign: true });
    data.append(FP8x23 { mag: 7233247, sign: true });
    data.append(FP8x23 { mag: 7374639, sign: true });
    data.append(FP8x23 { mag: 7471208, sign: true });
    data.append(FP8x23 { mag: 7400153, sign: true });
    data.append(FP8x23 { mag: 4067936, sign: false });
    data.append(FP8x23 { mag: 1369792, sign: false });
    data.append(FP8x23 { mag: 14639200, sign: false });
    data.append(FP8x23 { mag: 2899121, sign: true });
    data.append(FP8x23 { mag: 11357261, sign: true });
    data.append(FP8x23 { mag: 10575106, sign: true });
    data.append(FP8x23 { mag: 11940238, sign: false });
    data.append(FP8x23 { mag: 7253638, sign: false });
    data.append(FP8x23 { mag: 37733248, sign: false });
    data.append(FP8x23 { mag: 13071811, sign: false });
    data.append(FP8x23 { mag: 8123758, sign: false });
    data.append(FP8x23 { mag: 13889017, sign: false });
    data.append(FP8x23 { mag: 7267980, sign: true });
    data.append(FP8x23 { mag: 7725275, sign: true });
    data.append(FP8x23 { mag: 7985550, sign: true });
    data.append(FP8x23 { mag: 7210700, sign: true });
    data.append(FP8x23 { mag: 7563720, sign: true });
    data.append(FP8x23 { mag: 7472722, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
