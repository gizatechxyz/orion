use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 5612821, sign: false });
    data.append(FP8x23 { mag: 6015127, sign: true });
    data.append(FP8x23 { mag: 1074469, sign: true });
    data.append(FP8x23 { mag: 1804540, sign: true });
    data.append(FP8x23 { mag: 4808518, sign: false });
    data.append(FP8x23 { mag: 574150, sign: false });
    data.append(FP8x23 { mag: 4076462, sign: true });
    data.append(FP8x23 { mag: 2224269, sign: false });
    data.append(FP8x23 { mag: 3369341, sign: false });
    data.append(FP8x23 { mag: 3700297, sign: true });
    data.append(FP8x23 { mag: 8946182, sign: false });
    data.append(FP8x23 { mag: 281041, sign: true });
    data.append(FP8x23 { mag: 869145, sign: true });
    data.append(FP8x23 { mag: 4819824, sign: true });
    data.append(FP8x23 { mag: 1708213, sign: false });
    data.append(FP8x23 { mag: 1960672, sign: true });
    data.append(FP8x23 { mag: 8482395, sign: true });
    data.append(FP8x23 { mag: 1075301, sign: false });
    data.append(FP8x23 { mag: 13488073, sign: false });
    data.append(FP8x23 { mag: 11090812, sign: false });
    data.append(FP8x23 { mag: 2568531, sign: true });
    data.append(FP8x23 { mag: 4355375, sign: true });
    data.append(FP8x23 { mag: 235232, sign: true });
    data.append(FP8x23 { mag: 9703692, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
