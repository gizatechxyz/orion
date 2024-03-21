use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 11009153, sign: true });
    data.append(FP8x23 { mag: 8461836, sign: false });
    data.append(FP8x23 { mag: 6540115, sign: true });
    data.append(FP8x23 { mag: 10980352, sign: true });
    data.append(FP8x23 { mag: 4007365, sign: true });
    data.append(FP8x23 { mag: 3132520, sign: false });
    data.append(FP8x23 { mag: 11061564, sign: true });
    data.append(FP8x23 { mag: 4159334, sign: true });
    data.append(FP8x23 { mag: 10782781, sign: false });
    data.append(FP8x23 { mag: 12883323, sign: true });
    data.append(FP8x23 { mag: 12437596, sign: true });
    data.append(FP8x23 { mag: 2124169, sign: false });
    data.append(FP8x23 { mag: 13132270, sign: true });
    data.append(FP8x23 { mag: 7568067, sign: false });
    data.append(FP8x23 { mag: 6729016, sign: true });
    data.append(FP8x23 { mag: 5032391, sign: true });
    data.append(FP8x23 { mag: 15123860, sign: true });
    data.append(FP8x23 { mag: 8886124, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
