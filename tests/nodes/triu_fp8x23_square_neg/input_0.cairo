use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 503316480, sign: true });
    data.append(FP8x23 { mag: 494927872, sign: true });
    data.append(FP8x23 { mag: 427819008, sign: true });
    data.append(FP8x23 { mag: 629145600, sign: false });
    data.append(FP8x23 { mag: 956301312, sign: true });
    data.append(FP8x23 { mag: 452984832, sign: true });
    data.append(FP8x23 { mag: 1056964608, sign: true });
    data.append(FP8x23 { mag: 276824064, sign: true });
    data.append(FP8x23 { mag: 125829120, sign: false });
    data.append(FP8x23 { mag: 41943040, sign: false });
    data.append(FP8x23 { mag: 696254464, sign: false });
    data.append(FP8x23 { mag: 528482304, sign: true });
    data.append(FP8x23 { mag: 159383552, sign: true });
    data.append(FP8x23 { mag: 1056964608, sign: true });
    data.append(FP8x23 { mag: 864026624, sign: true });
    data.append(FP8x23 { mag: 671088640, sign: true });
    data.append(FP8x23 { mag: 25165824, sign: true });
    data.append(FP8x23 { mag: 1015021568, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
