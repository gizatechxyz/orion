use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 5171936, sign: false });
    data.append(FP8x23 { mag: 2660687, sign: false });
    data.append(FP8x23 { mag: 2533979, sign: false });
    data.append(FP8x23 { mag: 5576122, sign: false });
    data.append(FP8x23 { mag: 2013178, sign: false });
    data.append(FP8x23 { mag: 5819445, sign: false });
    data.append(FP8x23 { mag: 7588031, sign: false });
    data.append(FP8x23 { mag: 522071, sign: false });
    data.append(FP8x23 { mag: 523276, sign: false });
    data.append(FP8x23 { mag: 7309347, sign: false });
    data.append(FP8x23 { mag: 4826665, sign: false });
    data.append(FP8x23 { mag: 3283437, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
