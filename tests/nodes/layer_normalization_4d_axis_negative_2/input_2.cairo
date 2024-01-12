use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn input_2() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 14435005, sign: false });
    data.append(FP8x23 { mag: 275345, sign: false });
    data.append(FP8x23 { mag: 7948101, sign: true });
    data.append(FP8x23 { mag: 124471, sign: true });
    data.append(FP8x23 { mag: 11083371, sign: true });
    data.append(FP8x23 { mag: 2513924, sign: true });
    data.append(FP8x23 { mag: 6387124, sign: true });
    data.append(FP8x23 { mag: 5452904, sign: false });
    data.append(FP8x23 { mag: 12271809, sign: true });
    data.append(FP8x23 { mag: 15327354, sign: true });
    data.append(FP8x23 { mag: 3795402, sign: true });
    data.append(FP8x23 { mag: 2307268, sign: false });
    data.append(FP8x23 { mag: 5731544, sign: false });
    data.append(FP8x23 { mag: 4011370, sign: true });
    data.append(FP8x23 { mag: 3178152, sign: false });
    data.append(FP8x23 { mag: 14982171, sign: false });
    data.append(FP8x23 { mag: 2850000, sign: true });
    data.append(FP8x23 { mag: 9445099, sign: false });
    data.append(FP8x23 { mag: 8149556, sign: false });
    data.append(FP8x23 { mag: 8935026, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
