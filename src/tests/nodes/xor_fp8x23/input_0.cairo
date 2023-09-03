use array::ArrayTrait;
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::numbers::FP8x23;
use orion::operators::tensor::implementations::tensor_fp8x23::FP8x23Tensor;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 8388608, sign: true });
    data.append(FP8x23 { mag: 8388608, sign: true });
    data.append(FP8x23 { mag: 16777216, sign: true });
    data.append(FP8x23 { mag: 8388608, sign: true });
    data.append(FP8x23 { mag: 8388608, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 8388608, sign: true });
    data.append(FP8x23 { mag: 16777216, sign: true });
    data.append(FP8x23 { mag: 8388608, sign: true });
    data.append(FP8x23 { mag: 8388608, sign: true });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 8388608, sign: false });
    data.append(FP8x23 { mag: 16777216, sign: true });
    data.append(FP8x23 { mag: 8388608, sign: true });
    data.append(FP8x23 { mag: 16777216, sign: false });
    data.append(FP8x23 { mag: 16777216, sign: false });
    data.append(FP8x23 { mag: 16777216, sign: false });
    data.append(FP8x23 { mag: 25165824, sign: true });
    data.append(FP8x23 { mag: 16777216, sign: true });
    data.append(FP8x23 { mag: 25165824, sign: true });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 25165824, sign: true });
    data.append(FP8x23 { mag: 8388608, sign: true });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 8388608, sign: true });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 16777216, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}