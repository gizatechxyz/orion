use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::numbers::FP8x23;
use orion::operators::tensor::implementations::tensor_fp8x23::Tensor_fp8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 9774601, sign: false });
    data.append(FP8x23 { mag: 29725534, sign: false });
    data.append(FP8x23 { mag: 20607627, sign: false });
    data.append(FP8x23 { mag: 39982645, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}