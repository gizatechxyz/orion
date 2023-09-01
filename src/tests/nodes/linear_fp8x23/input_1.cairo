use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::numbers::FP8x23;
use orion::operators::tensor::implementations::tensor_fp8x23::Tensor_fp8x23;

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1236715, sign: false });
    data.append(FP8x23 { mag: 4771319, sign: false });
    data.append(FP8x23 { mag: 8392691, sign: false });
    data.append(FP8x23 { mag: 36629024, sign: true });
    data.append(FP8x23 { mag: 34768195, sign: false });
    data.append(FP8x23 { mag: 2858178, sign: false });

    
    TensorTrait::new(shape.span(), data.span())
}