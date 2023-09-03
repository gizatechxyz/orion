use array::ArrayTrait;

use orion::operators::tensor::{TensorTrait, Tensor};
use orion::numbers::fixed_point::core::{FixedTrait};
use orion::operators::tensor::implementations::tensor_fp8x23::FP8x23Tensor;use orion::numbers::FP8x23;


fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 905969664, sign: false });
    data.append(FP8x23 { mag: 838860800, sign: true });
    data.append(FP8x23 { mag: 142606336, sign: false });
    data.append(FP8x23 { mag: 159383552, sign: false });
    data.append(FP8x23 { mag: 721420288, sign: true });
    data.append(FP8x23 { mag: 276824064, sign: false });
    data.append(FP8x23 { mag: 226492416, sign: false });
    data.append(FP8x23 { mag: 629145600, sign: true });

    
    TensorTrait::new(shape.span(), data.span())
}