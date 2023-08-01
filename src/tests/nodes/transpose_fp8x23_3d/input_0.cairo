use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;

fn input_0() -> Tensor<FixedType> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FixedType { mag: 905969664, sign: false });
    data.append(FixedType { mag: 838860800, sign: true });
    data.append(FixedType { mag: 142606336, sign: false });
    data.append(FixedType { mag: 159383552, sign: false });
    data.append(FixedType { mag: 721420288, sign: true });
    data.append(FixedType { mag: 276824064, sign: false });
    data.append(FixedType { mag: 226492416, sign: false });
    data.append(FixedType { mag: 629145600, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP8x23) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}