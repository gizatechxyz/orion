use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::{FixedTrait, FixedType, FixedImpl};
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;

fn input_2() -> Tensor<FixedType> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FixedType { mag: 3538944, sign: false });
    data.append(FixedType { mag: 3604480, sign: false });
    data.append(FixedType { mag: 3670016, sign: false });
    data.append(FixedType { mag: 3735552, sign: false });
    data.append(FixedType { mag: 3801088, sign: false });
    data.append(FixedType { mag: 3866624, sign: false });
    data.append(FixedType { mag: 3932160, sign: false });
    data.append(FixedType { mag: 3997696, sign: false });
    data.append(FixedType { mag: 4063232, sign: false });
    data.append(FixedType { mag: 4128768, sign: false });
    data.append(FixedType { mag: 4194304, sign: false });
    data.append(FixedType { mag: 4259840, sign: false });
    data.append(FixedType { mag: 4325376, sign: false });
    data.append(FixedType { mag: 4390912, sign: false });
    data.append(FixedType { mag: 4456448, sign: false });
    data.append(FixedType { mag: 4521984, sign: false });
    data.append(FixedType { mag: 4587520, sign: false });
    data.append(FixedType { mag: 4653056, sign: false });
    data.append(FixedType { mag: 4718592, sign: false });
    data.append(FixedType { mag: 4784128, sign: false });
    data.append(FixedType { mag: 4849664, sign: false });
    data.append(FixedType { mag: 4915200, sign: false });
    data.append(FixedType { mag: 4980736, sign: false });
    data.append(FixedType { mag: 5046272, sign: false });
    data.append(FixedType { mag: 5111808, sign: false });
    data.append(FixedType { mag: 5177344, sign: false });
    data.append(FixedType { mag: 5242880, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}