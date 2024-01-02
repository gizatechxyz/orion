use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Array<Tensor<FP16x16>> {
    let mut sequence = ArrayTrait::new();

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 262144, sign: true });
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 131072, sign: false });
    data.append(FP16x16 { mag: 327680, sign: true });
    data.append(FP16x16 { mag: 131072, sign: true });
    data.append(FP16x16 { mag: 262144, sign: false });
    data.append(FP16x16 { mag: 262144, sign: true });
    data.append(FP16x16 { mag: 262144, sign: true });
    data.append(FP16x16 { mag: 262144, sign: true });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 327680, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 131072, sign: false });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 196608, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 327680, sign: true });
    data.append(FP16x16 { mag: 196608, sign: true });
    data.append(FP16x16 { mag: 262144, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 196608, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 131072, sign: false });
    data.append(FP16x16 { mag: 327680, sign: true });
    data.append(FP16x16 { mag: 131072, sign: true });
    data.append(FP16x16 { mag: 393216, sign: true });
    data.append(FP16x16 { mag: 262144, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 196608, sign: true });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 196608, sign: true });
    data.append(FP16x16 { mag: 327680, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 131072, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 196608, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 262144, sign: true });
    data.append(FP16x16 { mag: 196608, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 131072, sign: false });
    data.append(FP16x16 { mag: 393216, sign: true });
    data.append(FP16x16 { mag: 327680, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
