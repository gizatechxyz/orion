use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::Complex64Tensor;
use orion::numbers::{NumberTrait, complex64};
use orion::numbers::{FixedTrait, FP64x64};

fn output_0() -> Tensor<complex64> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data
        .append(
            complex64 {
                real: FP64x64 { mag: 78262906948318920704, sign: false },
                img: FP64x64 { mag: 0, sign: false }
            }
        );
    data
        .append(
            complex64 {
                real: FP64x64 { mag: 115199879809953955840, sign: false },
                img: FP64x64 { mag: 0, sign: false }
            }
        );
    data
        .append(
            complex64 {
                real: FP64x64 { mag: 109132409670155108352, sign: false },
                img: FP64x64 { mag: 0, sign: false }
            }
        );
    TensorTrait::new(shape.span(), data.span())
}
