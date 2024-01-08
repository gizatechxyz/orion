use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::Complex64Tensor;
use orion::numbers::{NumberTrait, complex64};
use orion::numbers::{FixedTrait, FP64x64};

fn input_0() -> Tensor<complex64> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data
        .append(
            complex64 {
                real: FP64x64 { mag: 18446744073709551616, sign: false },
                img: FP64x64 { mag: 36893488147419103232, sign: false }
            }
        );
    data
        .append(
            complex64 {
                real: FP64x64 { mag: 36893488147419103232, sign: false },
                img: FP64x64 { mag: 18446744073709551616, sign: true }
            }
        );
    data
        .append(
            complex64 {
                real: FP64x64 { mag: 55340232221128654848, sign: false },
                img: FP64x64 { mag: 55340232221128654848, sign: true }
            }
        );
    data
        .append(
            complex64 {
                real: FP64x64 { mag: 55340232221128654848, sign: false },
                img: FP64x64 { mag: 36893488147419103232, sign: true }
            }
        );
    data
        .append(
            complex64 {
                real: FP64x64 { mag: 55340232221128654848, sign: false },
                img: FP64x64 { mag: 92233720368547758080, sign: false }
            }
        );
    data
        .append(
            complex64 {
                real: FP64x64 { mag: 73786976294838206464, sign: false },
                img: FP64x64 { mag: 18446744073709551616, sign: true }
            }
        );
    TensorTrait::new(shape.span(), data.span())
}
