mod input_0;
mod output_0;


use orion::operators::tensor::Complex64Tensor;
use core::array::{ArrayTrait, SpanTrait};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::Complex64TensorPartialEq;

use orion::numbers::complex_number::complex64::Complex64Print;
use orion::numbers::{NumberTrait, complex64};

#[test]
#[available_gas(2000000000)]
fn test_reduce_l2_complex64_axis_0() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = input_0.reduce_l2(0, true);

    assert_eq(y_0, z_0);
}

