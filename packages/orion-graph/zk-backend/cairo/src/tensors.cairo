use orion_cairo::numbers::f16x16::f16x16;
use core::array::{ArrayTrait, SpanTrait};
use core::option::OptionTrait;


#[derive(Copy, Drop)]
pub struct Tensor {
    pub shape: Span<usize>,
    pub data: Span<f16x16>,
}

impl f16x16TensorPartialEq of PartialEq<Tensor> {
    fn eq(lhs: @Tensor, rhs: @Tensor) -> bool {
        tensor_eq(*lhs, *rhs)
    }

    fn ne(lhs: @Tensor, rhs: @Tensor) -> bool {
        !tensor_eq(*lhs, *rhs)
    }
}

// Internals
const PRECISION: i32 = 589; // 0.009

fn relative_eq(lhs: @f16x16, rhs: @f16x16) -> bool {
    let diff = *lhs - *rhs;

    let rel_diff = if *lhs != 0 {
        diff / *lhs
    } else {
        diff
    };

    rel_diff <= PRECISION
}

fn tensor_eq(mut lhs: Tensor, mut rhs: Tensor,) -> bool {
    let mut is_eq = true;

    while lhs.shape.len() != 0 && is_eq {
        is_eq = lhs.shape.pop_front().unwrap() == rhs.shape.pop_front().unwrap();
    };

    if !is_eq {
        return false;
    }

    while lhs.data.len() != 0 && is_eq {
        is_eq = relative_eq(lhs.data.pop_front().unwrap(), rhs.data.pop_front().unwrap());
    };

    is_eq
}
