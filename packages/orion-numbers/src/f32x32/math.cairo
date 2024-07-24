use core::integer;
use core::num::traits::{WideMul, Sqrt};
use orion_numbers::f32x32::core::{F32x32Impl, f32x32, ONE, HALF};



pub fn abs(a: f32x32) -> f32x32 {
    if a >= 0 {
        a
    } else {
        a * -1_i64
    }
}

pub fn div(a: f32x32, b: f32x32) -> f32x32 {
    let a_i128 = WideMul::wide_mul(a, ONE);
    let res_i128 = a_i128 / b.into();

    // Re-apply sign
    F32x32Impl::new(res_i128.try_into().unwrap())
}

pub fn mul(a: f32x32, b: f32x32) -> f32x32 {
    let prod_i128 = WideMul::wide_mul(a, b);

    // Re-apply sign
    F32x32Impl::new((prod_i128 / ONE.into()).try_into().unwrap())
}

pub fn round(a: f32x32) -> f32x32 {
    //let (div, rem) = DivRem::div_rem(a, ONE.try_into().unwrap());
    let div = Div::div(a, ONE);
    let rem = Rem::rem(a, ONE);

    if (HALF <= rem) {
        F32x32Impl::new_unscaled(div + 1)
    } else {
        F32x32Impl::new_unscaled(div)
    }
}

pub fn sign(a: f32x32) -> f32x32 {
    if a == 0 {
        F32x32Impl::new(0)
    } else if a > 0 {
        ONE
    } else {
        -ONE
    }
}

// Tests
//
// 
// --------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use orion_numbers::f32x32::helpers::{assert_precise, assert_relative};

    use super::{F32x32Impl, ONE, HALF, f32x32, integer, round, sign};


    #[test]
    fn test_sign() {
        let a = F32x32Impl::new(0);
        assert(a.sign() == 0, 'invalid sign (0)');

        let a = F32x32Impl::new(-HALF);
        assert(a.sign() == -ONE, 'invalid sign (-HALF)');

        let a = F32x32Impl::new(HALF);
        assert(a.sign() == ONE, 'invalid sign (HALF)');

        let a = F32x32Impl::new(-ONE);
        assert(a.sign() == -ONE, 'invalid sign (-ONE)');

        let a = F32x32Impl::new(ONE);
        assert(a.sign() == ONE, 'invalid sign (ONE)');
    }
}
