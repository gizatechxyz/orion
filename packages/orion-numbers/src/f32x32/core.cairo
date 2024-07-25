use orion_numbers::f32x32::math;
use orion_numbers::FixedTrait;

pub type f32x32 = i64;


pub impl F32x32Impl of FixedTrait<f32x32> {
    // CONSTANTS
    const ZERO: f32x32 = 0;
    const HALF: f32x32 = 2147483648; // 2 ** 31
    const ONE: f32x32 = 4294967296; // 2 ** 32
    const TWO: f32x32 = 8589934592; // 2 ** 33
    const MAX: f32x32 = 9223372036854775807; // 2 ** 63 -1 
    const MIN: f32x32 = -9223372036854775808; // -2 ** 63

    fn new_unscaled(x: i64) -> f32x32 {
        x * Self::ONE
    }

    fn new(x: i64) -> f32x32 {
        x
    }

    fn from_felt(x: felt252) -> f32x32 {
        x.try_into().unwrap()
    }

    fn from_unscaled_felt(x: felt252) -> f32x32 {
        return FixedTrait::from_felt(x * Self::ONE.into());
    }

    fn abs(self: f32x32) -> f32x32 {
        math::abs(self)
    }

    fn div(self: f32x32, rhs: f32x32) -> f32x32 {
        math::div(self, rhs)
    }

    fn mul(self: f32x32, rhs: f32x32) -> f32x32 {
        math::mul(self, rhs)
    }

    fn sign(self: f32x32) -> f32x32 {
        math::sign(self)
    }

    fn acos(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn acosh(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn asin(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn asinh(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn atan(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn atanh(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn add(lhs: f32x32, rhs: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn ceil(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn cos(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn cosh(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn exp(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn exp2(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn floor(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn ln(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn log2(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn log10(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn pow(self: f32x32, b: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn round(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn sin(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn sinh(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn sqrt(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn tan(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn tanh(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }


    fn sub(lhs: f32x32, rhs: f32x32) -> f32x32 {
        panic!("not implem yet")
    }

    fn NaN() -> f32x32 {
        -0
    }

    fn is_nan(self: f32x32) -> bool {
        self == -0
    }

    fn INF() -> f32x32 {
        Self::MAX
    }

    fn POS_INF() -> f32x32 {
        Self::MAX
    }

    fn NEG_INF() -> f32x32 {
        Self::MIN
    }

    fn is_inf(self: f32x32) -> bool {
        self == Self::MAX
    }

    fn is_pos_inf(self: f32x32) -> bool {
        self == Self::MAX
    }

    fn is_neg_inf(self: f32x32) -> bool {
        self == Self::MIN
    }

    fn erf(self: f32x32) -> f32x32 {
        panic!("not implem yet")
    }
}
