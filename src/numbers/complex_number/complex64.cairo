use core::debug::PrintTrait;

use orion::numbers::complex_number::complex_trait::ComplexTrait;
use orion::numbers::{FP64x64, FP64x64Impl, FP32x32, FP32x32Impl, FixedTrait};

// ====================== Complex 64 ======================

// complex64 represents a complex number in the Cartesian form z = a + bi where a and b are Fixed Points FP64x64.
// The real field holds the value of the real part.
// The img field holds the value of the imaginary part.
#[derive(Serde, Copy, Drop)]
struct complex64 {
    real: FP64x64,
    img: FP64x64,
}

// CONSTANTS for FP64x64

const PI: u128 = 57952155664616982739;
const HALF_PI: u128 = 28976077832308491370;
const TWO: u128 = 36893488147419103232;
const E: u128 = 50143449208471493718;

impl Complex64Impl of ComplexTrait<complex64, FP64x64> {
    fn new(real: FP64x64, img: FP64x64) -> complex64 {
        complex64 { real, img }
    }

    fn real(self: complex64) -> FP64x64 {
        self.real
    }

    fn img(self: complex64) -> FP64x64 {
        self.img
    }

    fn conjugate(self: complex64) -> complex64 {
        ComplexTrait::new(self.real, -self.img)
    }

    fn zero() -> complex64 {
        return complex64 { real: FixedTrait::ZERO(), img: FP64x64Impl::ZERO() };
    }

    fn one() -> complex64 {
        return complex64 { real: FP64x64Impl::ONE(), img: FP64x64Impl::ZERO() };
    }

    fn mag(self: complex64) -> FP64x64 {
        let two = FP64x64Impl::new(TWO, false);
        (self.real.pow(two) + self.img.pow(two)).sqrt()
    }

    fn arg(self: complex64) -> FP64x64 {
        atan2(self.real, self.img)
    }

    fn exp(self: complex64) -> complex64 {
        let real = self.real.exp() * self.img.cos();
        let img = self.real.exp() * self.img.sin();
        complex64 { real, img }
    }

    fn sqrt(self: complex64) -> complex64 {
        let x = self.real;
        let y = self.img;
        let two = FP64x64Impl::new(TWO, false);
        let real = (((x.pow(two) + y.pow(two)).sqrt() + x) / two).sqrt();
        let img = (((x.pow(two) + y.pow(two)).sqrt() - x) / two).sqrt();
        let img = FP64x64Impl::new(img.mag, y.sign);
        complex64 { real, img }
    }

    fn ln(self: complex64) -> complex64 {
        let real = self.mag().ln();
        let img = self.arg();
        complex64 { real, img }
    }

    fn pow(self: complex64, b: complex64) -> complex64 {
        let two = FP64x64Impl::new(TWO, false);
        let x = self.real;
        let y = self.img;

        //z^2=(a^2-b^2)+2abi
        if (b.real == two && b.img == FP64x64Impl::new(0, false)) {
            let real = x.pow(two) - y.pow(two);
            let img = two * x * y;
            return complex64 { real, img };
        }

        //(a+bi)^n=r^n(cos(nθ)+isin(nθ))
        if (b.img == FP64x64Impl::new(0, false)) {
            let mag_pow_n = self.mag().pow(b.real);
            let arg_mul_n = b.real * self.arg();
            let real = mag_pow_n * arg_mul_n.cos();
            let img = mag_pow_n * arg_mul_n.sin();
            return complex64 { real, img };
        }

        //let z = (a+bi)  (a+bi)^(c+di)= e^(c * ln(mag(z)) - d arg(z)) * cos (c * arg(z) + d ln(mag(r))) + i * e^(c * ln(mag(z)) - d arg(z)) * cos (c * arg(z) + d ln(mag(r)))
        //let A = e^(c * ln(mag(z)) - d arg(z)) and B = c * arg(z) + d ln(mag(r))
        //(a+bi)^(c+di)= A * cos (B) + i * A * sin B
        let A = FP64x64Impl::new(E, false).pow(b.real * self.mag().ln() - b.img * self.arg());
        let B = b.real * self.arg() + b.img * self.mag().ln();
        let real = A * B.cos();
        let img = A * B.sin();
        complex64 { real, img }
    }


    fn to_polar(self: complex64) -> (FP64x64, FP64x64) {
        let mag = self.mag();
        let arg = self.arg();
        return (mag, arg);
    }

    fn from_polar(mag: FP64x64, arg: FP64x64) -> complex64 {
        let real = mag * arg.cos();
        let img = mag * arg.sin();
        complex64 { real, img }
    }

    fn reciprocal(self: complex64) -> complex64 {
        let two = FP64x64Impl::new(TWO, false);
        let x = self.real;
        let y = self.img;

        let real = x / (x.pow(two) + y.pow(two));
        let img = -y / (x.pow(two) + y.pow(two));
        complex64 { real, img }
    }
}

fn atan2(x: FP64x64, y: FP64x64) -> FP64x64 {
    let two = FP64x64Impl::new(TWO, false);
    if (y != FP64x64Impl::ZERO() || x > FP64x64Impl::ZERO()) {
        return two * (y / (x + (x.pow(two) + y.pow(two)).sqrt())).atan();
    } else if x < FP64x64Impl::ZERO() {
        return FP64x64Impl::new(PI, false);
    } else {
        panic(array!['undifined'])
    }
}

impl complex64Print of PrintTrait<complex64> {
    fn print(self: complex64) {
        self.real.print();
        '+'.print();
        self.img.print();
        'i'.print();
    }
}

// Implements the Add trait for complex64.
impl complex64Add of Add<complex64> {
    fn add(lhs: complex64, rhs: complex64) -> complex64 {
        complex64_add(lhs, rhs)
    }
}

// Implements the AddEq trait for complex64.
impl complex64AddEq of AddEq<complex64> {
    #[inline(always)]
    fn add_eq(ref self: complex64, other: complex64) {
        self = Add::add(self, other);
    }
}

// Implements the Sub trait for complex64.
impl complex64Sub of Sub<complex64> {
    fn sub(lhs: complex64, rhs: complex64) -> complex64 {
        complex64_sub(lhs, rhs)
    }
}

// Implements the SubEq trait for complex64.
impl complex64SubEq of SubEq<complex64> {
    #[inline(always)]
    fn sub_eq(ref self: complex64, other: complex64) {
        self = Sub::sub(self, other);
    }
}

// Implements the Mul trait for complex64.
impl complex64Mul of Mul<complex64> {
    fn mul(lhs: complex64, rhs: complex64) -> complex64 {
        complex64_mul(lhs, rhs)
    }
}

// Implements the MulEq trait for complex64.
impl complex64MulEq of MulEq<complex64> {
    #[inline(always)]
    fn mul_eq(ref self: complex64, other: complex64) {
        self = Mul::mul(self, other);
    }
}

// Implements the Div trait for complex64.
impl complex64Div of Div<complex64> {
    fn div(lhs: complex64, rhs: complex64) -> complex64 {
        complex64_div(lhs, rhs)
    }
}

// Implements the DivEq trait for complex64.
impl complex64DivEq of DivEq<complex64> {
    #[inline(always)]
    fn div_eq(ref self: complex64, other: complex64) {
        self = Div::div(self, other);
    }
}


// Implements the PartialEq trait for complex64.
impl complex64PartialEq of PartialEq<complex64> {
    fn eq(lhs: @complex64, rhs: @complex64) -> bool {
        complex64_eq(*lhs, *rhs)
    }

    fn ne(lhs: @complex64, rhs: @complex64) -> bool {
        complex64_ne(*lhs, *rhs)
    }
}

// Implements the Neg trait for complex64.
impl i8Neg of Neg<complex64> {
    fn neg(a: complex64) -> complex64 {
        complex64_neg(a)
    }
}

/// Cf: ComplexTrait::new docstring

// Adds two complex64 complex numbers.
//
// The sum of two complex numbers (x + yi) + (u + vi) = (x + u) + (y + v)i.
// The result is a new complex number where the real part equals (x + u) and the imaginary part equals (y + v).
// # Arguments
// * `a` - The first complex64 to add.
// * `b` - The second complex64 to add.
// # Returns
// * `complex64` - The sum of `a` and `b`.
fn complex64_add(a: complex64, b: complex64) -> complex64 {
    let real = a.real + b.real;
    let img = a.img + b.img;
    return ComplexTrait::new(real, img);
}

// Subtracts complex64 complex numbers.
//
// The sum of two complex numbers (x + yi) - (u + vi) = (x - u) + (y - v)i.
// The result is a new complex number where the real part equals (x - u) and the imaginary part equals (y - v).
// # Arguments
// * `a` - The first complex64 to subtract.
// * `b` - The second complex64 to subtract.
// # Returns
// * `complex64` - The difference of `a` and `b`.
fn complex64_sub(a: complex64, b: complex64) -> complex64 {
    let real = a.real - b.real;
    let img = a.img - b.img;
    return ComplexTrait::new(real, img);
}

// Multiplies two complex64 integers.
// 
// The sum of two complex numbers (x + yi) * (u + vi) = (xu - yv) + (xv - yu)i.
// The result is a new complex number where the real part equals (xu - yv) and the imaginary part equals (xv - yu).
// # Arguments
//
// * `a` - The first complex64 to multiply.
// * `b` - The second complex64 to multiply.
//
// # Returns
//
// * `complex64` - The product of `a` and `b`.
fn complex64_mul(a: complex64, b: complex64) -> complex64 {
    let real = a.real * b.real - a.img * b.img;
    let img = a.real * b.img + a.img * b.real;
    return ComplexTrait::new(real, img);
}

// Divides the first complex64 by the second complex64.
// # Arguments
// * `a` - The complex64 dividend.
// * `b` - The complex64 divisor.
// # Returns
// * `complex64` - The quotient of `a` and `b`.
fn complex64_div(a: complex64, b: complex64) -> complex64 {
    complex64_mul(a, b.reciprocal())
}

// Compares two complex64 complex numbers for equality.
// # Arguments
// * `a` - The first complex64 complex number to compare.
// * `b` - The second complex64 complex number to compare.
// # Returns
// * `bool` - `true` if the two complex numbers are equal, `false` otherwise.
fn complex64_eq(a: complex64, b: complex64) -> bool {
    // Check if the two complex numbers have the same real part and the same imaginary part.
    if a.real == b.real && a.img == b.img {
        return true;
    }

    return false;
}

// Compares two complex64 complex numbers for inequality.
// # Arguments
// * `a` - The first complex64 complex number to compare.
// * `b` - The second complex64 complex number to compare.
// # Returns
// * `bool` - `true` if the two complex numbers are not equal, `false` otherwise.
fn complex64_ne(a: complex64, b: complex64) -> bool {
    // The result is the inverse of the equal function.
    return !complex64_eq(a, b);
}

// Negates the given complex64 complex number.
// # Arguments
// * `x` - The complex64 complex number to negate.
// # Returns
// * `complex64` - The negation of `x`.
fn complex64_neg(x: complex64) -> complex64 {
    // The negation of an complex number is obtained by negating its real part and its imaginary part.
    return ComplexTrait::new(-x.real, -x.img);
}
