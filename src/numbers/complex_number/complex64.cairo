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
const HALF: u128 = 9223372036854775808;

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
        complex64 { real: FixedTrait::ZERO(), img: FP64x64Impl::ZERO() }
    }

    fn one() -> complex64 {
        complex64 { real: FP64x64Impl::ONE(), img: FP64x64Impl::ZERO() }
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

    fn exp2(self: complex64) -> complex64 {
        let two = complex64 { real: FP64x64Impl::new(TWO, false), img: FP64x64Impl::ZERO() };

        two.pow(self)
    }

    fn sqrt(self: complex64) -> complex64 {
        let x = self.real;
        let y = self.img;
        let two = FP64x64Impl::new(TWO, false);
        let real = (((x.pow(two) + y.pow(two)).sqrt() + x) / two).sqrt();
        let img = if y == FP64x64Impl::ZERO() {
            FP64x64Impl::ZERO()
        } else {
            (((x.pow(two) + y.pow(two)).sqrt() - x) / two).sqrt()
        };
        let img = FP64x64Impl::new(img.mag, y.sign);

        complex64 { real, img }
    }

    fn ln(self: complex64) -> complex64 {
        let real = self.mag().ln();
        let img = self.arg();

        complex64 { real, img }
    }

    fn log2(self: complex64) -> complex64 {
        let ln_2 = FP64x64Impl::new(12786309186476892720, false);
        let ln = self.ln();

        complex64 { real: (ln.real / ln_2), img: (ln.img / ln_2) }
    }

    fn log10(self: complex64) -> complex64 {
        let ln_10 = FP64x64Impl::new(42475197399893398429, false);
        let ln = self.ln();

        complex64 { real: (ln.real / ln_10), img: (ln.img / ln_10) }
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

    //cos(z) = cos(a+bi) = cos(a)cosh(b)-isin(a)sinh(b)
    fn cos(self: complex64) -> complex64 {
        let a = self.real;
        let b = self.img;

        complex64 {
            real: FP64x64Impl::cos(a) * FP64x64Impl::cosh(b),
            img: -FP64x64Impl::sin(a) * FP64x64Impl::sinh(b)
        }
    }

    //sin(z) = sin(a+bi) = sin(a)cosh(b)+icos(a)sinh(b)
    fn sin(self: complex64) -> complex64 {
        let a = self.real;
        let b = self.img;

        complex64 {
            real: FP64x64Impl::sin(a) * FP64x64Impl::cosh(b),
            img: FP64x64Impl::cos(a) * FP64x64Impl::sinh(b)
        }
    }

    //tan(z) = tan(a+bi) = sin(2a) / (cosh(2b) + cos(2a)) + i sinh(2b) / (cosh(2b) + cos(2a))
    fn tan(self: complex64) -> complex64 {
        let two = FP64x64Impl::new(TWO, false);
        let a = self.real;
        let b = self.img;
        let den = FP64x64Impl::cosh(two * b) + FP64x64Impl::cos(two * a);

        complex64 { real: FP64x64Impl::sin(two * a) / den, img: FP64x64Impl::sinh(two * b) / den }
    }

    //acos(z) = pi/2 + i ln (iz sqrt(1 - z**2))
    fn acos(self: complex64) -> complex64 {
        let pi = Complex64Impl::new(FP64x64Impl::new(PI, false), FP64x64Impl::ZERO());
        let two = Complex64Impl::new(FP64x64Impl::new(TWO, false), FP64x64Impl::ZERO());
        let i = Complex64Impl::new(FP64x64Impl::ZERO(), FP64x64Impl::ONE());
        let one = Complex64Impl::new(FP64x64Impl::ONE(), FP64x64Impl::ZERO());
        let acos = pi / two
            + i * Complex64Impl::ln(i * self + Complex64Impl::sqrt(one - (self.pow(two))));

        acos
    }

    //asin(z) = - i ln (iz sqrt(1 - z**2))
    fn asin(self: complex64) -> complex64 {
        let two = Complex64Impl::new(FP64x64Impl::new(TWO, false), FP64x64Impl::ZERO());
        let i = Complex64Impl::new(FP64x64Impl::ZERO(), FP64x64Impl::ONE());
        let one = Complex64Impl::new(FP64x64Impl::ONE(), FP64x64Impl::ZERO());
        let asin = -i * Complex64Impl::ln(i * self + Complex64Impl::sqrt(one - (self.pow(two))));

        asin
    }

    //atan(z) = 1/2 * i[ln (1 - iz) - ln(1 + iz)]
    fn atan(self: complex64) -> complex64 {
        let two = Complex64Impl::new(FP64x64Impl::new(TWO, false), FP64x64Impl::ZERO());
        let i = Complex64Impl::new(FP64x64Impl::ZERO(), FP64x64Impl::ONE());
        let one = Complex64Impl::new(FP64x64Impl::ONE(), FP64x64Impl::ZERO());
        let atan = one
            / two
            * i
            * (Complex64Impl::ln(one - i * self) - Complex64Impl::ln(one + i * self));

        atan
    }

    //acosh(z) = ln (z + sqrt(z + 1) * sqrt(z - 1)) 
    fn acosh(self: complex64) -> complex64 {
        let one = Complex64Impl::new(FP64x64Impl::ONE(), FP64x64Impl::ZERO());
        let acosh = Complex64Impl::ln(
            self + Complex64Impl::sqrt(self + one) * Complex64Impl::sqrt(self - one)
        );

        acosh
    }

    //asinh(z) = ln (z + sqrt(z**2 + 1)) 
    fn asinh(self: complex64) -> complex64 {
        let one = Complex64Impl::new(FP64x64Impl::ONE(), FP64x64Impl::ZERO());
        let two = Complex64Impl::new(FP64x64Impl::new(TWO, false), FP64x64Impl::ZERO());
        let asinh = Complex64Impl::ln(self + Complex64Impl::sqrt(one + (self.pow(two))));

        asinh
    }

    //atanh(z) = 1/2 * [ln (1 + z) - ln(1 - z)]
    fn atanh(self: complex64) -> complex64 {
        let two = Complex64Impl::new(FP64x64Impl::new(TWO, false), FP64x64Impl::ZERO());
        let one = Complex64Impl::new(FP64x64Impl::ONE(), FP64x64Impl::ZERO());
        let atanh = (Complex64Impl::ln(one + self) - Complex64Impl::ln(one - self)) / two;

        atanh
    }

    //acos(z) = acos(a+bi) = cosh a * cos b + i sinh a * sin b
    fn cosh(self: complex64) -> complex64 {
        let a = self.real;
        let b = self.img;

        complex64 {
            real: FP64x64Impl::cosh(a) * FP64x64Impl::cos(b),
            img: FP64x64Impl::sinh(a) * FP64x64Impl::sin(b)
        }
    }

    //sinh(z) = sin(a+bi) = sinh(a)cos(b)+icosh(a)sin(b)
    fn sinh(self: complex64) -> complex64 {
        let a = self.real;
        let b = self.img;

        complex64 {
            real: FP64x64Impl::sinh(a) * FP64x64Impl::cos(b),
            img: FP64x64Impl::cosh(a) * FP64x64Impl::sin(b)
        }
    }

    //tanh(z) = tan(a+bi) = sin(2a) / (cosh(2a) + cos(2b)) + i sinh(2b) / (cosh(2a) + cos(2b))
    fn tanh(self: complex64) -> complex64 {
        let two = FP64x64Impl::new(TWO, false);
        let a = self.real;
        let b = self.img;
        let den = FP64x64Impl::cosh(two * a) + FP64x64Impl::cos(two * b);

        complex64 { real: FP64x64Impl::sinh(two * a) / den, img: FP64x64Impl::sin(two * b) / den }
    }


    fn to_polar(self: complex64) -> (FP64x64, FP64x64) {
        let mag = self.mag();
        let arg = self.arg();

        (mag, arg)
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

impl Complex64Print of PrintTrait<complex64> {
    fn print(self: complex64) {
        self.real.print();
        '+'.print();
        self.img.print();
        'i'.print();
    }
}

// Implements the Add trait for complex64.
impl Complex64Add of Add<complex64> {
    fn add(lhs: complex64, rhs: complex64) -> complex64 {
        complex64_add(lhs, rhs)
    }
}

// Implements the AddEq trait for complex64.
impl Complex64AddEq of AddEq<complex64> {
    #[inline(always)]
    fn add_eq(ref self: complex64, other: complex64) {
        self = Add::add(self, other);
    }
}

// Implements the Sub trait for complex64.
impl Complex64Sub of Sub<complex64> {
    fn sub(lhs: complex64, rhs: complex64) -> complex64 {
        complex64_sub(lhs, rhs)
    }
}

// Implements the SubEq trait for complex64.
impl Complex64SubEq of SubEq<complex64> {
    #[inline(always)]
    fn sub_eq(ref self: complex64, other: complex64) {
        self = Sub::sub(self, other);
    }
}

// Implements the Mul trait for complex64.
impl Complex64Mul of Mul<complex64> {
    fn mul(lhs: complex64, rhs: complex64) -> complex64 {
        complex64_mul(lhs, rhs)
    }
}

// Implements the MulEq trait for complex64.
impl Complex64MulEq of MulEq<complex64> {
    #[inline(always)]
    fn mul_eq(ref self: complex64, other: complex64) {
        self = Mul::mul(self, other);
    }
}

// Implements the Div trait for complex64.
impl Complex64Div of Div<complex64> {
    fn div(lhs: complex64, rhs: complex64) -> complex64 {
        complex64_div(lhs, rhs)
    }
}

// Implements the DivEq trait for complex64.
impl Complex64DivEq of DivEq<complex64> {
    #[inline(always)]
    fn div_eq(ref self: complex64, other: complex64) {
        self = Div::div(self, other);
    }
}

// Implements the PartialEq trait for complex64.
impl Complex64PartialEq of PartialEq<complex64> {
    fn eq(lhs: @complex64, rhs: @complex64) -> bool {
        complex64_eq(*lhs, *rhs)
    }

    fn ne(lhs: @complex64, rhs: @complex64) -> bool {
        complex64_ne(*lhs, *rhs)
    }
}

// Implements the Neg trait for complex64.
impl Complex64Neg of Neg<complex64> {
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

    ComplexTrait::new(real, img)
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

    ComplexTrait::new(real, img)
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

    ComplexTrait::new(real, img)
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

    false
}

// Compares two complex64 complex numbers for inequality.
// # Arguments
// * `a` - The first complex64 complex number to compare.
// * `b` - The second complex64 complex number to compare.
// # Returns
// * `bool` - `true` if the two complex numbers are not equal, `false` otherwise.
fn complex64_ne(a: complex64, b: complex64) -> bool {
    // The result is the inverse of the equal function.
    !complex64_eq(a, b)
}

// Negates the given complex64 complex number.
// # Arguments
// * `x` - The complex64 complex number to negate.
// # Returns
// * `complex64` - The negation of `x`.
fn complex64_neg(x: complex64) -> complex64 {
    // The negation of an complex number is obtained by negating its real part and its imaginary part.
    ComplexTrait::new(-x.real, -x.img)
}
