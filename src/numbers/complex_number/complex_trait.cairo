/// Trait
///
/// new - Constructs a new `complex_number`.
/// from_felt - Creates a new `complex_number` instance from two felt252 values.
/// real - Returns the real part of the `complex_number`.
/// img - Returns the imaginary part of the `complex_number`.
/// conjugate - Returns the conjugate of the `complex_number`.
/// zero - Returns the additive identity element zero.
/// one - Returns the multiplicative identity element one.
/// mag - Returns the magnitude of the `complex_number`.
/// arg - Returns the argument of the `complex_number`.
/// exp - Returns the value of e raised to the power of the `complex_number`.
/// exp2 - Returns the value of 2 raised to the power of the `complex_number`.
/// ln - Returns the natural logarithm of the `complex_number`.
/// log2 - Returns the base-2 logarithm of the `complex_number`.
/// log10 - Returns the base-10 logarithm of the `complex_number`.
/// pow - Returns the result of raising the `complex_number` to the power of another `complex_number`.
/// sqrt - Returns the value of the squre root of the `complex_number`.
/// acos - Returns the  arccosine (inverse of cosine) of the `complex_number`.
/// asin - Returns the  arcsine (inverse of sine) of the `complex_number`.
/// atan - Returns the arctangent (inverse of tangent) of the input `complex_number`.
/// cos - Returns the cosine of the `complex_number`.
/// sin - Returns the sine of the `complex_number`.
/// tan - Returns the tangent of the `complex_number`.
/// acosh - Returns the value of the inverse hyperbolic cosine of the `complex_number`.
/// asinh - Returns the value of the inverse hyperbolic sine of the `complex_number`.
/// atanh - Returns the value of the inverse hyperbolic tangent of the `complex_number`.
/// cosh - Returns the value of the hyperbolic cosine of the `complex_number`.
/// sinh - Returns the value of the hyperbolic sine of the `complex_number`.
/// tanh - Returns the value of the hyperbolic tangent of the `complex_number`.
/// to_polar - Returns the polar coordinates of the `complex_number`.
/// from_polar - Returns a `complex_number` from the polar coordinates of the `complex_number`.
/// reciprocal - Returns a the reciprocal of the `complex_number`.
///
trait ComplexTrait<T, F> {
    /// # ComplexTrait::new
    /// 
    /// ```rust
    /// fn new(real: F, img: F) -> T;
    /// ```
    ///
    /// ## Args
    ///
    /// * `real`(`F`) - The real part of the complex number.
    /// * `img`(`F`) - The imaginary part of the complex number.
    ///
    /// ## Returns
    /// 
    /// A new complex number.
    /// 
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// 
    /// fn new_complex64_example() -> complex64 {
    ///     ComplexTrait::new(FixedTrait::new(184467440737095516160, false), FixedTrait::new(18446744073709551616, false))
    /// }
    /// >>> {real: {mag: 184467440737095516160, sign: false}, im: {mag: 18446744073709551616, sign: false}} // 10 + i
    /// ```
    /// 
    fn new(real: F, img: F) -> T;
    /// # ComplexTrait::real
    /// 
    /// ```rust
    /// fn real(self: T) -> F;
    /// ```
    /// 
    /// Returns the real part of a complex number. The complex number is represented in Cartesian form `z = a + bi` where `a` is the real part.
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The complex number from which we want the real part.
    ///
    /// ## Returns
    /// 
    /// A fixed point number `<F>`, representing the real part of `self` .
    /// 
    /// ## Examples
    /// 
    /// ```rust    
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn real_complex64_example() -> FP64x64 {
    ///     let z: complex64 = ComplexTrait::new(FixedTrait::new(184467440737095516160, false), FixedTrait::new(18446744073709551616, false));
    ///     z.real()
    /// }
    /// >>> {mag: 184467440737095516160, sign: false} // 10
    /// ```
    /// 
    fn real(self: T) -> F;
    /// # ComplexTrait::img
    /// 
    /// ```rust
    /// fn img(self: T) -> F;
    /// ```
    /// 
    /// Returns the imaginary part of a complex number. The complex number is represented in Cartesian form `z = a + bi` where `b` is the imaginary part.
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The complex number from which we want the imaginary part.
    ///
    /// ## Returns
    /// 
    /// A fixed point number `<F>`, representing the imaginary part of `self` .
    /// 
    /// ## Examples
    ///
    /// ```rust    
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn img_complex64_example() -> FP64x64 {
    ///     let z: complex64 = ComplexTrait::new(FixedTrait::new(184467440737095516160, false), FixedTrait::new(18446744073709551616, false));
    ///     z.img()
    /// }
    /// >>> {mag: 18446744073709551616, sign: false} // 1
    /// ```
    /// 
    fn img(self: T) -> F;
    /// # ComplexTrait::conjugate
    /// 
    /// ```rust
    /// fn conjugate(self: T) -> T;
    /// ```
    ///   
    /// Returns the conjugate of a complex number. The complex number is represented in Cartesian form `z = a + bi`.
    /// The conjugate of `z = a + bi` is `z̅ = a - bi`
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The complex number from which we want the conjugate.
    ///
    /// ## Returns
    /// 
    /// A complex number `<T>`, representing the imaginary part of `self` .
    /// 
    /// ## Examples
    ///
    /// ```rust    
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn conjugate_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(FixedTrait::new(184467440737095516160, false), FixedTrait::new(18446744073709551616, false));
    ///     z.conjugate()
    /// }
    /// >>> {real: {mag: 184467440737095516160, sign: false}, im: {mag: 18446744073709551616, sign: true}} // 10 - i
    /// ```
    /// 
    fn conjugate(self: T) -> T;
    /// # ComplexTrait::zero
    /// 
    /// ```rust
    /// fn zero(self: T) -> T;
    /// ```
    ///   
    /// Returns the additive identity element zero
    ///
    /// ## Returns
    /// 
    /// A complex number `<T>`, representing the additive identity element of the complex field `0`.
    /// 
    /// ## Examples
    ///
    /// ```rust    
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn zero_complex64_example() -> complex64 {
    ///     ComplexTrait::zero()
    /// }
    /// >>> {real: {mag: 0, sign: false}, im: {mag: 0, sign: false}} // 0 + 0i
    /// ```
    /// 
    fn zero() -> T;
    /// # ComplexTrait::one
    /// 
    /// ```rust
    /// fn one(self: T) -> T;
    /// ```
    ///   
    /// Returns the multiplicative identity element one
    ///
    /// ## Returns
    /// 
    /// A complex number `<T>`, representing the multiplicative identity element of the complex field : `1 + 0i`. 
    /// 
    /// ## Examples
    ///
    /// ```rust    
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn one_complex64_example() -> complex64 {
    ///     ComplexTrait::one()
    /// }
    /// >>> {real: {mag: 18446744073709551616, sign: false}, im: {mag: 0, sign: false}} // 1 + 0i
    /// ```
    /// 
    fn one() -> T;
    /// # ComplexTrait::mag
    /// 
    /// ```rust
    /// fn mag(self: T) -> F;
    /// ```
    ///
    /// Returns the magnitude of the complex number
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number
    ///
    /// ## Returns
    /// 
    /// A fixed point number '<F>', representing the magnitude of the complex number. 
    /// 'mag(z) = sqrt(a^2 + b^2)'.
    /// 
    /// ## Examples
    ///
    /// ```rust    
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn mag_complex64_example() -> FP64x64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(73786976294838206464, false),
    ///         FixedTrait::new(774763251095801167872, false)
    ///     ); // 4 + 42i
    ///     z.mag()
    /// }
    /// >>> {mag: 0x2a30a6de7900000000, sign: false} // mag = 42.190046219457976
    /// ```
    /// 
    fn mag(self: T) -> F;
    /// # ComplexTrait::arg
    /// 
    /// ```rust
    /// fn arg(self: T) -> F;
    /// ```
    ///
    /// Returns the argument of the complex number
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number
    ///
    /// ## Returns
    /// 
    /// A fixed point number '<F>', representing the argument of the complex number in radian. 
    /// 'arg(z) = atan2(b, a)'.
    /// 
    /// ## Examples
    ///
    /// ```rust    
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn arg_complex64_example() -> FP64x64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(73786976294838206464, false),
    ///         FixedTrait::new(774763251095801167872, false)
    ///     ); // 4 + 42i
    ///     z.arg()
    /// }
    /// >>> {mag: 27224496882576083824, sign: false} // arg = 1.4758446204521403 (rad)
    /// ```
    /// 
    fn arg(self: T) -> F;
    /// # ComplexTrait::exp
    /// 
    /// ```rust
    /// fn exp(self: T) -> T;
    /// ```
    /// 
    /// Returns the value of e raised to the power of the complex number.
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number
    ///
    /// ## Returns
    ///
    /// The natural exponent of the input complex number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn exp_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(73786976294838206464, false),
    ///         FixedTrait::new(774763251095801167872, false)
    ///     ); // 4 + 42i
    ///     ComplexTrait::exp(z)
    /// }
    /// >>> {real: {mag: 402848450095324460000, sign: true}, im: {mag: 923082101320478400000, sign: true}} // -21.838458238788455-50.04038098170736 i
    /// ``` 
    ///
    fn exp(self: T) -> T;
    /// # ComplexTrait::exp2
    /// 
    /// ```rust
    /// fn exp2(self: T) -> T;
    /// ```
    /// 
    /// Returns the value of 2 raised to the power of the complex number.
    ///
    /// ## Args
    /// 
    /// * `self`(`T`) - The input complex number
    ///
    /// ## Returns
    ///
    /// The binary exponent of the input complex number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn exp2_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(73786976294838206464, false),
    ///         FixedTrait::new(774763251095801167872, false)
    ///     ); // 4 + 42i
    ///     ComplexTrait::exp2(z)
    /// }
    /// >>> {real: {mag: 197471674372309809080, sign: true}, im: {mag: 219354605088992285353, sign: true}} // -10.70502356986 -11.89127707 i
    /// ``` 
    ///
    fn exp2(self: T) -> T;
    /// # ComplexTrait::ln
    ///
    /// ```rust
    /// fn ln(self: T) -> T;
    /// ```
    /// 
    /// Returns the natural logarithm of the complex number.
    /// 
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number.
    ///
    /// ## Returns 
    ///
    /// A complex number representing the natural logarithm of the input number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn ln_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(73786976294838206464, false),
    ///         FixedTrait::new(774763251095801167872, false)
    ///     ); // 4 + 42i
    ///     z.ln()
    /// }
    /// >>> {real: {mag: 69031116512113681970, sign: false}, im: {mag: 27224496882576083824, sign: false}} // 3.7421843216430655 + 1.4758446204521403 i
    ///  ```
    ///
    fn ln(self: T) -> T;
    /// # ComplexTrait::log2
    /// 
    /// ```rust
    /// fn log2(self: T) -> T;
    /// ```
    /// 
    /// Returns the base-2 logarithm of the complex number.
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number.
    ///
    /// ## Panics
    ///
    /// * Panics if the input is negative.
    ///
    /// ## Returns
    ///
    /// A complex number representing the binary logarithm of the input number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn log2_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(36893488147419103232, false),
    ///         FixedTrait::new(55340232221128654848, false)
    ///     ); // 2 + 3i
    ///     z.log2()
    /// }
    /// >>> {real: {mag: 34130530934667840346, sign: false}, im: {mag: 26154904847122126193, sign: false}} // 1.85021986 + 1.41787163 i
    ///  ```
    /// 
    fn log2(self: T) -> T;
    /// # ComplexTrait::log10
    /// 
    /// ```rust
    /// fn log10(self: T) -> T;
    /// ```
    /// 
    /// Returns the base-10 logarithm of the complex number.
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number.
    ///
    /// ## Returns
    ///
    /// A complex number representing the base 10 logarithm of the input number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn log10_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(36893488147419103232, false),
    ///         FixedTrait::new(55340232221128654848, false)
    ///     ); // 2 + 3i
    ///     z.log10()
    /// }
    /// >>> {real: {mag: 10274314139629458970, sign: false}, im: {mag: 7873411322133748801, sign: false}} // 0.5569716761 + 0.4268218908 i
    ///  ```
    ///
    fn log10(self: T) -> T;
    /// # ComplexTrait::pow
    /// 
    /// ```rust
    /// fn pow(self: T, b: T) -> T;
    /// ```
    /// 
    /// Returns the result of raising the complex number to the power of another complex number.
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number.
    /// * `b`(`T`) - The exponent complex number.
    ///
    /// ## Returns
    ///
    /// A complex number representing the result of z^w.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::complex_trait::ComplexTrait;
    /// use orion::numbers::complex_number::complex64::{TWO, complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn pow_2_complex64_example() -> complex64 {
    ///     let two = ComplexTrait::new(FP64x64Impl::new(TWO, false),FP64x64Impl::new(0, false));
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(73786976294838206464, false),
    ///         FixedTrait::new(774763251095801167872, false)
    ///     ); // 4 + 42i
    ///     z.pow(two)
    /// }
    /// >>> {real: {mag: 32244908640844296224768, sign: true}, im: {mag: 6198106008766409342976, sign: false}} // -1748 + 336 i
    /// 
    /// fn pow_w_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(73786976294838206464, false),
    ///         FixedTrait::new(774763251095801167872, false)
    ///     ); // 4 + 42i
    /// 
    ///     let w: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(36893488147419103232, false),
    ///         FixedTrait::new(18446744073709551616, false)
    ///     ); // 2 + i
    ///     z.pow(w)
    /// }
    /// >>> {real: {mag: 6881545343236111419203, sign: false}, im: {mag: 2996539405459717736042, sign: false}} // -373.0485407816205 + 162.4438823807959 i
    /// ```
    ///
    fn pow(self: T, b: T) -> T;
    /// # ComplexTrait::sqrt
    /// 
    /// ```rust
    /// fn arg(self: T) -> F;
    /// ```
    /// 
    /// Returns the value of the squre root of the complex number.
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number
    ///
    /// ## Returns
    /// 
    /// A complex number '<T>', representing the square root of the complex number. 
    /// 'arg(z) = atan2(b, a)'.
    /// 
    /// ## Examples
    ///
    /// ```rust    
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn sqrt_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(73786976294838206464, false),
    ///         FixedTrait::new(774763251095801167872, false)
    ///     ); // 4 + 42i
    ///     z.sqrt()
    /// }
    /// >>> {real: {mag: 88650037379463118848, sign: false}, im: {mag: 80608310115317055488, sign: false}} // 4.80572815603723 + 4.369785247552674 i
    /// ```
    /// 
    fn sqrt(self: T) -> T;
    /// # ComplexTrait::acos
    /// 
    /// ```rust
    /// fn acos(self: T) -> T;
    /// ```
    /// 
    /// Returns the  arccosine (inverse of cosine) of the complex number.
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number.
    ///
    /// ## Returns
    ///
    /// A complex number representing the acos  of the input value.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn acos_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(36893488147419103232, false),
    ///         FixedTrait::new(55340232221128654848, false)
    ///     ); // 2 + 3i
    ///     z.acos()
    /// }
    /// >>> {real: {mag: 18449430688981877061, sign: false}, im: {mag: 36587032881711954470, sign: true}} //  1.000143542473797 - 1.98338702991653i
    ///  ```
    ///
    fn acos(self: T) -> T;
    /// # ComplexTrait::asin
    /// 
    /// ```rust
    /// fn asin(self: T) -> T;
    /// ```
    /// 
    /// Returns the  arcsine (inverse of sine) of the complex number.
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number.
    ///
    /// ## Returns
    ///
    /// A complex number representing the asin of the input value.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn asin_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(36893488147419103232, false),
    ///         FixedTrait::new(55340232221128654848, false)
    ///     ); // 2 + 3i
    ///     z.asin()
    /// }
    /// >>> {real: {mag: 10526647143326614308, sign: false}, im: {mag: 36587032881711954470, sign: false}} // 0.57065278432 + 1.9833870299i
    ///  ```
    ///
    fn asin(self: T) -> T;
    /// # ComplexTrait::atan
    /// 
    /// ```rust
    /// fn atan(self: T) -> T;
    /// ```
    /// 
    /// Returns the arctangent (inverse of tangent) of the input complex number.
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number.
    ///
    /// ## Returns
    ///
    /// A complex number representing the arctangent (inverse of tangent) of the input value.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn atan_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(36893488147419103232, false),
    ///         FixedTrait::new(55340232221128654848, false)
    ///     ); // 2 + 3i
    ///     z.atan()
    /// }
    /// >>> {real: {mag: 26008453796191787243, sign: false}, im: {mag: 4225645162986888119, sign: false}} // 1.40992104959 + 0.2290726829i
    ///  ```
    ///  
    fn atan(self: T) -> T;
    /// # ComplexTrait::cos
    /// 
    /// ```rust
    /// fn cos(self: T) -> T;
    /// ```
    /// 
    /// Returns the cosine of the complex number.
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number.
    ///
    /// ## Returns
    ///
    /// A complex number representing the cosine of the input value.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn cos_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(36893488147419103232, false),
    ///         FixedTrait::new(55340232221128654848, false)
    ///     ); // 2 + 3i
    ///     z.cos()
    /// }
    /// >>> {real: {mag: 77284883172661882094, sign: true}, im: {mag: 168035443352962049425, sign: true}} // -4.18962569 + -9.10922789375i
    ///  ```

    ///
    fn cos(self: T) -> T;
    /// # ComplexTrait::sin
    /// 
    /// ```rust
    /// fn sin(self: T) -> T;
    /// ```
    /// 
    /// Returns the sine of the complex number.
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number.
    ///
    /// ## Returns
    ///
    /// A complex number representing the sin of the input value.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn sin_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(36893488147419103232, false),
    ///         FixedTrait::new(55340232221128654848, false)
    ///     ); // 2 + 3i
    ///     z.sin()
    /// }
    /// >>> {real: {mag: 168870549816927860082, sign: false}, im: {mag: 76902690389051588309, sign: true}} // 9.15449914 - 4.168906959 i
    ///  ```
    ///
    fn sin(self: T) -> T;
    /// # ComplexTrait::tan
    /// 
    /// ```rust
    /// fn tan(self: T) -> T;
    /// ```
    /// 
    /// Returns the tangent of the complex number.
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number.
    ///
    /// ## Returns
    ///
    /// A complex number representing the tan of the input value.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn tan_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(36893488147419103232, false),
    ///         FixedTrait::new(55340232221128654848, false)
    ///     ); // 2 + 3i
    ///     z.tan()
    /// }
    /// >>> {real: {mag: 69433898428143694, sign: true}, im: {mag: 18506486100303669886, sign: false}} // -0.00376402 + 1.00323862i
    ///  ```
    ///
    fn tan(self: T) -> T;
    /// # ComplexTrait::acosh
    /// 
    /// ```rust
    /// fn acosh(self: T) -> T;
    /// ```
    /// 
    /// Returns the value of the inverse hyperbolic cosine of the complex number.
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number.
    ///
    /// ## Returns
    ///
    /// The inverse hyperbolic cosine of the input complex number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn acosh_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(36893488147419103232, false),
    ///         FixedTrait::new(55340232221128654848, false)
    ///     ); // 2 + 3i
    ///     z.acosh()
    /// }
    /// >>> {real: {mag: 36587032878947915965, sign: false}, im: {mag: 18449360714192945790, sign: false}} // 1.9833870 + 1.0001435424i
    ///  ```
    ///
    fn acosh(self: T) -> T;
    /// # ComplexTrait::asinh
    /// 
    /// ```rust
    /// fn asinh(self: T) -> T;
    /// ```
    /// 
    /// Returns the value of the inverse hyperbolic sine of the complex number.
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number.
    ///
    /// ## Returns
    ///
    /// The inverse hyperbolic sine of the input complex number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn asinh_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(36893488147419103232, false),
    ///         FixedTrait::new(55340232221128654848, false)
    ///     ); // 2 + 3i
    ///     z.asinh()
    /// }
    /// >>> {real: {mag: 36314960239770126586, sign: false}, im: {mag: 17794714057579789616, sign: false}} //1.9686379 + 0.964658504i
    ///  ```
    ///
    fn asinh(self: T) -> T;
    /// # ComplexTrait::atanh
    /// 
    /// ```rust
    /// fn atanh(self: T) -> T;
    /// ```
    /// 
    /// Returns the value of the inverse hyperbolic tangent of the complex number.
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number.
    ///
    /// ## Returns
    ///
    /// The inverse hyperbolic tangent of the input complex number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn atanh_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(36893488147419103232, false),
    ///         FixedTrait::new(55340232221128654848, false)
    ///     ); // 2 + 3i
    ///     z.atanh()
    /// }
    /// >>> {real: {mag: 2710687792925618924, sign: false}, im: {mag: 24699666646262346226, sign: false}} //  0.146946666 + 1.33897252i
    ///  ```

    ///
    fn atanh(self: T) -> T;
    /// # ComplexTrait::cosh
    /// 
    /// ```rust
    /// fn cosh(self: T) -> T;
    /// ```
    /// 
    /// Returns the value of the hyperbolic cosine of the complex number.
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number.
    ///
    /// ## Returns
    ///
    /// The hyperbolic cosine of the input complex number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn cosh_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(36893488147419103232, false),
    ///         FixedTrait::new(55340232221128654848, false)
    ///     ); // 2 + 3i
    ///     z.cosh()
    /// }
    /// >>> {real: {mag: 68705646899632870392, sign: true}, im: {mag: 9441447324287988702, sign: false}} // -3.72454550491 + 0.511822569987i
    ///  ```
    ///
    fn cosh(self: T) -> T;
    /// # ComplexTrait::sinh
    /// 
    /// ```rust
    /// fn sinh(self: T) -> T;
    /// ```
    /// 
    /// Returns the value of the hyperbolic sine of the complex number.
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number.
    ///
    /// ## Returns
    ///
    /// The hyperbolic sine of the input complex number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn sinh_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(36893488147419103232, false),
    ///         FixedTrait::new(55340232221128654848, false)
    ///     ); // 2 + 3i
    ///     z.sinh()
    /// }
    /// >>> {real: {mag: 66234138518106676624, sign: true}, im: {mag: 9793752294470951790, sign: false}} // -3.59056458998 + 0.530921086i
    ///  ```
    ///
    fn sinh(self: T) -> T;
    /// # ComplexTrait::tanh
    /// 
    /// ```rust
    /// fn tanh(self: T) -> T;
    /// ```
    /// 
    /// Returns the value of the hyperbolic tangent of the complex number.
    ///
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number.
    ///
    /// ## Returns
    ///
    /// The hyperbolic tangent of the input complex number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn tanh_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(36893488147419103232, false),
    ///         FixedTrait::new(55340232221128654848, false)
    ///     ); // 2 + 3i
    ///     z.tanh()
    /// }
    /// >>> {real: {mag: 17808227710002974080, sign: false}, im: {mag: 182334107030204896, sign: true}} // 0.96538587902 + 0.009884375i
    ///  ```
    ///
    fn tanh(self: T) -> T;
    /// # ComplexTrait::to_polar
    /// 
    /// ```rust
    /// fn to_polar(self: T) -> (F, F);
    /// ```
    /// 
    /// Returns the polar coordinates (magnitude and argument) of the complex number.
    /// 
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number.
    ///
    /// ## Returns 
    ///
    /// A tuple of two fixed point numbers representing the polar coordinates of the input number.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn to_polar_complex64_example() -> (FP64x64, FP64x64) {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(73786976294838206464, false),
    ///         FixedTrait::new(774763251095801167872, false)
    ///     ); // 4 + 42i
    ///     z.to_polar()
    /// }
    /// >>> ({mag: 778268985067028086784, sign: false},  {mag: 27224496882576083824, sign: false}) // mag : 42.190046219457976 + arg : 1.4758446204521403 
    ///  ```
    ///
    fn to_polar(self: T) -> (F, F);
    /// # ComplexTrait::from_polar
    ///
    /// 
    /// ```rust
    /// fn from_polar(mag: F, arg: F) -> T;
    /// ```
    /// 
    /// Returns a complex number (in the Cartesian form) from the polar coordinates of the complex number.
    /// 
    /// ## Args
    ///
    /// * `mag`(`F`) - The input fixed point number representing the magnitude.
    /// * `arg`(`F`) - The input fixed point number representing the argument.
    ///
    /// ## Returns 
    ///
    /// The complex number representing the Cartesian form calculated from the input polar coordinates.
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn from_polar_complex64_example() -> complex64 {
    ///     let mag: FP64x64 = FixedTrait::new(778268985067028086784, false); // 42.190046219457976
    ///     let arg: FP64x64 = FixedTrait::new(27224496882576083824, false); //1.4758446204521403
    ///     ComplexTrait::from_polar(mag,arg)
    /// }
    /// >>> {real: {mag: 73787936714814843012, sign: false}, im: {mag: 774759489569697723777, sign: false}} // 4 + 42 i
    ///  ```
    ///
    fn from_polar(mag: F, arg: F) -> T;
    /// # ComplexTrait::reciprocal
    ///
    /// 
    /// ```rust
    /// fn reciprocal(self: T) -> T;
    /// ```
    /// 
    /// Returns a the reciprocal of the complex number (i.e. 1/z).
    /// 
    /// ## Args
    ///
    /// * `self`(`T`) - The input complex number.
    ///
    /// ## Returns 
    ///
    /// The reciprocal of the complex number \(a + bi\) is given by:
    /// \[
    /// \frac{1}{a + bi} = \frac{a}{a^2 + b^2} - \frac{b}{a^2 + b^2}i
    /// \]
    ///
    /// ## Examples
    /// 
    /// ```rust
    /// use orion::numbers::complex_number::{complex_trait::ComplexTrait, complex64::complex64};
    /// use orion::numbers::{FP64x64, FP64x64Impl, FixedTrait};
    /// 
    /// fn reciprocal_complex64_example() -> complex64 {
    ///     let z: complex64 = ComplexTrait::new(
    ///         FixedTrait::new(73786976294838206464, false),
    ///         FixedTrait::new(774763251095801167872, false)
    ///     ); // 4 + 42i
    ///     z.reciprocal()
    /// }
    /// >>> {real: {mag: 41453357469010228, sign: false}, im: {mag: 435260253424607397, sign: true}} // 0.002247191011 - 0.0235955056 i
    ///  ```
    ///
    fn reciprocal(self: T) -> T;
}
