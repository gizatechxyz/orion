/// Trait
///
/// new - Constructs a new `complex_number`
/// real - Returns the real part of the `complex_number`
/// img - Returns the imaginary part of the `complex_number`
/// conjugate - Returns the conjugate of the `complex_number`
/// zero - Returns the additive identity element zero
/// one - Returns the multiplicative identity element one
/// mag - Returns the magnitude of the `complex_number`
/// arg - Returns the argument of the `complex_number`
/// exp - Returns the value of e raised to the power of the `complex_number`
/// sqrt - Returns the value of the squre root of the `complex_number`
/// pow - Returns the result of raising the `complex_number` to the power of another `complex_number`
/// ln - Returns the natural logarithm of the `complex_number`
/// to_polar - Returns the polar coordinates of the `complex_number`
/// from_polar - Returns a `complex_number` from the polar coordinates of the `complex_number`
/// reciprocal - Returns a the reciprocal of the `complex_number`
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
    ///
    //fn log2(self: T) -> T;
    ///
    //fn log10(self: T) -> T;
    ///
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
