# Fixed Point

{% hint style="info" %}
This library has been modified from [cubit](https://github.com/influenceth/cubit) library by [influenceth](https://github.com/influenceth) and adjusted to match with other fixed point implementations.
{% endhint %}

This API provides basic some operations for signed fixed point numbers. Fixed point numbers are represented as a struct with a magnitude and a sign.

The magnitude represents the absolute value of the number, and the sign indicates whether the number is positive or negative.

```rust
struct FP8x23 {
    mag: u32,
    sign: bool
}
```

### Data types

Orion supports currently these fixed point types:

| Data type | dtype     |
| --------- | --------- |
| Q8.23     | `FP8x23`  |
| Q16.16    | `FP16x16` |
| Q32.32    | `FP32x32` |
| Q64.64    | `FP64x64` |

### **`Fixed` Trait**

```rust
use orion::numbers::fixed_point::core::FixedTrait;
```

`Fixed` trait defines the operations that can be performed on a fixed point.

| function | description |
| --- | --- |
| [`fp.new`](fp.new.md) | Constructs a new fixed point instance. |
| [`fp.new_unscaled`](fp.new\_unscaled.md) | Creates a new fixed point instance with the specified unscaled magnitude and sign. |
| [`fp.from_felt`](fp.from\_felt.md) | Creates a new fixed point instance from a felt252 value. |
| [`fp.abs`](fp.abs.md) | Returns the absolute value of the fixed point number. |
| [`fp.ceil`](fp.ceil.md) | Returns the smallest integer greater than or equal to the fixed point number. |
| [`fp.exp`](fp.exp.md) | Returns the value of e raised to the power of the fixed point number. |
| [`fp.exp2`](fp.exp2.md) | Returns the value of 2 raised to the power of the fixed point number. |
| [`fp.floor`](fp.floor.md) | Returns the largest integer less than or equal to the fixed point number. |
| [`fp.ln`](fp.ln.md) | Returns the natural logarithm of the fixed point number. |
| [`fp.log2`](fp.log2.md) | Returns the base-2 logarithm of the fixed point number. |
| [`fp.log10`](fp.log10.md) | Returns the base-10 logarithm of the fixed point number. |
| [`fp.pow`](fp.pow.md) | Returns the result of raising the fixed point number to the power of another fixed point number. |
| [`fp.round`](fp.round.md) | Rounds the fixed point number to the nearest whole number. |
| [`fp.sqrt`](fp.sqrt.md) | Returns the square root of the fixed point number. |
| [`fp.acos`](fp.acos.md) | Returns the  arccosine (inverse of cosine) of the fixed point number. |
| [`fp.acos_fast`](fp.acos\_fast.md) | Returns the  arccosine (inverse of cosine) of the fixed point number faster with LUT. |
| [`fp.asin`](fp.asin.md) | Returns the  arcsine (inverse of sine) of the fixed point number. |
| [`fp.asin_fast`](fp.asin\_fast.md) | Returns the  arcsine (inverse of sine) of the fixed point number faster with LUT. |
| [`fp.atan`](fp.atan.md) | Returns the arctangent (inverse of tangent) of the input fixed point number. |
| [`fp.atan_fast`](fp.atan\_fast.md) | Returns the arctangent (inverse of tangent) of the input fixed point number faster with LUT. |
| [`fp.cos`](fp.cos.md) | Returns the cosine of the fixed point number. |
| [`fp.cos_fast`](fp.cos\_fast.md) | Returns the cosine of the fixed point number fast with LUT. |
| [`fp.sin`](fp.sin.md) | Returns the sine of the fixed point number. |
| [`fp.sin_fast`](fp.sin\_fast.md) | Returns the sine of the fixed point number faster with LUT. |
| [`fp.tan`](fp.tan.md) | Returns the tangent of the fixed point number. |
| [`fp.tan_fast`](fp.tan\_fast.md) | Returns the tangent of the fixed point number faster with LUT. |
| [`fp.acosh`](fp.acosh.md) | Returns the value of the inverse hyperbolic cosine of the fixed point number. |
| [`fp.asinh`](fp.asinh.md) | Returns the value of the inverse hyperbolic sine of the fixed point number. |
| [`fp.atanh`](fp.atanh.md) | Returns the value of the inverse hyperbolic tangent of the fixed point number. |
| [`fp.cosh`](fp.cosh.md) | Returns the value of the hyperbolic cosine of the fixed point number. |
| [`fp.sinh`](fp.sinh.md) | Returns the value of the hyperbolic sine of the fixed point number. |
| [`fp.tanh`](fp.tanh.md) | Returns the value of the hyperbolic tangent of the fixed point number. |
| [`fp.sign`](fp.sign.md) | Returns the element-wise indication of the sign of the input fixed point number. |
| [`fp.erf`](fp.erf.md) | Returns the error function of the input fixed point number computed element-wise. |

### Arithmetic & Comparison operators

`FixedType` implements arithmetic and comparison traits. This allows you to perform basic arithmetic operations using the associated operators. (`+`,`+=` `-`,`-=` `*`,`*=` `/` , `/=` ), as well as relational operators (`>`, `>=` ,`<` , `<=` , `==`, `!=` ).

#### Examples

```rust
fn add_fp_example() {
    // We instantiate two fixed point from here.
    // a = 1
    // b = 2
    let a = Fixed::new_unscaled(1, false);
    let b = Fixed::new_unscaled(2, false);

    // We can add two fixed point as follows.
    let result = a + b;

    assert(result == Fixed::new_unscaled(3), 'invalid result');
}
```

```rust
fn compare_fp_example() -> bool {
    // We instantiate two fixed point from here.
    // a = 42
    // b = -10
    let a = Fixed::new_unscaled(42, false);
    let b = Fixed::new_unscaled(10, true);

    // We can compare two fixed point as follows.
    return a > b;
}
>>> true
```
