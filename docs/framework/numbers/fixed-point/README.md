# Fixed Point

{% hint style="info" %}
This library has been modified from [cubit](https://github.com/influenceth/cubit) library by [influenceth](https://github.com/influenceth) and adjusted to match with Q8.23 and Q16.16 fixed point.
{% endhint %}

```rust
use orion::numbers::fixed_point;
```

This API provides basic some operations for signed fixed point Q8.23 and Q16.16 numbers. Fixed point numbers are represented as a struct `FixedType` with a magnitude and a sign.

The magnitude represents the absolute value of the number, and the sign indicates whether the number is positive or negative.

```rust
struct FixedType {
    mag: u128,
    sign: bool
}
```

### Data types

Orion supports currently one fixed point type.

| Data type | dtype       |
| --------- | ----------- |
| Q8.23     | `FixedType` |
| Q16.16    | `FixedType` |

### **`Fixed` Trait**

```rust
use orion::numbers::fixed_point::Fixed;
```

`Fixed` trait defines the operations that can be performed on a fixed point.

| function | description |
| --- | --- |
| [`fp.new`](fp.new.md) | Constructs a new fixed point instance. |
| [`fp.new_unscaled`](fp.new\_unscaled.md) | Creates a new fixed point instance with the specified unscaled magnitude and sign. |
| [`fp.from_felt`](fp.from\_felt.md) | Creates a new fixed point instance from a `felt252` value. |
| [`fp.from_unscaled_felt`](fp.from\_unscaled\_felt.md) | Creates a new fixed point instance from an unscaled `felt252` value. |
| [`fp.abs`](fp.abs.md) | Returns the absolute value of the fixed point number. |
| [`fp.ceil`](fp.ceil.md) | Returns the smallest integer greater than or equal to the fixed point number. |
| [`fp.floor`](fp.floor.md) | Returns the largest integer less than or equal to the fixed point number. |
| [`fp.exp`](fp.exp.md) | Returns the value of e raised to the power of the fixed point number.  |
| [`fp.exp2`](fp.exp2.md) | Returns the value of 2 raised to the power of the fixed point number. |
| [`fp.log`](fp.log.md) | Returns the natural logarithm of the fixed point number. |
| [`fp.log2`](fp.log2.md) | Returns the base-2 logarithm of the fixed point number. |
| [`fp.log10`](fp.log10.md) | Returns the base-10 logarithm of the fixed point number. |
| [`fp.pow`](fp.pow.md) | Returns the result of raising the fixed point number to the power of another fixed point number |
| [`fp.round`](fp.round.md) | Rounds the fixed point number to the nearest whole number. |
| [`fp.sqrt`](fp.sqrt.md) | Returns the square root of the fixed point number. |
| [`fp.sin`](fp.sin.md) | Returns the sine of the fixed point number. |
| [`fp.cos`](fp.cos.md) | Returns the cosine of the fixed point number. |
| [`fp.asin`](fp.asin.md) | Returns the arcsine (inverse of sine) of the fixed point number. |
| [`fp.sinh`](fp.sinh.md) | Returns the value of the hyperbolic sine of the fixed point number. |
| [`fp.tanh`](fp.tanh.md) | Returns the value of the hyperbolic tangent of the fixed point number. |
| [`fp.cosh`](fp.cosh.md) | Returns the value of the hyperbolic cosine of the fixed point number. |
| [`fp.acosh`](fp.acosh.md) | Returns the value of the inverse hyperbolic cosine of the fixed point number. |
| [`fp.asinh`](fp.asinh.md) | Returns the inverse hyperbolic sine of the input fixed point number. |
| [`fp.atan`](fp.atan.md) | Returns the arctangent (inverse of tangent) of the input fixed point number. |
| [`fp.acos`](fp.acos.md) | Returns the arccosine (inverse of cosine) of the fixed point number. |

### Arithmetic & Comparison operators

`FixedType` implements arithmetic and comparison traits. This allows you to perform basic arithmetic operations using the associated operators. (`+`,`+=` `-`,`-=` `*`,`*=` `/` , `/=` ), as well as relational operators (`>`, `>=` ,`<` , `<=` , `==`, `!=` ).

#### Examples

```rust
fn add_fp_example() {
    // We instantiate two fixed point from felt here.
    // a = 1
    // b = 2
    let a = Fixed::from_unscaled_felt(1);
    let b = Fixed::from_unscaled_felt(2);

    // We can add two fixed point as follows.
    let result = a + b;

    assert(result == Fixed::from_unscaled_felt(3), 'invalid result');
}
```

```rust
fn compare_fp_example() -> bool {
    // We instantiate two fixed point from felt here.
    // a = 42
    // b = -10
    let a = Fixed::from_unscaled_felt(42);
    let b = Fixed::from_unscaled_felt(-10);

    // We can compare two fixed point as follows.
    return a > b;
}
>>> true
```
