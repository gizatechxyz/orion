# Fixed Point

{% hint style="info" %}
This library has been modified from [cubit](https://github.com/influenceth/cubit) library by [influenceth](https://github.com/influenceth) and adjusted to match with Q5.26 fixed point.
{% endhint %}

```rust
use orion::numbers::fixed_point;
```

This API provides basic some operations for signed fixed point Q5.26 numbers. Fixed point numbers are represented as a struct `FixedType` with a magnitude and a sign.&#x20;

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
| Q5.26     | `FixedType` |

### **`Fixed` Trait**

```rust
use orion::numbers::fixed_point::Fixed;
```

`Fixed` trait defines the operations that can be performed on a fixed point.

| function | description |
| --- | --- |
| [`fp.new`](https://orion.gizatech.xyz/apis/numbers/fixed-point/fp.new.md) | Constructs a new `FixedType` instance. |
| [`fp.new_unscaled`](https://orion.gizatech.xyz/apis/numbers/fixed-point/fp.new_unscaled.md) | Creates a new `FixedType` instance with the specified unscaled magnitude and sign. |
| [`fp.from_felt`](https://orion.gizatech.xyz/apis/numbers/fixed-point/fp.from_felt.md) | Creates a new `FixedType` instance from a `felt252` value. |
| [`fp.from_unscaled_felt`](https://orion.gizatech.xyz/apis/numbers/fixed-point/fp.from_unscaled_felt.md) | Creates a new `FixedType` instance from an unscaled `felt252` value. |
| [`fp.abs`](https://orion.gizatech.xyz/apis/numbers/fixed-point/fp.abs.md) | Returns the absolute value of the fixed point number. |
| [`fp.ceil`](https://orion.gizatech.xyz/apis/numbers/fixed-point/fp.ceil.md) | Returns the smallest integer greater than or equal to the fixed point number. |
| [`fp.floor`](https://orion.gizatech.xyz/apis/numbers/fixed-point/fp.floor.md) | Returns the largest integer less than or equal to the fixed point number. |
| [`fp.exp`](https://orion.gizatech.xyz/apis/numbers/fixed-point/fp.exp.md) | Returns the value of e raised to the power of the fixed point number.  |
| [`fp.exp2`](https://orion.gizatech.xyz/apis/numbers/fixed-point/fp.exp2.md) | Returns the value of 2 raised to the power of the fixed point number. |
| [`fp.ln`](https://orion.gizatech.xyz/apis/numbers/fixed-point/fp.ln.md) | Returns the natural logarithm of the fixed point number. |
| [`fp.log2`](https://orion.gizatech.xyz/apis/numbers/fixed-point/fp.log2.md) | Returns the base-2 logarithm of the fixed point number. |
| [`fp.log10`](https://orion.gizatech.xyz/apis/numbers/fixed-point/fp.log10.md) | Returns the base-10 logarithm of the fixed point number. |
| [`fp.pow`](https://orion.gizatech.xyz/apis/numbers/fixed-point/fp.pow.md) | Returns the result of raising the fixed point number to the power of another fixed point number |
| [`fp.round`](https://orion.gizatech.xyz/apis/numbers/fixed-point/fp.round.md) | Rounds the fixed point number to the nearest whole number. |
| [`fp.sqrt`](https://orion.gizatech.xyz/apis/numbers/fixed-point/fp.sqrt.md) | Returns the square root of the fixed point number. |

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
