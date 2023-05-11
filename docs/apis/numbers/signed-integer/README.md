# Signed Integer

```rust
use onnx_cairo::numbers::signed_integer;
```

A `signed_integer` is represented by a structure containing both the magnitude and its sign as a boolean.

The magnitude represents the absolute value of the number, and the sign indicates whether the number is positive or negative.

```rust
// Example of an i32.
struct i32 {
    mag: u32,
    sign: bool, // true means a negative sign.
}
```

### Data types

ONNX-Cairo supports currently five `signed_integer` types.

| Data type       | dtype  |
| --------------- | ------ |
| 8-bit integer   | `i8`   |
| 16-bit integer  | `i16`  |
| 32-bit integer  | `i32`  |
| 64-bit integer  | `i64`  |
| 128-bit integer | `i128` |

### **IntegerTrait**

```rust
use onnx_cairo::numbers::signed_integer::IntegerTrait;
```

`IntegerTrait` defines the operations that can be performed on an integer.

| function                                   | description                                                   |
| ------------------------------------------ | ------------------------------------------------------------- |
| [`IntegerTrait::new`](integertrait-new.md) | Constructs a new `signed_integer`                             |
| [`int.div_rem`](int.div\_rem.md)           | Computes `signed_integer` division and modulus simultaneously |
| [`int.abs`](int.abs.md)                    | Computes the absolute value of the given `signed_integer`     |
| [`int.max`](int.max.md)                    | Returns the maximum between two `signed_integer`              |
| [`int.min`](int.min.md)                    | Returns the minimum between two `signed_integer`              |

### Arithmetic & Comparison operators

`signed_integer` implements arithmetic and comparison traits. This allows you to perform basic arithmetic operations using the associated operators. (`+`, `-`, `*`, `/` ), as well as relational operators (`>`, `>=` ,`<` , `<=` , `==`, `!=` ).

#### Examples

```rust
fn add_i32_example() -> i32 {
    // We instantiate two signed integer here.
    // a = 42
    // b = -10
    let a = IntegerTrait::<i32>::new(42, false);
    let b = IntegerTrait::<i32>::new(10, true);

    // We can add two signed integer as follows.
    return a + b;
}
>>> 32
```

```rust
fn compare_i32_example() -> bool {
    // We instantiate two signed integer here.
    // a = 42
    // b = -10
    let a = IntegerTrait::<i32>::new(42, false);
    let b = IntegerTrait::<i32>::new(10, true);

    // We can compare two signed integer as follows.
    return a > b;
}
>>> true
```
