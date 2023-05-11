# int.div\_rem

Computes signed\_integer division and modulus simultaneously

```rust
fn div_rem(self: T, other: T) -> (T, T);
```

#### Args

| Name    | Type | Description  |
| ------- | ---- | ------------ |
| `self`  | `T`  | The dividend |
| `other` | `T`  | The divisor  |

> _`<T>` generic type depends on `signed_integer` dtype._

#### Panics

| TypeError                      |
| ------------------------------ |
| Panics if the divisor is zero. |

#### Returns

A tuple of signed integer `<T>`, containing the quotient and the remainder of the division.

> _`<T>` generic type depends on `signed_integer` dtype._

#### Examples

```rust
fn div_rem_example() -> (i32, i32) {
    // We instantiate signed integers here.
    let a = IntegerTrait::<i32>::new(13_u32, false);
    let b = IntegerTrait::<i32>::new(5_u32, false);
    
    // We can call `div_rem` function as follows.
    a.div_rem(b)
}
>>> ({mag: 2, sign: false}, {mag: 3, sign: false}) // = (2, 3)
```
