# IntegerTrait::new

Returns a new signed integer.

```rust
fn new(mag: U, sign: bool) -> T;
```

#### Args

| Name   | Type   | Description                                                         |
| ------ | ------ | ------------------------------------------------------------------- |
| `mag`  | `U`    | The magnitude of the integer.                                       |
| `sign` | `bool` | The sign of the integer, where `true` represents a negative number. |

> _`<T>` generic type depends on `signed_integer` dtype._
>
> _`<U>` generic type depends on the uint type (u8, u16, u32, u64, u128)._

#### Panics

| TypeError                        |
| -------------------------------- |
| Panics if `mag` is out of range. |

#### Returns

A new signed integer.

#### Examples

```rust
fn new_i8_example() -> i8 {
    IntegerTrait::<i8>::new(42_u8, true)
}
>>> {mag: 42, sign: true} // = -42
```

```rust
fn panic_i8_example() -> i8 {
    IntegerTrait::<i8>::new(129_u8, true)
}
>>> panics with "int: out of range"
```
