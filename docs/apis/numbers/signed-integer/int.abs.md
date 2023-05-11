# int.abs

Computes the absolute value of a signed\_integer.

```rust
fn abs(self: T) -> T;
```

#### Args

| Name   | Type | Description                                               |
| ------ | ---- | --------------------------------------------------------- |
| `self` | `T`  | The signed integer to which the absolute value is applied |

> _`<T>` generic type depends on `signed_integer` dtype._

#### Returns

A signed integer `<T>`, representing the absolute value of `self` .

> _`<T>` generic type depends on `signed_integer` dtype._

#### Examples

```rust
fn abs_example() -> i32 {
    // We instantiate signed integers here.
    let int = IntegerTrait::<i32>::new(42_u32, true);
    
    // We can call `abs` function as follows.
    a.abs()
}
>>> {mag: 42, sign: false} // = 42
```
