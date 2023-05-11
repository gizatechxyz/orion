# int.max

Returns the maximum between two signed\_integer.

```rust
fn max(self: T, other: T) -> T;
```

#### Args

| Name    | Type | Description                           |
| ------- | ---- | ------------------------------------- |
| `self`  | `T`  | The first signed integer to compare.  |
| `other` | `T`  | The second signed integer to compare. |

> _`<T>` generic type depends on `signed_integer` dtype._

#### Returns

A signed integer `<T>`, The maximum between `self` and `other`.

> _`<T>` generic type depends on `signed_integer` dtype._

#### Examples

```rust
fn max_example() -> i32 {
    // We instantiate signed integer here.
    let a = IntegerTrait::<i32>::new(42_u32, true);
    let b = IntegerTrait::<i32>::new(13_u32, false);
    
    // We can call `max` function as follows.
    a.max(b)
}
>>> {mag: 13, sign: false} // as 13 > -42
```
