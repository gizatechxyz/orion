# tensor.center_crop_pad

```rust
fn center_crop_pad(
    self: @Tensor<T>, shape: Tensor<usize>, axes: Option<Array<i64>>, zero: T
) -> Tensor<T>
```

Center crop or pad an input to given dimensions.

## Args

* `self`(`@Tensor<T>`) - Input to extract the centered crop from.
* `shape`(Tensor<usize>) - 1-D tensor representing the cropping window dimensions.
* `axes`(Option<Array<i64>) - If provided, it specifies a subset of axes that ‘shape’ refer to.

## Panics

* Panics if axes is a negative number, axis+rank (self) is less than 0.

## Returns

Output data of tensors after crop/pad.

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
use core::option::OptionTrait;
fn center_crop_pad_example() -> Tensor<u32> {
    let tensor: Tensor<u32> = TensorTrait::<u32>::new(
        shape: array![5,4,1].span(), 
        data: array![
            1, 2, 3, 4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20
            ].span(),
    );
    // We can call `center_crop_pad` function as follows.
    return tensor.center_crop_pad(TensorTrait::new(array![3].span(), array![5,2,1].span()), Option::None(()));
}
>>> [[2,3],[6,7],[10,11],[14,15],[18,19]]
```
