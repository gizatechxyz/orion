# NNTrait::space_to_depth

```rust
    fn space_to_depth(tensor: @Tensor<T>, blocksize: usize) -> Tensor<T>;
```

SpaceToDepth rearranges blocks of spatial data into depth. More specifically, this op outputs a copy of the input tensor where values from the height and width dimensions are moved to the depth dimension.

## Args

* `tensor`(`@Tensor<T>`) - The input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth, H is the height and W is the width.
* `blocksize`(`usize`) - The size of the blocks to move along [blocksize, blocksize].

## Returns

A `Tensor<T>` of [N, C * blocksize * blocksize, H/blocksize, W/blocksize].

## Examples

```rust
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{I8Tensor, I8TensorAdd};
use orion::numbers::NumberTrait;
use orion::operators::nn::NNTrait;
use orion::operators::nn::I8NN;
use orion::numbers::FixedTrait;

fn space_to_depth_example() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(-3);
    data.append(0);
    data.append(0);
    data.append(0);
    data.append(-1);
    data.append(1);
    data.append(-2);
    data.append(-3);
    data.append(2);
    data.append(-2);
    data.append(-3);
    data.append(-3);
    data.append(-1);
    data.append(0);
    data.append(1);
    data.append(-3);
    let tensor = TensorTrait::new(shape.span(), data.span());
    return NNTrait::space_to_depth(@tensor, 2);
}
>>> [[[[-3, 0]], [[2, -3]], [[0, 0]], [[-2, -3]], [[-1, -2]], [[-1, 1]], [[1, -3]], [[0, -3]]]]
```
