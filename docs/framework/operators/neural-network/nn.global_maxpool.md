#NNTrait::global_maxpool 

```rust
    fn global_maxpool(
    X: @Tensor<T>,
    ) -> Tensor<T>;
```

Given an input tensor X, cmputes the global maxpooling.

## Args

* `X`(`@Tensor<T>`) - Input tensor of shape (N, C, H, W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.

## Returns

A `Tensor<T>` of shape (N, C, 1, 1). For non-image case, a `Tensor<T>` of shape (N, C, 1,..,1_n).

## Example

 ```rust
 use orion::operators::nn::NNTrait;
 use orion::operators::tensor::{Tensor, TensorTrait};

fn example_global_maxpool() -> Tensor<u32> {

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::<u32>::new();
    data.append(36);
    data.append(63);
    data.append(57);
    data.append(62);
    data.append(13);
    data.append(87);
    data.append(44);
    data.append(6);
    data.append(35);
    data.append(35);
    data.append(75);
    data.append(63);
    data.append(49);
    data.append(11);
    data.append(45);
    data.append(11);

    let mut X = TensorTrait::new(shape.span(), data.span());

    return NNTrait::global_maxpool(
        @X
    );
}

>>>  [[[87]
    [75]]]

````