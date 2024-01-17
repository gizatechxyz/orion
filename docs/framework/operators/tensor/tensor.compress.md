# tensor.compress

```rust 
   fn compress(self: @Tensor<T>, condition: Tensor<T>, axis: Option<usize>) -> Tensor<T>;
```

Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index. In case axis is not provided, input is flattened before elements are selected.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `condition`(`Tensor<T>`) - Rank 1 tensor of booleans to indicate which slices or data elements to be selected. Its length can be less than the input length along the axis or the flattened input size if axis is not specified. In such cases data slices or elements exceeding the condition length are discarded.
* `axis`(`Option<usize>`) - (Optional) Axis along which to take slices. If not specified, input is flattened before elements being selected. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).

## Panics

* Panics if condition rank is not equal to 1.

## Returns 

A new `Tensor<T>` .

## Example
fn compress_example() -> Tensor<u32> {
    let tensor = TensorTrait::<u32>::new(
        shape: array![3, 2].span(), 
        data: array![[1, 2], [3, 4], [5, 6]].span(), 
    );
    let condition = TensorTrait::<u32>::new(
        shape: array![3].span(), 
        data: array![0, 1, 1].span(), 
    );

    return tensor.compress(
        condition: condition, 
        axis: Option::Some((0)), 
    );
}
>>> [[3, 4],
     [5, 6]]
```
