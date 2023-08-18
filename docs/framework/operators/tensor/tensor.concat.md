# tensor.concat

```rust 
   fn concat(tensors: Span<Tensor<T>>, axis: usize,  ) -> Tensor<T>;
```

Concatenate a list of tensors into a single tensor..

## Args

* `tensors`(` Span<Tensor<T>>,`) - Array of the input tensors.
* `axis`(`usize`) -  Axis to concat on.

## Panics

* Panics if lenght of tensors is not equal greater than 1.
* Panics if dimension is not greater than axis

## Returns 

A new `Tensor<T>` concatenated tensor of the input tensors.

## Example

```rust
fn concat_example() -> Tensor<FixedType> {
   let tensor1 = u32_tensor_2x2_helper();
   let tensor2 = u32_tensor_2x2_helper();

   let mut data = ArrayTrait::new();
   data.append(tensor1);
   data.append(tensor2);
   axis = 0 
   let result = TensorTrait::concat(tensors: data.span(), axis: axis);
   return result;
}
>>> [[0. 1.]
     [2. 3.],
     [0. 1.]
     [2. 3.]]

    result.shape
>>> (4, 2)

   axis = 1
   let result = TensorTrait::concat(tensors: data.span(), axis: axis);
   return result;
}
>>> [[0. 1., 0., 1.]
     [2. 3., 2., 3.]]

    result.shape
>>> (2, 4 ) 
```
