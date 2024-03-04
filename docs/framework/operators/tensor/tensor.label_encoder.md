# tensor.label_encoder

```rust
fn label_encoder(self: @Tensor<T>, default_list: Option<Span<T>>, default_tensor: Option<Tensor<T>>, keys: Option<Span<T>>, keys_tensor: Option<Tensor<T>>, values: Option<Span<T>>, values_tensor: Option<Tensor<T>>) -> Tensor<T>;
```

Maps each element in the input tensor to another value.

The mapping is determined by the two parallel attributes, 'keys_' and 'values_' attribute. 
The i-th value in the specified 'keys_' attribute would be mapped to the i-th value in the specified 'values_' attribute.
 It implies that input's element type and the element type of the specified 'keys_' should be identical while the output type is identical to the specified 'values_' attribute.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `default_list`(`Option<Span<T>>`) - The default span.
* `default_tensor`(`Option<Tensor<T>>`) - The default tensor.
* `keys`(`Option<Span<T>>`) - The keys span.
* `keys_tensor`(`Option<Tensor<T>>`) - The keys tensor.
* `values`(` Option<Span<T>>`) - The values span.
* `values_tensor`(`Option<Tensor<T>>`) - The values tensor.

One and only one of 'default_*'s should be set
One and only one of 'keys*'s should be set
 One and only one of 'values*'s should be set.

## Panics

* Panics if the len/shape of keys and values are not the same.

## Returns

A new `Tensor<T>` which maps each element in the input tensor to another value..

## Type Constraints

* `T` in (`Tensor<FP>`, `Tensor<i8>`, `Tensor<i32>`, `tensor<u32>,`)

## Examples

```rust
use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::U32Tensor;
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};

fn label_encoder_example() -> Tensor<T>,  {
   fn data() -> Tensor<u32> {
       let mut sizes = ArrayTrait::new();
       sizes.append(2);
       sizes.append(3);
       let mut data = ArrayTrait::new();
       data.append(1);
       data.append(2);
       data.append(3);
       data.append(1);
       data.append(4);
       data.append(5);

       let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());
       return tensor;
   }

   fn keys() -> Tensor<u32> {
       let mut sizes = ArrayTrait::new();
       sizes.append(3);
       sizes.append(1);

       let mut data = ArrayTrait::new();
       data.append(1);
       data.append(2);
       data.append(1);

       let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());
       return tensor;
   }

   fn values() -> Tensor<u32> {
       let mut sizes = ArrayTrait::new();
       sizes.append(3);
       sizes.append(1);

       let mut data = ArrayTrait::new();
       data.append(8);
       data.append(9);
       data.append(7);

       let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());
       return tensor;
   }

   fn default() -> Tensor<u32> {
       let mut sizes = ArrayTrait::new();
       sizes.append(1);

       let mut data = ArrayTrait::new();
       data.append(999);

       let tensor = TensorTrait::<u32>::new(sizes.span(), data.span());
       return tensor;
   }

   let data = data();
   let keys = keys();
   let values = values();
   let default = default();
   return data.label_encoder(default_list: Option::None, default_tensor: Option::Some(default),
        keys: Option::None, keys_tensor: Option::Some(keys),  
        values: Option::None, values_tensor: Option::Some(values));
>>> [7, 9, 999, 7, 999, 999],
```
