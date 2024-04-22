
# NNTrait::max_pool

```rust
    fn max_pool(
    X: @Tensor<T>,
    auto_pad: Option<AUTO_PAD>,
    ceil_mode: Option<usize>,
    dilations: Option<Span<usize>>,
    kernel_shape: Span<usize>,
    pads: Option<Span<usize>>,
    storage_order: Option<usize>,
    strides: Option<Span<usize>>,
    output_len: usize,
) -> (Tensor<T>, Option<Tensor<usize>>);
```

MaxPool consumes an input tensor X and applies max pooling across the tensor according to kernel sizes, stride sizes, and pad lengths. max pooling consisting of computing the max on all values of a subset of the input tensor according to the kernel size and downsampling the data into the output tensor Y for further processing. The output spatial shape is calculated differently depending on whether explicit padding is used, where pads is employed, or auto padding is used, where auto_pad is utilized.

## Args

* `X`(`@Tensor<T>`) - Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size. 
* `auto_pad`(`Option<AUTO_PAD>`) - Default is NOTSET, auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. NOTSET means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that `output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`.
* `ceil_mode`(`Option<usize>`) - Default is 1, Whether to use ceil or floor (default) to compute the output shape.
* `dilations`(`Option<Span<usize>>`) - Dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each spatial axis.
* `kernel_shape`(`Span<usize>`) - The size of the kernel along each axis.
* `pads`(`Option<Span<usize>>`) - Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis.
* `storage_order`(`Option<usize>`) - Default is 0, The storage order of the tensor. 0 is row major, and 1 is column major. 
* `strides`(`Option<Span<usize>>`) - Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
* `output_len`(`Option<usize>`) - Default is 1, If set to 2, return the indices tensor.

## Returns

A `Tensor<T>` that contains the result of the max pool.
A `Option<Tensor<usize>>` with the indices tensor from max pooling across the input tensor. The dimensions of indices are the same as output tensor. 
## Examples
    
```rust
use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;
use orion::operators::nn::FP16x16NN;
use orion::numbers::FP16x16;
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};


fn example_max_pool() -> (Tensor<FP16x16>, Option<Tensor<usize>>) {
    let mut shape = ArrayTrait::<usize>::new();
   shape.append(1);
   shape.append(1);
   shape.append(5);
   shape.append(5);
   let mut data = ArrayTrait::new();
   data.append(FP16x16 { mag: 65536, sign: false });
   data.append(FP16x16 { mag: 131072, sign: false });
   data.append(FP16x16 { mag: 196608, sign: false });
   data.append(FP16x16 { mag: 262144, sign: false });
   data.append(FP16x16 { mag: 327680, sign: false });
   data.append(FP16x16 { mag: 393216, sign: false });
   data.append(FP16x16 { mag: 458752, sign: false });
   data.append(FP16x16 { mag: 524288, sign: false });
   data.append(FP16x16 { mag: 589824, sign: false });
   data.append(FP16x16 { mag: 655360, sign: false });
   data.append(FP16x16 { mag: 720896, sign: false });
   data.append(FP16x16 { mag: 786432, sign: false });
   data.append(FP16x16 { mag: 851968, sign: false });
   data.append(FP16x16 { mag: 917504, sign: false });
   data.append(FP16x16 { mag: 983040, sign: false });
   data.append(FP16x16 { mag: 1048576, sign: false });
   data.append(FP16x16 { mag: 1114112, sign: false });
   data.append(FP16x16 { mag: 1179648, sign: false });
   data.append(FP16x16 { mag: 1245184, sign: false });
   data.append(FP16x16 { mag: 1310720, sign: false });
   data.append(FP16x16 { mag: 1376256, sign: false });
   data.append(FP16x16 { mag: 1441792, sign: false });
   data.append(FP16x16 { mag: 1507328, sign: false });
   data.append(FP16x16 { mag: 1572864, sign: false });
   data.append(FP16x16 { mag: 1638400, sign: false });
   let mut X = TensorTrait::new(shape.span(), data.span());
   return NNTrait::max_pool(
       @X,
       Option::None,
       Option::None,
       Option::None,
       array![5, 5, 5].span(),
       Option::Some(array![2, 2, 2, 2].span()),
       Option::None,
       Option::None,
       1
   );

}

>>> ([
           [
               [
                   [13, 14, 15, 15, 15],
                   [18, 19, 20, 20, 20],
                   [23, 24, 25, 25, 25],
                   [23, 24, 25, 25, 25],
                   [23, 24, 25, 25, 25],
               ]
           ]
       ], 
       Option::None)


````
