# tensor.conv_integer

```rust
    
fn conv_integer(
    X: @Tensor<T>,
    W: @Tensor<T>,
    X_zero_point: Option<@Tensor<T>>,
    W_zero_point: Option<@Tensor<T>>,
    auto_pad: Option<AUTO_PAD>,
    dilations: Option<Span<usize>>,
    group: Option<usize>,
    kernel_shape: Option<Span<usize>>,
    pads: Option<Span<usize>>,
    strides: Option<Span<usize>>,
) -> Tensor<T>;
```

Performs integer convolution 

The integer convolution operator consumes an input tensor, its zero-point, a filter, and its zero-point, and computes the output.

## Args

* `X`(`@Tensor<i8>`) - Input data tensor, has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ... x Dn)..
* `W`(`@Tensor<i8>`) - Weight tensor that will be used in the convolutions; has size (M x C/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be (M x C/group x k1 x k2 x ... x kn), where (k1 x k2 x ... kn) is the dimension of the kernel. 
* `X_zero_point`(`@Tensor<T>`) - Zero point for input `X`
* `W_zero_point`(`@Tensor<T>`) - Zero point for input `W`. 
* `auto_pad`(`Option<AUTO_PAD>`) - Default is NOTSET, auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. NOTSET means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that `output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`.
* `dilations`(`Option<Span<usize>>`) - Dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each spatial axis.
* `group`(`Option<usize>`) - Default is 1, number of groups input channels and output channels are divided into.
* `kernel_shape`(`Option<Span<usize>>`) - The shape of the convolution kernel. If not present, should be inferred from input W.
* `pads`(`Option<Span<usize>>`) - Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis.
* `strides`(`Option<Span<usize>>`) - Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis. 

## Returns

A new `Tensor<usize>`, containing the result of the convolution of the inputs.

 
## Example

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, I8Tensor, U32Tensor};
use orion::operators::nn::NNTrait;
use orion::operators::nn::U32NN;

fn example_conv_integer() -> Tensor<usize> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(2);
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(6);
    data.append(7);
    data.append(8);
    data.append(9);
    data.append(10);
    let mut X = TensorTrait::<i8>::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(1);
    data.append(1);
    data.append(1);
    data.append(1);
    let mut W = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(1);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(1);
    let X_zero_point = TensorTrait::<i8>::new(shape.span(), data.span());
    
    
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(12);
    data.append(16);
    data.append(24);
    data.append(28);
    let expected_output = TensorTrait::new(shape.span(), data.span());

    'data ok'.print();

    return NNTrait::conv_integer(
        @X,
        @W,
        Option::Some(@X_zero_point),
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
    );
}
>>> [[12, 16], [24, 28]]
```
