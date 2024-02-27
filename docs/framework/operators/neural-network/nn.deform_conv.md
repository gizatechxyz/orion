# NNTrait::deform_conv

```rust
   fn deform_conv(
       X: @Tensor<T>,
       W: @Tensor<T>,
       offset: @Tensor<T>,
       B: Option<Span<T>>,
       mask: Option<Tensor<T>>,
       dilations: Option<Span<usize>>,
       group: Option<usize>,
       kernel_shape: Option<Span<usize>>,
       offset_group: Option<usize>,
       pads: Option<Span<usize>>,
       strides: Option<Span<usize>>,
   ) -> Tensor<T>
```

Performs deformable convolution as described in https://arxiv.org/abs/1703.06211 and https://arxiv.org/abs/1811.11168. This operator specification supports the 2-D case.

## Args

   X: @Tensor<T>,
   W: @Tensor<T>,
   offset: @Tensor<T>,
   B: Option<Span<T>>,
   mask: Option<Tensor<T>>,
   dilations: Option<Span<usize>>,
   group: Option<usize>,
   kernel_shape: Option<Span<usize>>,
   offset_group: Option<usize>,
   pads: Option<Span<usize>>,
   strides: Option<Span<usize>>,

* `X`(`@Tensor<T>`) - Input data tensor. For 2D image data, it has shape (N, C, H, W) where N is the batch size, C is the number of input channels, and H and W are the height and width. 
* `W`(`@Tensor<T>`) - Weight tensor that will be used in the convolutions. It has shape (oC, C/group, kH, kW), where oC is the number of output channels and kH and kW are the kernel height and width.
* `offset`(`@Tensor<T>`) - Offset tensor denoting the offset for the sampling locations in the convolution kernel. It has shape (N, offset_group * kH * kW * 2, oH, oW) for 2D data
* `B`(`Option<Span<T>>`) - Default is a tensor of zeros, optional 1D bias of length oC to be added to the convolution.
* `mask`(`Option<Tensor<T>>`) -  Default is a tensor of ones, the mask tensor to be applied to each position in the convolution kernel. It has shape (N, offset_group * kH * kW, oH, oW) for 2D data.
* `dilations`(`Option<Span<usize>>`) - Default is 1 along each axis, dilation value along each spatial axis of the kernel.
* `group`(`usize`) - Default is 1, number of groups the input and output channels, C and oC, are divided into.
* `kernel_shape`(`Option<Span<usize>>`) - Shape of the convolution kernel. If not present, it is inferred from the shape of input W.
* `offset_group`(`Option<usize>`) - Default is 1, number of groups of offset. C must be divisible by offset_group.
* `pads`(`Option<Span<usize>>`) - Default is 0 along each axis, padding for the beginning and end along each spatial axis. The values represent the number of pixels added to the beginning and end of the corresponding axis and can take any nonnegative value.
* `strides`(`Option<Span<usize>>`) - Default is 1 along each axis, stride along each spatial axis.

## Returns

A `Tensor<T>` output tensor that contains the result of convolution.

## Examples
    
```rust
fn example_deform_conv() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 131072, sign: false });
    data.append(FP16x16 { mag: 196608, sign: false });
    data.append(FP16x16 { mag: 262144, sign: false });
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 393216, sign: false });
    data.append(FP16x16 { mag: 458752, sign: false });
    data.append(FP16x16 { mag: 524288, sign: false });
    let mut X = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    let mut W = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(8);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 32768, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 6553, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    let mut offset = TensorTrait::new(shape.span(), data.span());


    return NNTrait::deform_conv(
        @X,
        @W,
        @offset,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::Some(array![2, 2].span()),
        Option::None,
        Option::Some(array![0, 0, 0, 0].span()),
        Option::None,
    );
}

>>> [
        [
            [
                [9.5, 11.9],  
                [20.0, 24.0],
            ]
        ]
    ]

````