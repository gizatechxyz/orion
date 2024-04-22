# tensor.qlinear_conv

```rust
    
qlinear_conv(
    self: @Tensor<Q>,
    X_scale: @Tensor<T>,
    X_zero_point: @Tensor<T>,
    W: @Tensor<Q>,
    W_scale: @Tensor<T>,
    W_zero_point: @Tensor<T>,
    B: Option<Span<Q>>,
    auto_pad: Option<AUTO_PAD>,
    dilations: Option<Span<usize>>,
    group: Option<usize>,
    kernel_shape: Option<Span<usize>>,
    pads: Option<Span<usize>>,
    strides: Option<Span<usize>>,
    y_scale: @Tensor<T>,
    y_zero_point: @Tensor<T>,
) -> Tensor<Q> 
```

Performs convolution on quantized Tensors

The convolution operator consumes a quantized input tensor, its scale and zero point, a quantized filter, its scale and zero point, 
and output's scale and zero point, and computes the quantized output. Each scale and zero-point pair must have same shape. 
It means they must be either scalars (per tensor) or 1-D tensors (per output channel). Each input or output and its related zero point must have same type. 

## Args

* `self`(`@Tensor<i8>`) - Quantized input data tensor, has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ... x Dn).
* `X_scale`(`@Tensor<T>`) - Scale for input `X`.
* `X_zero_point`(`@Tensor<T>`) - Zero point for input `X`.
* `W`(`@Tensor<i8>`) - Quantized weight tensor that will be used in the convolutions; has size (M x C/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be (M x C/group x k1 x k2 x ... x kn), where (k1 x k2 x ... kn) is the dimension of the kernel. 
* `W_scale`(`@Tensor<T>`) - Scale for input `W`.
* `W_zero_point`(`@Tensor<T>`) - Zero point for input `W`. 
* `B`(`Option<@Tensor<T>>`) - Optional 1D bias to be added to the convolution, has size of M. Bias must be quantized using scale = x_scale * w_scale and zero_point = 0.
* `auto_pad`(`Option<AUTO_PAD>`) - Default is NOTSET, auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. NOTSET means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that `output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`.
* `dilations`(`Option<Span<usize>>`) - Dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each spatial axis.
* `group`(`Option<usize>`) - Default is 1, number of groups input channels and output channels are divided into.
* `kernel_shape`(`Option<Span<usize>>`) - The shape of the convolution kernel. If not present, should be inferred from input W.
* `pads`(`Option<Span<usize>>`) - Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis.
* `strides`(`Option<Span<usize>>`) - Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.
* `y_scale`(`@Tensor<T>`) - Scale for output.
* `y_zero_point`(`@Tensor<T>`) - Zero point for output.   

## Returns

A new `Tensor<i8>`, containing the quantized result of the convolution of the dequantized inputs.

## Type Constraints

u32 tensor, not supported.
fp8x23wide tensor, not supported.
fp16x16wide tensor, not supported.
 
## Example

```rust
   use orion::operators::tensor::{TensorTrait, Tensor};
   use orion::operators::tensor::I8TensorPartialEq;
   use orion::utils::{assert_eq, assert_seq_eq};
   use orion::operators::tensor::{I8Tensor, I8TensorAdd};
   use orion::operators::tensor::FP16x16TensorPartialEq;
   use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
   use core::array::{ArrayTrait, SpanTrait};
   use orion::operators::tensor::implementations::tensor_fp16x16::{TensorI8IntoTensorFP16x16, FP16x16TensorSub,FP16x16TensorDiv,FP16x16TensorMul};
   use orion::numbers::{FP16x16, I8IntoFP16x16};

   fn qlinear_conv_example() -> Tensor<i8> {
       let mut shape = ArrayTrait::<usize>::new();
       shape.append(1);
       shape.append(1);
       shape.append(3);
       shape.append(3);

       let mut data = ArrayTrait::new();
       data.append(1);
       data.append(2);
       data.append(3);
       data.append(4);
       data.append(5);
       data.append(6);
       data.append(7);
       data.append(8);
       data.append(9);
       let mut X = TensorTrait::new(shape.span(), data.span());

       let mut shape = ArrayTrait::<usize>::new();
       shape.append(1);
       shape.append(1);
       shape.append(1);
       shape.append(1);

       let mut data = ArrayTrait::new();
       data.append(0_i8);
       let mut W = TensorTrait::new(shape.span(), data.span());

       let mut shape = ArrayTrait::<usize>::new();
       shape.append(6);

       let mut data = ArrayTrait::new();
       data.append(FP16x16 { mag: 32768, sign: false });
       data.append(FP16x16 { mag: 131072, sign: false });
       data.append(FP16x16 { mag: 26214, sign: false });
       data.append(FP16x16 { mag: 196608, sign: false });
       data.append(FP16x16 { mag: 13107, sign: false });
       data.append(FP16x16 { mag: 262144, sign: false });
       let mut param = TensorTrait::new(shape.span(), data.span());
   
       let X_scale = TensorTrait::new(
           shape: array![1].span(), data: array![*param.data.at(0)].span(),
       );
       let X_zero_point = TensorTrait::new(
           shape: array![1].span(), data: array![*param.data.at(1)].span(),
       );
       let W_scale = TensorTrait::new(
           shape: array![1].span(), data: array![*param.data.at(2)].span(),
       );
       let W_zero_point = TensorTrait::new(
           shape: array![1].span(), data: array![*param.data.at(3)].span(),
       );
       let y_scale = TensorTrait::new(
           shape: array![1].span(), data: array![*param.data.at(4)].span(),
       );
       let y_zero_point = TensorTrait::new(
           shape: array![1].span(), data: array![*param.data.at(5)].span(),
       );
   
       return X
           .qlinear_conv(
               @X_scale,
               @X_zero_point,
               @W,
               @W_scale,
               @W_zero_point,
               Option::None,
               Option::None,
               Option::None,
               Option::None,
               Option::None,
               Option::None,
               Option::None,
               @y_scale,
               @y_zero_point,
           );
   }
   
>>> [
        [
            [
                [  7,   4,   1],
                [ -2,  -5,  -8],
                [-11, -14, -17],
            ]
        ]
    ]
```
