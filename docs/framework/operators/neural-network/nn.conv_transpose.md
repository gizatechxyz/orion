# NNTrait::conv_transpose

```rust
    conv_transpose(
    X: @Tensor<T>,
    W: @Tensor<T>,
    B: Option<@Tensor<T>>,
    auto_pad: Option<AUTO_PAD>,
    dilations: Option<Span<usize>>,
    group: Option<usize>,
    kernel_shape: Option<Span<usize>>,
    output_padding: Option<Span<usize>>,
    output_shape: Option<Span<usize>>,
    pads: Option<Span<usize>>,
    strides: Option<Span<usize>>,
) -> Tensor<T>
```

The convolution transpose operator consumes an input tensor and a input weigth tensor, and computes the output.

## Args

* `X`(`@Tensor<T>`) - Input data tensor, has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W if 2D, otherwise the size is (N x C x D1 x D2 ... x Dn).
* `W`(`@Tensor<T>`) - The weight tensor, has size (C x M/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps if 2D, for more than 2 dimensions, the weight shape will be (C x M/group x k1 x k2 x ... x kn).
* `B`(`Option<@Tensor<T>>`) - Optional 1D bias to be added to the convolution, has size of M.
* `auto_pad`(`Option<AUTO_PAD>`) - Default is NOTSET, auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. NOTSET means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that `output_shape[i] = input_shape[i] * strides[i]` for each axis `i`.
* `dilations`(`Option<Span<usize>>`) - Dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each spatial axis.
* `group`(`Option<usize>`) - Default is 1, number of groups input channels and output channels are divided into.
* `kernel_shape`(`Option<Span<usize>>`) - The shape of the convolution kernel. If not present, should be inferred from input W.
* `output_padding`(`Option<Span<usize>>`) - Additional elements added to the side with higher coordinate indices in the output. Each padding value in "output_padding" must be less than the corresponding stride/dilation dimension. By default, this attribute is a zero vector. 
* `output_shape`(`Option<Span<usize>>`) - The shape of the output can be explicitly set which will cause pads values to be auto generated. If output_shape is specified pads values are ignored. See doc for details for equations to generate pads.
* `pads`(`Option<Span<usize>>`) - Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis.
* `strides`(`Option<Span<usize>>`) - Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.

## Returns

A `Tensor<T>` of shape (M, N).

## Examples
    
```rust
use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;
use orion::operators::nn::FP16x16NN;
use orion::numbers::FP16x16;
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};

fn example_conv_transpose() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    let W = TensorTrait::new(shape.span(), data.span());

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

    return NNTrait::conv_transpose(
        @X,
        @W,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
        Option::None,
    );

}
>>> [
        [
            [
                [0.0, 1.0, 3.0, 3.0, 2.0],  
                [3.0, 8.0, 15.0, 12.0, 7.0],
                [9.0, 21.0, 36.0, 27.0, 15.0],
                [9.0, 20.0, 33.0, 24.0, 13.0],
                [6.0, 13.0, 21.0, 15.0, 8.0],
            ],
            [
                [0.0, 1.0, 3.0, 3.0, 2.0],
                [3.0, 8.0, 15.0, 12.0, 7.0],
                [9.0, 21.0, 36.0, 27.0, 15.0],
                [9.0, 20.0, 33.0, 24.0, 13.0],
                [6.0, 13.0, 21.0, 15.0, 8.0],
            ],
        ]
    ]

````
