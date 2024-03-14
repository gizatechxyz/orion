# NNTrait::lrn

```rust
        fn lrn(tensor: @Tensor<T>, size: usize, alpha: Option<T>, beta: Option<T>, bias: Option<T>) -> Tensor<T>;
```

Local Response Normalization proposed in the AlexNet paper. It normalizes over local input regions. The local region is defined across the channels. 

## Args

* `tensor`(`@Tensor<T>`) - Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. 
* `size`(usize) - The number of channels to sum over.
* `alpha`(Option<T>) - Scaling parameter.
* `beta`(Option<T>) - The exponent. 
* `bias`(Option<T>)

## Returns

* A `Tensor<T>` with the same shape as the input tensor.

## Examples

```rust
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::numbers::{FixedTrait, FP16x16};
use orion::operators::nn::NNTrait;
use orion::operators::nn::FP16x16NN;

fn example() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 1400832, sign: true });
    data.append(FP16x16 { mag: 85184, sign: true });
    data.append(FP16x16 { mag: 399616, sign: false });
    data.append(FP16x16 { mag: 1828864, sign: false });
    data.append(FP16x16 { mag: 962560, sign: true });
    data.append(FP16x16 { mag: 1190912, sign: false });
    data.append(FP16x16 { mag: 1355776, sign: true });
    data.append(FP16x16 { mag: 946176, sign: false });
    data.append(FP16x16 { mag: 502528, sign: true });
    data.append(FP16x16 { mag: 85824, sign: false });
    data.append(FP16x16 { mag: 1333248, sign: false });
    data.append(FP16x16 { mag: 39200, sign: true });
    data.append(FP16x16 { mag: 712192, sign: false });
    data.append(FP16x16 { mag: 1497088, sign: false });
    data.append(FP16x16 { mag: 1521664, sign: false });
    data.append(FP16x16 { mag: 606720, sign: true });
    data.append(FP16x16 { mag: 848384, sign: false });
    data.append(FP16x16 { mag: 1732608, sign: true });
    data.append(FP16x16 { mag: 1158144, sign: true });
    data.append(FP16x16 { mag: 1806336, sign: false });
    data.append(FP16x16 { mag: 935424, sign: true });
    data.append(FP16x16 { mag: 1106944, sign: true });
    data.append(FP16x16 { mag: 1180672, sign: true });
    data.append(FP16x16 { mag: 1509376, sign: true });
    data.append(FP16x16 { mag: 856064, sign: false });
    data.append(FP16x16 { mag: 1841152, sign: false });
    data.append(FP16x16 { mag: 75008, sign: false });
    data.append(FP16x16 { mag: 19504, sign: true });
    data.append(FP16x16 { mag: 1326080, sign: false });
    data.append(FP16x16 { mag: 1423360, sign: true });
    data.append(FP16x16 { mag: 1258496, sign: true });
    data.append(FP16x16 { mag: 671232, sign: true });
    data.append(FP16x16 { mag: 548352, sign: false });
    data.append(FP16x16 { mag: 797696, sign: true });
    data.append(FP16x16 { mag: 1245184, sign: true });
    data.append(FP16x16 { mag: 1404928, sign: false });
    let tensor1 = TensorTrait::new(shape.span(), data.span());

    return NNTrait::lrn(@tensor1, 3, Option::Some(FP16x16 { mag: 13, sign: false }),Option::Some(FP16x16 { mag: 32768, sign: false }),Option::Some(FP16x16 { mag: 131072, sign: false }));
}
>>> [FP16x16 { mag: 964915, sign: true }, FP16x16 { mag: 60080, sign: true }, FP16x16 { mag: 282570, sign: false }, FP16x16 { mag: 1276847, sign: false }, FP16x16 { mag: 680331, sign: true }, FP16x16 { mag: 838194, sign: false }, FP16x16 { mag: 933879, sign: true }, FP16x16 { mag: 667340, sign: false }, FP16x16 { mag: 355340, sign: true }, FP16x16 { mag: 59919, sign: false }, FP16x16 { mag: 942330, sign: false }, FP16x16 { mag: 27589, sign: true }, FP16x16 { mag: 503258, sign: false }, 
    FP16x16 { mag: 1019961, sign: false }, FP16x16 { mag: 1074737, sign: false }, FP16x16 { mag: 424644, sign: true }, FP16x16 { mag: 599690, sign: false }, FP16x16 { mag: 1182030, sign: true }, FP16x16 { mag: 818383, sign: true }, FP16x16 { mag: 1230651, sign: false }, FP16x16 { mag: 660681, sign: true }, FP16x16 { mag: 774752, sign: true }, FP16x16 { mag: 834572, sign: true }, FP16x16 { mag: 1029735, sign: true }, FP16x16 { mag: 605008, sign: false }, FP16x16 { mag: 1295696, sign: false },
    FP16x16 { mag: 52967, sign: false }, FP16x16 { mag: 13759, sign: true }, FP16x16 { mag: 937699, sign: false }, FP16x16 { mag: 1006488, sign: true }, FP16x16 { mag: 889419, sign: true }, FP16x16 { mag: 472374, sign: true }, FP16x16 { mag: 387220, sign: false }, FP16x16 { mag: 562768, sign: true }, FP16x16 { mag: 880496, sign: true }, FP16x16 { mag: 993454, sign: false }]
```