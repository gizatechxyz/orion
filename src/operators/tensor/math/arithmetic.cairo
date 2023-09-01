use core::option::OptionTrait;
use core::traits::TryInto;
use array::ArrayTrait;
use array::SpanTrait;

use orion::operators::tensor::helpers::broadcast_shape;

use orion::operators::tensor::core::{Tensor, TensorTrait, unravel_index,};
use orion::operators::tensor::helpers::{broadcast_index_mapping, len_from_shape,};
use orion::utils::saturate;

fn add<
    T,
    F,
    impl TTensor: TensorTrait<T, F>,
    impl TAdd: Add<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    self: @Tensor<T>, other: @Tensor<T>
) -> Tensor<T> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result.append(*(*self.data)[indices_self] + *(*other.data)[indices_other]);

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<T>::new(broadcasted_shape, result.span());
}

fn saturated_add<
    T,
    F,
    Q,
    impl TTensor: TensorTrait<T, F>,
    impl QTensor: TensorTrait<Q, F>,
    impl TAdd: Add<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TTryInto: TryInto<T, Q>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl QDrop: Drop<Q>,
>(
    self: @Tensor<T>, other: @Tensor<T>, min_saturation: T, max_saturation: T
) -> Tensor<Q> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result
            .append(
                saturate(
                    min_saturation,
                    max_saturation,
                    *(*self.data)[indices_self] + *(*other.data)[indices_other]
                )
                    .try_into()
                    .unwrap()
            );

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<Q, F>::new(broadcasted_shape, result.span());
}

fn sub<
    T,
    F,
    impl TTensor: TensorTrait<T, F>,
    impl TSub: Sub<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    self: @Tensor<T>, other: @Tensor<T>
) -> Tensor<T> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result.append(*(*self.data)[indices_self] - *(*other.data)[indices_other]);

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<T>::new(broadcasted_shape, result.span());
}

fn saturated_sub<
    T,
    F,
    Q,
    impl TTensor: TensorTrait<T, F>,
    impl QTensor: TensorTrait<Q, F>,
    impl TSub: Sub<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TTryInto: TryInto<T, Q>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl QDrop: Drop<Q>,
>(
    self: @Tensor<T>, other: @Tensor<T>, min_saturation: T, max_saturation: T
) -> Tensor<Q> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result
            .append(
                saturate(
                    min_saturation,
                    max_saturation,
                    *(*self.data)[indices_self] - *(*other.data)[indices_other]
                )
                    .try_into()
                    .unwrap()
            );

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<Q, F>::new(broadcasted_shape, result.span());
}

fn mul<
    T,
    F,
    impl TTensor: TensorTrait<T, F>,
    impl TMul: Mul<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    self: @Tensor<T>, other: @Tensor<T>
) -> Tensor<T> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result.append(*(*self.data)[indices_self] * *(*other.data)[indices_other]);

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<T>::new(broadcasted_shape, result.span());
}

fn saturated_mul<
    T,
    F,
    Q,
    impl TTensor: TensorTrait<T, F>,
    impl QTensor: TensorTrait<Q, F>,
    impl TMul: Mul<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TTryInto: TryInto<T, Q>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl QDrop: Drop<Q>,
>(
    self: @Tensor<T>, other: @Tensor<T>, min_saturation: T, max_saturation: T
) -> Tensor<Q> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result
            .append(
                saturate(
                    min_saturation,
                    max_saturation,
                    *(*self.data)[indices_self] * *(*other.data)[indices_other]
                )
                    .try_into()
                    .unwrap()
            );

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<Q, F>::new(broadcasted_shape, result.span());
}

fn div<
    T,
    F,
    impl TTensor: TensorTrait<T, F>,
    impl TMul: Div<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    self: @Tensor<T>, other: @Tensor<T>
) -> Tensor<T> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result.append(*(*self.data)[indices_self] / *(*other.data)[indices_other]);

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<T>::new(broadcasted_shape, result.span());
}

fn saturated_div<
    T,
    F,
    Q,
    impl TTensor: TensorTrait<T, F>,
    impl QTensor: TensorTrait<Q, F>,
    impl TDiv: Div<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TTryInto: TryInto<T, Q>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl QDrop: Drop<Q>,
>(
    self: @Tensor<T>, other: @Tensor<T>, min_saturation: T, max_saturation: T
) -> Tensor<Q> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result
            .append(
                saturate(
                    min_saturation,
                    max_saturation,
                    *(*self.data)[indices_self] / *(*other.data)[indices_other]
                )
                    .try_into()
                    .unwrap()
            );

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<Q, F>::new(broadcasted_shape, result.span());
}
