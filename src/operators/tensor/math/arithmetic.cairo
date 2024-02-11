use core::option::OptionTrait;
use core::traits::TryInto;
use core::array::ArrayTrait;
use core::array::SpanTrait;

use orion::operators::tensor::helpers::broadcast_shape;

use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait, unravel_index,};
use orion::operators::tensor::helpers::{broadcast_index_mapping, len_from_shape,};
use orion::utils::saturate;

fn add<
    T, impl TTensor: TensorTrait<T>, impl TAdd: Add<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>
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

fn add_by_scalar<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TAdd: Add<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TPartialEq: PartialEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    self: @Tensor<T>, val: T
) -> Tensor<T> {
    if val == NumberTrait::zero() {
        return *self;
    }

    let mut input_data = *self.data;
    let mut data_result = ArrayTrait::<T>::new();
    loop {
        match input_data.pop_front() {
            Option::Some(ele) => { data_result.append(*ele + val); },
            Option::None => { break; }
        };
    };

    return TensorTrait::<T>::new(*self.shape, data_result.span());
}

fn saturated_add<
    T,
    Q,
    impl TTensor: TensorTrait<T>,
    impl QTensor: TensorTrait<Q>,
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

    return TensorTrait::<Q>::new(broadcasted_shape, result.span());
}

fn sub<
    T, impl TTensor: TensorTrait<T>, impl TSub: Sub<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>
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

fn sub_by_scalar<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TSub: Sub<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TPartialEq: PartialEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    self: @Tensor<T>, val: T
) -> Tensor<T> {
    if val == NumberTrait::zero() {
        return *self;
    }

    let mut input_data = *self.data;
    let mut data_result = ArrayTrait::<T>::new();
    loop {
        match input_data.pop_front() {
            Option::Some(ele) => { data_result.append(*ele - val); },
            Option::None => { break; }
        };
    };

    return TensorTrait::<T>::new(*self.shape, data_result.span());
}

fn saturated_sub<
    T,
    Q,
    impl TTensor: TensorTrait<T>,
    impl QTensor: TensorTrait<Q>,
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

    return TensorTrait::<Q>::new(broadcasted_shape, result.span());
}

fn mul<
    T, impl TTensor: TensorTrait<T>, impl TMul: Mul<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>
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

fn mul_by_scalar<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TMul: Mul<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TPartialEq: PartialEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    self: @Tensor<T>, val: T
) -> Tensor<T> {
    if val == NumberTrait::one() {
        return *self;
    }

    let mut input_data = *self.data;
    let mut data_result = ArrayTrait::<T>::new();
    loop {
        match input_data.pop_front() {
            Option::Some(ele) => { data_result.append(*ele * val); },
            Option::None => { break; }
        };
    };

    return TensorTrait::<T>::new(*self.shape, data_result.span());
}

fn saturated_mul<
    T,
    Q,
    impl TTensor: TensorTrait<T>,
    impl QTensor: TensorTrait<Q>,
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

    return TensorTrait::<Q>::new(broadcasted_shape, result.span());
}

fn div<
    T, impl TTensor: TensorTrait<T>, impl TMul: Div<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>
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

fn div_by_scalar<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TDiv: Div<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TPartialEq: PartialEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    self: @Tensor<T>, val: T
) -> Tensor<T> {
    if val == NumberTrait::one() {
        return *self;
    }

    let mut input_data = *self.data;
    let mut data_result = ArrayTrait::<T>::new();
    loop {
        match input_data.pop_front() {
            Option::Some(ele) => { data_result.append(*ele / val); },
            Option::None => { break; }
        };
    };

    return TensorTrait::<T>::new(*self.shape, data_result.span());
}

fn saturated_div<
    T,
    Q,
    impl TTensor: TensorTrait<T>,
    impl QTensor: TensorTrait<Q>,
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

    return TensorTrait::<Q>::new(broadcasted_shape, result.span());
}

fn div_downcast<
    T,
    D,
    impl TTensor: TensorTrait<T>,
    impl DTensor: TensorTrait<D>,
    impl DDiv: Div<D>,
    impl TTryIntoD: TryInto<T, D>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl DCopy: Copy<D>,
    impl DDrop: Drop<D>
>(
    self: @Tensor<T>, other: @Tensor<T>
) -> Tensor<D> {
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
                (*(*self.data)[indices_self]).try_into().unwrap()
                    / (*(*other.data)[indices_other]).try_into().unwrap()
            );

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::<D>::new(broadcasted_shape, result.span());
}
