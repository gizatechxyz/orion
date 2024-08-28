use core::nullable::NullableImpl;
use core::num::traits::WrappingAdd;
use core::ops::index::Index;

pub trait VecTrait<V, T> {
    /// Creates a new V instance.
    /// Parameters
    /// * size The size of the vec to initialize.
    /// Returns
    /// * V The new vec instance.
    fn new(size: usize) -> V;

    /// Returns the item at the given index, or None if the index is out of bounds.
    /// Parameters
    /// * self The vec instance.
    /// * index The index of the item to get.
    /// Returns
    /// * Option<T> The item at the given index, or None if the index is out of bounds.
    fn get(ref self: V, index: usize) -> Option<T>;

    /// Returns the item at the given index, or panics if the index is out of bounds.
    /// Parameters
    /// * self The vec instance.
    /// * index The index of the item to get.
    /// Returns
    /// * T The item at the given index.
    fn at(ref self: V, index: usize) -> T;

    /// Pushes a new item to the vec.
    /// Parameters
    /// * self The vec instance.
    /// * value The value to push onto the vec.
    fn push(ref self: V, value: T);

    /// Sets the item at the given index to the given value.
    /// Panics if the index is out of bounds.
    /// Parameters
    /// * self The vec instance.
    /// * index The index of the item to set.
    /// * value The value to set the item to.
    fn set(ref self: V, index: usize, value: T);

    /// Returns the length of the vec.
    /// Parameters
    /// * self The vec instance.
    /// Returns
    /// * usize The length of the vec.
    fn len(self: @V) -> usize;
}

impl VecIndex<V, T, +VecTrait<V, T>> of Index<V, usize> {
    type Target = T;

    #[inline(always)]
    fn index(ref self: V, index: usize) -> T {
        self.at(index)
    }
}

pub struct NullableVec<T> {
    pub items: Felt252Dict<Nullable<T>>,
    pub len: usize,
}

impl DestructNullableVec<T, +Drop<T>> of Destruct<NullableVec<T>> {
    fn destruct(self: NullableVec<T>) nopanic {
        self.items.squash();
    }
}

use core::num::traits::Zero;

impl NullableVecImpl<T, +Zero<T>, +Drop<T>, +Copy<T>> of VecTrait<NullableVec<T>, T> {
    fn new(size: usize) -> NullableVec<T> {
        NullableVec { items: Default::default(), len: size }
    }

    fn get(ref self: NullableVec<T>, index: usize) -> Option<T> {
        if index < self.len() {
            Option::Some(self.items.get(index.into()).deref_or(Zero::zero()))
        } else {
            Option::None
        }
    }

    fn at(ref self: NullableVec<T>, index: usize) -> T {
        assert(index < self.len(), 'Index out of bounds');
        self.items.get(index.into()).deref_or(Zero::zero())
    }

    fn push(ref self: NullableVec<T>, value: T) {
        self.items.insert(self.len.into(), NullableImpl::new(value));
        self.len = self.len.wrapping_add(1);
    }

    fn set(ref self: NullableVec<T>, index: usize, value: T) {
        assert(index < self.len(), 'Index out of bounds');
        self.items.insert(index.into(), NullableImpl::new(value));
    }

    fn len(self: @NullableVec<T>) -> usize {
        *self.len
    }
}
