use core::box::BoxTrait;
use core::traits::Into;
use core::nullable::{Nullable, match_nullable, FromNullableResult, nullable_from_box};
use alexandria_data_structures::vec::{VecTrait};
use orion::numbers::NumberTrait;

struct NullableVec<T> {
    items: Felt252Dict<Nullable<T>>,
    len: usize,
}

impl DestructNullableVec<T, impl TDrop: Drop<T>> of Destruct<NullableVec<T>> {
    fn destruct(self: NullableVec<T>) nopanic {
        self.items.squash();
    }
}

impl NullableVecImpl<
    T, MAG, impl TDrop: Drop<T>, impl TCopy: Copy<T>, +NumberTrait<T, MAG>
> of VecTrait<NullableVec<T>, T> {
    fn new() -> NullableVec<T> {
        NullableVec { items: Default::default(), len: 0 }
    }

    fn get(ref self: NullableVec<T>, index: usize) -> Option<T> {
        if (index < self.len()) {
            return match match_nullable(self.items.get(index.into())) {
                FromNullableResult::Null(()) => { Option::Some(NumberTrait::zero()) },
                FromNullableResult::NotNull(val) => { Option::Some(val.unbox()) },
            };
        } else {
            Option::<T>::None
        }
    }

    fn at(ref self: NullableVec<T>, index: usize) -> T {
        assert(index < self.len(), 'Index out of bounds');

        return match self.get(index) {
            Option::Some(val) => val,
            Option::None => NumberTrait::zero(),
        };
    }

    fn push(ref self: NullableVec<T>, value: T) -> () {
        self.items.insert(self.len.into(), nullable_from_box(BoxTrait::new(value)));
        self.len = core::integer::u32_wrapping_add(self.len, 1_usize);
    }

    fn set(ref self: NullableVec<T>, index: usize, value: T) {
        if index >= self.len() {
            self.len = index + 1;
        }
        self.items.insert(index.into(), nullable_from_box(BoxTrait::new(value)));
    }

    fn len(self: @NullableVec<T>) -> usize {
        *self.len
    }
}
