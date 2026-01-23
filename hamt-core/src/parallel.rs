#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
pub(crate) type Iter<'a, T> = rayon::slice::Iter<'a, T>;
#[cfg(not(feature = "parallel"))]
pub(crate) type Iter<'a, T> = std::slice::Iter<'a, T>;

#[cfg(feature = "parallel")]
pub(crate) type IterMut<'a, T> = rayon::slice::IterMut<'a, T>;
#[cfg(not(feature = "parallel"))]
pub(crate) type IterMut<'a, T> = std::slice::IterMut<'a, T>;

#[cfg(feature = "parallel")]
pub(crate) fn par_or_seq<T: Sync>(slice: &[T]) -> Iter<'_, T> {
    slice.par_iter()
}

#[cfg(not(feature = "parallel"))]
pub(crate) fn par_or_seq<T>(slice: &[T]) -> Iter<'_, T> {
    slice.iter()
}

#[cfg(feature = "parallel")]
pub(crate) fn par_or_seq_mut<T: Send>(slice: &mut [T]) -> IterMut<'_, T> {
    slice.par_iter_mut()
}

#[cfg(not(feature = "parallel"))]
pub(crate) fn par_or_seq_mut<T>(slice: &mut [T]) -> IterMut<'_, T> {
    slice.iter_mut()
}
