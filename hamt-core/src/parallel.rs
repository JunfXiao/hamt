//! Per-cell loops that use rayon only when the caller asks for it AND the `parallel`
//! feature is enabled; otherwise they run sequentially on the calling thread.
//!
//! [`Surface`](crate::surface::Surface) asks for it only on 2-D / 3-D lattices
//! ([`Surface::is_multi_dimensional`](crate::surface::Surface::is_multi_dimensional)).
//! A 1-D wall has too little work per loop. Replaying a 188-cell TRNSYS wall (one
//! macro-step = 34 calls of `sync_state_from` / `update_cell_properties` at
//! 2.4–4.5 µs each, 18 % of the solve time), a solve step took 610 µs sequentially,
//! 895 µs with `RAYON_NUM_THREADS=1` and 2042 µs with the default 24 threads. The
//! TRNSYS kernel calls the DLL from its own thread, which is not a rayon worker, so
//! even a one-thread pool hands every loop to a pool thread and back.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Applies `op` to every element of `slice`, in parallel only if `parallel` is set
/// and the `parallel` feature is enabled.
pub(crate) fn for_each_mut<T, F>(slice: &mut [T], parallel: bool, op: F)
where
    T: Send,
    F: Fn(&mut T) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        if parallel {
            return slice.par_iter_mut().for_each(op);
        }
    }
    let _ = parallel;
    slice.iter_mut().for_each(op)
}

/// Applies `op` to every pair `(dst[i], src[i])`, in parallel only if `parallel` is
/// set and the `parallel` feature is enabled.
pub(crate) fn zip_for_each_mut<T, U, F>(dst: &mut [T], src: &[U], parallel: bool, op: F)
where
    T: Send,
    U: Sync,
    F: Fn(&mut T, &U) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        if parallel {
            return dst.par_iter_mut().zip(src).for_each(|(d, s)| op(d, s));
        }
    }
    let _ = parallel;
    dst.iter_mut().zip(src).for_each(|(d, s)| op(d, s))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Mutex;
    use std::thread::{self, ThreadId};

    fn record_thread(seen: &Mutex<HashSet<ThreadId>>) {
        seen.lock().unwrap().insert(thread::current().id());
    }

    #[test]
    fn sequential_for_each_mut_stays_on_the_calling_thread() {
        let seen = Mutex::new(HashSet::new());
        let mut v = vec![0u32; 10_000];
        for_each_mut(&mut v, false, |x| {
            *x += 1;
            record_thread(&seen);
        });
        assert!(v.iter().all(|&x| x == 1));
        assert_eq!(
            *seen.lock().unwrap(),
            HashSet::from([thread::current().id()])
        );
    }

    #[test]
    fn sequential_zip_for_each_mut_stays_on_the_calling_thread() {
        let seen = Mutex::new(HashSet::new());
        let src: Vec<u32> = (0..10_000).collect();
        let mut dst = vec![0u32; src.len()];
        zip_for_each_mut(&mut dst, &src, false, |d, s| {
            *d = *s;
            record_thread(&seen);
        });
        assert_eq!(dst, src);
        assert_eq!(
            *seen.lock().unwrap(),
            HashSet::from([thread::current().id()])
        );
    }

    #[test]
    fn parallel_loops_visit_every_element_exactly_once() {
        let mut v = vec![0u32; 10_000];
        for_each_mut(&mut v, true, |x| *x += 1);
        assert!(v.iter().all(|&x| x == 1));

        let src: Vec<u32> = (0..10_000).collect();
        let mut dst = vec![0u32; src.len()];
        zip_for_each_mut(&mut dst, &src, true, |d, s| *d = *s);
        assert_eq!(dst, src);
    }
}
