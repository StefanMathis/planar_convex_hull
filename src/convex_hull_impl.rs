//! This module contains implementations of [`ConvexHull`] for various foreign types.
//! Some implementations are hidden between feature flags.

use std::{collections::HashMap, hash::BuildHasher};

use super::{ConvexHull, Index};

impl<P: Into<[f64; 2]> + std::marker::Sync + Clone> ConvexHull for Vec<P> {
    fn convex_hull_get(&self, key: Index) -> [f64; 2] {
        // SAFETY: Index is only generated within the convex_hull method out of indices
        // returned by convex_hull_iter (which are known to be valid)
        return unsafe { self.get_unchecked(usize::from(key)) }
            .clone()
            .into();
    }

    fn convex_hull_iter(&self) -> impl Iterator<Item = (usize, [f64; 2])> {
        return self.iter().cloned().map(Into::into).enumerate();
    }
}

impl<P: Into<[f64; 2]> + std::marker::Sync + Clone, const N: usize> ConvexHull for [P; N] {
    fn convex_hull_get(&self, key: Index) -> [f64; 2] {
        // SAFETY: Index is only generated within the convex_hull method out of indices
        // returned by convex_hull_iter (which are known to be valid)
        return unsafe { self.get_unchecked(usize::from(key)) }
            .clone()
            .into();
    }

    fn convex_hull_iter(&self) -> impl Iterator<Item = (usize, [f64; 2])> {
        return self.iter().cloned().map(Into::into).enumerate();
    }
}

impl<P: Into<[f64; 2]> + std::marker::Sync + Clone> ConvexHull for &[P] {
    fn convex_hull_get(&self, key: Index) -> [f64; 2] {
        // SAFETY: Index is only generated within the convex_hull method out of indices
        // returned by convex_hull_iter (which are known to be valid)
        return unsafe { self.get_unchecked(usize::from(key)) }
            .clone()
            .into();
    }

    fn convex_hull_iter(&self) -> impl Iterator<Item = (usize, [f64; 2])> {
        return self.iter().cloned().map(Into::into).enumerate();
    }
}

impl<S: BuildHasher + std::marker::Sync, P: Into<[f64; 2]> + std::marker::Sync + Clone> ConvexHull
    for HashMap<usize, P, S>
{
    fn convex_hull_get(&self, key: Index) -> [f64; 2] {
        return self.get(&(key.into())).unwrap().clone().into();
    }

    fn convex_hull_iter(&self) -> impl Iterator<Item = (usize, [f64; 2])> {
        return self
            .iter()
            .map(|(key, val)| (key.clone(), val.clone().into()));
    }
}

#[cfg(feature = "slab")]
impl<P: Into<[f64; 2]> + std::marker::Sync + Clone> ConvexHull for slab::Slab<P> {
    fn convex_hull_get(&self, key: Index) -> [f64; 2] {
        // SAFETY: Index is only generated within the convex_hull method out of indices
        // returned by convex_hull_iter (which are known to be valid)
        return unsafe { self.get_unchecked(usize::from(key)) }
            .clone()
            .into();
    }

    fn convex_hull_iter(&self) -> impl Iterator<Item = (usize, [f64; 2])> {
        return self
            .iter()
            .map(|(key, val)| (key.clone(), val.clone().into()));
    }
}

#[cfg(feature = "ahash")]
impl<S: BuildHasher + std::marker::Sync, P: Into<[f64; 2]> + std::marker::Sync + Clone> ConvexHull
    for ahash::AHashMap<usize, P, S>
{
    fn convex_hull_get(&self, key: Index) -> [f64; 2] {
        // SAFETY: Index is only generated within the convex_hull method out of indices
        // returned by convex_hull_iter (which are known to be valid)
        return unsafe { self.get(&usize::from(key)).unwrap_unchecked() }
            .clone()
            .into();
    }

    fn convex_hull_iter(&self) -> impl Iterator<Item = (usize, [f64; 2])> {
        return self
            .iter()
            .map(|(key, val)| (key.clone(), val.clone().into()));
    }
}
