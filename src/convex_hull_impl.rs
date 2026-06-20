//! This module contains implementations of [`ConvexHull`] for various
//! foreign types. Some implementations are hidden behind feature flags, see
//! the [`crate`] module documentation.

use std::{
    collections::{HashMap, HashSet},
    hash::BuildHasher,
};

use super::ConvexHull;

impl<P: Into<[f64; 2]> + std::marker::Sync + Clone> ConvexHull for Vec<P> {
    fn convex_hull_iter(&self) -> impl Iterator<Item = (usize, [f64; 2])> {
        return self.iter().cloned().map(Into::into).enumerate();
    }
}

impl<P: Into<[f64; 2]> + std::marker::Sync + Clone, const N: usize> ConvexHull for [P; N] {
    fn convex_hull_iter(&self) -> impl Iterator<Item = (usize, [f64; 2])> {
        return self.iter().cloned().map(Into::into).enumerate();
    }
}

impl<P: Into<[f64; 2]> + std::marker::Sync + Clone> ConvexHull for &[P] {
    fn convex_hull_iter(&self) -> impl Iterator<Item = (usize, [f64; 2])> {
        return self.iter().cloned().map(Into::into).enumerate();
    }
}

impl<S: BuildHasher + std::marker::Sync, P: Into<[f64; 2]> + std::marker::Sync + Clone> ConvexHull
    for HashMap<usize, P, S>
{
    fn convex_hull_iter(&self) -> impl Iterator<Item = (usize, [f64; 2])> {
        return self
            .iter()
            .map(|(key, val)| (key.clone(), val.clone().into()));
    }
}

impl<S: BuildHasher + std::marker::Sync, P: Into<[f64; 2]> + std::marker::Sync + Clone> ConvexHull
    for HashSet<P, S>
{
    fn convex_hull_iter(&self) -> impl Iterator<Item = (usize, [f64; 2])> {
        return self
            .iter()
            .enumerate()
            .map(|(i, val)| (i, val.clone().into()));
    }
}

#[cfg(feature = "slab")]
impl<P: Into<[f64; 2]> + std::marker::Sync + Clone> ConvexHull for slab::Slab<P> {
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
    fn convex_hull_iter(&self) -> impl Iterator<Item = (usize, [f64; 2])> {
        return self
            .iter()
            .map(|(key, val)| (key.clone(), val.clone().into()));
    }
}

#[cfg(feature = "ahash")]
impl<S: BuildHasher + std::marker::Sync, P: Into<[f64; 2]> + std::marker::Sync + Clone> ConvexHull
    for ahash::AHashSet<P, S>
{
    fn convex_hull_iter(&self) -> impl Iterator<Item = (usize, [f64; 2])> {
        return self
            .iter()
            .enumerate()
            .map(|(i, val)| (i, val.clone().into()));
    }
}
