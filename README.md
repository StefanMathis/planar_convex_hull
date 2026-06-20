planar_convex_hull
==================

<!-- This file has ben generated with build.rs by concatenating docs/links.md,
docs/main.md and (if available docs/end.md). Do not modify this file, instead
modify the components. -->

[`ConvexHull`]: https://docs.rs/planar_convex_hull/0.3.0/planar_convex_hull/trait.ConvexHull.html
[`convex_hull`]: https://docs.rs/planar_convex_hull/0.3.0/planar_convex_hull/trait.ConvexHull.html#method.convex_hull
[`Index`]: https://docs.rs/planar_convex_hull/0.3.0/planar_convex_hull/struct.Index.html

[![Documentation](https://docs.rs/planar_convex_hull/badge.svg)](https://docs.rs/planar_convex_hull)

A lightweight library providing a trait for implementing a divide-and-conquer
planar convex hull algorithm for your own datatype.

The full API documentation is available at https://docs.rs/planar_convex_hull/0.3.0/planar_convex_hull.

> **Feedback welcome!**  
> Found a bug, missing docs, or have a feature request?  
> Please open an issue on [GitHub](https://github.com/StefanMathis/planar_convex_hull.git).

This library offers the [`ConvexHull`] trait which provides a divide-and-conquer
convex hull algorithm in O(n log h) [1, 2] via the [`convex_hull`] method. The
trait can be implemented easily for any collection type holding planar
point-like types which fulfills the following conditions:
- The point-like type implements `Into<[f64; 2]>`, `Sync` and `Clone`,
- The elements and their keys / indices can be iterated over.

# Examples

Let's assume we want to implement [`ConvexHull`] for a
[`newtype`](https://doc.rust-lang.org/rust-by-example/generics/new_types.html)
wrapper around a slice of `[f64; 2]`. All we need to do is to tell the trait how
how to iterate over the collection elements and their keys / indices:

```rust
use planar_convex_hull::ConvexHull;

struct MySlice<'a>(&'a[[f64; 2]]);

impl<'a> ConvexHull for MySlice<'a> {
    fn convex_hull_iter(&self) -> impl Iterator<Item = (usize, [f64; 2])> {
        return self.0.iter().cloned().map(Into::into).enumerate();
    }
}

// Rhombus with two points in its middle
let my_slice = MySlice(&[
    [10.0, 4.0],
    [-10.0, 4.0],
    [0.0, 6.0],
    [0.0, 2.0],
    [4.0, 4.0], // Not part of the convex hull
    [-4.0, 4.0], // Not part of the convex hull
]);

// The convex hull is the rhombus formed by the points 0, 1, 2 and 3. The
// points 4 and 5 are not part of the convex hull, because they are located
// on the line between points 0 and 1.
let mut hull = my_slice.convex_hull();
assert_eq!(hull.next(), Some((0, [10.0, 4.0])));
assert_eq!(hull.next(), Some((2, [0.0, 6.0])));
assert_eq!(hull.next(), Some((1, [-10.0, 4.0])));
assert_eq!(hull.next(), Some((3, [0.0, 2.0])));
assert_eq!(hull.next(), None);
```

The following example shows that even a collection type which has no concept of
"keys" or "indices" can still be used, provided that it has a stable iteration
order over its elements:

```rust
use std::collections::HashSet;
use ordered_float::OrderedFloat;
use planar_convex_hull::ConvexHull;

// Custom point type is needed because HashSet requires its elements to
// implement Eq and ConvexHull requires the elements to implement Into<[f64; 2]>
// (which [OrderedFloat<f64>; 2] does not do).
#[derive(Clone, Hash, PartialEq, Eq)]
struct MyPoint([OrderedFloat<f64>; 2]);

impl From<MyPoint> for [f64; 2] {
    fn from(value: MyPoint) -> Self {
        return [value.0[0].into_inner(), value.0[1].into_inner()];
    }
}

let data = &[
    [-3.0, -1.0],
    [-2.0, 2.0],
    [0.0, 0.0],
    [1.0, 3.0],
    [5.0, -1.0],
    [6.0, 2.0],
    [7.0, -4.0],
    [8.0, -1.0],
];
let hashset: HashSet<MyPoint> = HashSet::from_iter(
    data.iter().map(|[x, y]| MyPoint([OrderedFloat(x.clone()), OrderedFloat(y.clone())]))
);

// Keys are meaningless, so we focus on the actual points
let mut hull = hashset.convex_hull();
assert_eq!(hull.next().map(|(_, p)| p), Some([8.0, -1.0]));
assert_eq!(hull.next().map(|(_, p)| p), Some([6.0, 2.0]));
assert_eq!(hull.next().map(|(_, p)| p), Some([1.0, 3.0]));
assert_eq!(hull.next().map(|(_, p)| p), Some([-2.0, 2.0]));
assert_eq!(hull.next().map(|(_, p)| p), Some([-3.0, -1.0]));
assert_eq!(hull.next().map(|(_, p)| p), Some([7.0, -4.0]));
assert_eq!(hull.next(), None);
```

# Predefined implementations

The `imp` module contains implementations of [`ConvexHull`] for the following
collection types with `P: Into<[f64; 2]>`:
* [`Vec<P>`](https://doc.rust-lang.org/std/vec/struct.Vec.html)
* [`HashMap<usize, P>`](https://doc.rust-lang.org/std/collections/struct.HashSet.html)
* [`HashSet<P>`](https://doc.rust-lang.org/std/collections/struct.HashMap.html)
* [`[P; N]`](https://doc.rust-lang.org/std/primitive.array.html) with `N` being
the size of the array
* [`&[P]`](https://doc.rust-lang.org/std/primitive.slice.html)
* [`Slab<P>`](https://docs.rs/slab/latest/slab/struct.Slab.html) (only available
with feature flag  `slab ` enabled)
* [`AHashMap<usize, P>`](https://docs.rs/ahash/0.8.12/ahash/struct.AHashMap.html)
(only available with feature flag ahash` enabled)
* [`AHashSet<P>`](https://docs.rs/ahash/0.8.12/ahash/struct.AHashSet.html)
(only available with feature flag `ahash` enabled)

Please open an issue on the repository website
[https://github.com/StefanMathis/planar_convex_hull](https://github.com/StefanMathis/planar_convex_hull)
if you need an implementation of [`ConvexHull`] for additional collection types.
You can also use the
[`newtype`](https://doc.rust-lang.org/rust-by-example/generics/new_types.html)
idiom as shown in the `MySlice` implementation instead.

# Feature flags

All features are disabled by default.

## Parallelizing the divide-and-conquer algorithm

Enabling the `rayon` feature parallelizes the divide-and-conquer algorithm.

## Implementations for foreign datatypes

The flags `slab` and `ahash` provide [`ConvexHull`] implementations for
foreign data types. See [Predefined implementations](#predefined-implementations).

# Literature

1. Liu, Gh., Chen, Cb: A new algorithm for computing the convex hull of a planar
point set.
J. Zhejiang Univ. - Sci. A 8, 1210–1217 (2007). <https://doi.org/10.1631/jzus.2007.A1210>
2. Saad, Omar: A Convex Hull Algorithm and its implementation in O(n log h)
(2017). <https://www.codeproject.com/Articles/1210225/Fast-and-improved-D-Convex-Hull-algorithm-and-its>

**Note**: As of June 2026, \[2\] is unfortunately offline, but can still be
reached using the fantastic Wayback machine:
<https://web.archive.org/web/20250818231303/https://www.codeproject.com/Articles/1210225/Fast-and-improved-D-Convex-Hull-algorithm-and-its>.
A full copy of the website fetched from the Wayback machine is stored in the
repo (docs/convex_hull_algorithm.html).
