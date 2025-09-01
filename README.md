planar_convex_hull
==================

A lightweight library providing a trait for implementing convex hull algorithm on your own datatype.

[`ConvexHull`]: https://docs.rs/planar_convex_hull/0.1.2/planar_convex_hull/trait.ConvexHull.html
[`convex_hull`]: https://docs.rs/planar_convex_hull/0.1.2/planar_convex_hull/trait.ConvexHull.html#method.convex_hull
[`Index`]: https://docs.rs/planar_convex_hull/0.1.2/planar_convex_hull/struct.Index.html

This library offers the [`ConvexHull`] trait which provides a divide-and-conquer convex hull algorithm in O(n log h) [1, 2]
via the [`convex_hull`] method. The trait can be implemented easily for any collection type holding point-like types 
which fulfills the following conditions:
- The point-like type implements `Into<[f64; 2]>`, `Sync` and `Clone`,
- The elements of the collections can be randomly accessed via an `usize` index,
- The elements and their indices can be iterated over.

# Example implementation

Let's assume we want to implement [`ConvexHull`] for a [`newtype`](https://doc.rust-lang.org/rust-by-example/generics/new_types.html) with an underlying slice of `[f64; 2]`. All we need to do is to tell the trait how to randomly access the data and how to iterate over the collection:

```rust
use planar_convex_hull::{ConvexHull, Index, reinterpret};

struct MySlice<'a>(&'a[[f64; 2]]);

impl<'a> ConvexHull for MySlice<'a> {
    /// Index is a newtype of usize and is used to make sure that only indices returned
    /// by convex_hull_iter can be used for random data access.
    fn convex_hull_get(&self, key: Index) -> [f64; 2] {
        // SAFETY: Index is only generated within the convex_hull method out of indices
        // returned by convex_hull_iter (which are known to be valid)
        return unsafe { self.0.get_unchecked(usize::from(key)) }
            .clone()
            .into();
    }

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

// Returns a `Vec<Index>`. This vector can now be used to access the points via `convex_hull_get`:
let hull_i = my_slice.convex_hull();
let pts: Vec<[f64; 2]> = hull_i.iter().map(|i| my_slice.convex_hull_get(*i)).collect();
assert_eq!(pts, vec![[10.0, 4.0], [0.0, 6.0], [-10.0, 4.0], [0.0, 2.0]]);

// Now we want to use the raw usize indices for something else
let hull = reinterpret(my_slice.convex_hull());
assert_eq!(hull, vec![0, 2, 1, 3]);
```

# Predefined implementations

The `imp` module contains implementations of [`ConvexHull`] for the following collection types with `P: Into<[f64; 2]>`:
* [`Vec<P>`](https://doc.rust-lang.org/std/vec/struct.Vec.html)
* [`HashMap<usize, P>`](https://doc.rust-lang.org/std/collections/struct.HashMap.html)
* [`[P; N]`](https://doc.rust-lang.org/std/primitive.array.html) with `N` being the size of the array
* [`&[P]`](https://doc.rust-lang.org/std/primitive.slice.html)
* [`Slab<P>`](https://docs.rs/slab/latest/slab/struct.Slab.html) (only available with feature flag **slab** enabled)
* [`AHashMap<usize, P>`](https://docs.rs/ahash/0.8.12/ahash/struct.AHashMap.html) (only available with feature flag **ahash** enabled)

Please open an issue on the repository website [https://github.com/StefanMathis/planar_convex_hull](https://github.com/StefanMathis/planar_convex_hull) if you need an implementation of [`ConvexHull`] for additional collection types. You can also use
the [`newtype`](https://doc.rust-lang.org/rust-by-example/generics/new_types.html) idiom as shown in the example for a reference
of a foreign collection instead (since all methods of [`ConvexHull`] operate on shared references).

# Feature flags

All features are disabled by default.

## Parallelizing the divide-and-conquer algorithm

Enabling the **rayon** feature parallelizes the divide-and-conquer algorithm.

## Implementations for foreign datatypes

The flags **slab** and **ahash** provide [`ConvexHull`] implementations for foreign data types. See [Predefined implementations](#predefined-implementations).

# Literature

1. Liu, Gh., Chen, Cb: A new algorithm for computing the convex hull of a planar point set.
J. Zhejiang Univ. - Sci. A 8, 1210–1217 (2007). https://doi.org/10.1631/jzus.2007.A1210
2. Saad, Omar: A Convex Hull Algorithm and its implementation in O(n log h) (2017). https://www.codeproject.com/Articles/1210225/Fast-and-improved-D-Convex-Hull-algorithm-and-its
