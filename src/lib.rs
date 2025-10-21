#![doc = include_str!("../README.md")]

use ordered_float::OrderedFloat;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::f64::INFINITY;
use std::f64::NEG_INFINITY;
use std::ops::Bound::Excluded;
use std::ops::Bound::Unbounded;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub mod convex_hull_impl;

/**
A trait for implementing a planar convex hull algorithm for a collection type.

This trait is meant to be implemented on a collection (e.g. a vector, slice, hashmap, ...) which
stores instances of a type representing a 2-dimensional point in cartesian coordinates. The collection also
needs to allow accessing this data via an `usize` index (so. e.g. a hashset is not suitable).
The type needs to implement `Clone` and `Into<[f64; 2]>`; the first array element is treated as the x-coordinate
and the second element is treated as the y-coordinate.

Implementing the trait provides the [`ConvexHull::convex_hull`] method which returns a vector of indices describing
the convex hull of a point set. To do so, the methods [`ConvexHull::convex_hull_get`] (for random data access)
and [`ConvexHull::convex_hull_iter`] (for iterating through the points and the corresponding indices) need to
be implemented.

The [README / module documentation](crate) shows an example
how to implement these two methods for a custom data collection.
 */
pub trait ConvexHull: std::marker::Sync {
    /**
    Returns a point using the given index.

    As described in its docstring, instances of [`Index`] are created within [`ConvexHull::convex_hull`]
    and cannot be created manually. The underlying `usize` index can be read out via `usize::from`.
    As long as [`ConvexHull::convex_hull_iter`] is implemented correctly, it can be assumed that
    the underlying `usize` is always valid for the collection. This allows access optimization:

    ```ignore
    impl<P: Into<[f64; 2]> Clone + std::marker::Sync> ConvexHull for Vec<P> {
        fn convex_hull_get(&self, key: Index) -> [f64; 2] {
            // SAFETY: Index is only generated within the convex_hull method out of indices
            // returned by convex_hull_iter (which are known to be valid)
            return unsafe { self.get_unchecked(usize::from(key)) }
                .clone()
                .into();
    }
    ```
     */
    fn convex_hull_get(&self, key: Index) -> [f64; 2];

    /**
    Iterates over all indices of a collection and the associated points in any order.

    The following example shows how this function is implemented for the `Vec` type:
    ```ignore
    impl<P: Into<[f64; 2]> + std::marker::Sync + Clone> ConvexHull for Vec<P> {
        fn convex_hull_iter(&self) -> impl Iterator<Item = (usize, [f64; 2])> {
            return self.iter().cloned().map(Into::into).enumerate();
        }
    }
    ```
    */
    fn convex_hull_iter(&self) -> impl Iterator<Item = (usize, [f64; 2])>;

    // ==================================================================================

    /**
    Calculates the convex hull for the given collection

    This function calculates the convex hull of a set of points using the divide-and-conquer algorithm presented in \[1, 2\].
    If the input contains duplicate points which are part of the convex hull, one of the points is selected arbitrarily.
    Nonreal points (points containing NaN or infinite values) are ignored.

    The returned vector `Vec<Index>` contains the convex hull indices in counter-clockwise order. The underlying points
    can be accessed using the [`ConvexHull::convex_hull_get`] method, the inderlying `usize` indices can be retrieved via [`reinterpret`].
    The convex hull is defined by connecting neighboring points as defined by the indices
    (including the last and the first index) by straight lines.

    In addition to the original algorithm descriptions, this implementation also covers edge cases such as all points being located on a
    single line or multiple hull points having the same x- or y- coordinate (see examples below).

    When the  `rayon ` feature is enabled, the divide-and-conquer part of the algorithm is parallelized.

    # Literature

    1. Liu, Gh., Chen, Cb: A new algorithm for computing the convex hull of a planar point set.
    J. Zhejiang Univ. - Sci. A 8, 1210–1217 (2007). [https://doi.org/10.1631/jzus.2007.A1210](https://doi.org/10.1631/jzus.2007.A1210)
    2. Saad, Omar: A Convex Hull Algorithm and its implementation in O(n log h) (2017).
    [https://www.codeproject.com/Articles/1210225/Fast-and-improved-D-Convex-Hull-algorithm-and-its](https://www.codeproject.com/Articles/1210225/Fast-and-improved-D-Convex-Hull-algorithm-and-its)

    # Examples
    ```
    use planar_convex_hull::{ConvexHull, reinterpret};

    // Rhombus with two points in its middle
    let slice = &[
        [10.0, 4.0],
        [-10.0, 4.0],
        [0.0, 6.0],
        [0.0, 2.0],
        [4.0, 4.0], // Not part of the convex hull
        [-4.0, 4.0], // Not part of the convex hull
    ];

    // Returns a `Vec<Index>`. This vector can now be used to access the points via `convex_hull_get`:
    let hull_i = slice.convex_hull();
    let pts: Vec<[f64; 2]> = hull_i.iter().map(|i| slice.convex_hull_get(*i)).collect();
    assert_eq!(pts, vec![[10.0, 4.0], [0.0, 6.0], [-10.0, 4.0], [0.0, 2.0]]);

    // Now we want to use the raw usize indices for something else
    let hull = reinterpret(slice.convex_hull());
    assert_eq!(hull, vec![0, 2, 1, 3]);

    // All points on a single line with the same y-value everywhere. The points 2 and 3 in the middle of the line show up twice, because the convex hull goes "up and down" the x-axis
    let slice = &[
        [10.0, -2.0],
        [-10.0, -2.0],
        [0.0, -2.0],
        [3.0, -2.0],
    ];
    let hull = reinterpret(slice.convex_hull());
    assert_eq!(hull, vec![0, 3, 2, 1, 2, 3]);

    // Triangle with a point on the diagonal
    let slice = &[
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.5, 0.5], // Part of the convex hull
    ];
    let hull = reinterpret(slice.convex_hull());
    assert_eq!(hull, vec![0, 3, 1, 2]);

    // Triangle with a point in the middle
    let slice = &[
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.25, 0.25], // Not part of the convex hull
    ];
    let hull = reinterpret(slice.convex_hull());
    assert_eq!(hull, vec![0, 1, 2]);
    ```
     */
    fn convex_hull(&self) -> Vec<Index> {
        // Step 1: Identify the four point-pairs defining each quadrant. A quadrant is defined by the x-value of one point and the y-value of the other point.
        let mut q1x: usize = usize::MAX;
        let mut q1y: usize = usize::MAX;
        let mut q2x: usize = usize::MAX;
        let mut q2y: usize = usize::MAX;
        let mut q3x: usize = usize::MAX;
        let mut q3y: usize = usize::MAX;
        let mut q4x: usize = usize::MAX;
        let mut q4y: usize = usize::MAX;
        let mut q1x_pt: [f64; 2] = [NEG_INFINITY, NEG_INFINITY];
        let mut q1y_pt: [f64; 2] = [NEG_INFINITY, NEG_INFINITY];
        let mut q2x_pt: [f64; 2] = [INFINITY, NEG_INFINITY];
        let mut q2y_pt: [f64; 2] = [INFINITY, NEG_INFINITY];
        let mut q3x_pt: [f64; 2] = [INFINITY, INFINITY];
        let mut q3y_pt: [f64; 2] = [INFINITY, INFINITY];
        let mut q4x_pt: [f64; 2] = [NEG_INFINITY, INFINITY];
        let mut q4y_pt: [f64; 2] = [NEG_INFINITY, INFINITY];

        let mut num_real_points = 0;

        for (idx, point) in self.convex_hull_iter() {
            // Skip any non-real points
            if point[0].is_infinite()
                || point[1].is_infinite()
                || point[0].is_nan()
                || point[1].is_nan()
            {
                continue;
            }

            num_real_points += 1;

            // q1x
            if let Some(cmp) = q1x_pt[0].partial_cmp(&point[0]) {
                match cmp {
                    Ordering::Less => {
                        q1x_pt = point.clone();
                        q1x = idx;
                    }
                    Ordering::Equal => {
                        if point[1] > q1x_pt[1] {
                            q1x_pt = point.clone();
                            q1x = idx;
                        }
                    }
                    Ordering::Greater => (),
                }
            }

            // q1y
            if let Some(cmp) = q1y_pt[1].partial_cmp(&point[1]) {
                match cmp {
                    Ordering::Less => {
                        q1y_pt = point.clone();
                        q1y = idx;
                    }
                    Ordering::Equal => {
                        if point[0] > q1y_pt[0] {
                            q1y_pt = point.clone();
                            q1y = idx;
                        }
                    }
                    Ordering::Greater => (),
                }
            }

            // q2x
            if let Some(cmp) = q2x_pt[0].partial_cmp(&point[0]) {
                match cmp {
                    Ordering::Greater => {
                        q2x_pt = point.clone();
                        q2x = idx;
                    }
                    Ordering::Equal => {
                        if point[1] > q2x_pt[1] {
                            q2x_pt = point.clone();
                            q2x = idx;
                        }
                    }
                    Ordering::Less => (),
                }
            }

            // q2y
            if let Some(cmp) = q2y_pt[1].partial_cmp(&point[1]) {
                match cmp {
                    Ordering::Less => {
                        q2y_pt = point.clone();
                        q2y = idx;
                    }
                    Ordering::Equal => {
                        if point[0] < q2y_pt[0] {
                            q2y_pt = point.clone();
                            q2y = idx;
                        }
                    }
                    Ordering::Greater => (),
                }
            }

            // q3x
            if let Some(cmp) = q3x_pt[0].partial_cmp(&point[0]) {
                match cmp {
                    Ordering::Greater => {
                        q3x_pt = point.clone();
                        q3x = idx;
                    }
                    Ordering::Equal => {
                        if point[1] < q3x_pt[1] {
                            q3x_pt = point.clone();
                            q3x = idx;
                        }
                    }
                    Ordering::Less => (),
                }
            }

            // q3y
            if let Some(cmp) = q3y_pt[1].partial_cmp(&point[1]) {
                match cmp {
                    Ordering::Greater => {
                        q3y_pt = point.clone();
                        q3y = idx;
                    }
                    Ordering::Equal => {
                        if point[0] < q3y_pt[0] {
                            q3y_pt = point.clone();
                            q3y = idx;
                        }
                    }
                    Ordering::Less => (),
                }
            }

            // q4x
            if let Some(cmp) = q4x_pt[0].partial_cmp(&point[0]) {
                match cmp {
                    Ordering::Less => {
                        q4x_pt = point.clone();
                        q4x = idx;
                    }
                    Ordering::Equal => {
                        if point[1] < q4x_pt[1] {
                            q4x_pt = point.clone();
                            q4x = idx;
                        }
                    }
                    Ordering::Greater => (),
                }
            }

            // q4y
            if let Some(cmp) = q4y_pt[1].partial_cmp(&point[1]) {
                match cmp {
                    Ordering::Greater => {
                        q4y_pt = point.clone();
                        q4y = idx;
                    }
                    Ordering::Equal => {
                        if point[0] > q4x_pt[0] {
                            q4x_pt = point.clone();
                            q4x = idx;
                        }
                    }
                    Ordering::Less => (),
                }
            }
        }

        // Cover the special case of a collection having only one point
        if num_real_points == 1 {
            return vec![Index(q1x)];
        }

        // Step 2: Construct the convex hull in each quadrant. Filter all points which are not in the initial point set
        let mut partial_hull_q1: BTreeMap<OrderedFloat<f64>, usize> = BTreeMap::new();
        if q1x != usize::MAX {
            partial_hull_q1.insert(OrderedFloat(-q1x_pt[0]), q1x);
        }
        if q1y != usize::MAX {
            partial_hull_q1.insert(OrderedFloat(-q1y_pt[0]), q1y);
        }

        let mut partial_hull_q2: BTreeMap<OrderedFloat<f64>, usize> = BTreeMap::new();
        if q2x != usize::MAX {
            partial_hull_q2.insert(OrderedFloat(-q2x_pt[0]), q2x);
        }
        if q2y != usize::MAX {
            partial_hull_q2.insert(OrderedFloat(-q2y_pt[0]), q2y);
        }

        let mut partial_hull_q3: BTreeMap<OrderedFloat<f64>, usize> = BTreeMap::new();
        if q3x != usize::MAX {
            partial_hull_q3.insert(OrderedFloat(q3x_pt[0]), q3x);
        }
        if q3y != usize::MAX {
            partial_hull_q3.insert(OrderedFloat(q3y_pt[0]), q3y);
        }

        let mut partial_hull_q4: BTreeMap<OrderedFloat<f64>, usize> = BTreeMap::new();
        if q4x != usize::MAX {
            partial_hull_q4.insert(OrderedFloat(q4x_pt[0]), q4x);
        }
        if q4y != usize::MAX {
            partial_hull_q4.insert(OrderedFloat(q4y_pt[0]), q4y);
        }

        let mut partial_hulls = [
            partial_hull_q1,
            partial_hull_q2,
            partial_hull_q3,
            partial_hull_q4,
        ];

        let degenerate_quadrant = [
            partial_hulls[0].len() < 2,
            partial_hulls[1].len() < 2,
            partial_hulls[2].len() < 2,
            partial_hulls[3].len() < 2,
        ];

        let end_points = [q1x, q1y, q2x, q2y, q3x, q3y, q4x, q4y];

        fn loop_body<T: ConvexHull + ?Sized>(
            this: &T,
            partial_hull: &mut BTreeMap<OrderedFloat<f64>, usize>,
            quadrant: usize,
            is_degenerate: bool,
            end_points: [usize; 8],
            q1y_pt: [f64; 2],
            q2x_pt: [f64; 2],
            q3y_pt: [f64; 2],
            q4x_pt: [f64; 2],
        ) {
            // In q1 and q2, the search for new convex hull points starts with the largest x-value and stops with the smallest x-value of the quadrant.
            // In q3 and q4, the search starts with the smallest x-value and ends with the largest. To use the same code inside the loop,
            // the signs of the x-values in q1 and q2 are flipped.
            let orientation = 1.0 - (2.0 * (quadrant < 2) as i32 as f64);

            for (c, pt_c) in this.convex_hull_iter() {
                // Skip any non-real points
                // Inverting "is_finite" also catches NaN (is_infinite only catches infinite values, not NaN)
                if !pt_c[0].is_finite() || !pt_c[1].is_finite() {
                    continue;
                }

                match quadrant {
                    0 => {
                        // Skip test if c == a or c == b
                        if end_points.contains(&c) {
                            continue;
                        }

                        // Quadrant 1 -> 2
                        if q1y_pt[1] == pt_c[1] {
                            partial_hull.insert(OrderedFloat(pt_c[0] * orientation), c);
                            continue;
                        }
                    }
                    1 => {
                        // Skip test if c == a or c == b
                        if end_points.contains(&c) {
                            continue;
                        }

                        // Quadrant 2 -> 3
                        if q2x_pt[0] == pt_c[0] {
                            partial_hull.insert(
                                OrderedFloat((pt_c[0] + pt_c[1] - q2x_pt[1]) * orientation),
                                c,
                            );
                            continue;
                        }
                    }
                    2 => {
                        // Skip test if c == a or c == b
                        if end_points.contains(&c) {
                            continue;
                        }

                        // Quadrant 3 -> 4
                        if q3y_pt[1] == pt_c[1] {
                            partial_hull.insert(OrderedFloat(pt_c[0] * orientation), c);
                            continue;
                        }
                    }
                    3 => {
                        // Skip test if c == a or c == b
                        if end_points.contains(&c) {
                            continue;
                        }

                        // Quadrant 4 -> 1
                        if q4x_pt[0] == pt_c[0] {
                            partial_hull.insert(
                                OrderedFloat((pt_c[0] + pt_c[1] - q4x_pt[1]) * orientation),
                                c,
                            );
                            continue;
                        }
                    }
                    _ => unreachable!(),
                }

                /*
                Exclude all degenerate partial hulls. A partial hull is one which only has one entry.
                 */
                if is_degenerate {
                    continue;
                }

                let x = OrderedFloat(orientation * pt_c[0]);

                // Find the two points inside the current partial hull whose x-values form the closest bracket around the x-value of pt_c
                // If one of the range methods yields an empty iterator, pt_c is not inside the current quadrant and can therefore be skipped.
                let lower_clamp = match partial_hull.range((Unbounded, Excluded(x))).next() {
                    Some(val) => val,
                    None => continue,
                };
                let upper_clamp = match partial_hull.range((Excluded(x), Unbounded)).next() {
                    Some(val) => val,
                    None => continue,
                };

                let a = *lower_clamp.1;
                let b = *upper_clamp.1;

                /*
                SAFETY: Since any of the indices in the partial hull is a valid index for the input vector, this operation is never out of bounds.
                 */
                let mut pt_a = this.convex_hull_get(Index(a));
                let mut pt_b = this.convex_hull_get(Index(b));

                /*
                Calculate the cross product which tells us whether C is on the left of the line AB, directly on the line or right of it:
                If (cross_prod > 0) then C is to the left => C can be discarded
                If (cross_prod = 0) then C is on the line => C is part of the convex hull but does not invalidate any of the previous convex hull points
                If (cross_prod < 0) then C is to the right => C is part of the convex hull and possibly invalidates A and/or B as well as neighboring points of A and B

                The last step is done by recursively reading the left / right neighbor of A / B (called D) from here on. If A / B is located on the left of DC / CD,
                A / B is discarded and D is assigned as the next A / B. If A / B has no neighbors or if A / B is not located on the left of DC / CD, the main loop continues.
                 */
                let cross_prod = (pt_b[0] - pt_a[0]) * (pt_c[1] - pt_a[1])
                    - (pt_b[1] - pt_a[1]) * (pt_c[0] - pt_a[0]);

                if let Some(ordering) = cross_prod.partial_cmp(&0.0) {
                    match ordering {
                        Ordering::Less => {
                            // Check all neighbors on the left of A
                            loop {
                                let d = match partial_hull
                                    .range((
                                        Unbounded,
                                        Excluded(OrderedFloat(pt_a[0] * orientation)),
                                    ))
                                    .next()
                                {
                                    Some(val) => *val.1,
                                    None => break, // A / B has no neighbor in search direction
                                };
                                let pt_d = this.convex_hull_get(Index(d));

                                // Line DC with A
                                let cross_prod = (pt_c[0] - pt_d[0]) * (pt_a[1] - pt_d[1])
                                    - (pt_c[1] - pt_d[1]) * (pt_a[0] - pt_d[0]);

                                // If true, A / B is on the left of DC / CD and is therefore discarded.
                                if cross_prod > 0.0 {
                                    partial_hull.remove(&OrderedFloat(pt_a[0] * orientation));

                                    // Replace A with D
                                    pt_a = this.convex_hull_get(Index(d));
                                } else {
                                    break;
                                }
                            }

                            // Check all neighbors on the right of B
                            loop {
                                let d = match partial_hull
                                    .range((
                                        Excluded(OrderedFloat(pt_b[0] * orientation)),
                                        Unbounded,
                                    ))
                                    .next()
                                {
                                    Some(val) => *val.1,
                                    None => break, // A / B has no neighbor in search direction
                                };
                                let pt_d = this.convex_hull_get(Index(d));

                                // Line CD with B
                                let cross_prod = (pt_d[0] - pt_c[0]) * (pt_b[1] - pt_c[1])
                                    - (pt_d[1] - pt_c[1]) * (pt_b[0] - pt_c[0]);

                                // If true, A / B is on the left of DC / CD and is therefore discarded.
                                if cross_prod > 0.0 {
                                    partial_hull.remove(&OrderedFloat(pt_b[0] * orientation));

                                    // Replace B with D
                                    pt_b = this.convex_hull_get(Index(d));
                                } else {
                                    break;
                                }
                            }

                            // Add C to the partial hull
                            partial_hull.insert(OrderedFloat(pt_c[0] * orientation), c);
                        }
                        Ordering::Equal => {
                            partial_hull.insert(OrderedFloat(pt_c[0] * orientation), c);
                        }
                        Ordering::Greater => continue,
                    }
                }
            }
        }

        /*
        Loop for hull construction
         */
        #[cfg(not(feature = "rayon"))]
        {
            partial_hulls
                .iter_mut()
                .zip(degenerate_quadrant.into_iter())
                .enumerate()
                .for_each(|(quadrant, (partial_hull, is_degenerate))| {
                    loop_body(
                        self,
                        partial_hull,
                        quadrant,
                        is_degenerate,
                        end_points.clone(),
                        q1y_pt,
                        q2x_pt,
                        q3y_pt,
                        q4x_pt,
                    );
                });
        }
        #[cfg(feature = "rayon")]
        {
            partial_hulls
                .par_iter_mut()
                .zip(degenerate_quadrant.into_par_iter())
                .enumerate()
                .for_each(|(quadrant, (partial_hull, is_degenerate))| {
                    loop_body(
                        self,
                        partial_hull,
                        quadrant,
                        is_degenerate,
                        end_points.clone(),
                        q1y_pt,
                        q2x_pt,
                        q3y_pt,
                        q4x_pt,
                    );
                });
        }

        // Step 3: Combine the hulls
        let mut resulting_hull: Vec<Index> = Vec::new();

        for mut partial_hull in partial_hulls.into_iter() {
            // Check if the first value of the new partial hull equals the current last value of the assembled hull.
            // If so, it is discarded
            if let Some(last) = resulting_hull.last() {
                if let Some(first) = partial_hull.pop_first() {
                    if first.1 != usize::from(last.clone()) {
                        resulting_hull.push(Index(first.1));
                    }
                }
            }

            for (_, idx) in partial_hull.iter() {
                resulting_hull.push(Index(*idx));
            }
        }

        // If the last value of partial_hull equals the first one, discard it
        while resulting_hull.last() == resulting_hull.first() {
            if let None = resulting_hull.pop() {
                break;
            }
        }

        return resulting_hull;
    }
}

/**
An index known to be valid.

This index is generated within the [`ConvexHull::convex_hull`] method from the indices provided
by the [`ConvexHull::convex_hull_iter`] and is therefore known to be valid.
This allows to optimize the [`ConvexHull::convex_hull_get`] method
(e.g. the implementation for `Vec<P: Into<[f64;2]>`
uses [`get_unchecked`](https://doc.rust-lang.org/std/vec/struct.Vec.html#method.get_unchecked)).

There is no way to create this struct outside this crate in order to prevent the accidental
use of invalid indices in [`ConvexHull::convex_hull_get`]. However, `usize`
implements `From<Index> for usize` to make the underlying `usize` value accessible inside
custom implementations of the [`ConvexHull::convex_hull_get`] method.
*/
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Index(usize);

impl From<Index> for usize {
    fn from(value: Index) -> Self {
        return value.0;
    }
}

/**
Reinterprets a `Vec<Index>` as a `Vec<usize>`.

Since [`Index`] is a [newtype](https://doc.rust-lang.org/rust-by-example/generics/new_types.html) of `usize`,
it can be reinterpreted as a `Vec<usize>` without the need for allocations. This is useful if the output
indices of [`ConvexHull::convex_hull`] should be used for other purposes than just accessing the points
of the convex hull via [`ConvexHull::convex_hull_get`].

# Examples

```
use planar_convex_hull::{ConvexHull, reinterpret};

let vec: Vec<[f64; 2]> = vec![
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
];

// Returns a `Vec<Index>`. This vector can now be used to access the points via `convex_hull_get`:
let hull = vec.convex_hull();
let pts: Vec<[f64; 2]> = hull.iter().map(|i| vec.convex_hull_get(*i)).collect();
assert_eq!(pts, vec![[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]);

// Now we want to use the raw usize indices for something else
let hull_usize = reinterpret(hull);
assert_eq!(hull_usize, vec![3, 2, 0, 1]);
```
 */
pub fn reinterpret(index_vec: Vec<Index>) -> Vec<usize> {
    // Safety:
    // - Index is #[repr(transparent)] over usize
    // - Vec<Index> and Vec<usize> have the same layout
    // - Therefore, we can safely transmute the Vec
    let ptr = index_vec.as_ptr() as *mut usize;
    let len = index_vec.len();
    let cap = index_vec.capacity();

    // Prevent dropping the original Vec
    std::mem::forget(index_vec);

    // SAFETY: the above conditions are met
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

/**
Like [`reinterpret`], but reinterprets a slice instead of a vector.

# Examples

```
use planar_convex_hull::{ConvexHull, reinterpret_ref};

let vec: Vec<[f64; 2]> = vec![
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
];

// Returns a `Vec<Index>`. This vector can now be used to access the points via `convex_hull_get`:
let hull = vec.convex_hull();
let pts: Vec<[f64; 2]> = hull.iter().map(|i| vec.convex_hull_get(*i)).collect();
assert_eq!(pts, vec![[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]);

// Now we want to use the raw usize indices for something else
let hull_usize = reinterpret_ref(hull.as_slice());
assert_eq!(hull_usize, &[3, 2, 0, 1]);
```
 */
pub fn reinterpret_ref(index_slice: &[Index]) -> &[usize] {
    // SAFETY:
    // - Index is #[repr(transparent)] over usize, so they have the same memory layout
    // - A slice is a fat pointer (ptr + len), and we are only changing the type from Index to usize
    // - Thus, reinterpretation is safe as long as Index contains only a usize
    unsafe { std::slice::from_raw_parts(index_slice.as_ptr() as *const usize, index_slice.len()) }
}
