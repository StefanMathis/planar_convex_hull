/*!
[`ConvexHull`]: crate::ConvexHull
[`convex_hull`]: crate::ConvexHull::convex_hull

A lightweight library providing a trait for implementing a divide-and-conquer
planar convex hull algorithm for your own datatype.

 */
#![doc = include_str!("../docs/main.md")]
#![deny(missing_docs)]

use ordered_float::OrderedFloat;
use std::cmp::Ordering;
use std::collections::{BTreeMap, btree_map::IntoIter};
use std::f64::INFINITY;
use std::f64::NEG_INFINITY;
use std::ops::Bound::Excluded;
use std::ops::Bound::Unbounded;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub mod convex_hull_impl;

/**
A trait for implementing a planar convex hull algorithm for a collection type.

This trait is meant to be implemented on a collection (e.g. a vector, slice,
hashmap, ...) which stores instances of a type representing a 2-dimensional
point in cartesian coordinates. The type needs to implement `Clone` and
`Into<[f64; 2]>`; the first array element is treated as the x-coordinate and the
second element is treated as the y-coordinate.

Implementing the trait requires the [`ConvexHull::convex_hull_iter`] method,
which returns an iterator over the collection's keys and the associated points.
In return, the trait provides the [`ConvexHull::convex_hull`] method which
creates a [`ConvexHullIter`] iterator over the convex hull points and keys.

See the docstring of [`ConvexHull::convex_hull_iter`] and the
[README / module documentation](crate) for implementation examples.

# Examples

```
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
 */
pub trait ConvexHull: std::marker::Sync {
    /**
    Iterates over all keys of a collection and the associated points in any
    order.

    The following examples show how this function is implemented for different
    collection types (see source code of the [`convex_hull_impl`] module).

    # Vector

    The "keys" of a [`Vec`] are the element indices.

    ```ignore
    impl<P: Into<[f64; 2]> + std::marker::Sync + Clone> ConvexHull for Vec<P> {
        fn convex_hull_iter(&self) -> impl Iterator<Item = (usize, [f64; 2])> {
            return self.iter().cloned().map(Into::into).enumerate();
        }
    }
    ```

    # HashMap

    The "keys" of a [`HashMap`](std::collections::HashMap) are the keys of the
    map.

    ```ignore
    impl<S: BuildHasher + std::marker::Sync, P: Into<[f64; 2]> + std::marker::Sync + Clone> ConvexHull
        for HashMap<usize, P, S>
    {
        fn convex_hull_iter(&self) -> impl Iterator<Item = (usize, [f64; 2])> {
            return self
                .iter()
                .map(|(key, val)| (key.clone(), val.clone().into()));
        }
    }
    ```

    # HashSet

    The "keys" of a [`HashSet`](std::collections::HashSet) is the enumeration of
    the set elements. This implementation relies on the fact that the iteration
    order of a [`HashSet`](std::collections::HashSet) is stable and
    deterministic.

    ```ignore
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
    ```
    */
    fn convex_hull_iter(&self) -> impl Iterator<Item = (usize, [f64; 2])>;

    // ==================================================================================

    /**
    Calculates the convex hull for `self` and returns a [`ConvexHullIter`] which
    can be used to iterate through the convex hull points and their
    corresponding keys.

    This function calculates the convex hull of a set of points using the
    divide-and-conquer algorithm presented in \[1, 2\]. If the input contains
    duplicate points which are part of the convex hull, one of the points is
    selected arbitrarily. Nonreal points (points containing NaN or infinite
    values) are ignored.

    The produced [`ConvexHullIter`] returns `(key, point)` pairs representing
    the corners of the convex hull in counter-clockwise (mathematically
    positive) order. Connecting the points with straight lines forms the convex
    hull.

    In addition to the original algorithm descriptions, this implementation also
    covers edge cases such as all points being located on a single line or
    multiple hull points having the same x- or y-coordinate (see examples
    below).

    When the `rayon` feature is enabled, the divide-and-conquer part of the
    algorithm is parallelized.

    # Literature

    1. Liu, Gh., Chen, Cb: A new algorithm for computing the convex hull of a planar point set.
    J. Zhejiang Univ. - Sci. A 8, 1210–1217 (2007). [https://doi.org/10.1631/jzus.2007.A1210](https://doi.org/10.1631/jzus.2007.A1210)
    2. Saad, Omar: A Convex Hull Algorithm and its implementation in O(n log h) (2017).
    See docs/convex_hull_algorithm.html.

    # Examples
    ```
    use planar_convex_hull::ConvexHull;

    // Rhombus with two points in its middle
    let slice = &[
        [10.0, 4.0],
        [-10.0, 4.0],
        [0.0, 6.0],
        [0.0, 2.0],
        [4.0, 4.0], // Not part of the convex hull
        [-4.0, 4.0], // Not part of the convex hull
    ];

    // The convex hull is the rhombus formed by the points 0, 1, 2 and 3. The
    // points 4 and 5 are not part of the convex hull, because they are located
    // on the line between points 0 and 1.
    let mut hull = slice.convex_hull();
    assert_eq!(hull.next(), Some((0, [10.0, 4.0])));
    assert_eq!(hull.next(), Some((2, [0.0, 6.0])));
    assert_eq!(hull.next(), Some((1, [-10.0, 4.0])));
    assert_eq!(hull.next(), Some((3, [0.0, 2.0])));
    assert_eq!(hull.next(), None);

    // All points on a single line with the same y-value everywhere. The points
    // 2 and 3 in the middle of the line show up twice, because the convex hull
    // goes "up and down" the x-axis
    let slice = &[
        [10.0, -2.0],
        [-10.0, -2.0],
        [0.0, -2.0],
        [3.0, -2.0],
    ];
    let mut hull = slice.convex_hull();
    assert_eq!(hull.next(), Some((0, [10.0, -2.0])));
    assert_eq!(hull.next(), Some((3, [3.0, -2.0])));
    assert_eq!(hull.next(), Some((2, [0.0, -2.0])));
    assert_eq!(hull.next(), Some((1, [-10.0, -2.0])));
    assert_eq!(hull.next(), Some((2, [0.0, -2.0])));
    assert_eq!(hull.next(), Some((3, [3.0, -2.0])));
    assert_eq!(hull.next(), None);

    // Triangle with a collinear point on the hull edge
    let slice = &[
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.5, 0.5], // Collinear point on hull edge
    ];
    let mut hull = slice.convex_hull();
    assert_eq!(hull.next(), Some((0, [1.0, 0.0])));
    assert_eq!(hull.next(), Some((1, [0.0, 1.0])));
    assert_eq!(hull.next(), Some((2, [0.0, 0.0])));
    assert_eq!(hull.next(), None);

    // Triangle with a point in the middle
    let slice = &[
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.25, 0.25], // Not part of the convex hull
    ];
    let mut hull = slice.convex_hull();
    assert_eq!(hull.next(), Some((0, [1.0, 0.0])));
    assert_eq!(hull.next(), Some((1, [0.0, 1.0])));
    assert_eq!(hull.next(), Some((2, [0.0, 0.0])));
    assert_eq!(hull.next(), None);
    ```
     */
    fn convex_hull(&self) -> ConvexHullIter {
        /*
        Step 1: Identify the four point-pairs defining each quadrant.

        We search for the points containing one extremum x- or y-value. The
        quadrant borders are defined by the other value of the point. For
        example, if the point with the largest x-value is [2, 1] and that with
        the largest y-value is [1, 3], all points where x >= 1 and y >= 1 belong
        to the q1 quadrant. Similarily, the q2 quadrant is defined by x <= 1 and
        y >= 1, the q3 quadrant by x <= 1 and y <= 3, and the q4 quadrant by
        x >= 1 and y <= 3.
         */
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

        // This variable is used to catch the special case of a collection
        // having only one real point.
        let mut num_real_points = 0;

        for (idx, point) in self.convex_hull_iter() {
            // Skip any non-real points
            if !point[0].is_finite() || !point[1].is_finite() {
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
                        if point[0] > q4y_pt[0] {
                            q4y_pt = point.clone();
                            q4y = idx;
                        }
                    }
                    Ordering::Less => (),
                }
            }
        }

        // Cover the special case of a collection having only one point
        if num_real_points == 1 {
            let mut q1 = BTreeMap::new();
            q1.insert(OrderedFloat(0.0), (q1x, q1x_pt));

            let q2 = BTreeMap::new();
            let q3 = BTreeMap::new();
            let q4 = BTreeMap::new();
            return ConvexHullIter::new([q1, q2, q3, q4]);
        }

        // Step 2: Insert the found extremum points into the quadrant hulls. If
        // a quadrant has less than two points, it is considered degenerate and
        // will be ignored in the next step.
        //
        // The hulls are represented by BTreeMaps, where the key is the x-value
        // of the point and the value is a tuple containing the index and the
        // point itself. The BTreeMap is used to keep the points sorted by their
        // x-values, which is necessary for the next step of the algorithm.
        let mut partial_hull_q1: BTreeMap<OrderedFloat<f64>, (usize, [f64; 2])> = BTreeMap::new();
        if q1x != usize::MAX {
            partial_hull_q1.insert(OrderedFloat(-q1x_pt[0]), (q1x, q1x_pt));
        }
        if q1y != usize::MAX {
            partial_hull_q1.insert(OrderedFloat(-q1y_pt[0]), (q1y, q1y_pt));
        }

        let mut partial_hull_q2: BTreeMap<OrderedFloat<f64>, (usize, [f64; 2])> = BTreeMap::new();
        if q2x != usize::MAX {
            partial_hull_q2.insert(OrderedFloat(-q2x_pt[0]), (q2x, q2x_pt));
        }
        if q2y != usize::MAX {
            partial_hull_q2.insert(OrderedFloat(-q2y_pt[0]), (q2y, q2y_pt));
        }

        let mut partial_hull_q3: BTreeMap<OrderedFloat<f64>, (usize, [f64; 2])> = BTreeMap::new();
        if q3x != usize::MAX {
            partial_hull_q3.insert(OrderedFloat(q3x_pt[0]), (q3x, q3x_pt));
        }
        if q3y != usize::MAX {
            partial_hull_q3.insert(OrderedFloat(q3y_pt[0]), (q3y, q3y_pt));
        }

        let mut partial_hull_q4: BTreeMap<OrderedFloat<f64>, (usize, [f64; 2])> = BTreeMap::new();
        if q4x != usize::MAX {
            partial_hull_q4.insert(OrderedFloat(q4x_pt[0]), (q4x, q4x_pt));
        }
        if q4y != usize::MAX {
            partial_hull_q4.insert(OrderedFloat(q4y_pt[0]), (q4y, q4y_pt));
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
            partial_hull: &mut BTreeMap<OrderedFloat<f64>, (usize, [f64; 2])>,
            quadrant: usize,
            is_degenerate: bool,
            end_points: [usize; 8],
            q1y_pt: [f64; 2],
            q2x_pt: [f64; 2],
            q3y_pt: [f64; 2],
            q4x_pt: [f64; 2],
        ) {
            // In q1 and q2, the search for new convex hull points starts with
            // the largest x-value and stops with the smallest x-value of the
            // quadrant (counter-clockwise search along the point set). In q3
            // and q4, the search starts with the smallest x-value and ends with
            // the largest. To use the same code inside the loop, the signs of
            // the x-values in q1 and q2 are flipped. The orientation variable
            // is used to flip the signs of the x-values in q1 and q2.
            let orientation = 1.0 - (2.0 * (quadrant < 2) as i32 as f64);

            for (c, pt_c) in this.convex_hull_iter() {
                // Skip any non-real points. Inverting "is_finite" also catches
                // NaN (is_infinite only catches infinite values, not NaN).
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
                            partial_hull.insert(OrderedFloat(pt_c[0] * orientation), (c, pt_c));
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
                                (c, pt_c),
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
                            partial_hull.insert(OrderedFloat(pt_c[0] * orientation), (c, pt_c));
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
                                (c, pt_c),
                            );
                            continue;
                        }
                    }
                    _ => unreachable!(),
                }

                // Exclude all degenerate partial hulls. A partial hull is one
                // which only has one entry.
                if is_degenerate {
                    continue;
                }

                let x = OrderedFloat(orientation * pt_c[0]);

                /*
                Find the two points inside the current partial hull whose
                x-values form the closest bracket around the x-value of pt_c,
                using the fact that the BTreeMap is sorted by x-value. The two
                points are called A and B. If C is located to the right of the
                line AB, C is part of the convex hull and possibly invalidates
                A and/or B as well as neighboring points of A and B. If C is
                located to the left of the line AB or directly on it, C is not
                part of the convex hull and can be discarded. The cross product
                of the vectors AB and AC is used to determine the relative
                position of C to the line AB. The cross product is positive if C
                is to the left of AB, negative if C is to the right, and zero if
                C is on the line AB. The cross product is calculated as follows:
                cross_prod_abc = (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x)
                 */
                let mut pt_a = match partial_hull.range((Unbounded, Excluded(x))).last() {
                    Some(lower_clamp) => lower_clamp.1.1,
                    None => continue,
                };
                let mut pt_b = match partial_hull.range((Excluded(x), Unbounded)).next() {
                    Some(upper_clamp) => upper_clamp.1.1,
                    None => continue,
                };

                /*
                Calculate the cross product which tells us whether C is on the
                left of the line AB, directly on the line or right of it:
                If (cross_prod > 0) then C is to the left => C can be discarded.
                If (cross_prod = 0) then C is on the line => C is collinear
                to A and B and can be discarded.
                If (cross_prod < 0) then C is to the right => C is part of the
                convex hull and possibly invalidates A and/or B as well as
                neighboring points of A and B.

                The last step is done by repeatedly reading the left / right
                neighbor of A / B (called D) from here on. If A / B is located
                on the left of DC / CD, A / B is discarded and D is assigned as
                the next A / B. If A / B has no neighbors or if A / B is not
                located on the left of DC / CD, the main loop continues.
                 */
                let cross_prod_abc = (pt_b[0] - pt_a[0]) * (pt_c[1] - pt_a[1])
                    - (pt_b[1] - pt_a[1]) * (pt_c[0] - pt_a[0]);

                if let Some(ordering) = cross_prod_abc.partial_cmp(&0.0) {
                    match ordering {
                        Ordering::Less => {
                            // Check all neighbors on the left of A: [-INF, A)
                            loop {
                                let (_, pt_d) = match partial_hull
                                    .range((
                                        Unbounded,
                                        Excluded(OrderedFloat(pt_a[0] * orientation)),
                                    ))
                                    .last()
                                {
                                    Some(val) => *val.1,
                                    None => break, // A has no neighbor in search direction
                                };

                                // Line DC with A
                                let cross_prod = (pt_c[0] - pt_d[0]) * (pt_a[1] - pt_d[1])
                                    - (pt_c[1] - pt_d[1]) * (pt_a[0] - pt_d[0]);

                                // If true, A is on the left of DC and is therefore discarded.
                                if cross_prod >= 0.0 {
                                    partial_hull.remove(&OrderedFloat(pt_a[0] * orientation));

                                    // Replace A with D.
                                    pt_a = pt_d;
                                } else {
                                    break;
                                }
                            }

                            // Check all neighbors on the right of B
                            loop {
                                let (_, pt_d) = match partial_hull
                                    .range((
                                        Excluded(OrderedFloat(pt_b[0] * orientation)),
                                        Unbounded,
                                    ))
                                    .next()
                                {
                                    Some(val) => *val.1,
                                    None => break, // B has no neighbor in search direction
                                };

                                // Line CD with B
                                let cross_prod = (pt_d[0] - pt_c[0]) * (pt_b[1] - pt_c[1])
                                    - (pt_d[1] - pt_c[1]) * (pt_b[0] - pt_c[0]);

                                // If true, B is on the left of CD and is therefore discarded.
                                if cross_prod >= 0.0 {
                                    partial_hull.remove(&OrderedFloat(pt_b[0] * orientation));

                                    // Replace B with D
                                    pt_b = pt_d;
                                } else {
                                    break;
                                }
                            }

                            // Add C to the partial hull
                            partial_hull.insert(OrderedFloat(pt_c[0] * orientation), (c, pt_c));
                        }
                        _ => continue,
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

        // Step 3: Combine the hulls inside an iterator which goes over the four
        // quadrants in counter-clockwise order. The iterator also filters out
        // duplicate points at the boundaries of the quadrants.
        return ConvexHullIter::new(partial_hulls);
    }
}

/**
An owning iterator over the convex hull points and their corresponding keys.

This `struct` is created by [`ConvexHull::convex_hull`]. See its documentation
for more.
 */
#[derive(Debug)]
pub struct ConvexHullIter {
    quadrant_iterators: [IntoIter<OrderedFloat<f64>, (usize, [f64; 2])>; 4],
    hull_idx: usize,
    first_returned_idx: Option<usize>,
    last_returned_idx: Option<usize>,
}

impl ConvexHullIter {
    fn new(quadrants: [BTreeMap<OrderedFloat<f64>, (usize, [f64; 2])>; 4]) -> Self {
        let quadrant_iterators = quadrants.map(|q| q.into_iter());

        return Self {
            quadrant_iterators,
            hull_idx: 0,
            first_returned_idx: None,
            last_returned_idx: None,
        };
    }
}

impl Iterator for ConvexHullIter {
    type Item = (usize, [f64; 2]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.hull_idx >= 4 {
            return None;
        }

        let current_hull = &mut self.quadrant_iterators[self.hull_idx];
        match current_hull.next() {
            Some(item) => {
                // This check prevents that points are returned twice at the
                // boundary of two hull iterators.
                let idx = Some(item.1.0);
                if idx == self.last_returned_idx || idx == self.first_returned_idx {
                    return self.next();
                }
                self.last_returned_idx = idx;

                if self.first_returned_idx.is_none() {
                    self.first_returned_idx = idx;
                }

                return Some(item.1);
            }
            None => {
                self.hull_idx += 1;
                return self.next();
            }
        }
    }
}
