use std::{
    collections::{HashMap, HashSet},
    f64::{INFINITY, NAN, NEG_INFINITY},
};

use nalgebra::Point2;
use ordered_float::OrderedFloat;
use slab::Slab;

use planar_convex_hull::ConvexHull;

#[test]
fn test_zero_points() {
    let slice: &[[f64; 2]] = &[];
    assert_eq!(slice.convex_hull().collect::<Vec<_>>(), vec![]);
}

#[test]
fn test_one_point() {
    let slice = &[[-3.0, -1.0]];
    assert_eq!(
        slice.convex_hull().collect::<Vec<_>>(),
        vec![(0, [-3.0, -1.0])]
    );
}

#[test]
fn test_two_points() {
    {
        let slice = &[[-3.0, -1.0], [-2.0, 2.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((1, [-2.0, 2.0])));
        assert_eq!(hull.next(), Some((0, [-3.0, -1.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let slice = &[[-3.0, -1.0], [-3.0, 2.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((1, [-3.0, 2.0])));
        assert_eq!(hull.next(), Some((0, [-3.0, -1.0])));
        assert_eq!(hull.next(), None);
    }
}

#[test]
fn test_three_points() {
    {
        let slice = &[[-3.0, -1.0], [-2.0, 2.0], [5.0, -1.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((2, [5.0, -1.0])));
        assert_eq!(hull.next(), Some((1, [-2.0, 2.0])));
        assert_eq!(hull.next(), Some((0, [-3.0, -1.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let slice = &[[-3.0, -1.0], [-2.0, 2.0], [5.0, -2.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((2, [5.0, -2.0])));
        assert_eq!(hull.next(), Some((1, [-2.0, 2.0])));
        assert_eq!(hull.next(), Some((0, [-3.0, -1.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let slice = &[[-3.0, -1.0], [-2.0, 2.0], [-2.5, 0.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((1, [-2.0, 2.0])));
        assert_eq!(hull.next(), Some((0, [-3.0, -1.0])));
        assert_eq!(hull.next(), Some((2, [-2.5, 0.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let slice = &[[0.0, 1.0], [0.0, 2.0], [0.0, -1.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((1, [0.0, 2.0])));
        assert_eq!(hull.next(), Some((0, [0.0, 1.0])));
        assert_eq!(hull.next(), Some((2, [0.0, -1.0])));
        assert_eq!(hull.next(), Some((0, [0.0, 1.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let slice = &[[0.0, 1.0], [0.0, 2.0], [0.5, -1.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((2, [0.5, -1.0])));
        assert_eq!(hull.next(), Some((1, [0.0, 2.0])));
        assert_eq!(hull.next(), Some((0, [0.0, 1.0])));
        assert_eq!(hull.next(), None);
    }
}

#[test]
fn test_four_points() {
    {
        let slice = &[[1.0, 1.0], [0.0, 2.0], [-1.0, 3.0], [0.0, 0.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((0, [1.0, 1.0])));
        assert_eq!(hull.next(), Some((2, [-1.0, 3.0])));
        assert_eq!(hull.next(), Some((3, [0.0, 0.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let slice = &[[0.0, 1.0], [0.0, 2.0], [0.0, 0.0], [1.0, 1.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((3, [1.0, 1.0])));
        assert_eq!(hull.next(), Some((1, [0.0, 2.0])));
        assert_eq!(hull.next(), Some((0, [0.0, 1.0])));
        assert_eq!(hull.next(), Some((2, [0.0, 0.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let slice = &[[-3.0, -1.0], [0.0, 2.0], [0.0, 0.0], [5.0, -1.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((3, [5.0, -1.0])));
        assert_eq!(hull.next(), Some((1, [0.0, 2.0])));
        assert_eq!(hull.next(), Some((0, [-3.0, -1.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let slice = &[[1.0, 0.0], [0.0, 1.0], [4.0, 3.0], [3.0, 4.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((2, [4.0, 3.0])));
        assert_eq!(hull.next(), Some((3, [3.0, 4.0])));
        assert_eq!(hull.next(), Some((1, [0.0, 1.0])));
        assert_eq!(hull.next(), Some((0, [1.0, 0.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let slice = &[[0.0, 0.0], [1.0, 1.0], [0.0, 2.0], [-1.0, 1.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((1, [1.0, 1.0])));
        assert_eq!(hull.next(), Some((2, [0.0, 2.0])));
        assert_eq!(hull.next(), Some((3, [-1.0, 1.0])));
        assert_eq!(hull.next(), Some((0, [0.0, 0.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let slice = &[[0.0, -4.0], [1.0, -3.0], [0.0, -3.0], [-1.0, -3.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((1, [1.0, -3.0])));
        assert_eq!(hull.next(), Some((2, [0.0, -3.0])));
        assert_eq!(hull.next(), Some((3, [-1.0, -3.0])));
        assert_eq!(hull.next(), Some((0, [0.0, -4.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let slice = &[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 1.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((1, [1.0, 1.0])));
        assert_eq!(hull.next(), Some((2, [0.0, 1.0])));
        assert_eq!(hull.next(), Some((3, [-1.0, 1.0])));
        assert_eq!(hull.next(), Some((0, [0.0, 0.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let slice = &[[0.0, 0.0], [-1.0, 3.0], [-4.0, 2.0], [-5.0, 4.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((0, [0.0, 0.0])));
        assert_eq!(hull.next(), Some((1, [-1.0, 3.0])));
        assert_eq!(hull.next(), Some((3, [-5.0, 4.0])));
        assert_eq!(hull.next(), Some((2, [-4.0, 2.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let slice = &[[0.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, -2.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((1, [0.0, 1.0])));
        assert_eq!(hull.next(), Some((0, [0.0, 0.0])));
        assert_eq!(hull.next(), Some((2, [0.0, -1.0])));
        assert_eq!(hull.next(), Some((3, [0.0, -2.0])));
        assert_eq!(hull.next(), Some((2, [0.0, -1.0])));
        assert_eq!(hull.next(), Some((0, [0.0, 0.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let slice = &[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((3, [1.0, 1.0])));
        assert_eq!(hull.next(), Some((2, [0.0, 1.0])));
        assert_eq!(hull.next(), Some((0, [0.0, 0.0])));
        assert_eq!(hull.next(), Some((1, [1.0, 0.0])));
        assert_eq!(hull.next(), None);
    }
}

#[test]
fn test_five_points() {
    {
        let slice = &[[1.0, 0.0], [0.0, 1.0], [4.0, 3.0], [3.0, 4.0], [-1.0, 5.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((2, [4.0, 3.0])));
        assert_eq!(hull.next(), Some((3, [3.0, 4.0])));
        assert_eq!(hull.next(), Some((4, [-1.0, 5.0])));
        assert_eq!(hull.next(), Some((1, [0.0, 1.0])));
        assert_eq!(hull.next(), Some((0, [1.0, 0.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let slice = &[[1.0, 0.0], [0.0, 1.0], [4.0, 3.0], [3.0, 4.0], [2.0, 2.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((2, [4.0, 3.0])));
        assert_eq!(hull.next(), Some((3, [3.0, 4.0])));
        assert_eq!(hull.next(), Some((1, [0.0, 1.0])));
        assert_eq!(hull.next(), Some((0, [1.0, 0.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let slice = &[[1.0, 0.0], [0.0, 1.0], [4.0, 3.0], [3.0, 4.0], [-1.0, 1.0]];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((2, [4.0, 3.0])));
        assert_eq!(hull.next(), Some((3, [3.0, 4.0])));
        assert_eq!(hull.next(), Some((4, [-1.0, 1.0])));
        assert_eq!(hull.next(), Some((0, [1.0, 0.0])));
        assert_eq!(hull.next(), None);
    }
}

#[test]
fn test_eight_points() {
    {
        let slice = &[
            [-3.0, -1.0],
            [-2.0, 2.0],
            [0.0, 0.0],
            [1.0, 3.0],
            [5.0, -1.0],
            [6.0, 2.0],
            [7.0, -4.0],
            [8.0, -1.0],
        ];

        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((7, [8.0, -1.0])));
        assert_eq!(hull.next(), Some((5, [6.0, 2.0])));
        assert_eq!(hull.next(), Some((3, [1.0, 3.0])));
        assert_eq!(hull.next(), Some((1, [-2.0, 2.0])));
        assert_eq!(hull.next(), Some((0, [-3.0, -1.0])));
        assert_eq!(hull.next(), Some((6, [7.0, -4.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let slice = &[
            [6.0, 2.0],
            [7.0, -4.0],
            [8.0, -1.0],
            [-3.0, -1.0],
            [-2.0, 2.0],
            [0.0, 0.0],
            [1.0, 3.0],
            [5.0, -1.0],
        ];
        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((2, [8.0, -1.0])));
        assert_eq!(hull.next(), Some((0, [6.0, 2.0])));
        assert_eq!(hull.next(), Some((6, [1.0, 3.0])));
        assert_eq!(hull.next(), Some((4, [-2.0, 2.0])));
        assert_eq!(hull.next(), Some((3, [-3.0, -1.0])));
        assert_eq!(hull.next(), Some((1, [7.0, -4.0])));
        assert_eq!(hull.next(), None);
    }
}

#[test]
fn test_twelve_points() {
    {
        let slice = &[
            [0.5, 0.5],
            [0.5, -0.5],
            [-0.5, -0.5],
            [-0.5, 0.5],
            [-1.5, 0.5],
            [-1.5, -0.5],
            [0.5, 1.5],
            [-0.5, 1.5],
            [1.5, 0.5],
            [1.5, -0.5],
            [0.5, -1.5],
            [-0.5, -1.5],
        ];

        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((8, [1.5, 0.5])));
        assert_eq!(hull.next(), Some((6, [0.5, 1.5])));
        assert_eq!(hull.next(), Some((7, [-0.5, 1.5])));
        assert_eq!(hull.next(), Some((4, [-1.5, 0.5])));
        assert_eq!(hull.next(), Some((5, [-1.5, -0.5])));
        assert_eq!(hull.next(), Some((11, [-0.5, -1.5])));
        assert_eq!(hull.next(), Some((10, [0.5, -1.5])));
        assert_eq!(hull.next(), Some((9, [1.5, -0.5])));
        assert_eq!(hull.next(), None);
    }
}

#[test]
fn test_newtype() {
    #[derive(Clone)]
    struct PointWrapper([f64; 2]);
    impl From<PointWrapper> for [f64; 2] {
        fn from(value: PointWrapper) -> Self {
            return value.0;
        }
    }

    let vec = vec![
        PointWrapper([0.0, 0.0]),
        PointWrapper([1.0, 0.0]),
        PointWrapper([0.0, 1.0]),
        PointWrapper([1.0, 1.0]),
    ];

    let mut hull = vec.convex_hull();
    assert_eq!(hull.next(), Some((3, [1.0, 1.0])));
    assert_eq!(hull.next(), Some((2, [0.0, 1.0])));
    assert_eq!(hull.next(), Some((0, [0.0, 0.0])));
    assert_eq!(hull.next(), Some((1, [1.0, 0.0])));
    assert_eq!(hull.next(), None);
}

#[test]
fn test_vec() {
    {
        let vec: Vec<Point2<f64>> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
        ];
        let mut hull = vec.convex_hull();
        assert_eq!(hull.next(), Some((3, [1.0, 1.0])));
        assert_eq!(hull.next(), Some((2, [0.0, 1.0])));
        assert_eq!(hull.next(), Some((0, [0.0, 0.0])));
        assert_eq!(hull.next(), Some((1, [1.0, 0.0])));
        assert_eq!(hull.next(), None);
    }
}

#[test]
fn test_slab() {
    {
        let mut slab: Slab<Point2<f64>> = Slab::new();
        slab.insert(Point2::new(0.0, 0.0));
        slab.insert(Point2::new(1.0, 0.0));
        slab.insert(Point2::new(0.0, 1.0));
        slab.insert(Point2::new(1.0, 1.0));

        let mut hull = slab.convex_hull();
        assert_eq!(hull.next(), Some((3, [1.0, 1.0])));
        assert_eq!(hull.next(), Some((2, [0.0, 1.0])));
        assert_eq!(hull.next(), Some((0, [0.0, 0.0])));
        assert_eq!(hull.next(), Some((1, [1.0, 0.0])));
        assert_eq!(hull.next(), None);
    }
}

#[test]
fn test_hashmap() {
    {
        let mut hashmap = HashMap::new();
        hashmap.insert(10, [0.0, 0.0]);

        let mut hull = hashmap.convex_hull();
        assert_eq!(hull.next(), Some((10, [0.0, 0.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let mut hashmap = HashMap::new();
        hashmap.insert(10, [0.0, 0.0]);
        hashmap.insert(5, [0.0, 1.0]);

        let mut hull = hashmap.convex_hull();
        assert_eq!(hull.next(), Some((5, [0.0, 1.0])));
        assert_eq!(hull.next(), Some((10, [0.0, 0.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let mut hashmap = HashMap::new();
        hashmap.insert(0, [0.0, 0.0]);
        hashmap.insert(1, [1.0, 0.0]);
        hashmap.insert(2, [0.0, 1.0]);
        hashmap.insert(3, [1.0, 1.0]);

        let mut hull = hashmap.convex_hull();
        assert_eq!(hull.next(), Some((3, [1.0, 1.0])));
        assert_eq!(hull.next(), Some((2, [0.0, 1.0])));
        assert_eq!(hull.next(), Some((0, [0.0, 0.0])));
        assert_eq!(hull.next(), Some((1, [1.0, 0.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let mut hashmap = ahash::AHashMap::new();
        hashmap.insert(0, [0.0, 0.0]);
        hashmap.insert(1, [1.0, 0.0]);
        hashmap.insert(2, [0.0, 1.0]);
        hashmap.insert(3, [1.0, 1.0]);

        let mut hull = hashmap.convex_hull();
        assert_eq!(hull.next(), Some((3, [1.0, 1.0])));
        assert_eq!(hull.next(), Some((2, [0.0, 1.0])));
        assert_eq!(hull.next(), Some((0, [0.0, 0.0])));
        assert_eq!(hull.next(), Some((1, [1.0, 0.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let mut hashmap = HashMap::new();
        hashmap.insert(1, [0.0, 0.0]);
        hashmap.insert(2, [1.0, 0.0]);
        hashmap.insert(3, [0.0, 1.0]);
        hashmap.insert(4, [1.0, 1.0]);

        let mut hull = hashmap.convex_hull();
        assert_eq!(hull.next(), Some((4, [1.0, 1.0])));
        assert_eq!(hull.next(), Some((3, [0.0, 1.0])));
        assert_eq!(hull.next(), Some((1, [0.0, 0.0])));
        assert_eq!(hull.next(), Some((2, [1.0, 0.0])));
        assert_eq!(hull.next(), None);
    }
    {
        let mut hashmap = HashMap::new();
        hashmap.insert(10, [0.0, 0.0]);
        hashmap.insert(20, [1.0, 0.0]);
        hashmap.insert(30, [0.0, 1.0]);
        hashmap.insert(40, [1.0, 1.0]);

        let mut hull = hashmap.convex_hull();
        assert_eq!(hull.next(), Some((40, [1.0, 1.0])));
        assert_eq!(hull.next(), Some((30, [0.0, 1.0])));
        assert_eq!(hull.next(), Some((10, [0.0, 0.0])));
        assert_eq!(hull.next(), Some((20, [1.0, 0.0])));
        assert_eq!(hull.next(), None);
    }
}

#[test]
fn test_hashset() {
    {
        let slice = &[
            [-3.0, -1.0],
            [-2.0, 2.0],
            [0.0, 0.0],
            [1.0, 3.0],
            [5.0, -1.0],
            [6.0, 2.0],
            [7.0, -4.0],
            [8.0, -1.0],
        ];

        #[derive(Clone, Hash, PartialEq, Eq)]
        struct MyPoint([OrderedFloat<f64>; 2]);

        impl From<MyPoint> for [f64; 2] {
            fn from(value: MyPoint) -> Self {
                return [value.0[0].into_inner(), value.0[1].into_inner()];
            }
        }

        let hashset: HashSet<MyPoint> = HashSet::from_iter(
            slice
                .iter()
                .map(|[x, y]| MyPoint([OrderedFloat(x.clone()), OrderedFloat(y.clone())])),
        );

        let mut hull = hashset.convex_hull();
        assert_eq!(hull.next().map(|(_, p)| p), Some([8.0, -1.0]));
        assert_eq!(hull.next().map(|(_, p)| p), Some([6.0, 2.0]));
        assert_eq!(hull.next().map(|(_, p)| p), Some([1.0, 3.0]));
        assert_eq!(hull.next().map(|(_, p)| p), Some([-2.0, 2.0]));
        assert_eq!(hull.next().map(|(_, p)| p), Some([-3.0, -1.0]));
        assert_eq!(hull.next().map(|(_, p)| p), Some([7.0, -4.0]));
        assert_eq!(hull.next(), None);
    }
}

#[test]
fn test_nonreal_points() {
    let last_points = [
        [INFINITY, -1.0],
        [0.0, INFINITY],
        [NAN, -1.0],
        [0.0, NAN],
        [INFINITY, NAN],
        [NAN, INFINITY],
        [INFINITY, NEG_INFINITY],
        [NAN, NEG_INFINITY],
    ];

    for last_point in last_points {
        let slice = &[[-3.0, -1.0], [-2.0, 2.0], [5.0, -1.0], last_point];

        let mut hull = slice.convex_hull();
        assert_eq!(hull.next(), Some((2, [5.0, -1.0])));
        assert_eq!(hull.next(), Some((1, [-2.0, 2.0])));
        assert_eq!(hull.next(), Some((0, [-3.0, -1.0])));
        assert_eq!(hull.next(), None);
    }
}

#[test]
fn test_issue_1() {
    let points = [
        [14.0, 27.0],
        [12.0, 21.0],
        [18.0, 39.0],
        [16.0, 33.0],
        [8.0, 13.0],
        [6.0, 7.0],
        [12.0, 25.0],
        [10.0, 19.0],
        [2.0, -1.0],
        [0.0, -7.0],
        [6.0, 11.0],
        [4.0, 5.0],
        [-4.0, -15.0],
        [-6.0, -21.0],
        [0.0, -3.0],
        [-2.0, -9.0],
        [4.0, 13.0],
        [2.0, 7.0],
        [8.0, 25.0],
        [6.0, 19.0],
        [-2.0, -1.0],
        [-4.0, -7.0],
        [2.0, 11.0],
        [0.0, 5.0],
        [-8.0, -15.0],
        [-10.0, -21.0],
        [-4.0, -3.0],
        [-6.0, -9.0],
        [-14.0, -29.0],
        [-16.0, -35.0],
        [-10.0, -17.0],
        [-12.0, -23.0],
    ];

    let mut hull = points.convex_hull();
    assert_eq!(hull.next(), Some((2, [18.0, 39.0])));
    assert_eq!(hull.next(), Some((18, [8.0, 25.0])));
    assert_eq!(hull.next(), Some((30, [-10.0, -17.0])));
    assert_eq!(hull.next(), Some((29, [-16.0, -35.0])));
    assert_eq!(hull.next(), Some((13, [-6.0, -21.0])));
    assert_eq!(hull.next(), Some((1, [12.0, 21.0])));
    assert_eq!(hull.next(), None);
}
