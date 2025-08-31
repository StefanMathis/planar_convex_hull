use std::{
    collections::HashMap,
    f64::{INFINITY, NAN, NEG_INFINITY},
};

use nalgebra::Point2;
use slab::Slab;

use planar_convex_hull::{ConvexHull, reinterpret};

#[test]
fn test_zero_points() {
    let slice: &[[f64; 2]] = &[];
    let hull = reinterpret(slice.convex_hull());
    assert_eq!(hull, vec![]);
}

#[test]
fn test_one_point() {
    let slice = &[[-3.0, -1.0]];
    let hull = reinterpret(slice.convex_hull());
    assert_eq!(hull, vec![0]);
}

#[test]
fn test_two_points() {
    {
        let slice = &[[-3.0, -1.0], [-2.0, 2.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![1, 0]);
    }
    {
        let slice = &[[-3.0, -1.0], [-3.0, 2.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![1, 0]);
    }
}

#[test]
fn test_three_points() {
    {
        let slice = &[[-3.0, -1.0], [-2.0, 2.0], [5.0, -1.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![2, 1, 0]);
    }
    {
        let slice = &[[-3.0, -1.0], [-2.0, 2.0], [5.0, -2.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![2, 1, 0]);
    }
    {
        let slice = &[[-3.0, -1.0], [-2.0, 2.0], [-2.5, 0.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![1, 0, 2]);
    }
    {
        let slice = &[[0.0, 1.0], [0.0, 2.0], [0.0, -1.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![1, 0, 2, 0]);
    }
    {
        let slice = &[[0.0, 1.0], [0.0, 2.0], [0.5, -1.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![2, 1, 0]);
    }
}

#[test]
fn test_four_points() {
    {
        let slice = &[[1.0, 1.0], [0.0, 2.0], [-1.0, 3.0], [0.0, 0.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![0, 1, 2, 3]);
    }
    {
        let slice = &[[0.0, 1.0], [0.0, 2.0], [0.0, 0.0], [1.0, 1.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![3, 1, 0, 2]);
    }
    {
        let slice = &[[-3.0, -1.0], [0.0, 2.0], [0.0, 0.0], [5.0, -1.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![3, 1, 0]);
    }
    {
        let slice = &[[1.0, 0.0], [0.0, 1.0], [4.0, 3.0], [3.0, 4.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![2, 3, 1, 0]);
    }
    {
        let slice = &[[0.0, 0.0], [1.0, 1.0], [0.0, 2.0], [-1.0, 1.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![1, 2, 3, 0]);
    }
    {
        let slice = &[[0.0, -4.0], [1.0, -3.0], [0.0, -3.0], [-1.0, -3.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![1, 2, 3, 0]);
    }
    {
        let slice = &[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 1.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![1, 2, 3, 0]);
    }
    {
        let slice = &[[0.0, 0.0], [-1.0, 3.0], [-4.0, 2.0], [-5.0, 4.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![0, 1, 3, 2]);
    }
    {
        let slice = &[[0.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, -2.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![1, 0, 2, 3, 2, 0]);
    }
    {
        let slice = &[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![3, 2, 0, 1]);
    }
}

#[test]
fn test_five_points() {
    {
        let slice = &[[1.0, 0.0], [0.0, 1.0], [4.0, 3.0], [3.0, 4.0], [-1.0, 5.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![2, 3, 4, 1, 0]);
    }
    {
        let slice = &[[1.0, 0.0], [0.0, 1.0], [4.0, 3.0], [3.0, 4.0], [2.0, 2.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![2, 3, 1, 0]);
    }
    {
        let slice = &[[1.0, 0.0], [0.0, 1.0], [4.0, 3.0], [3.0, 4.0], [-1.0, 1.0]];
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![2, 3, 4, 0]);
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
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![7, 5, 3, 1, 0, 6]);
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
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![2, 0, 6, 4, 3, 1]);
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
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![8, 6, 7, 4, 5, 11, 10, 9]);
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
    let hull = reinterpret(vec.convex_hull());
    assert_eq!(hull, vec![3, 2, 0, 1]);
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
        let hull = reinterpret(vec.convex_hull());
        assert_eq!(hull, vec![3, 2, 0, 1]);
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
        let hull = reinterpret(slab.convex_hull());
        assert_eq!(hull, vec![3, 2, 0, 1]);
    }
}

#[test]
fn test_hashmap() {
    {
        let mut hashmap = HashMap::new();
        hashmap.insert(10, [0.0, 0.0]);
        let hull = reinterpret(hashmap.convex_hull());
        assert_eq!(hull, vec![10]);
    }
    {
        let mut hashmap = HashMap::new();
        hashmap.insert(10, [0.0, 0.0]);
        hashmap.insert(5, [0.0, 1.0]);
        let hull = reinterpret(hashmap.convex_hull());
        assert_eq!(hull, vec![5, 10]);
    }
    {
        let mut hashmap = HashMap::new();
        hashmap.insert(0, [0.0, 0.0]);
        hashmap.insert(1, [1.0, 0.0]);
        hashmap.insert(2, [0.0, 1.0]);
        hashmap.insert(3, [1.0, 1.0]);
        let hull = reinterpret(hashmap.convex_hull());
        assert_eq!(hull, vec![3, 2, 0, 1]);
    }
    {
        let mut hashmap = ahash::AHashMap::new();
        hashmap.insert(0, [0.0, 0.0]);
        hashmap.insert(1, [1.0, 0.0]);
        hashmap.insert(2, [0.0, 1.0]);
        hashmap.insert(3, [1.0, 1.0]);
        let hull = reinterpret(hashmap.convex_hull());
        assert_eq!(hull, vec![3, 2, 0, 1]);
    }
    {
        let mut hashmap = HashMap::new();
        hashmap.insert(1, [0.0, 0.0]);
        hashmap.insert(2, [1.0, 0.0]);
        hashmap.insert(3, [0.0, 1.0]);
        hashmap.insert(4, [1.0, 1.0]);
        let hull = reinterpret(hashmap.convex_hull());
        assert_eq!(hull, vec![4, 3, 1, 2]);
    }
    {
        let mut hashmap = HashMap::new();
        hashmap.insert(10, [0.0, 0.0]);
        hashmap.insert(20, [1.0, 0.0]);
        hashmap.insert(30, [0.0, 1.0]);
        hashmap.insert(40, [1.0, 1.0]);
        let hull = reinterpret(hashmap.convex_hull());
        assert_eq!(hull, vec![40, 30, 10, 20]);
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
        let hull = reinterpret(slice.convex_hull());
        assert_eq!(hull, vec![2, 1, 0]);
    }
}
