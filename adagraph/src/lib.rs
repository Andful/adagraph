//#![warn(missing_docs)]
#![feature(associated_type_defaults)]

//! Traits for graph theory
//!
//! Adagraph name is derived from ADAptive GRAPH.
//! Many problems can be represented as a graph. The aim of this library is to adapt your problem into a graph representation by implementing the traits provided by adagraph.
//! This library, instead of providing a graph data structure it allows you to represent your problem as a graph

pub mod algos;
pub mod prelude;

use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::iter::FromIterator;

/// Graph trait
pub trait Graph {
    /// Index used to reference a vertex
    type VertexIndex: Copy + Ord;
    /// Index used to reference a edge
    type EdgeIndex: Copy + Ord;
    /// Get vertices adjacent to an edge
    fn adjacent_vertives(&self, edge: Self::EdgeIndex) -> (Self::VertexIndex, Self::VertexIndex);
    /// Get edge connectin v1 and v2 if it exists
    fn get_edge(&self, v1: Self::VertexIndex, v2: Self::VertexIndex) -> Option<Self::EdgeIndex>;
}

pub trait GraphCount: Graph {
    /// Number of vertices
    fn n_vertices(&self) -> usize;
    /// Number of edges
    fn n_edges(&self) -> usize;
    /// Degree of the vertex
    fn degree(&self, v: Self::VertexIndex) -> usize;
}

/// Trait to iterate through graph elements
pub trait GraphIterator: Graph {
    /// Type that iterates though vertices
    type Vertices<'b>: Iterator<Item = Self::VertexIndex>
    where
        Self: 'b;
    /// Type that iterates though edges
    type Edges<'b>: Iterator<Item = Self::EdgeIndex>
    where
        Self: 'b;
    /// Type that iterates though the neighbors of a vertex
    type Neighbors<'b>: Iterator<Item = (Self::VertexIndex, Self::EdgeIndex)>
    where
        Self: 'b;
    /// Iterator to iterate the vertices of a graph
    fn vertices(&self) -> Self::Vertices<'_>;
    /// Iterator to iterate the edges of a graph
    fn edges(&self) -> Self::Edges<'_>;
    /// Iterator to iterate the neighbors of a graph
    fn neighbors(&self, v: Self::VertexIndex) -> Self::Neighbors<'_>;
}

/// SAFETY: The Set struct must have a correct implementation of set. Unsafe code can rely on a correct implementation
pub unsafe trait Set:
    Default
    + std::cmp::PartialEq
    + Clone
    + IntoIterator<Item = Self::Element>
    + FromIterator<Self::Element>
{
    /// Type of the element of the set
    type Element: Copy + Eq;
    /// Iterator type to iterate though the elements of the set
    type Iter<'a>: Iterator<Item = Self::Element>
    where
        Self: 'a;
    type IterIncreasing<'a>: Iterator<Item = Self::Element>
    where
        Self::Element: Ord,
        Self: 'a;
    type IterDecreasing<'a>: Iterator<Item = Self::Element>
    where
        Self::Element: Ord,
        Self: 'a;
    type IntoIterIncreasing: Iterator<Item = Self::Element>
    where
        Self::Element: Ord;
    type IntoIterDecreasing: Iterator<Item = Self::Element>
    where
        Self::Element: Ord;
    /// New empty set
    fn empty() -> Self;
    fn singleton(e: Self::Element) -> Self;
    /// Check whether set contains such element
    fn contains(&self, v: Self::Element) -> bool;
    /// Insert element
    fn insert<'b>(&'b mut self, v: Self::Element) -> &'b mut Self;
    /// Insert elements of set into this set
    fn insert_set<'b>(&'b mut self, set: &Self) -> &'b mut Self;
    /// Remove element from this set
    fn remove<'b>(&'b mut self, v: Self::Element) -> &'b mut Self;
    fn remove_set<'b>(&'b mut self, set: &Self) -> &'b mut Self;
    fn intersect_set<'a>(&'a mut self, other: &Self) -> &'a mut Self;
    fn is_subset_of(&self, other: &Self) -> bool;
    fn is_superset_of(&self, other: &Self) -> bool {
        other.is_subset_of(self)
    }
    fn is_intersecting(&self, other: &Self) -> bool;
    fn is_empty(&self) -> bool;
    fn clear<'b>(&'b mut self) -> &'b mut Self;
    fn iter<'b>(&'b self) -> Self::Iter<'b>;
    fn iter_increasing<'b>(&'b self) -> Self::IterIncreasing<'b>
    where
        Self::Element: std::cmp::Ord;
    fn iter_decreasing<'b>(&'b self) -> Self::IterDecreasing<'b>
    where
        Self::Element: std::cmp::Ord;
    fn into_iter_increasing(self) -> Self::IntoIterIncreasing
    where
        Self::Element: std::cmp::Ord;
    fn into_iter_decreasing(self) -> Self::IntoIterDecreasing
    where
        Self::Element: std::cmp::Ord;
    fn len(&self) -> usize;
    fn min_value(&self) -> Option<Self::Element>
    where
        Self::Element: std::cmp::Ord;
    fn max_value(&self) -> Option<Self::Element>
    where
        Self::Element: std::cmp::Ord;
}

pub trait VertexSetGraph: Graph {
    type VertexSet: Set<Element = Self::VertexIndex>;
    fn vertex_set(&self) -> Self::VertexSet;
    fn vertex_neighbor_set(&self, v: Self::VertexIndex) -> Self::VertexSet;
    fn neighbor_set_simple(&self, set: &Self::VertexSet) -> Self::VertexSet {
        let mut result = Self::VertexSet::empty();
        for v in set.iter() {
            result.insert_set(&self.vertex_neighbor_set(v));
        }
        result
    }
    fn neighbor_set_inclusive(&self, set: &Self::VertexSet) -> Self::VertexSet {
        let mut result = self.neighbor_set_simple(set);
        result.insert_set(set);
        result
    }
    fn neighbor_set_exclusive(&self, set: &Self::VertexSet) -> Self::VertexSet {
        let mut result = self.neighbor_set_simple(set);
        result.remove_set(set);
        result
    }
}

pub trait VertexStoreGraph: Graph {
    type VertexStore<E>: Store<Self::VertexIndex, E>;
}

/// A general map data structure
/// SAFETY: the data structure has to ensure its correctness, unsafe blocks can assume its correct implementation
pub unsafe trait Store<K, V>:
    IntoIterator<Item = (K, V)> + FromIterator<(K, V)> + Default
where
    K: Copy,
{
    type Iter<'a>: Iterator<Item = (K, &'a V)>
    where
        V: 'a,
        Self: 'a;
    type IterMut<'a>: Iterator<Item = (K, &'a mut V)>
    where
        V: 'a,
        Self: 'a;
    fn new() -> Self;
    fn add(&mut self, k: K, v: V) -> Option<V>;
    fn remove(&mut self, k: K) -> Option<V>;
    fn get(&self, k: K) -> Option<V>
    where
        V: Copy;
    fn get_ref(&self, k: K) -> Option<&V>;
    fn get_mut(&mut self, k: K) -> Option<&mut V>;
    fn get_mul_mut<'a, S>(&'a mut self, set: &'_ S) -> Vec<(K, Option<&'a mut V>)>
    where
        S: Set<Element = K>;
    fn iter(&self) -> Self::Iter<'_>;
    fn iter_mut(&mut self) -> Self::IterMut<'_>;
    fn clone(&self) -> Self
    where
        V: Clone;
}

pub trait StoreWithKey<K>
where
    K: Copy,
{
    type Store<V>: Store<K, V>;
}

pub type GraphEditingResult<R> = std::result::Result<R, GraphEditingError>;

#[derive(Debug)]
pub enum GraphEditingError {
    TooManyVertices(String),
    RepeatedVertex(String),
    ParallelEdge(String),
    SelfLoopEdge(String),
    NonExistingEdge(String),
    NonExistingVertex(String),
    UnsupportedOperation(String),
    UnsupportedArgument(String),
    UnknownError(String),
}

impl Display for GraphEditingError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
}

impl Error for GraphEditingError {}

pub trait GraphBuilder {
    type Graph: Graph;
    type VertexSet: Set<Element = <Self::Graph as Graph>::VertexIndex>;
    fn new<Args>(args: Args) -> Self
    where
        Args: 'static;
    fn vertex_set(&self) -> Self::VertexSet;
    fn add_vertex<Args>(self, args: Args) -> Self
    where
        Args: 'static;
    fn add_edge<Args>(
        self,
        v: <Self::Graph as Graph>::VertexIndex,
        w: <Self::Graph as Graph>::VertexIndex,
        args: Args,
    ) -> Self
    where
        Args: 'static;
    fn build(self) -> GraphEditingResult<Self::Graph>;
}

pub trait BuildableGraph: Graph {
    type Builder: GraphBuilder<Graph = Self>;

    fn builder<Args>(args: Args) -> Self::Builder
    where
        Args: 'static,
    {
        Self::Builder::new(args)
    }
}

pub trait VertexAddableGraph: Graph {
    fn add_vertex<Args>(&mut self, args: Args) -> GraphEditingResult<&mut Self>
    where
        Args: 'static;
}

pub trait EdgeAddableGraph: Graph {
    fn add_edge<Args>(
        &mut self,
        v: Self::VertexIndex,
        w: Self::VertexIndex,
        args: Args,
    ) -> GraphEditingResult<&mut Self>
    where
        Args: 'static;
}

pub trait EdgeRemovableGraph: Graph {
    fn remove_edge<Args>(
        &mut self,
        e: Self::EdgeIndex,
        args: Args,
    ) -> GraphEditingResult<&mut Self>
    where
        Args: 'static;
}
