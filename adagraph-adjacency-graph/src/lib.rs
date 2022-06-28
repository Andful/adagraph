#![feature(generic_associated_types)]
#![feature(drain_filter)]
use adagraph::prelude::*;
use adagraph_induced_subgraph::{AsInducedSubgraph, InducedSubgraph};

pub struct AdjGraph<V, Vertices, Adjacency>
where
    V: Copy + Ord,
    Vertices: Set<Element = V>,
    Adjacency: Store<V, Vertices>,
{
    vertices: Vertices,
    adjacency: Adjacency,
}

impl<V, Vertices, Adjacency> Clone for AdjGraph<V, Vertices, Adjacency>
where
    V: Copy + Ord,
    Vertices: Set<Element = V>,
    Adjacency: Store<V, Vertices>,
{
    fn clone(&self) -> Self {
        Self {
            vertices: self.vertices.clone(),
            adjacency: self.adjacency.clone(),
        }
    }
}

impl<V, Vertices, Adjacency> AsInducedSubgraph for AdjGraph<V, Vertices, Adjacency>
where
    V: Copy + Ord,
    Vertices: Set<Element = V>,
    Adjacency: Store<V, Vertices>,
{
    type OriginalGraph = Self;
    fn as_induced_subgraph<'b>(
        &'b self,
        selected: Self::VertexSet,
    ) -> InducedSubgraph<'b, Self::OriginalGraph> {
        InducedSubgraph::new(self, selected)
    }
}

impl<V, Vertices, Adjacency> AdjGraph<V, Vertices, Adjacency>
where
    V: Copy + Ord,
    Vertices: Set<Element = V>,
    Adjacency: Store<V, Vertices>,
{
    pub fn new(vertices: Vertices) -> Self {
        let adjacency = vertices.iter().map(|v| (v, Vertices::empty())).collect();
        Self {
            vertices,
            adjacency,
        }
    }

    pub fn new_from_graph<G>(graph: &G, vertices: Vertices) -> Self where G: VertexSetGraph<VertexIndex=V, VertexSet=Vertices>, {
        let adjacency = vertices.iter().map(|v| {
            let mut neighbors = graph.vertex_neighbor_set(v);
            neighbors.intersect_set(&vertices);
            (v, neighbors)
        }).collect();
        Self {
            vertices,
            adjacency,
        }
    }

    pub fn add_edge(&mut self, v: V, u: V) {
        if self.vertices.contains(v) && self.vertices.contains(u) {
            self.adjacency
                .get_mut(v)
                .unwrap_or_else(|| unreachable!())
                .insert(u);
            self.adjacency
                .get_mut(u)
                .unwrap_or_else(|| unreachable!())
                .insert(v);
        }
    }

    pub fn saturate(&mut self, mut vs: Vertices) -> &mut Self {
        vs.intersect_set(&self.vertices);
        for v in vs.iter() {
            self.adjacency
                .get_mut(v)
                .unwrap_or_else(|| unreachable!())
                .insert_set(&vs)
                .remove(v);
        }
        self
    }

    pub fn subgraph(&mut self, selected: &Vertices) -> &mut Self {
        self.vertices.intersect_set(selected);
        self.vertices.iter().for_each(|v| {
            self.adjacency = std::mem::replace(&mut self.adjacency, std::iter::empty().collect())
                .into_iter()
                .filter_map(|(v, mut adj)| {
                    if !selected.contains(v) {
                        return None;
                    }
                    adj.intersect_set(&selected);
                    Some((v, adj))
                })
                .collect()
        });
        self
    }
}

impl<V, Vertices, Adjacency> Graph for AdjGraph<V, Vertices, Adjacency>
where
    V: Copy + Ord,
    Vertices: Set<Element = V>,
    Adjacency: Store<V, Vertices>,
{
    type VertexIndex = V;
    type EdgeIndex = (V, V);
    fn adjacent_vertives(&self, e: Self::EdgeIndex) -> (Self::VertexIndex, Self::VertexIndex) {
        e
    }

    fn get_edge(&self, v1: Self::VertexIndex, v2: Self::VertexIndex) -> Option<Self::EdgeIndex> {
        if let Some(vs) = self.adjacency.get_ref(v1) {
            if vs.contains(v2) {
                Some((v1, v2))
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl<V, Vertices, Adjacency> VertexSetGraph for AdjGraph<V, Vertices, Adjacency>
where
    V: Copy + Ord,
    Vertices: Set<Element = V>,
    Adjacency: Store<V, Vertices>,
{
    type VertexSet = Vertices;
    fn vertex_set(&self) -> Self::VertexSet {
        self.vertices.clone()
    }

    fn vertex_neighbor_set(&self, v: Self::VertexIndex) -> Self::VertexSet {
        self.adjacency
            .get_ref(v)
            .unwrap_or_else(|| panic!("vertex not in graph"))
            .clone()
    }

    fn neighbor_set_exclusive(&self, set: &Self::VertexSet) -> Self::VertexSet {
        let mut result = self.neighbor_set_inclusive(set);
        result.remove_set(set);
        result
    }
}

impl<V, Vertices, Adjacency> ConnectedComponentFrom for AdjGraph<V, Vertices, Adjacency>
where
    V: Copy + Ord,
    Vertices: Set<Element = V>,
    Adjacency: Store<V, Vertices>,
{
}

impl<V, Vertices, Adjacency> ConnectedComponents for AdjGraph<V, Vertices, Adjacency>
where
    V: Copy + Ord,
    Vertices: Set<Element = V>,
    Adjacency: Store<V, Vertices>,
{
}

impl<V, Vertices, Adjacency> BreathFirstSearch for AdjGraph<V, Vertices, Adjacency>
where
    V: Copy + Ord,
    Vertices: Set<Element = V>,
    Adjacency: Store<V, Vertices>,
{
    type BreathFirstSearchIterator<'b>
    where
        Self: 'b,
    = adagraph::algos::DefaultBreathFirstSearchIteratorVertexSet<'b, Self>;
}
