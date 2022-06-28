#![feature(generic_associated_types)]
use adagraph::prelude::*;

pub trait AsInducedSubgraph: VertexSetGraph {
    type OriginalGraph: AsInducedSubgraph<
        VertexIndex = Self::VertexIndex,
        EdgeIndex = Self::EdgeIndex,
        VertexSet = Self::VertexSet,
    >;
    fn as_induced_subgraph<'a>(
        &'a self,
        selected: Self::VertexSet,
    ) -> InducedSubgraph<'a, Self::OriginalGraph>;
}

impl<'a, G> AsInducedSubgraph for InducedSubgraph<'a, G>
where
    G: AsInducedSubgraph,
{
    type OriginalGraph = G;
    fn as_induced_subgraph<'b>(
        &'b self,
        selected: <Self as VertexSetGraph>::VertexSet,
    ) -> InducedSubgraph<'b, G> {
        InducedSubgraph::new(self.get_original_graph(), selected)
    }
}

pub struct InducedSubgraph<'a, G>
where
    G: AsInducedSubgraph,
{
    graph: &'a G,
    selected: G::VertexSet,
}

impl<'a, G> Clone for InducedSubgraph<'a, G>
where
    G: AsInducedSubgraph,
{
    fn clone(&self) -> Self {
        Self {
            graph: self.graph,
            selected: self.selected.clone(),
        }
    }
}

impl<'a, G> InducedSubgraph<'a, G>
where
    G: AsInducedSubgraph,
{
    pub fn new(graph: &'a G, selected: G::VertexSet) -> Self {
        Self { graph, selected }
    }

    pub fn get_original_graph(&self) -> &'a G {
        self.graph
    }

    pub fn get_selected(&self) -> &G::VertexSet {
        &self.selected
    }
}

impl<'a, G> Graph for InducedSubgraph<'a, G>
where
    G: AsInducedSubgraph,
{
    type VertexIndex = G::VertexIndex;
    type EdgeIndex = G::EdgeIndex;
    fn adjacent_vertives(&self, e: Self::EdgeIndex) -> (Self::VertexIndex, Self::VertexIndex) {
        self.graph.adjacent_vertives(e)
    }

    fn get_edge(&self, v1: Self::VertexIndex, v2: Self::VertexIndex) -> Option<Self::EdgeIndex> {
        if self.selected.contains(v1) && self.selected.contains(v2) {
            self.graph.get_edge(v1, v2)
        } else {
            None
        }
    }
}

impl<'a, G> GraphCount for InducedSubgraph<'a, G>
where
    G: AsInducedSubgraph,
{
    fn n_vertices(&self) -> usize {
        self.selected.len()
    }

    fn n_edges(&self) -> usize {
        self.edges().count()
    }

    fn degree(&self, v: Self::VertexIndex) -> usize {
        self.vertex_neighbor_set(v).len()
    }
}

pub struct InducedSubgraphEdges<'a, 'b, G>
where
    G: AsInducedSubgraph,
    'a: 'b,
{
    induced_subgraph: &'b InducedSubgraph<'a, G>,
    v: Option<G::VertexIndex>,
    to_visit: G::VertexSet,
    neighbors: <G::VertexSet as IntoIterator>::IntoIter,
}

impl<'a, 'b, G> InducedSubgraphEdges<'a, 'b, G>
where
    G: AsInducedSubgraph,
    'a: 'b,
{
    fn new(induced_subgraph: &'b InducedSubgraph<'a, G>) -> Self {
        let mut to_visit = induced_subgraph.selected.clone();
        let v = to_visit.iter().next();

        if let Some(v) = v {
            to_visit.remove(v);
            let neighbors = induced_subgraph.vertex_neighbor_set(v).into_iter();
            Self {
                v: Some(v),
                induced_subgraph,
                to_visit,
                neighbors: neighbors,
            }
        } else {
            Self {
                v: None,
                induced_subgraph,
                to_visit: G::VertexSet::empty(),
                neighbors: G::VertexSet::empty().into_iter(),
            }
        }
    }
}

impl<'a, 'b, G> Iterator for InducedSubgraphEdges<'a, 'b, G>
where
    G: AsInducedSubgraph,
{
    type Item = G::EdgeIndex;
    fn next(&mut self) -> Option<Self::Item> {
        match self.v {
            Some(v) => {
                match self.neighbors.next() {
                    Some(w) => self.induced_subgraph.graph.get_edge(v, w), //This should always be Some
                    None => {
                        let v = self.to_visit.iter().next();
                        match v {
                            Some(v) => {
                                self.v = Some(v);
                                self.to_visit.remove(v);
                                let mut neighbors =
                                    self.induced_subgraph.graph.vertex_neighbor_set(v);
                                neighbors.intersect_set(&self.to_visit);
                                self.neighbors = neighbors.into_iter();
                                self.next()
                            }
                            None => None,
                        }
                    }
                }
            }
            None => None,
        }
    }
}

pub struct InducedSubgraphNeighbors<'a, 'b, G>
where
    G: AsInducedSubgraph,
    'a: 'b,
{
    v: G::VertexIndex,
    graph: &'b InducedSubgraph<'a, G>,
    neighbors: <G::VertexSet as IntoIterator>::IntoIter,
}

impl<'a, 'b, G> InducedSubgraphNeighbors<'a, 'b, G>
where
    G: AsInducedSubgraph,
    'a: 'b,
{
    fn new(graph: &'b InducedSubgraph<'a, G>, v: G::VertexIndex) -> Self {
        Self {
            v,
            graph,
            neighbors: graph.vertex_neighbor_set(v).into_iter(),
        }
    }
}

impl<'a, 'b, G> Iterator for InducedSubgraphNeighbors<'a, 'b, G>
where
    G: AsInducedSubgraph,
    'a: 'b,
{
    type Item = (G::VertexIndex, G::EdgeIndex);
    fn next(&mut self) -> Option<Self::Item> {
        match self.neighbors.next() {
            Some(w) => Some((w, self.graph.get_edge(self.v, w).unwrap())),
            None => None,
        }
    }
}

impl<'a, G> GraphIterator for InducedSubgraph<'a, G>
where
    G: AsInducedSubgraph,
{
    type Vertices<'b>
    where
        'a: 'b,
    = <<G as VertexSetGraph>::VertexSet as Set>::Iter<'b>;
    type Edges<'b>
    where
        'a: 'b,
    = InducedSubgraphEdges<'a, 'b, G>;
    type Neighbors<'b>
    where
        'a: 'b,
    = InducedSubgraphNeighbors<'a, 'b, G>;

    fn vertices(&self) -> Self::Vertices<'_> {
        self.selected.iter()
    }

    fn edges(&self) -> Self::Edges<'_> {
        Self::Edges::<'_>::new(self)
    }

    fn neighbors(&self, v: Self::VertexIndex) -> Self::Neighbors<'_> {
        Self::Neighbors::<'_>::new(self, v)
    }
}

impl<'a, G> VertexSetGraph for InducedSubgraph<'a, G>
where
    G: AsInducedSubgraph,
{
    type VertexSet = G::VertexSet;
    fn vertex_set(&self) -> Self::VertexSet {
        self.selected.clone()
    }

    fn vertex_neighbor_set(&self, v: Self::VertexIndex) -> Self::VertexSet {
        let mut result = self.graph.vertex_neighbor_set(v);
        result.intersect_set(&self.selected);
        result
    }
}

impl<'a, G> ConnectedComponentFrom for InducedSubgraph<'a, G> where G: AsInducedSubgraph {}

impl<'a, G> ConnectedComponents for InducedSubgraph<'a, G> where G: AsInducedSubgraph {}

impl<'a, G> BreathFirstSearch for InducedSubgraph<'a, G>
where
    G: AsInducedSubgraph,
{
    type BreathFirstSearchIterator<'b>
    where
        Self: 'b,
    = adagraph::algos::DefaultBreathFirstSearchIteratorVertexSet<'b, Self>;
}
