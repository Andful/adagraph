use crate::{Graph, Set, VertexSetGraph};

pub trait ConnectedComponentFrom: VertexSetGraph {
    fn connected_component_from(&self, start: Self::VertexSet) -> Self::VertexSet {
        let mut island = Self::VertexSet::empty();
        let mut border = start;

        while !border.is_empty() {
            island.insert_set(&border);
            border = self.neighbor_set_exclusive(&island)
        }

        island
    }
}

pub trait ConnectedComponents: VertexSetGraph + ConnectedComponentFrom {
    type ConnectedComponentsIterator<'a>: ConnectedComponentsIterator<'a, Self>
    where
        Self: 'a,
    = DefaultConnectedComponentsIterator<'a, Self>;

    fn connected_components(&'_ self) -> Self::ConnectedComponentsIterator<'_> {
        Self::ConnectedComponentsIterator::<'_>::new(self)
    }
}

pub trait ConnectedComponentsIterator<'a, G>: Iterator<Item = G::VertexSet>
where
    G: VertexSetGraph + ?Sized,
{
    fn new(graph: &'a G) -> Self;
}

pub struct DefaultConnectedComponentsIterator<'a, G>
where
    G: VertexSetGraph + ConnectedComponentFrom + ?Sized,
{
    graph: &'a G,
    to_visit: G::VertexSet,
}

impl<'a, G> Iterator for DefaultConnectedComponentsIterator<'a, G>
where
    G: VertexSetGraph + ConnectedComponentFrom + ?Sized,
{
    type Item = G::VertexSet;
    fn next(&mut self) -> Option<Self::Item> {
        let v = self.to_visit.iter().next();
        match v {
            Some(v) => {
                let set = G::VertexSet::singleton(v);
                let component = self.graph.connected_component_from(set);
                self.to_visit.remove_set(&component);
                Some(component)
            }
            None => None,
        }
    }
}

impl<'a, G> ConnectedComponentsIterator<'a, G> for DefaultConnectedComponentsIterator<'a, G>
where
    G: VertexSetGraph + ConnectedComponentFrom + ?Sized,
{
    fn new(graph: &'a G) -> Self {
        Self {
            to_visit: graph.vertex_set(),
            graph,
        }
    }
}

pub trait BreathFirstSearch: Graph {
    type BreathFirstSearchIterator<'a>: BreathFirstSearchIterator<'a, Self>
    where
        Self: 'a;
    fn breath_first_search(&'_ self, v: Self::VertexIndex) -> Self::BreathFirstSearchIterator<'_> {
        Self::BreathFirstSearchIterator::<'_>::new(self, v)
    }
}

pub trait BreathFirstSearchIterator<'a, G>: Iterator<Item = G::VertexIndex>
where
    G: Graph + ?Sized,
{
    fn new(graph: &'a G, v: G::VertexIndex) -> Self;
}

pub struct DefaultBreathFirstSearchIteratorVertexSet<'a, G>
where
    G: VertexSetGraph + ?Sized,
{
    graph: &'a G,
    visited: G::VertexSet,
    to_output: <G::VertexSet as IntoIterator>::IntoIter,
}

impl<'a, G> Iterator for DefaultBreathFirstSearchIteratorVertexSet<'a, G>
where
    G: VertexSetGraph,
{
    type Item = G::VertexIndex;

    fn next(&mut self) -> Option<Self::Item> {
        match self.to_output.next() {
            Some(v) => Some(v),
            None => {
                let neighbors = self.graph.neighbor_set_exclusive(&self.visited);
                if neighbors.is_empty() {
                    None
                } else {
                    self.visited.insert_set(&neighbors);
                    self.to_output = neighbors.into_iter();
                    self.next()
                }
            }
        }
    }
}

impl<'a, G> BreathFirstSearchIterator<'a, G> for DefaultBreathFirstSearchIteratorVertexSet<'a, G>
where
    G: VertexSetGraph,
{
    fn new(graph: &'a G, v: G::VertexIndex) -> Self {
        let mut visited = G::VertexSet::empty();
        visited.insert(v);
        Self {
            graph,
            to_output: visited.clone().into_iter(),
            visited,
        }
    }
}
