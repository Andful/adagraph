#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(type_alias_impl_trait)]

use std::collections::VecDeque;

use adagraph::prelude::*;
use adagraph_induced_subgraph::{AsInducedSubgraph, InducedSubgraph};

pub struct MinimumDegreeOrdering<G, S>
where
    G: VertexSetGraph,
    S: StoreWithKey<G::VertexIndex>,
{
    considered: G::VertexSet,
    adjacency: S::Store<G::VertexSet>,
}

impl<'a, G, S> Iterator for MinimumDegreeOrdering<G, S>
where
    G: VertexSetGraph,
    S: StoreWithKey<G::VertexIndex>,
{
    type Item = G::VertexIndex;
    fn next(&mut self) -> Option<Self::Item> {
        let result = self
            .considered
            .iter()
            .min_by_key(|v| self.adjacency.get_ref(*v).unwrap().len());

        if let Some(v) = result {
            self.considered.remove(v);
            let adj = self.adjacency.get_ref(v).unwrap().clone();
            for w in adj.iter() {
                self.adjacency.get_mut(w).unwrap().insert_set(&adj);
            }
        }

        result
    }
}

impl<G, S> MinimumDegreeOrdering<G, S>
where
    G: VertexSetGraph,
    S: StoreWithKey<G::VertexIndex>
{
    pub fn new(graph: &G) -> Self {
        let considered = graph.vertex_set();
        let adjacency: S::Store<G::VertexSet> = considered.iter().map(|v| (v, graph.vertex_neighbor_set(v))).collect();
        Self {
            considered,
            adjacency,
        }
    }
}

///http://www.ii.uib.no/~pinar/MCS-M.pdf
pub struct MaximumCardinalitySearchForMinimalTriangulation<'a, G, S>
where
    G: AsInducedSubgraph,
    S: StoreWithKey<G::VertexIndex>,
{
    graph: InducedSubgraph<'a, G::OriginalGraph>,
    weights: S::Store<usize>,
}

impl<'a, G, S> Iterator for MaximumCardinalitySearchForMinimalTriangulation<'a, G, S>
where
    G: AsInducedSubgraph,
    S: StoreWithKey<G::VertexIndex>,
{
    type Item = G::VertexIndex;
    fn next(&mut self) -> Option<G::VertexIndex> {
        let z = self
            .graph
            .vertex_set()
            .into_iter()
            .max_by_key(|v| self.weights.get(*v));
        match z {
            Some(z) => {
                let mut new_vertex_set = self.graph.vertex_set();
                new_vertex_set.remove(z);

                let mut neighbors = self.graph.vertex_neighbor_set(z);
                let mut queue = neighbors
                    .iter()
                    .map(|v| (0, v))
                    .collect::<VecDeque<(usize, G::VertexIndex)>>();

                let mut path_weights: S::Store<usize> = new_vertex_set.iter().map(|v| (v, usize::MAX)).collect();
                self.graph =
                    InducedSubgraph::new(self.graph.get_original_graph(), new_vertex_set.clone());

                while let Some((path_weight, y)) = queue.pop_front() {
                    let y_weight = self.weights.get(y).unwrap();
                    if y_weight > path_weight {
                        neighbors.insert(y);
                    }
                    let new_path_weight = path_weight.max(y_weight);
                    for (x, _) in self.graph.neighbors(y) {
                        assert!(new_vertex_set.contains(x));
                        if path_weights.get(x).unwrap() > new_path_weight {
                            queue.push_back((new_path_weight, x));
                            *path_weights.get_mut(x).unwrap() = new_path_weight;
                        }
                    }
                }
                neighbors
                    .iter()
                    .for_each(|v| *self.weights.get_mut(v).unwrap() += 1);

                Some(z)
            }
            None => None,
        }
    }
}

impl<'a, G, S> MaximumCardinalitySearchForMinimalTriangulation<'a, G, S>
where
    G: AsInducedSubgraph,
    S: StoreWithKey<G::VertexIndex>,
{
    pub fn new(graph: &'a G) -> Self {
        let graph = graph.as_induced_subgraph(graph.vertex_set());
        let weights: S::Store<usize> = graph.get_selected().iter().map(|v| (v, 0)).collect();
        Self { graph, weights }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use adagraph_nauty::{NautyGraph, NautyGraphVertexStoreWrapper};

    #[test]
    fn test_mcs_m() {
        let graph = NautyGraph::builder(7)
            .add_edge(0, 1, ())
            .add_edge(0, 2, ())
            .add_edge(1, 3, ())
            .add_edge(2, 3, ())
            .add_edge(3, 4, ())
            .add_edge(3, 5, ())
            .add_edge(4, 6, ())
            .add_edge(5, 6, ())
            .build()
            .unwrap();

        let mut msc = MaximumCardinalitySearchForMinimalTriangulation::<NautyGraph, NautyGraphVertexStoreWrapper>::new(&graph);

        while let Some(v) = msc.next() {
            println!("{}", v);
        }
    }

    #[test]
    fn test_md() {
        let graph = NautyGraph::builder(7)
            .add_edge(0, 1, ())
            .add_edge(0, 2, ())
            .add_edge(1, 3, ())
            .add_edge(2, 3, ())
            .add_edge(3, 4, ())
            .add_edge(3, 5, ())
            .add_edge(4, 6, ())
            .add_edge(5, 6, ())
            .build()
            .unwrap();

        let mut md =
            MinimumDegreeOrdering::<NautyGraph, NautyGraphVertexStoreWrapper>::new(&graph);

        while let Some(v) = md.next() {
            println!("{}", v);
        }
    }
}
