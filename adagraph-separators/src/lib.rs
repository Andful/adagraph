use adagraph::prelude::*;
use adagraph_induced_subgraph::AsInducedSubgraph;
use adagraph_triangulation::MaximumCardinalitySearchForMinimalTriangulation;
use std::collections::BTreeSet;

enum MaximalParallelMinimalSeparatorsPhases<'a, G, S, I>
where
    G: VertexSetGraph,
    S: StoreWithKey<G::VertexIndex>,
    I: Iterator<Item = G::VertexIndex>,
{
    FirstPhase {
        graph: &'a G,
        ordering: I,
        peo: S::Store<Option<usize>>,
        current_order: usize,
        degrees: S::Store<Option<usize>>,
        to_reprocess: S::Store<Vec<G::VertexSet>>,
        lower: G::VertexSet,
    },
    SecondPhase {
        to_reprocess: Vec<Vec<G::VertexSet>>,
        currently_reprocessing: std::vec::IntoIter<G::VertexSet>,
        already_seen_degree: BTreeSet<usize>,
    },
}

///A linear time algorithm to list the minimal separators of chordal graphs
pub struct MaximalParallelMinimalSeparators<'a, G, S, I>
where
    G: AsInducedSubgraph,
    S: StoreWithKey<G::VertexIndex>,
    I: Iterator<Item = G::VertexIndex>,
    G::VertexSet: Ord,
{
    separators: &'a mut BTreeSet<G::VertexSet>,
    phase: MaximalParallelMinimalSeparatorsPhases<'a, G, S, I>,
}

impl<'a, G, S, I> MaximalParallelMinimalSeparators<'a, G, S, I>
where
    G: AsInducedSubgraph,
    S: StoreWithKey<G::VertexIndex>,
    I: Iterator<Item = G::VertexIndex>,
    G::VertexSet: Ord,
{
    pub fn new(graph: &'a G, ordering: I, separators: &'a mut BTreeSet<G::VertexSet>) -> Self {
        let current_order = graph.vertex_set().len();
        Self {
            separators,
            phase: MaximalParallelMinimalSeparatorsPhases::FirstPhase {
                graph,
                ordering,
                peo: graph.vertex_set().iter().map(|v| (v, None)).collect(),
                current_order,
                degrees: graph.vertex_set().iter().map(|v| (v, None)).collect(),
                to_reprocess: graph.vertex_set().iter().map(|v| (v, vec![])).collect(),
                lower: graph.vertex_set(),
            },
        }
    }

    fn get_heigher_order_neighbors(
        v: G::VertexIndex,
        graph: &'a G,
        lower: &mut G::VertexSet,
    ) -> G::VertexSet {
        lower.remove(v);
        let mut v_set = G::VertexSet::empty();
        v_set.insert(v);

        let component = graph
            .as_induced_subgraph(lower.clone())
            .connected_component_from(v_set);

        graph.neighbor_set_exclusive(&component)
    }

    fn get_z(
        neighbors: &G::VertexSet,
        peo: &S::Store<Option<usize>>,
    ) -> Option<G::VertexIndex> {
        neighbors
            .iter()
            .filter_map(|z| peo.get(z).map(|o| (z, o)))
            .min_by_key(|(_, v)| *v)
            .map(|(z, _)| z)
    }

    fn update_peo(
        v: G::VertexIndex,
        peo: &mut S::Store<Option<usize>>,
        current_order: &mut usize,
    ) {
        *peo.get_mut(v).unwrap() = Some(*current_order);
        *current_order -= 1;
    }

    fn update_degree(
        v: G::VertexIndex,
        degrees: &mut S::Store<Option<usize>>,
        v_degree: usize,
    ) {
        *degrees.get_mut(v).unwrap() = Some(v_degree);
    }
}

impl<'a, G, S, I> Iterator for MaximalParallelMinimalSeparators<'a, G, S, I>
where
    G: AsInducedSubgraph,
    S: StoreWithKey<G::VertexIndex>,
    I: Iterator<Item = G::VertexIndex>,
    G::VertexSet: Ord,
{
    type Item = G::VertexSet;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match &mut self.phase {
                MaximalParallelMinimalSeparatorsPhases::FirstPhase {
                    graph,
                    ordering,
                    current_order,
                    to_reprocess,
                    peo,
                    degrees,
                    lower,
                } => match ordering.next() {
                    Some(v) => {
                        let heigher_order_neighbors =
                            Self::get_heigher_order_neighbors(v, graph, lower);
                        let z = Self::get_z(&heigher_order_neighbors, peo);
                        match z {
                            Some(z) => {
                                let z_degree = degrees.get(z).unwrap().unwrap();
                                let v_degree = heigher_order_neighbors.len();
                                Self::update_peo(v, peo, current_order);
                                Self::update_degree(v, degrees, v_degree);
                                if v_degree <= z_degree {
                                    if self.separators.insert(heigher_order_neighbors.clone()) {
                                        return Some(heigher_order_neighbors);
                                    } else {
                                        continue;
                                    }
                                } else {
                                    to_reprocess
                                        .get_mut(z)
                                        .unwrap()
                                        .push(heigher_order_neighbors);
                                    continue;
                                }
                            }
                            None => {
                                Self::update_peo(v, peo, current_order);
                                Self::update_degree(v, degrees, 0);
                                continue;
                            }
                        }
                    }
                    None => {
                        let mut to_reprocess = to_reprocess
                            .iter_mut()
                            .filter(|(_, s)| !s.is_empty())
                            .map(|(_, s)| std::mem::take(s))
                            .collect::<Vec<Vec<G::VertexSet>>>();
                        if let Some(currently_reprocessing) = to_reprocess.pop() {
                            self.phase = MaximalParallelMinimalSeparatorsPhases::SecondPhase {
                                to_reprocess,
                                currently_reprocessing: currently_reprocessing.into_iter(),
                                already_seen_degree: BTreeSet::new(),
                            };
                            continue;
                        } else {
                            return None;
                        }
                    }
                },
                MaximalParallelMinimalSeparatorsPhases::SecondPhase {
                    to_reprocess,
                    currently_reprocessing,
                    already_seen_degree,
                } => match currently_reprocessing.next() {
                    Some(neighbors) => {
                        if !already_seen_degree.insert(neighbors.len()) {
                            if self.separators.insert(neighbors.clone()) {
                                return Some(neighbors);
                            } else {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    }
                    None => match to_reprocess.pop() {
                        Some(new_to_reprocess) => {
                            *currently_reprocessing = new_to_reprocess.into_iter();
                            *already_seen_degree = BTreeSet::new();
                            continue;
                        }
                        None => return None,
                    },
                },
            }
        }
    }
}

pub struct CliqueMinimalSeparators<'a, G, S>
where
    G: AsInducedSubgraph,
    S: StoreWithKey<G::VertexIndex>,
    G::VertexSet: Ord,
{
    graph: &'a G,
    mpms: MaximalParallelMinimalSeparators<
        'a,
        G,
        S,
        MaximumCardinalitySearchForMinimalTriangulation<'a, G, S>,
    >,
}

impl<'a, G, S> CliqueMinimalSeparators<'a, G, S>
where
    G: AsInducedSubgraph,
    S: StoreWithKey<G::VertexIndex>,
    G::VertexSet: Ord,
{
    pub fn new(graph: &'a G, checked: &'a mut BTreeSet<G::VertexSet>) -> Self {
        Self {
            graph,
            mpms: MaximalParallelMinimalSeparators::new(
                graph,
                MaximumCardinalitySearchForMinimalTriangulation::new(graph),
                checked,
            ),
        }
    }

    fn is_clique(&self, separator: &G::VertexSet) -> bool {
        let mut mask = separator.clone();
        for v in separator.iter() {
            mask.remove(v);
            if !self.graph.vertex_neighbor_set(v).is_superset_of(&mask) {
                return false;
            }
        }

        true
    }
}

impl<'a, G, S> Iterator for CliqueMinimalSeparators<'a, G, S>
where
    G: AsInducedSubgraph,
    S: StoreWithKey<G::VertexIndex>,
    G::VertexSet: Ord,
{
    type Item = G::VertexSet;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let e = self.mpms.next();
            match e {
                Some(s) => {
                    if self.is_clique(&s) {
                        return Some(s);
                    } else {
                        continue;
                    }
                }
                None => return None,
            }
        }
    }
}

pub struct CloseMinimalSeparators<'a, G>
where
    G: AsInducedSubgraph,
    G::VertexSet: Ord,
{
    graph: &'a G,
    separator: &'a G::VertexSet,
    separator_vertices: <G::VertexSet as IntoIterator>::IntoIter,
    queued_result_separators: Vec<G::VertexSet>,
}

impl<'a, G> CloseMinimalSeparators<'a, G>
where
    G: AsInducedSubgraph,
    G::VertexSet: Ord,
{
    pub fn new(graph: &'a G, separator: &'a G::VertexSet) -> Self {
        let separator_vertices = separator.clone().into_iter();

        Self {
            graph,
            separator,
            separator_vertices,
            queued_result_separators: Vec::new(),
        }
    }
}

impl<'a, G> Iterator for CloseMinimalSeparators<'a, G>
where
    G: AsInducedSubgraph,
    G::VertexSet: Ord,
{
    type Item = G::VertexSet;
    fn next(&mut self) -> Option<Self::Item> {
        match self.queued_result_separators.pop() {
            Some(s) => Some(s),
            None => match self.separator_vertices.next() {
                Some(v) => {
                    let mut separator = self.separator.clone();
                    let mut vertex_set = G::VertexSet::empty();
                    vertex_set.insert(v);
                    separator.insert_set(&self.graph.vertex_neighbor_set(v));
                    let mut components = self.graph.vertex_set();
                    components.remove_set(&separator);
                    self.queued_result_separators = self
                        .graph
                        .as_induced_subgraph(components)
                        .connected_components()
                        .map(|s| self.graph.neighbor_set_exclusive(&s))
                        .collect();
                    self.next()
                }
                None => None,
            },
        }
    }
}

pub struct MinimalSeparators<'a, G>
where
    G: AsInducedSubgraph,
    G::VertexSet: Ord,
{
    graph: &'a G,
    to_process: Vec<G::VertexSet>,
    minimal_separators: &'a mut BTreeSet<G::VertexSet>,
}

impl<'a, G> Iterator for MinimalSeparators<'a, G>
where
    G: AsInducedSubgraph,
    G::VertexSet: Ord,
{
    type Item = G::VertexSet;
    fn next(&mut self) -> Option<Self::Item> {
        match self.to_process.pop() {
            Some(separator) => {
                for close_separators in CloseMinimalSeparators::<G>::new(self.graph, &separator) {
                    if self.minimal_separators.insert(close_separators.clone()) {
                        self.to_process.push(close_separators);
                    }
                }
                Some(separator)
            }
            None => None,
        }
    }
}

impl<'a, G> MinimalSeparators<'a, G>
where
    G: AsInducedSubgraph,
    G::VertexSet: Ord,
{
    pub fn new(graph: &'a G, minimal_separators: &'a mut BTreeSet<G::VertexSet>) -> Self {
        graph
            .vertex_set()
            .iter()
            .flat_map(|v| {
                let mut subgraph = graph.vertex_set();
                subgraph.remove_set(&graph.vertex_neighbor_set(v));
                subgraph.remove(v);
                graph
                    .as_induced_subgraph(subgraph)
                    .connected_components()
                    .collect::<Vec<G::VertexSet>>()
            })
            .map(|s| graph.neighbor_set_exclusive(&s))
            .for_each(|s| {
                minimal_separators.insert(s);
            });

        let to_process = minimal_separators
            .iter()
            .map(|separator| separator.clone())
            .collect();

        MinimalSeparators {
            graph,
            to_process,
            minimal_separators,
        }
    }
}

pub struct MinimalABSeparators<'a, G>
where
    G: AsInducedSubgraph,
    G::VertexSet: Ord,
{
    graph: &'a G,
    minimal_separators: &'a mut BTreeSet<G::VertexSet>,
    to_process: Vec<G::VertexSet>,
    a: G::VertexSet,
    na: G::VertexSet,
}

impl<'a, G> MinimalABSeparators<'a, G>
where
    G: AsInducedSubgraph,
    G::VertexSet: Ord,
{
    pub fn new(
        graph: &'a G,
        a: G::VertexSet,
        b: G::VertexSet,
        minimal_separators: &'a mut BTreeSet<G::VertexSet>,
    ) -> Self {
        let na = graph.neighbor_set_exclusive(&a);
        let mut result = MinimalABSeparators {
            graph,
            minimal_separators,
            to_process: vec![b],
            a,
            na,
        };
        result.next();
        result
    }
}

impl<'a, G> Iterator for MinimalABSeparators<'a, G>
where
    G: AsInducedSubgraph,
    G::VertexSet: Ord,
{
    type Item = G::VertexSet;
    fn next(&mut self) -> Option<Self::Item> {
        match self.to_process.pop() {
            Some(s) => {
                let mut components = self.graph.vertex_set();
                components.remove_set(&s);
                let ca = self
                    .graph
                    .as_induced_subgraph(components)
                    .connected_component_from(self.a.clone());
                let na = &self.na;
                for x in s.iter().filter(|v| !na.contains(*v)) {
                    let mut delta = self.graph.vertex_neighbor_set(x);
                    delta.intersect_set(&ca);
                    let mut ca_component = ca.clone();
                    ca_component.remove_set(&delta);
                    let ca_prime = self
                        .graph
                        .as_induced_subgraph(ca_component)
                        .connected_component_from(self.a.clone());
                    let mut new_separator = delta;
                    new_separator
                        .insert_set(&s)
                        .intersect_set(&self.graph.neighbor_set_exclusive(&ca_prime));
                    if self.minimal_separators.insert(new_separator.clone()) {
                        self.to_process.push(new_separator);
                    }
                }
                Some(s)
            }
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use adagraph_nauty::{BitSet, NautyGraph, NautyGraphVertexStoreWrapper};
    type G = NautyGraph;
    type VS = <NautyGraph as VertexSetGraph>::VertexSet;
    type S = NautyGraphVertexStoreWrapper;

    #[test]
    fn test_minimal_ab_separator() {
        let graph = G::builder(11)
            .add_edge(0, 1, ())
            .add_edge(0, 4, ())
            .add_edge(0, 7, ())
            .add_edge(1, 2, ())
            .add_edge(2, 3, ())
            .add_edge(4, 5, ())
            .add_edge(5, 6, ())
            .add_edge(7, 8, ())
            .add_edge(8, 9, ())
            .add_edge(3, 10, ())
            .add_edge(6, 10, ())
            .add_edge(9, 10, ())
            .build()
            .unwrap();

        let a = VS::singleton(0);
        let b = VS::singleton(10);
        for s in MinimalABSeparators::new(&graph, a, b, &mut BTreeSet::new()) {
            println!("{}", s);
        }
    }

    #[test]
    fn test_lexbfs() {
        let graph = G::builder(4)
            .add_edge(0, 1, ())
            .add_edge(1, 2, ())
            .add_edge(2, 3, ())
            .add_edge(3, 0, ())
            .build()
            .unwrap();

        //graph.add_edge(0, 2, ()).unwrap();

        let mut ordering = MaximumCardinalitySearchForMinimalTriangulation::<'_, G, S>::new(&graph)
            .collect::<Vec<usize>>();
        ordering.reverse();
        println!("ordering: {:?}", ordering);
        let separators = MaximalParallelMinimalSeparators::<
            '_,
            G,
            S,
            MaximumCardinalitySearchForMinimalTriangulation<'_, G, S>,
        >::new(&graph, MaximumCardinalitySearchForMinimalTriangulation::new(&graph), &mut BTreeSet::new())
        .collect::<Vec<VS>>();
        println!("separators: {:?}", separators);
    }

    #[test]
    fn test_lexbfs2() {
        let graph = G::builder(8)
            .add_edge(0, 1, ())
            .add_edge(0, 2, ())
            .add_edge(1, 2, ())
            .add_edge(1, 4, ())
            .add_edge(1, 5, ())
            .add_edge(1, 6, ())
            .add_edge(2, 3, ())
            .add_edge(2, 4, ())
            .add_edge(3, 4, ())
            .add_edge(4, 6, ())
            .add_edge(4, 7, ())
            .add_edge(5, 6, ())
            .add_edge(6, 7, ())
            .build()
            .unwrap();

        let mut ordering = MaximumCardinalitySearchForMinimalTriangulation::<G, S>::new(&graph)
            .collect::<Vec<usize>>();
        ordering.reverse();
        println!("ordering: {:?}", ordering);

        let separators = MaximalParallelMinimalSeparators::<
            '_,
            G,
            S,
            MaximumCardinalitySearchForMinimalTriangulation<'_, G, S>,
        >::new(&graph, MaximumCardinalitySearchForMinimalTriangulation::new(&graph), &mut BTreeSet::new())
        .collect::<Vec<VS>>();
        println!("separators: {:?}", separators);
    }
}
