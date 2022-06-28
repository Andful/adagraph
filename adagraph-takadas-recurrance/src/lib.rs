use adagraph::prelude::*;
use adagraph_induced_subgraph::AsInducedSubgraph;

struct TakatasRecurranceFrame<G>
where
    G: AsInducedSubgraph,
{
    a_side: G::VertexSet,
    b_side: G::VertexSet,
    separator: G::VertexSet,
    fixed: G::VertexSet,
    exclude: G::VertexSet,
}

pub struct TakatasRecurrance<'a, G, F>
where
    G: AsInducedSubgraph,
    F: FnMut(&G::VertexSet, &G::VertexSet) -> bool,
{
    graph: &'a G,
    condition: F,
    stack: Vec<TakatasRecurranceFrame<G>>,
}

impl<'a, G, F> TakatasRecurrance<'a, G, F>
where
    G: AsInducedSubgraph,
    F: FnMut(&G::VertexSet, &G::VertexSet) -> bool,
{
    pub fn new(graph: &'a G, mut condition: F) -> Self {
        let mut a_excluded = G::VertexSet::empty();
        Self {
            graph,
            stack: graph
                .vertex_set()
                .iter()
                .flat_map(|a| {
                    let a_neighbors = graph.vertex_neighbor_set(a);
                    let mut rest = graph.vertex_set();
                    rest.remove_set(&a_neighbors).remove(a);
                    let mut fixed = graph.vertex_neighbor_set(a);
                    fixed.intersect_set(&a_excluded);
                    let exclude = a_excluded.clone();
                    a_excluded.insert(a);
                    graph
                        .as_induced_subgraph(rest)
                        .connected_components()
                        .filter_map(|b_side| {
                            let separator = graph.neighbor_set_exclusive(&b_side);
                            let mut without_separator = graph.vertex_set();
                            without_separator.remove_set(&separator);
                            let a_side = graph
                                .as_induced_subgraph(without_separator)
                                .connected_component_from(G::VertexSet::singleton(a));
                            if !a_side.clone().intersect_set(&exclude).is_empty() {
                                return None;
                            }
                            if !condition(&a_side, &fixed) {
                                return None;
                            }
                            Some(TakatasRecurranceFrame {
                                a_side,
                                b_side,
                                separator,
                                fixed: fixed.clone(),
                                exclude: exclude.clone(),
                            })
                        })
                        .collect::<Vec<TakatasRecurranceFrame<G>>>()
                })
                .collect(),
            condition,
        }
    }
}

impl<'a, G, F> Iterator for TakatasRecurrance<'a, G, F>
where
    G: AsInducedSubgraph,
    F: FnMut(&G::VertexSet, &G::VertexSet) -> bool,
{
    type Item = G::VertexSet;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let TakatasRecurranceFrame {
                a_side,
                b_side,
                separator,
                fixed,
                exclude,
            } = self.stack.pop()?;

            debug_assert!(self.graph.neighbor_set_exclusive(&a_side) == separator);
            debug_assert!(self.graph.neighbor_set_exclusive(&b_side) == separator);
            debug_assert!(fixed.is_subset_of(&separator));
            debug_assert!(a_side.clone().intersect_set(&exclude).is_empty());

            let mut to_decide = separator.clone();
            to_decide.remove_set(&fixed);

            debug_assert!(to_decide.clone().intersect_set(&exclude).is_empty());

            let v = if let Some(v) = to_decide.iter().next() {
                v
            } else {
                return Some(separator);
            };

            {
                //v excluded in the separator
                let mut new_a_side = a_side.clone();
                new_a_side.insert(v);

                let mut rest = self.graph.vertex_set();
                rest.remove_set(&self.graph.neighbor_set_inclusive(&new_a_side));
                let mut fixed = fixed.clone();
                fixed.insert_set(self.graph.vertex_neighbor_set(v).intersect_set(&exclude));

                self.graph
                    .as_induced_subgraph(rest)
                    .connected_components()
                    .for_each(|b_side| {
                        let separator = self.graph.neighbor_set_exclusive(&b_side);
                        let mut fixed = fixed.clone();
                        fixed.intersect_set(&separator);
                        let mut without_separator = self.graph.vertex_set();
                        without_separator.remove_set(&separator);
                        let a_side = self
                            .graph
                            .as_induced_subgraph(without_separator)
                            .connected_component_from(new_a_side.clone());

                        if !a_side.clone().intersect_set(&exclude).is_empty() {
                            return;
                        }

                        if !(self.condition)(&a_side, &fixed) {
                            return;
                        }

                        self.stack.push(TakatasRecurranceFrame {
                            a_side,
                            b_side,
                            separator,
                            fixed,
                            exclude: exclude.clone(),
                        })
                    });
            }
            {
                //v included in the separator
                let mut new_fixed = fixed.clone();
                new_fixed.insert(v);
                self.stack.push(TakatasRecurranceFrame {
                    a_side,
                    b_side,
                    separator,
                    fixed: new_fixed,
                    exclude,
                })
            }
        }
    }
}
