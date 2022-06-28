#![feature(drain_filter)]
#![feature(assert_matches)]

mod disjoint_set;

use adagraph::prelude::*;
use adagraph_adjacency_graph::AdjGraph;
use adagraph_induced_subgraph::AsInducedSubgraph;
use disjoint_set::{DisjointSet, DisjointSetMutRef, DisjointSetRef};
use std::{
    convert::TryInto,
    fmt::{self, Debug},
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum DisjointSetType<VS>
where
    VS: Set,
{
    None,
    Separator,
    Component(Option<VS>),
    FullComponent,
}

impl<VS> Default for DisjointSetType<VS>
where
    VS: Set,
{
    fn default() -> Self {
        Self::None
    }
}

pub struct Separator<V, VS, S>
where
    V: Copy + Ord,
    VS: Set<Element = V>,
    S: StoreWithKey<V>,
{
    separator_vertex: Option<V>,
    separator_size: usize,
    number_full_components: usize,
    sets: DisjointSet<V, VS, S, DisjointSetType<VS>>,
}

impl<V, VS, S> Debug for Separator<V, VS, S>
where
    V: Copy + Ord + Debug,
    VS: Set<Element = V> + Debug,
    S: StoreWithKey<V>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Separator")
            .field("separator_vertex", &self.separator_vertex)
            .field("separator_size", &self.separator_size)
            .field("number_full_components", &self.number_full_components)
            .field("sets", &self.sets)
            .finish()
    }
}

impl<V, VS, S> Clone for Separator<V, VS, S>
where
    V: Copy + Ord,
    VS: Set<Element = V>,
    S: StoreWithKey<V>,
{
    fn clone(&self) -> Self {
        Self {
            separator_vertex: self.separator_vertex,
            separator_size: self.separator_size,
            number_full_components: self.number_full_components,
            sets: self.sets.clone(),
        }
    }
}

impl<V, VS, S> Separator<V, VS, S>
where
    V: Copy + Ord,
    VS: Set<Element = V>,
    S: StoreWithKey<V>,
{
    fn add_edge(mut self, v: V, u: V) -> (Option<Self>, Option<Self>) {
        let sets = self.sets.get_mul_mut(VS::singleton(v).insert(u));
        if sets.len() < 2 {
            return (Some(self), None);
        }
        let sets: [DisjointSetMutRef<'_, V, VS, DisjointSetType<VS>>; 2] =
            match sets.try_into() {
                Ok(e) => e,
                Err(_) => unreachable!(),
            };

        let [DisjointSetMutRef(_, v_set, mut v_type), DisjointSetMutRef(_, _, mut u_type)] = sets;

        let (v, u) = if v_set.contains(v) { (v, u) } else { (u, v) };

        match (&mut v_type, &mut u_type) {
            (DisjointSetType::Separator, DisjointSetType::FullComponent)
            | (DisjointSetType::FullComponent, DisjointSetType::Separator) => (Some(self), None),
            (DisjointSetType::Separator, DisjointSetType::Component(neighbors)) => {
                let mut unwrapped_neighbors =
                    std::mem::take(neighbors).unwrap_or_else(|| unreachable!());
                unwrapped_neighbors.insert(v);
                if self.separator_size == unwrapped_neighbors.len() {
                    self.number_full_components += 1;
                    *u_type = DisjointSetType::FullComponent;
                } else {
                    *neighbors = Some(unwrapped_neighbors);
                }
                (Some(self), None)
            }
            (DisjointSetType::Component(neighbors), DisjointSetType::Separator) => {
                let mut unwrapped_neighbors =
                    std::mem::take(neighbors).unwrap_or_else(|| unreachable!());
                unwrapped_neighbors.insert(u);
                if self.separator_size == unwrapped_neighbors.len() {
                    self.number_full_components += 1;
                    *v_type = DisjointSetType::FullComponent;
                } else {
                    *neighbors = Some(unwrapped_neighbors);
                }
                (Some(self), None)
            }
            (DisjointSetType::Component(v_neighbors), DisjointSetType::Component(u_neighbors)) => {
                let mut unwrapped_neighbors =
                    std::mem::take(v_neighbors).unwrap_or_else(|| unreachable!());
                unwrapped_neighbors
                    .insert_set(u_neighbors.as_ref().unwrap_or_else(|| unreachable!()));
                if self.separator_size == unwrapped_neighbors.len() {
                    self.number_full_components += 1;
                    self.sets.union(v, u, |_, _| DisjointSetType::FullComponent);
                } else {
                    self.sets.union(v, u, |_, _| {
                        DisjointSetType::Component(Some(unwrapped_neighbors))
                    });
                }
                (Some(self), None)
            }
            (DisjointSetType::Component(_), DisjointSetType::FullComponent)
            | (DisjointSetType::FullComponent, DisjointSetType::Component(_)) => {
                self.sets.union(v, u, |_, _| DisjointSetType::FullComponent);
                (Some(self), None)
            }
            (DisjointSetType::FullComponent, DisjointSetType::FullComponent) => {
                let result = if self.number_full_components > 2 {
                    let mut result = self.clone();
                    result
                        .sets
                        .union(v, u, |_, _| DisjointSetType::FullComponent);
                    result.number_full_components -= 1;
                    Some(result)
                } else {
                    None
                };
                (result, Some(self))
            }
            _ => unreachable!(),
        }
    }

    fn with_vertex(&mut self, u: V, v: V, graph: &AdjGraph<V, VS, S::Store<VS>>) -> Option<Self> {
        let separator = self
            .separator_vertex
            .map(|sv| {
                self.sets
                    .get_mut(sv)
                    .unwrap_or_else(|| unreachable!())
                    .1
                    .clone()
            })
            .unwrap_or(VS::empty());
        let mut new_separator = separator.clone();
        new_separator.insert(v);

        let original_set = self
            .sets
            .get_mut(v)
            .unwrap_or_else(|| unreachable!())
            .1
            .clone();
        let mut set = original_set.clone();
        set.remove(v);

        let mut sets = Self::reconstruct_component(&set, self.separator_size + 1, graph)?;
        let mut result = self.clone();
        result.number_full_components = 2;
        result
            .sets
            .iter_mut()
            .for_each(|DisjointSetMutRef(_, set, e)| match e {
                DisjointSetType::FullComponent => {
                    if !set.contains(u) && !set.contains(v) {
                        *e = DisjointSetType::Component(Some(separator.clone()));
                    }
                }
                _ => (),
            });
        debug_assert!(!sets.iter().any(|DisjointSetRef(_, _, e)| {
            if let DisjointSetType::None = e {
                true
            } else {
                false
            }
        }));
        debug_assert!(!result.sets.iter().any(|DisjointSetRef(_, _, e)| {
            if let DisjointSetType::None = e {
                true
            } else {
                false
            }
        }));
        result.sets.intersect_sets(
            &mut sets,
            v,
            |DisjointSetMutRef(_, _, _), DisjointSetMutRef(_, _, e), _| std::mem::take(e),
            |_, _, _| DisjointSetType::Separator,
        );
        debug_assert!(!result.sets.iter().any(|DisjointSetRef(_, _, e)| {
            if let DisjointSetType::None = e {
                true
            } else {
                false
            }
        }));

        if let Some(sv) = self.separator_vertex {
            result.sets.union(v, sv, |_, _| DisjointSetType::Separator);
        } else {
            result.separator_vertex = Some(v);
        }
        result.separator_size += 1;
        Some(result)
    }

    fn combine(
        &mut self,
        other: &mut Self,
        v: V,
        w: V,
        graph: &AdjGraph<V, VS, S::Store<VS>>,
    ) -> Option<Self> {
        let v_set = self
            .sets
            .get_mut(v)
            .unwrap_or_else(|| unreachable!())
            .1
            .clone();
        let w_set = self
            .sets
            .get_mut(w)
            .unwrap_or_else(|| unreachable!())
            .1
            .clone();
        debug_assert!(v_set.contains(v));
        debug_assert!(w_set.contains(w));
        let separator = if let Some(v) = self.separator_vertex {
            self.sets
                .get_mut(v)
                .unwrap_or_else(|| unreachable!())
                .1
                .clone()
        } else {
            VS::empty()
        };
        let other_v_set = other
            .sets
            .get_mut(v)
            .unwrap_or_else(|| unreachable!())
            .1
            .clone();
        let other_w_set = other
            .sets
            .get_mut(w)
            .unwrap_or_else(|| unreachable!())
            .1
            .clone();
        debug_assert!(other_v_set.contains(v));
        debug_assert!(other_w_set.contains(w));
        let other_separator = if let Some(v) = other.separator_vertex {
            other
                .sets
                .get_mut(v)
                .unwrap_or_else(|| unreachable!())
                .1
                .clone()
        } else {
            VS::empty()
        };

        let mut new_separator = separator.clone();
        new_separator.insert_set(&other_separator);
        let separator_size = new_separator.len();

        let mut vw_set = v_set.clone();
        vw_set.intersect_set(&other_w_set);

        let mut wv_set = w_set.clone();
        wv_set.intersect_set(&other_v_set);

        let set = if !vw_set.is_empty()
            && wv_set.is_empty()
            && graph.neighbor_set_exclusive(&vw_set).len() == separator_size
        {
            vw_set
        } else if vw_set.is_empty()
            && !wv_set.is_empty()
            && graph.neighbor_set_exclusive(&wv_set).len() == separator_size
        {
            wv_set
        } else {
            return None;
        };

        let mut components = Self::reconstruct_component(&set, new_separator.len(), graph)?;
        let mut result = self.clone();

        if other.separator_vertex.is_some() {
            result
                .sets
                .iter_mut()
                .for_each(|DisjointSetMutRef(_, _, e)| {
                    if let DisjointSetType::FullComponent = e {
                        *e = DisjointSetType::Component(Some(separator.clone()));
                    }
                });
        }

        let u = set.iter().next().unwrap_or_else(|| unreachable!());
        result.sets.intersect_sets(
            &mut other.sets,
            u,
            |_, DisjointSetMutRef(_, _, e), _| match e {
                DisjointSetType::FullComponent => {
                    if new_separator != other_separator {
                        DisjointSetType::Component(Some(other_separator.clone()))
                    } else {
                        e.clone()
                    }
                }
                _ => e.clone(),
            },
            |_, _, _| unreachable!(),
        );

        result
            .sets
            .union_set(&new_separator, DisjointSetType::Separator); // I can combine the two sets

        result.sets.intersect_sets(
            &mut components,
            u,
            |_, DisjointSetMutRef(_, _, e), _| e.clone(),
            |_, _, _| unreachable!(),
        ); //TODO check if it is a full component
        result
            .sets
            .union(v, w, |_, _| DisjointSetType::FullComponent);

        if let (Some(v1), Some(v2)) = (self.separator_vertex, other.separator_vertex) {
            result.sets.union(v1, v2, |_, _| DisjointSetType::Separator);
        }

        if result.separator_vertex.is_none() {
            if other.separator_vertex.is_none() {
                unreachable!()
            } else {
                result.separator_vertex = other.separator_vertex;
            }
        }
        result.separator_size = separator_size;
        result.number_full_components = 2;
        debug_assert!(!result.sets.iter().any(|DisjointSetRef(_, _, e)| {
            if let DisjointSetType::None = e {
                true
            } else {
                false
            }
        }));
        Some(result)
    }

    fn reconstruct_component(
        set: &VS,
        new_separator_size: usize,
        graph: &AdjGraph<V, VS, S::Store<VS>>,
    ) -> Option<DisjointSet<V, VS, S, DisjointSetType<VS>>> {
        let mut result = DisjointSet::new(set, |_| DisjointSetType::None);
        let mut there_is_a_fullcomponent = false;

        for c in graph
            .as_induced_subgraph(set.clone())
            .connected_components()
        {
            let neighbors = graph.neighbor_set_exclusive(&c);

            if neighbors.len() == new_separator_size {
                if !there_is_a_fullcomponent {
                    result.union_set(&c, DisjointSetType::FullComponent);
                    there_is_a_fullcomponent = true;
                } else {
                    return None;
                }
            } else {
                result.union_set(&c, DisjointSetType::Component(Some(neighbors.clone())));
            }
        }

        if there_is_a_fullcomponent {
            Some(result)
        } else {
            None
        }
    }
}

pub struct Stage<'a, V, VS, S>
where
    V: Copy + Ord,
    VS: Set<Element = V>,
    S: StoreWithKey<V>,
{
    graph: &'a AdjGraph<V, VS, S::Store<VS>>,
    v: V,
    u: V,
    vu_separators: Vec<Separator<V, VS, S>>,
}

impl<'a, V, VS, S> Stage<'a, V, VS, S>
where
    V: Copy + Ord,
    VS: Set<Element = V>,
    S: StoreWithKey<V>,
{
    pub fn new(graph: &'a AdjGraph<V, VS, S::Store<VS>>, v: V, u: V) -> Self {
        Self {
            graph,
            v,
            u,
            vu_separators: Vec::new(),
        }
    }

    pub fn process<Source, ToExpand, AddVertex, Combine>(
        &mut self,
        source: Source,
        to_expand: ToExpand,
        add_vertex: AddVertex,
        combine: Combine,
    ) -> StageIterator<'a, '_, V, VS, S, Source, ToExpand, AddVertex, Combine>
    where
        Source: Iterator<Item = Separator<V, VS, S>>,
        ToExpand: FnMut(&mut Separator<V, VS, S>) -> bool,
        AddVertex: FnMut(&mut Separator<V, VS, S>, V) -> bool,
        Combine: FnMut(&mut Separator<V, VS, S>, &mut Separator<V, VS, S>) -> bool,
    {
        StageIterator {
            stage: self,
            to_expand,
            add_vertex,
            combine,
            source,
            expansion_phase: SeparatorExpanderPhase::None,
        }
    }
}

fn test_separator_consistency<V, VS, S>(separator: &mut Separator<V, VS, S>, graph: &AdjGraph<V, VS, S::Store<VS>>) -> bool
where
    V: Copy + Ord,
    VS: Set<Element = V>,
    S: StoreWithKey<V>,
{
    if let Some(sv) = separator.separator_vertex {
        let separator_vertices = separator.sets.get_mut(sv).unwrap().1.clone();
        for c in graph
            .as_induced_subgraph(graph.vertex_set().remove_set(&separator_vertices).clone())
            .connected_components()
        {
            let neighbors = graph.neighbor_set_exclusive(&c);
            if neighbors.len() == separator_vertices.len() {
                if let DisjointSetType::FullComponent =
                    separator.sets.get_mut(c.iter().next().unwrap()).unwrap().2
                {
                } else {
                    return false;
                }
            } else {
                if let DisjointSetType::Component(other_neighbors) =
                    separator.sets.get_mut(c.iter().next().unwrap()).unwrap().2
                {
                    let other_neighbors = (other_neighbors.as_ref().unwrap()).clone();
                    if neighbors != other_neighbors {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }
    } else {
        for c in graph
            .as_induced_subgraph(graph.vertex_set())
            .connected_components()
        {
            if let DisjointSetType::FullComponent =
                separator.sets.get_mut(c.iter().next().unwrap()).unwrap().2
            {
            } else {
                return false;
            }
        }
    }
    true
}

enum SeparatorExpanderPhase<V, VS, S>
where
    V: Copy + Ord,
    VS: Set<Element = V>,
    S: StoreWithKey<V>,
{
    FirstVertex(Separator<V, VS, S>),
    SecondVertex(Separator<V, VS, S>),
    Combine(Separator<V, VS, S>, usize),
    None,
}

impl<V, VS, S> Default for SeparatorExpanderPhase<V, VS, S>
where
    V: Copy + Ord,
    VS: Set<Element = V>,
    S: StoreWithKey<V>,
{
    fn default() -> Self {
        Self::None
    }
}

pub struct StageIterator<'a, 'b, V, VS, S, Source, ToExpand, AddVertex, Combine>
where
    V: Copy + Ord,
    VS: Set<Element = V>,
    S: StoreWithKey<V>,
    Source: Iterator<Item = Separator<V, VS, S>>,
    ToExpand: FnMut(&mut Separator<V, VS, S>) -> bool,
    AddVertex: FnMut(&mut Separator<V, VS, S>, V) -> bool,
    Combine: FnMut(&mut Separator<V, VS, S>, &mut Separator<V, VS, S>) -> bool,
    'b: 'a,
{
    stage: &'a mut Stage<'b, V, VS, S>,
    source: Source,
    to_expand: ToExpand,
    add_vertex: AddVertex,
    combine: Combine,
    expansion_phase: SeparatorExpanderPhase<V, VS, S>,
}

impl<'a, 'b, V, VS, S, Source, ToExpand, AddVertex, Combine> Iterator
    for StageIterator<'a, 'b, V, VS, S, Source, ToExpand, AddVertex, Combine>
where
    V: Copy + Ord,
    VS: Set<Element = V>,
    S: StoreWithKey<V>,
    Source: Iterator<Item = Separator<V, VS, S>>,
    ToExpand: FnMut(&mut Separator<V, VS, S>) -> bool,
    AddVertex: FnMut(&mut Separator<V, VS, S>, V) -> bool,
    Combine: FnMut(&mut Separator<V, VS, S>, &mut Separator<V, VS, S>) -> bool,
    'b: 'a,
{
    type Item = Separator<V, VS, S>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match std::mem::take(&mut self.expansion_phase) {
                SeparatorExpanderPhase::FirstVertex(mut separator) => {
                    let result = if (self.add_vertex)(&mut separator, self.stage.v) {
                        separator.with_vertex(self.stage.u, self.stage.v, &self.stage.graph)
                    } else {
                        None
                    };

                    self.expansion_phase = SeparatorExpanderPhase::SecondVertex(separator);

                    if let Some(mut result) = result {
                        debug_assert!(test_separator_consistency(&mut result, self.stage.graph));
                        return Some(result);
                    } else {
                        continue;
                    }
                }
                SeparatorExpanderPhase::SecondVertex(mut separator) => {
                    let result = if (self.add_vertex)(&mut separator, self.stage.u) {
                        separator.with_vertex(self.stage.v, self.stage.u, &self.stage.graph)
                    } else {
                        None
                    };

                    self.expansion_phase = SeparatorExpanderPhase::Combine(separator, 0);

                    if let Some(mut result) = result {
                        debug_assert!(test_separator_consistency(&mut result, self.stage.graph));
                        return Some(result);
                    } else {
                        continue;
                    }
                }
                SeparatorExpanderPhase::Combine(mut separator, i) => {
                    if let Some(vu_separator) = self.stage.vu_separators.get_mut(i) {
                        let result = if (self.combine)(&mut separator, vu_separator) {
                            separator.combine(
                                vu_separator,
                                self.stage.v,
                                self.stage.u,
                                &self.stage.graph,
                            )
                        } else {
                            None
                        };

                        self.expansion_phase = SeparatorExpanderPhase::Combine(separator, i + 1);

                        if let Some(mut result) = result {
                            debug_assert!(test_separator_consistency(
                                &mut result,
                                self.stage.graph
                            ));
                            return Some(result);
                        } else {
                            continue;
                        }
                    } else {
                        self.expansion_phase = SeparatorExpanderPhase::None;
                        self.stage.vu_separators.push(separator);
                        continue;
                    }
                }
                SeparatorExpanderPhase::None => {
                    let separator = self.source.next()?;
                    let (result, vu_separator) = separator.add_edge(self.stage.v, self.stage.u);
                    if let Some(mut vu_separator) = vu_separator {
                        if (self.to_expand)(&mut vu_separator) {
                            self.expansion_phase =
                                SeparatorExpanderPhase::FirstVertex(vu_separator);
                        }
                    }
                    if let Some(mut result) = result {
                        debug_assert!(test_separator_consistency(&mut result, self.stage.graph));
                        return Some(result);
                    } else {
                        continue;
                    }
                }
            }
        }
    }
}

pub fn enumerate_conditional_separators<V, VS, S, I, F>(vs: VS, edges: I, condition: F) -> Vec<VS>
where
    V: Copy + Ord,
    VS: Set<Element = V>,
    S: StoreWithKey<V>,
    I: Iterator<Item = (V, V)>,
    F: Fn(&VS) -> bool,
{
    let mut separators: Vec<Separator<V, VS, S>> = vec![Separator {
        separator_vertex: None,
        separator_size: 0,
        number_full_components: vs.len(),
        sets: DisjointSet::<V, VS, S, DisjointSetType<VS>>::new(&vs, |_| DisjointSetType::FullComponent),
    }];

    let mut graph = AdjGraph::<V, VS, S::Store<VS>>::new(vs);
    for (v, u) in edges {
        graph.add_edge(v, u);
        let mut stage = Stage::new(&graph, v, u);
        separators = stage
            .process(
                separators.into_iter(),
                |_| true,
                |s, v| {
                    s.separator_vertex
                        .map(|sv| {
                            let mut ss = s.sets.get_mut(sv).unwrap().1.clone();
                            ss.insert(v);
                            condition(&ss)
                        })
                        .unwrap_or(true)
                },
                |s1, s2| {
                    s1.separator_vertex
                        .map(|vs1| {
                            s2.separator_vertex
                                .map(|vs2| {
                                    let mut ss1 = s1.sets.get_mut(vs1).unwrap().1.clone();
                                    ss1.insert_set(s2.sets.get_mut(vs2).unwrap().1);
                                    condition(&ss1)
                                })
                                .unwrap_or(true)
                        })
                        .unwrap_or(true)
                },
            )
            .collect();
    }

    separators
        .into_iter()
        .map(|mut s| {
            s.separator_vertex
                .map(|vs| s.sets.get_mut(vs).unwrap().1.clone())
                .unwrap_or(VS::empty())
        })
        .collect()
}
