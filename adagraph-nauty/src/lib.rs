#![allow(incomplete_features)]
#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(core_intrinsics)]
#![feature(assert_matches)]

mod bit_set;
pub use bit_set::{BitSet, BitSetArray};

//mod trie;
//pub use trie::Trie;

use std::{
    any::Any,
    collections::{vec_deque, VecDeque},
    iter::{Enumerate, FilterMap, FromIterator, Iterator},
    marker::PhantomData,
    ops::Range,
};

use adagraph::{prelude::*, GraphEditingResult};

use adagraph_induced_subgraph::{AsInducedSubgraph, InducedSubgraph};

use bit_set::{BitStore, IncreasingBitSetIterator};

#[derive(Clone, Debug)]
pub struct NautyGraph {
    n: usize,
    data: BitSetArray<usize>,
}

impl NautyGraph {
    fn new(n: usize) -> Self {
        Self {
            n,
            data: BitSetArray::<usize>::new(n, (n + usize::size() - 1) / usize::size()),
        }
    }
}

impl VertexAddableGraph for NautyGraph {
    fn add_vertex<Args>(&mut self, _: Args) -> GraphEditingResult<&mut Self> {
        self.n += 1;
        self.data
            .reshape(self.n, (self.n + usize::size() - 1) / usize::size());
        Ok(self)
    }
}

impl EdgeAddableGraph for NautyGraph {
    fn add_edge<Args>(&mut self, v: usize, w: usize, _: Args) -> GraphEditingResult<&mut Self> {
        //TODO: check self loops and parallel edges
        if v == w {
            Err(GraphEditingError::SelfLoopEdge("".to_string()))
        } else {
            if v >= self.n {
                Err(GraphEditingError::NonExistingVertex(format!("{}", v)))?;
            }
            if w >= self.n {
                Err(GraphEditingError::NonExistingVertex(format!("{}", w)))?;
            }

            let mut adjv = self.data.get_mut(v);

            if adjv.contains(w) {
                Err(GraphEditingError::ParallelEdge("".to_string()))?;
            }
            adjv.insert(w);
            let mut adjw = self.data.get_mut(w);
            debug_assert!(!adjw.contains(v));
            adjw.insert(v);
            Ok(self)
        }
    }
}

impl EdgeRemovableGraph for NautyGraph {
    fn remove_edge<Args>(&mut self, e: (usize, usize), _: Args) -> GraphEditingResult<&mut Self> {
        let (v, w) = e;
        if v >= self.n {
            return Err(GraphEditingError::NonExistingVertex(format!("{}", v)));
        }
        if w >= self.n {
            return Err(GraphEditingError::NonExistingVertex(format!("{}", w)));
        }
        let mut adjv = self.data.get_mut(v);
        if !adjv.contains(w) {
            Err(GraphEditingError::NonExistingEdge("".to_string()))?;
        }
        adjv.remove(w);
        let mut adjw = self.data.get_mut(w);
        debug_assert!(adjw.contains(v));
        adjw.remove(v);
        Ok(self)
    }
}

impl Graph for NautyGraph {
    type VertexIndex = usize;
    type EdgeIndex = (usize, usize);

    fn adjacent_vertives(
        &self,
        e: <Self as Graph>::EdgeIndex,
    ) -> (<Self as Graph>::VertexIndex, <Self as Graph>::VertexIndex) {
        e
    }
    fn get_edge(&self, v: Self::VertexIndex, w: Self::VertexIndex) -> Option<Self::EdgeIndex> {
        if v >= self.n || w >= self.n {
            None?
        }
        if !self.data.get_ref(v).contains(w) {
            debug_assert!(!self.data.get_ref(w).contains(v));
            None?
        }
        Some((v, w))
    }
}

impl GraphCount for NautyGraph {
    fn n_vertices(&self) -> usize {
        self.n
    }
    fn n_edges(&self) -> usize {
        self.data.iter().map(|d| d.iter().count()).sum::<usize>() / 2
    }

    fn degree(&self, v: Self::VertexIndex) -> usize {
        assert!(v < self.n);
        self.data.get_ref(v).iter().count()
    }
}

pub struct NautyGraphVertices<'a> {
    iter: Range<usize>,
    pd: PhantomData<&'a ()>,
}

impl std::iter::Iterator for NautyGraphVertices<'_> {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

pub struct NautyGraphEdges<'a> {
    graph: &'a NautyGraph,
    v: usize,
    vertices: Range<usize>,
    neighbors: Option<IncreasingBitSetIterator<'a, usize>>,
}

impl<'a> std::iter::Iterator for NautyGraphEdges<'a> {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        match self.neighbors.as_mut()?.next() {
            Some(w) => Some((self.v, w)),
            None => match self.vertices.next() {
                Some(v) => {
                    self.v = v;
                    self.neighbors = Some(self.graph.data.get_ref(v).iter_from(v + 1));
                    self.next()
                }
                None => None,
            },
        }
    }
}

pub struct NautyGraphNeighbors<'a> {
    i: usize,
    iter: IncreasingBitSetIterator<'a, usize>,
}

impl<'a> std::iter::Iterator for NautyGraphNeighbors<'a> {
    type Item = (usize, (usize, usize));
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(j) = self.iter.next() {
            Some((j, (self.i, j)))
        } else {
            None
        }
    }
}

impl GraphIterator for NautyGraph {
    type Vertices<'a> = NautyGraphVertices<'a>;
    type Edges<'a> = NautyGraphEdges<'a>;
    type Neighbors<'a> = NautyGraphNeighbors<'a>;
    fn vertices(&self) -> NautyGraphVertices<'_> {
        NautyGraphVertices {
            iter: (0..self.n),
            pd: PhantomData,
        }
    }
    fn edges(&self) -> Self::Edges<'_> {
        Self::Edges::<'_> {
            v: 0,
            vertices: (1..self.n),
            neighbors: if self.n > 0 {
                Some(self.data.get_ref(0).iter_from(1))
            } else {
                None
            },
            graph: self,
        }
    }

    fn neighbors(&self, i: <Self as Graph>::VertexIndex) -> Self::Neighbors<'_> {
        Self::Neighbors::<'_> {
            i,
            iter: self.data.get_ref(i).iter(),
        }
    }
}

impl VertexSetGraph for NautyGraph {
    type VertexSet = BitSet<usize>;

    fn vertex_set(&self) -> Self::VertexSet {
        let mut result = BitSet::<usize>::new();
        result.fill(self.n);
        result
    }

    fn vertex_neighbor_set(&self, v: Self::VertexIndex) -> Self::VertexSet {
        BitSet {
            data: self.data.get_ref(v).data.to_vec(),
        }
    }
    fn neighbor_set_simple(&self, set: &Self::VertexSet) -> Self::VertexSet {
        let mut result = BitSet::<usize>::new();
        for v in set.iter() {
            result.insert_set(&self.data.get(v));
        }
        result
    }
}

pub struct NautyGraphVertexStoreWrapper;

impl StoreWithKey<usize> for NautyGraphVertexStoreWrapper {
    type Store<V> = NautyGraphVertexStore<V>;
}

pub struct NautyGraphVertexStore<V> {
    offset: usize,
    data: VecDeque<Option<V>>,
}

pub struct NautyGraphVertexStoreFilterMap<V> {
    offset: usize,
    pd: std::marker::PhantomData<V>,
}

impl<V> FnMut<((usize, Option<V>),)> for NautyGraphVertexStoreFilterMap<V> {
    extern "rust-call" fn call_mut(&mut self, ((e, v),): ((usize, Option<V>),)) -> Self::Output {
        match v {
            Some(v) => Some((e + self.offset, v)),
            None => None,
        }
    }
}

impl<V> FnOnce<((usize, Option<V>),)> for NautyGraphVertexStoreFilterMap<V> {
    type Output = Option<(usize, V)>;
    extern "rust-call" fn call_once(mut self, args: ((usize, Option<V>),)) -> Self::Output {
        self.call_mut(args)
    }
}

pub struct NautyGraphVertexStoreRefFilterMap<'a, V> {
    offset: usize,
    pd: std::marker::PhantomData<&'a V>,
}

impl<'a, V> FnMut<((usize, &'a Option<V>),)> for NautyGraphVertexStoreRefFilterMap<'a, V> {
    extern "rust-call" fn call_mut(
        &mut self,
        ((e, v),): ((usize, &'a Option<V>),),
    ) -> Self::Output {
        match v.as_ref() {
            Some(v) => Some((e + self.offset, v)),
            None => None,
        }
    }
}

impl<'a, V> FnOnce<((usize, &'a Option<V>),)> for NautyGraphVertexStoreRefFilterMap<'a, V> {
    type Output = Option<(usize, &'a V)>;
    extern "rust-call" fn call_once(mut self, args: ((usize, &'a Option<V>),)) -> Self::Output {
        self.call_mut(args)
    }
}

pub struct NautyGraphVertexStoreMutFilterMap<'a, V> {
    offset: usize,
    pd: std::marker::PhantomData<&'a V>,
}

impl<'a, V> FnMut<((usize, &'a mut Option<V>),)> for NautyGraphVertexStoreMutFilterMap<'a, V> {
    extern "rust-call" fn call_mut(
        &mut self,
        ((e, v),): ((usize, &'a mut Option<V>),),
    ) -> Self::Output {
        match v.as_mut() {
            Some(v) => Some((e + self.offset, v)),
            None => None,
        }
    }
}

impl<'a, V> FnOnce<((usize, &'a mut Option<V>),)> for NautyGraphVertexStoreMutFilterMap<'a, V> {
    type Output = Option<(usize, &'a mut V)>;
    extern "rust-call" fn call_once(mut self, args: ((usize, &'a mut Option<V>),)) -> Self::Output {
        self.call_mut(args)
    }
}

impl<V> IntoIterator for NautyGraphVertexStore<V> {
    type IntoIter =
        FilterMap<Enumerate<vec_deque::IntoIter<Option<V>>>, NautyGraphVertexStoreFilterMap<V>>;
    type Item = (usize, V);
    fn into_iter(self) -> Self::IntoIter {
        self.data
            .into_iter()
            .enumerate()
            .filter_map(NautyGraphVertexStoreFilterMap {
                offset: self.offset,
                pd: PhantomData,
            })
    }
}

impl<V> FromIterator<(usize, V)> for NautyGraphVertexStore<V> {
    fn from_iter<T: IntoIterator<Item = (usize, V)>>(iter: T) -> Self {
        let temp: Vec<(usize, V)> = iter.into_iter().collect();
        if temp.len() == 0 {
            return NautyGraphVertexStore {
                offset: 0,
                data: Default::default(),
            };
        }
        let offset = temp.iter().map(|(e, _)| *e).min().unwrap();
        let len = temp.iter().map(|(e, _)| *e).max().unwrap() - offset + 1;
        let mut data: VecDeque<Option<V>> = (0..len).map(|_| None).collect();

        for (i, e) in temp {
            *data.get_mut(i - offset).unwrap() = Some(e)
        }

        Self { offset, data }
    }
}

impl<V> Default for NautyGraphVertexStore<V> {
    fn default() -> Self {
        NautyGraphVertexStore {
            offset: 0,
            data: Default::default(),
        }
    }
}

impl<V> Clone for NautyGraphVertexStore<V>
where
    V: Clone,
{
    fn clone(&self) -> Self {
        let Self { offset, data } = self;
        Self {
            offset: *offset,
            data: data.clone(),
        }
    }
}

unsafe impl<V> Store<usize, V> for NautyGraphVertexStore<V> {
    type Iter<'a>
    where
        V: 'a,
    = FilterMap<
        Enumerate<vec_deque::Iter<'a, Option<V>>>,
        NautyGraphVertexStoreRefFilterMap<'a, V>,
    >;
    type IterMut<'a>
    where
        V: 'a,
    = FilterMap<
        Enumerate<vec_deque::IterMut<'a, Option<V>>>,
        NautyGraphVertexStoreMutFilterMap<'a, V>,
    >;

    fn new() -> Self {
        Self::default()
    }

    fn add(&mut self, k: usize, v: V) -> Option<V> {
        if self.data.is_empty() {
            self.offset = k;
        }

        if k < self.offset {
            self.data
                .resize_with(self.data.len() + self.offset - k - 1, || None);
            self.data.rotate_right(self.offset - k - 1);
            self.data.push_front(Some(v));
            self.offset = k;
            return None;
        }

        if k - self.offset >= self.data.len() {
            self.data.resize_with(k - self.offset, || None);
            self.data.push_back(Some(v));
            return None;
        }

        std::mem::replace(self.data.get_mut(k - self.offset).unwrap(), Some(v))
    }

    fn remove(&mut self, k: usize) -> Option<V> {
        if k < self.offset || k - self.offset >= self.data.len() {
            return None;
        }

        let result = std::mem::take(self.data.get_mut(k - self.offset).unwrap());
        while let Some(None) = self.data.back() {
            self.data.pop_back();
        }
        while let Some(None) = self.data.front() {
            self.data.pop_front();
            self.offset += 1;
        }
        result
    }

    fn get(&self, k: usize) -> Option<V>
    where
        V: Copy,
    {
        self.get_ref(k).map(|e| *e)
    }

    fn get_ref(&self, k: usize) -> Option<&V> {
        self.data.get(k - self.offset).map_or(None, |v| v.as_ref())
    }

    fn get_mut(&mut self, k: usize) -> Option<&mut V> {
        self.data
            .get_mut(k - self.offset)
            .map_or(None, |v| v.as_mut())
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.data
            .iter()
            .enumerate()
            .filter_map(NautyGraphVertexStoreRefFilterMap {
                offset: self.offset,
                pd: PhantomData,
            })
    }

    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        self.data
            .iter_mut()
            .enumerate()
            .filter_map(NautyGraphVertexStoreMutFilterMap {
                offset: self.offset,
                pd: PhantomData,
            })
    }

    fn get_mul_mut<S>(&mut self, set: &S) -> Vec<(usize, Option<&mut V>)>
    where
        S: Set<Element = usize>,
    {
        let mut offset = self.offset;
        let (mut front, mut back) = self.data.as_mut_slices();

        let mut result: Vec<(usize, Option<&mut V>)> = Vec::new();

        let mut iter = set.iter_increasing().peekable();

        while let Some(i) = iter.peek() {
            if offset <= *i {
                break;
            }
            result.push((*i, None));
            iter.next();
        }

        while let Some(i) = iter.peek() {
            if offset + front.len() <= *i {
                break;
            }
            let (toss, new_front) = front.split_at_mut(*i - offset + 1);
            result.push((*i, toss.last_mut().map(|e| e.as_mut()).unwrap()));
            front = new_front;
            offset = *i + 1;
            iter.next();
        }

        offset += front.len();

        while let Some(i) = iter.peek() {
            if offset + back.len() <= *i {
                break;
            }
            let (toss, new_back) = back.split_at_mut(*i - offset + 1);
            result.push((*i, toss.last_mut().map(|e| e.as_mut()).unwrap()));
            back = new_back;
            offset = *i + 1;
            iter.next();
        }

        for i in iter {
            result.push((i, None));
        }

        result
    }

    fn clone(&self) -> Self
    where
        V: Clone,
    {
        Self {
            offset: self.offset,
            data: self.data.clone(),
        }
    }
}

impl AsInducedSubgraph for NautyGraph {
    type OriginalGraph = Self;

    fn as_induced_subgraph<'a>(
        &'a self,
        selected: Self::VertexSet,
    ) -> InducedSubgraph<'a, Self::OriginalGraph> {
        InducedSubgraph::new(self, selected)
    }
}

pub struct NautyGraphInducedSubgraphIntoIter<V> {
    iter: Enumerate<std::vec::IntoIter<Option<V>>>,
}

impl<V> Iterator for NautyGraphInducedSubgraphIntoIter<V> {
    type Item = (usize, V);
    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some((i, mv)) => match mv {
                Some(v) => Some((i, v)),
                None => self.next(),
            },
            None => None,
        }
    }
}

pub struct NautyGraphInducedSubgraphIter<'a, V> {
    iter: Enumerate<std::slice::Iter<'a, Option<V>>>,
}

impl<'a, V> Iterator for NautyGraphInducedSubgraphIter<'a, V> {
    type Item = (usize, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some((i, mv)) => match mv {
                Some(v) => Some((i, v)),
                None => self.next(),
            },
            None => None,
        }
    }
}

pub struct NautyGraphInducedSubgraphIterMut<'a, V> {
    iter: Enumerate<std::slice::IterMut<'a, Option<V>>>,
}

impl<'a, V> Iterator for NautyGraphInducedSubgraphIterMut<'a, V> {
    type Item = (usize, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some((i, mv)) => match mv {
                Some(v) => Some((i, v)),
                None => self.next(),
            },
            None => None,
        }
    }
}

impl BreathFirstSearch for NautyGraph {
    type BreathFirstSearchIterator<'a> =
        adagraph::algos::DefaultBreathFirstSearchIteratorVertexSet<'a, Self>;
}

impl ConnectedComponentFrom for NautyGraph {}

impl ConnectedComponents for NautyGraph {}

impl BuildableGraph for NautyGraph {
    type Builder = NautyGraphBuilder;
}

pub struct NautyGraphBuilder {
    result: std::result::Result<NautyGraph, GraphEditingError>,
}

impl GraphBuilder for NautyGraphBuilder {
    type Graph = NautyGraph;
    type VertexSet = BitSet<usize>;
    fn new<Args>(args: Args) -> Self
    where
        Args: 'static,
    {
        let args: &dyn Any = &args;
        if args.is::<u8>() {
            Self::new_with_n(*args.downcast_ref::<u8>().unwrap() as usize)
        } else if args.is::<u16>() {
            Self::new_with_n(*args.downcast_ref::<u16>().unwrap() as usize)
        } else if args.is::<u32>() {
            Self::new_with_n(*args.downcast_ref::<u32>().unwrap() as usize)
        } else if args.is::<u64>() {
            Self::new_with_n(*args.downcast_ref::<u64>().unwrap() as usize)
        } else if args.is::<u128>() {
            Self::new_with_n(*args.downcast_ref::<u128>().unwrap() as usize)
        } else if args.is::<usize>() {
            Self::new_with_n(*args.downcast_ref::<usize>().unwrap())
        } else if args.is::<i8>() {
            Self::new_with_n(*args.downcast_ref::<i8>().unwrap() as usize)
        } else if args.is::<i16>() {
            Self::new_with_n(*args.downcast_ref::<i16>().unwrap() as usize)
        } else if args.is::<i32>() {
            Self::new_with_n(*args.downcast_ref::<i32>().unwrap() as usize)
        } else if args.is::<i64>() {
            Self::new_with_n(*args.downcast_ref::<i64>().unwrap() as usize)
        } else if args.is::<i128>() {
            Self::new_with_n(*args.downcast_ref::<i128>().unwrap() as usize)
        } else if args.is::<isize>() {
            Self::new_with_n(*args.downcast_ref::<isize>().unwrap() as usize)
        } else if args.is::<()>() {
            Self::new_with_n(0)
        } else {
            Self {
                result: Err(GraphEditingError::UnsupportedArgument(format!(
                    "Unsupported argument type{:?}",
                    std::any::type_name::<Args>()
                ))),
            }
        }
    }

    fn vertex_set(&self) -> Self::VertexSet {
        match &self.result {
            Ok(graph) => graph.vertex_set(),
            Err(_) => BitSet::new(),
        }
    }

    fn add_vertex<Args>(mut self, args: Args) -> Self
    where
        Args: 'static,
    {
        match &mut self.result {
            Ok(graph) => match graph.add_vertex(args) {
                Ok(_) => self,
                Err(e) => {
                    self.result = Err(e);
                    self
                }
            },
            Err(_) => self,
        }
    }

    fn add_edge<Args>(mut self, v: usize, w: usize, args: Args) -> Self
    where
        Args: 'static,
    {
        match &mut self.result {
            Ok(graph) => match graph.add_edge(v, w, args) {
                Ok(_) => self,
                Err(e) => {
                    self.result = Err(e);
                    self
                }
            },
            Err(_) => self,
        }
    }

    fn build(self) -> std::result::Result<NautyGraph, GraphEditingError> {
        self.result
    }
}

impl NautyGraphBuilder {
    fn new_with_n(n: usize) -> Self {
        Self {
            result: Ok(NautyGraph::new(n)),
        }
    }
}

pub type NautyGraphVertexSet = BitSet<u64>;

#[cfg(test)]
mod tests {
    use std::assert_matches;

    use adagraph::Set;

    #[test]
    fn test_graph_iterator() {
        use super::*;
        let graph = NautyGraph::builder(4)
            .add_edge(0, 1, ())
            .add_edge(1, 2, ())
            .add_edge(2, 3, ())
            .add_edge(3, 0, ())
            .add_edge(1, 3, ())
            .build()
            .unwrap();

        println!("{:?}", graph.data.get_ref(0));
        println!("{:?}", graph.data.get_ref(1));
        println!("{:?}", graph.data.get_ref(2));

        let vertices = graph
            .vertices()
            .map(|v| format!("{}", v))
            .collect::<Vec<String>>()
            .join(",");
        let edges = graph
            .edges()
            .map(|v| format!("{:?}", v))
            .collect::<Vec<String>>()
            .join(",");
        let neighbors = graph
            .neighbors(0)
            .map(|v| format!("{:?}", v))
            .collect::<Vec<String>>()
            .join(",");

        assert_eq!(Some(3), graph.data.get_ref(2).iter_from(3).next());
        assert_eq!("0,1,2,3", vertices);
        assert_eq!("(0, 1),(0, 3),(1, 2),(1, 3),(2, 3)", edges);
        assert_eq!("(1, (0, 1)),(3, (0, 3))", neighbors);
    }

    #[test]
    fn test_nauty_induced() {
        use super::*;
        use adagraph_induced_subgraph::AsInducedSubgraph;

        let graph = NautyGraph::builder(9)
            .add_edge(0, 1, ())
            .add_edge(0, 8, ())
            .add_edge(1, 2, ())
            .add_edge(2, 3, ())
            .add_edge(3, 0, ())
            .add_edge(1, 3, ())
            .add_edge(1, 8, ())
            .build()
            .unwrap();

        println!("{:?}", graph.vertex_neighbor_set(0));

        println!(
            "{:?}",
            graph
                .as_induced_subgraph(BitSet::empty().insert(1).insert(8).clone())
                .vertex_neighbor_set(1)
        );
    }

    #[test]
    fn test_nauty_store() {
        use super::{BitSet, NautyGraphVertexStore};
        use adagraph::Store;
        use std::assert_matches::assert_matches;

        let mut dut = NautyGraphVertexStore::<&str>::new();

        dut.add(20, "1 more");

        assert_eq!(dut.offset, 20);
        assert_eq!(dut.data.len(), 1);

        dut.add(10, "test");

        assert_eq!(dut.offset, 10);
        assert_eq!(dut.data.len(), 11);

        dut.add(11, "test1");

        assert_eq!(dut.offset, 10);
        assert_eq!(dut.data.len(), 11);

        dut.add(30, "2 more");

        assert_eq!(dut.offset, 10);
        assert_eq!(dut.data.len(), 21);

        assert_matches!(
            dut.get_mul_mut(BitSet::<u64>::empty().insert(11).insert(30))
                .as_slice(),
            [(11, Some(&mut "test1")), (30, Some(&mut "2 more"))]
        );

        assert_eq!(dut.remove(20), Some("1 more"));

        assert_eq!(dut.offset, 10);
        assert_eq!(dut.data.len(), 21);

        assert_eq!(dut.remove(30), Some("2 more"));

        assert_eq!(dut.offset, 10);
        assert_eq!(dut.data.len(), 2);
    }
}
