use crate::bit_set::{BitSet, DecreasingBitSetIterator, BitStore};
use adagraph::Set;

pub struct Trie<E> {
    root: TrieNode<E>,
}

struct TrieNode<E> {
    value: Option<E>,
    children: Vec<Option<TrieNode<E>>>,
}

struct SliceFilterIter<'a, WORD, E>
where
    WORD: BitStore,
{
    i: usize,
    exclude: Option<&'a BitSet<WORD>>,
    slice: &'a [Option<TrieNode<E>>],
}

impl<'a, WORD, E> SliceFilterIter<'a, WORD, E>
where
    WORD: BitStore,
{
    fn new(slice: &'a [Option<TrieNode<E>>], exclude: Option<&'a BitSet<WORD>>) -> Self {
        Self {
            i: 0,
            exclude,
            slice,
        }
    }
}

impl<'a, WORD, E> Iterator for SliceFilterIter<'a, WORD, E>
where
    WORD: BitStore,
{
    type Item = &'a TrieNode<E>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.slice.get(self.i) {
            Some(e) => {
                let old_i = self.i;
                self.i += 1;
                if let Some(exclude) = self.exclude {
                    if exclude.contains(old_i) {
                        return self.next();
                    }
                }
                match e {
                    Some(e) => Some(e),
                    None => self.next(),
                }
            }
            None => None,
        }
    }
}

pub struct TrieSupersetExcludeIterator<'a, WORD, E>
where
    WORD: BitStore,
{
    current_value: BitSet<WORD>,
    values_needed_to_be_included: Vec<usize>,
    cursor: Option<usize>,
    exclude: Option<&'a BitSet<WORD>>,
    trie_iters: Vec<(&'a TrieNode<E>, SliceFilterIter<'a, WORD, E>)>,
}

impl<'a, WORD, E> Iterator for TrieSupersetExcludeIterator<'a, WORD, E>
where
    WORD: BitStore,
{
    type Item = (BitSet<WORD>, &'a E);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((_, iter)) = self.trie_iters.last_mut() {
            if let Some(e) = iter.next() {
                self.current_value.insert(e.children.len() - 1);
                if let Some(cursor) = self.cursor.as_mut() {
                    if self.values_needed_to_be_included[*cursor] == e.children.len() - 1 {
                        if *cursor > 0 {
                            *cursor -= 1;
                        } else {
                            self.cursor = None;
                        }
                    }
                }
                let iter_slice = if let Some(cursor) = self.cursor {
                    &e.children[self.values_needed_to_be_included[cursor]..]
                } else {
                    &e.children[..]
                };
                self.trie_iters
                    .push((e, SliceFilterIter::new(iter_slice, self.exclude)));
                self.next()
            } else {
                let (node, _) = self.trie_iters.pop().unwrap();
                let new_cursor = self.cursor.map(|i| i + 1).unwrap_or_default();
                if Some(&(node.children.len() - 1))
                    == self.values_needed_to_be_included.get(new_cursor)
                {
                    self.cursor = Some(new_cursor)
                }
                match node.value.as_ref() {
                    Some(value) => {
                        let result = Some((self.current_value.clone(), value));
                        self.current_value.remove(node.children.len() - 1);
                        result
                    }
                    None => {
                        self.current_value.remove(node.children.len() - 1);
                        self.next()
                    }
                }
            }
        } else {
            None
        }
    }
}

impl<E> Trie<E> {
    pub fn new(n: usize) -> Self {
        Trie {
            root: TrieNode::new(n),
        }
    }

    pub fn iter<WORD>(&'_ mut self) -> TrieSupersetExcludeIterator<'_, WORD, E>
    where
        WORD: BitStore,
    {
        TrieSupersetExcludeIterator {
            current_value: BitSet::new(),
            values_needed_to_be_included: vec![],
            cursor: None,
            exclude: None,
            trie_iters: vec![(&self.root, SliceFilterIter::new(&self.root.children, None))],
        }
    }

    pub fn superset_iter<WORD>(
        &'_ self,
        set: &BitSet<WORD>,
    ) -> TrieSupersetExcludeIterator<'_, WORD, E>
    where
        WORD: BitStore,
    {
        let values_needed_to_be_included: Vec<usize> = set.iter().collect();
        let cursor = Some(values_needed_to_be_included.len() - 1);
        TrieSupersetExcludeIterator {
            current_value: BitSet::new(),
            values_needed_to_be_included,
            cursor,
            exclude: None,
            trie_iters: vec![(&self.root, SliceFilterIter::new(&self.root.children, None))],
        }
    }

    pub fn superset_and_exclude_iter<'a, WORD>(
        &'a self,
        set: &BitSet<WORD>,
        exclude: &'a BitSet<WORD>,
    ) -> TrieSupersetExcludeIterator<'a, WORD, E>
    where
        WORD: BitStore,
    {
        let values_needed_to_be_included: Vec<usize> = set.iter().collect();
        let cursor = Some(values_needed_to_be_included.len() - 1);
        TrieSupersetExcludeIterator {
            current_value: BitSet::new(),
            values_needed_to_be_included,
            cursor,
            exclude: Some(exclude),
            trie_iters: vec![(
                &self.root,
                SliceFilterIter::new(&self.root.children, Some(exclude)),
            )],
        }
    }

    pub fn insert<WORD>(&mut self, set: &BitSet<WORD>, elem: E) -> Option<E>
    where
        WORD: BitStore,
    {
        self.root.insert(set.iter_decreasing(), elem)
    }
}

impl<E> TrieNode<E> {
    fn new(n: usize) -> Self {
        TrieNode {
            value: None,
            children: (0..n).map(|_| None).collect(),
        }
    }

    pub fn insert<WORD>(
        &mut self,
        mut set_iter: DecreasingBitSetIterator<WORD>,
        elem: E,
    ) -> Option<E>
    where
        WORD: BitStore,
    {
        match set_iter.next() {
            Some(v) => {
                let child = match &mut self.children[v] {
                    Some(child) => child,
                    None => {
                        self.children[v] = Some(TrieNode::new(v + 1));
                        self.children[v].as_mut().unwrap()
                    }
                };

                child.insert(set_iter, elem)
            }
            None => std::mem::replace(&mut self.value, Some(elem)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BitSet;
    type VertexSet = BitSet<u64>;

    #[test]
    fn test_trie() {
        let mut vs1 = VertexSet::new();
        vs1.insert(0).insert(1).insert(2).insert(3).insert(4);

        let mut vs2 = VertexSet::new();
        vs2.insert(2).insert(4);

        let mut vs3 = VertexSet::new();
        vs3.insert(1).insert(5);

        let mut trie = Trie::<()>::new(7);

        println!("{:?}", vs1.iter_decreasing().collect::<Vec<usize>>());

        trie.insert(&vs1, ());
        trie.insert(&vs2, ());
        trie.insert(&vs3, ());

        println!(
            "{:?}",
            trie.root
                .children
                .iter()
                .map(Option::is_some)
                .collect::<Vec<bool>>()
        );

        println!("{}", trie.iter::<u64, 1>().count());

        for (s, _) in trie.iter::<u64, 1>() {
            println!("{}", s);
        }

        println!("iter");
        for (s, _) in trie.superset_iter(VertexSet::new().insert(2)) {
            println!("{}", s);
        }

        println!("iter superset");
        for (s, _) in trie.superset_iter(VertexSet::new().insert(2)) {
            println!("{}", s);
        }

        println!("iter superset and exclude");
        for (s, _) in
            trie.superset_and_exclude_iter(VertexSet::new().insert(2), VertexSet::new().insert(1))
        {
            println!("{}", s);
        }
    }

    #[test]
    fn test_trie_superset_iterator() {
        let mut vs1 = VertexSet::new();
        vs1.insert(0).insert(1).insert(2).insert(3).insert(4);

        let mut vs2 = VertexSet::new();
        vs2.insert(2).insert(4);

        let mut vs3 = VertexSet::new();
        vs3.insert(1).insert(5);

        let mut trie = Trie::<()>::new(7);

        trie.insert(&vs1, ());
        trie.insert(&vs2, ());
        trie.insert(&vs3, ());

        println!(
            "{:?}",
            trie.root
                .children
                .iter()
                .map(Option::is_some)
                .collect::<Vec<bool>>()
        );

        println!("{}", trie.iter::<u64, 1>().count());

        for (s, _) in trie.superset_iter(VertexSet::new().insert(2)) {
            println!("{}", s);
        }
    }
}
