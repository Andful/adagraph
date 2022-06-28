use std::{cmp::Ordering, fmt::Debug};

use adagraph::prelude::*;
pub struct DisjointSetMutRef<'a, V, VS, E>(pub V, pub &'a VS, pub &'a mut E)
where
    V: Ord + Copy,
    VS: Set<Element = V>;

pub struct DisjointSetRef<'a, V, VS, E>(pub V, pub &'a VS, pub &'a E)
where
    V: Ord + Copy,
    VS: Set<Element = V>;

impl<'a, V, VS, E> Debug for DisjointSetRef<'a, V, VS, E>
where
    V: Ord + Copy + Debug,
    VS: Set<Element = V> + Debug,
    E: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}", self))?;
        Ok(())
    }
}

enum DisjointSetNode<V, VS, E>
where
    V: Ord + Copy,
    VS: Set<Element = V>,
{
    Root { element: E, set: VS },
    Child { parent: V },
}

impl<V, VS, E> Clone for DisjointSetNode<V, VS, E>
where
    V: Ord + Copy,
    VS: Set<Element = V>,
    E: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Self::Root { element, set } => Self::Root {
                element: element.clone(),
                set: set.clone(),
            },
            Self::Child { parent } => Self::Child { parent: *parent },
        }
    }
}

pub struct DisjointSet<V, VS, S, E>
where
    V: Copy + Ord,
    VS: Set<Element = V>,
    S: StoreWithKey<V>,
{
    elements: S::Store<DisjointSetNode<V, VS, E>>,
}

impl<V, VS, S, E> Debug for DisjointSet<V, VS, S, E>
where
    V: Copy + Ord + Debug,
    VS: Set<Element = V> + Debug,
    S: StoreWithKey<V>,
    E: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{{ {} }}",
            self.iter()
                .map(|e| format!("{:?}", e))
                .collect::<Vec<String>>()
                .join(",")
        ))
    }
}

impl<V, VS, S, E> Clone for DisjointSet<V, VS, S, E>
where
    V: Copy + Ord,
    VS: Set<Element = V>,
    S: StoreWithKey<V>,
    E: Clone,
{
    fn clone(&self) -> Self {
        Self {
            elements: self.elements.clone(),
        }
    }
}

impl<V, VS, S, E> DisjointSet<V, VS, S, E>
where
    V: Copy + Ord,
    VS: Set<Element = V>,
    S: StoreWithKey<V>,
{
    pub fn new<F>(set: &VS, mut f: F) -> Self
    where
        F: FnMut(V) -> E,
    {
        let elements: S::Store<DisjointSetNode<V, VS, E>> = set
            .iter()
            .map(|v| {
                (
                    v,
                    DisjointSetNode::Root {
                        element: f(v),
                        set: VS::singleton(v),
                    },
                )
            })
            .collect();

        Self { elements }
    }

    pub fn get_mut(&mut self, v: V) -> Option<DisjointSetMutRef<V, VS, E>> {
        self.elements.get_ref(v)?;
        let root = self.get_internal(v);
        if let DisjointSetNode::Root { element, set } = self
            .elements
            .get_mut(root)
            .unwrap_or_else(|| unreachable!())
        {
            Some(DisjointSetMutRef(root, set, element))
        } else {
            unreachable!()
        }
    }

    pub fn get_mul_mut(&mut self, vs: &VS) -> Vec<DisjointSetMutRef<V, VS, E>> {
        let roots = vs
            .iter()
            .map(|v| self.get_internal(v))
            .collect::<VS>();
        self.elements
            .get_mul_mut(&roots)
            .into_iter()
            .map(|(root, element)| {
                if let DisjointSetNode::Root { set, element } =
                    element.unwrap_or_else(|| unreachable!())
                {
                    DisjointSetMutRef(root, &*set, element)
                } else {
                    unreachable!()
                }
            })
            .collect()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = DisjointSetMutRef<V, VS, E>> {
        self.elements.iter_mut().filter_map(|(v, e)| {
            if let DisjointSetNode::Root { set, element } = e {
                Some(DisjointSetMutRef(v, set, element))
            } else {
                None
            }
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = DisjointSetRef<V, VS, E>> {
        self.elements.iter().filter_map(|(v, e)| {
            if let DisjointSetNode::Root { set, element } = e {
                Some(DisjointSetRef(v, set, element))
            } else {
                None
            }
        })
    }

    pub fn get_internal(&mut self, mut v: V) -> V {
        let mut traversed = VS::empty();
        loop {
            match self.elements.get_mut(v).unwrap_or_else(|| unreachable!()) {
                DisjointSetNode::Child { parent } => {
                    traversed.insert(v);
                    v = *parent;
                }
                _ => {
                    for w in traversed {
                        *self.elements.get_mut(w).unwrap_or_else(|| unreachable!()) =
                            DisjointSetNode::Child { parent: v };
                    }
                    return v;
                }
            }
        }
    }

    pub fn union<F>(
        &mut self,
        v: V,
        w: V,
        f: F,
    ) -> Option<DisjointSetMutRef<V, VS, E>>
    where
        F: FnOnce(E, E) -> E,
    {
        self.elements.get_ref(v)?;
        self.elements.get_ref(w)?;
        let x = self.get_internal(v);
        let y = self.get_internal(w);

        if x == y {
            if let Some(DisjointSetNode::Root { element, set }) = self.elements.get_mut(x) {
                return Some(DisjointSetMutRef(x, set, element));
            } else {
                unreachable!();
            }
        }

        if let (
            DisjointSetNode::Root {
                element: element1,
                set: set1,
            },
            DisjointSetNode::Root {
                element: element2,
                set: set2,
            },
        ) = (
            std::mem::replace(
                self.elements.get_mut(x).unwrap_or_else(|| unreachable!()),
                DisjointSetNode::Child { parent: y },
            ),
            std::mem::replace(
                self.elements.get_mut(y).unwrap_or_else(|| unreachable!()),
                DisjointSetNode::Child { parent: x },
            ),
        ) {
            let cmp = set1.len().cmp(&set2.len());
            let mut set = set1;
            set.insert_set(&set2);
            match cmp {
                Ordering::Greater | Ordering::Equal => {
                    *self.elements.get_mut(x).unwrap_or_else(|| unreachable!()) =
                        DisjointSetNode::Root {
                            element: f(element1, element2),
                            set,
                        };
                    if let Some(DisjointSetNode::Root { element, set }) = self.elements.get_mut(x) {
                        Some(DisjointSetMutRef(x, set, element))
                    } else {
                        unreachable!();
                    }
                }
                Ordering::Less => {
                    *self.elements.get_mut(y).unwrap_or_else(|| unreachable!()) =
                        DisjointSetNode::Root {
                            element: f(element1, element2),
                            set,
                        };
                    if let Some(DisjointSetNode::Root { element, set }) = self.elements.get_mut(y) {
                        Some(DisjointSetMutRef(y, set, element))
                    } else {
                        unreachable!();
                    }
                }
            }
        } else {
            unreachable!();
        }
    }

    pub fn union_set(&mut self, vs: &VS, element: E) -> Option<DisjointSetMutRef<V, VS, E>> {
        for v in vs.iter() {
            self.elements.get_ref(v)?;
        }

        let mut iter = vs.iter();
        let u = if let Some(u) = iter.next() {
            u
        } else {
            return None;
        };

        let DisjointSetMutRef(v, vs, e) = if let Some(v) = iter.next() {
            for w in iter {
                self.union(u, w, |e, _| e);
            }

            self.union(u, v, |e, _| e).unwrap_or_else(|| unreachable!())
        } else {
            self.get_mut(u).unwrap_or_else(|| unreachable!())
        };
        *e = element;
        Some(DisjointSetMutRef(v, vs, e))
    }

    pub fn intersect_sets<F, FUnused>(
        &mut self,
        other: &mut Self,
        v: V,
        mut f: F,
        f_unused: FUnused,
    ) -> bool
    where
        F: FnMut(DisjointSetMutRef<V, VS, E>, DisjointSetMutRef<V, VS, E>, &VS) -> E,
        FUnused: FnOnce(V, &VS, E) -> E,
    {
        if self.elements.get_ref(v).is_none() {
            return false;
        }
        let root = self.get_internal(v);
        if let DisjointSetNode::Root {
            mut element,
            mut set,
        } = std::mem::replace(
            self.elements
                .get_mut(root)
                .unwrap_or_else(|| unreachable!()),
            DisjointSetNode::Child { parent: v },
        ) {
            let old_set = set.clone();
            let mut to_sets: Vec<(VS, E)> = other
                .iter_mut()
                .filter_map(|DisjointSetMutRef(other_v, other_set, other_element)| {
                    let mut new_set = set.clone();
                    new_set.intersect_set(other_set);
                    if !new_set.is_empty() {
                        set.remove_set(&new_set);
                        let new_element = f(
                            DisjointSetMutRef(root, &old_set, &mut element),
                            DisjointSetMutRef(other_v, other_set, other_element),
                            &new_set,
                        );
                        Some((new_set, new_element))
                    } else {
                        None
                    }
                })
                .collect();

            if !set.is_empty() {
                let new_element = f_unused(root, &set, element);
                to_sets.push((set, new_element));
            }

            for (new_set, new_element) in to_sets {
                let mut iter = new_set.clone().into_iter();
                let u = iter.next().unwrap_or_else(|| unreachable!());
                *self.elements.get_mut(u).unwrap_or_else(|| unreachable!()) =
                    DisjointSetNode::Root {
                        element: new_element,
                        set: new_set,
                    };
                for v in iter {
                    *self.elements.get_mut(v).unwrap_or_else(|| unreachable!()) =
                        DisjointSetNode::Child { parent: u };
                }
            }

            true
        } else {
            unreachable!();
        }
    }
}

#[cfg(test)]
mod tests {
    use adagraph::prelude::*;
    use adagraph_nauty::{self, NautyGraph};
}
