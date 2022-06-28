use crate::Set;
use num::{One, Zero};
use std::fmt::{Binary, Debug, Display};
use std::iter::{FromIterator, IntoIterator};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

pub unsafe trait BitStore:
    Copy
    + BitAnd<Output = Self>
    + BitAndAssign
    + BitOr<Output = Self>
    + BitOrAssign
    + BitXor<Output = Self>
    + BitXorAssign
    + Not<Output = Self>
    + Zero
    + One
    + Ord
    + Binary
    + Debug
    + Default
    + 'static
{
    fn leading_zeros(&self) -> usize;
    fn trailing_zeros(&self) -> usize;
    fn count_ones(&self) -> usize;
    fn shl(&self, s: usize) -> Self;
    fn shr(&self, s: usize) -> Self;
    fn size() -> usize;
}

unsafe impl BitStore for u8 {
    fn leading_zeros(&self) -> usize {
        u8::leading_zeros(*self) as usize
    }

    fn trailing_zeros(&self) -> usize {
        u8::trailing_zeros(*self) as usize
    }

    fn count_ones(&self) -> usize {
        u8::count_ones(*self) as usize
    }

    fn shl(&self, s: usize) -> Self {
        Self::checked_shl(*self, s as u32).unwrap_or_default()
    }

    fn shr(&self, s: usize) -> Self {
        Self::checked_shr(*self, s as u32).unwrap_or_default()
    }

    fn size() -> usize {
        8
    }
}

unsafe impl BitStore for u16 {
    fn leading_zeros(&self) -> usize {
        u16::leading_zeros(*self) as usize
    }

    fn trailing_zeros(&self) -> usize {
        u16::trailing_zeros(*self) as usize
    }

    fn count_ones(&self) -> usize {
        u16::count_ones(*self) as usize
    }

    fn shl(&self, s: usize) -> Self {
        Self::checked_shl(*self, s as u32).unwrap_or_default()
    }

    fn shr(&self, s: usize) -> Self {
        Self::checked_shr(*self, s as u32).unwrap_or_default()
    }

    fn size() -> usize {
        16
    }
}

unsafe impl BitStore for u32 {
    fn leading_zeros(&self) -> usize {
        u32::leading_zeros(*self) as usize
    }

    fn trailing_zeros(&self) -> usize {
        u32::trailing_zeros(*self) as usize
    }

    fn count_ones(&self) -> usize {
        u32::count_ones(*self) as usize
    }

    fn shl(&self, s: usize) -> Self {
        Self::checked_shl(*self, s as u32).unwrap_or_default()
    }

    fn shr(&self, s: usize) -> Self {
        Self::checked_shr(*self, s as u32).unwrap_or_default()
    }

    fn size() -> usize {
        32
    }
}

unsafe impl BitStore for u64 {
    fn leading_zeros(&self) -> usize {
        u64::leading_zeros(*self) as usize
    }

    fn trailing_zeros(&self) -> usize {
        u64::trailing_zeros(*self) as usize
    }

    fn count_ones(&self) -> usize {
        u64::count_ones(*self) as usize
    }

    fn shl(&self, s: usize) -> Self {
        Self::checked_shl(*self, s as u32).unwrap_or_default()
    }

    fn shr(&self, s: usize) -> Self {
        Self::checked_shr(*self, s as u32).unwrap_or_default()
    }

    fn size() -> usize {
        64
    }
}

unsafe impl BitStore for usize {
    fn leading_zeros(&self) -> usize {
        usize::leading_zeros(*self) as usize
    }

    fn trailing_zeros(&self) -> usize {
        usize::trailing_zeros(*self) as usize
    }

    fn count_ones(&self) -> usize {
        usize::count_ones(*self) as usize
    }

    fn shl(&self, s: usize) -> Self {
        Self::checked_shl(*self, s as u32).unwrap_or_default()
    }

    fn shr(&self, s: usize) -> Self {
        Self::checked_shr(*self, s as u32).unwrap_or_default()
    }

    fn size() -> usize {
        std::mem::size_of::<usize>() * 8
    }
}

pub(crate) fn leading_one<D>() -> D
where
    D: BitStore,
{
    !((!D::zero()).shr(1))
}

#[derive(Default)]
pub struct BitSet<D>
where
    D: BitStore,
{
    pub(crate) data: Vec<D>,
}

impl<D> Eq for BitSet<D> where D: BitStore {}
impl<D> Ord for BitSet<D>
where
    D: BitStore,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        for i in 0..self.data.len().max(other.data.len()) {
            let cmp = self
                .data
                .get(i)
                .unwrap_or(&D::zero())
                .cmp(other.data.get(i).unwrap_or(&D::zero()));
            if cmp != std::cmp::Ordering::Equal {
                return cmp;
            }
        }
        std::cmp::Ordering::Equal
    }
}
impl<D> PartialEq for BitSet<D>
where
    D: BitStore,
{
    fn eq(&self, other: &Self) -> bool {
        for i in 0..self.data.len().max(other.data.len()) {
            if self.data.get(i).unwrap_or(&D::zero()) != other.data.get(i).unwrap_or(&D::zero()) {
                return false;
            }
        }
        true
    }
}
impl<D> PartialOrd for BitSet<D>
where
    D: BitStore,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub struct DecreasingBitSetIterator<'a, D>
where
    D: BitStore,
{
    pub iter: std::iter::Rev<std::slice::Iter<'a, D>>,
    pub i: usize,
    pub j: usize,
    pub data: D,
}

impl<'a, D> DecreasingBitSetIterator<'a, D>
where
    D: BitStore,
{
    fn new(data: &'a [D]) -> Self {
        let mut iter = data.iter().rev();
        let i = data.len() - 1;
        let data = if let Some(e) = iter.next() {
            *e
        } else {
            D::zero()
        };
        Self {
            iter,
            i,
            j: D::size() - 1,
            data,
        }
    }
}

impl<'a, D> Iterator for DecreasingBitSetIterator<'a, D>
where
    D: BitStore,
{
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        while self.data == D::zero() && self.i != usize::MAX {
            self.i = self.i.wrapping_sub(1);
            self.j = D::size() - 1;
            self.data = *(self.iter.next()?);
        }
        if self.i == usize::MAX {
            None?
        }
        let trailing_zeros = self.data.trailing_zeros();
        self.data = self.data.shr(trailing_zeros + 1);
        self.j = self.j.wrapping_sub(trailing_zeros + 1);
        Some(self.i * D::size() + self.j.wrapping_add(1))
    }
}

impl<D> BitSet<D>
where
    D: BitStore,
{
    pub fn iter_decreasing<'a>(&'a self) -> DecreasingBitSetIterator<'a, D> {
        DecreasingBitSetIterator::new(&self.data)
    }

    pub fn iter_from<'a>(&'a self, p: usize) -> IncreasingBitSetIterator<'a, D> {
        IncreasingBitSetIterator::new_from(&self.data, p)
    }

    pub fn into_iter_from(self, p: usize) -> OwnedIncreasingBitSetIterator<D> {
        OwnedIncreasingBitSetIterator::<D>::new_from(self.data, p)
    }
}

impl<D> Display for BitSet<D>
where
    D: BitStore,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        fmt.write_str(&format!(
            "{{{}}}",
            (0..self.data.len() * D::size())
                .filter(|i| self.contains(*i))
                .map(|i| i.to_string())
                .collect::<Vec<String>>()
                .join(",")
        ))?;
        Ok(())
    }
}

impl<D> Debug for BitSet<D>
where
    D: BitStore,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        <Self as Display>::fmt(self, fmt)
    }
}

impl<D> FromIterator<usize> for BitSet<D>
where
    D: BitStore,
{
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        let mut result = Self::new();
        for v in iter {
            result.insert(v);
        }
        result
    }
}

unsafe impl<D> Set for BitSet<D>
where
    D: BitStore,
{
    type Element = usize;
    type Iter<'a> = IncreasingBitSetIterator<'a, D>;
    type IterIncreasing<'a> = IncreasingBitSetIterator<'a, D>;
    type IterDecreasing<'a> = DecreasingBitSetIterator<'a, D>;
    type IntoIterIncreasing = OwnedIncreasingBitSetIterator<D>;
    type IntoIterDecreasing = OwnedDecreasingBitSetIterator<D>;

    fn empty() -> Self {
        Self::new()
    }

    fn singleton(e: Self::Element) -> Self {
        let mut result = Self::new();
        result.insert(e);
        result
    }

    fn clear(&mut self) -> &mut Self {
        self.data.truncate(0);
        self
    }

    fn contains(&self, i: usize) -> bool {
        BitSetRef::new(&self.data).contains(i)
    }

    fn insert(&mut self, i: usize) -> &mut Self {
        self.resize_to_fit(i);
        BitSetMut::new(&mut self.data).insert(i);
        self
    }

    fn remove(&mut self, i: usize) -> &mut Self {
        BitSetMut::new(&mut self.data).remove(i);
        self.shrink();
        self
    }

    fn insert_set<'a>(&'a mut self, other: &Self) -> &'a mut Self {
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(e1, e2)| *e1 |= *e2);
        if self.data.len() < other.data.len() {
            self.data
                .extend(other.data[self.data.len()..].iter().map(Clone::clone))
        }
        self
    }

    fn remove_set<'a>(&'a mut self, other: &Self) -> &'a mut Self {
        BitSetMut::new(&mut self.data).remove_set(BitSetRef::new(&other.data));
        self.shrink();
        self
    }

    fn intersect_set<'a>(&'a mut self, other: &Self) -> &'a mut Self {
        BitSetMut::new(&mut self.data).intersect_set(BitSetRef::new(&other.data));
        self.shrink();
        self
    }

    fn iter<'a>(&'a self) -> IncreasingBitSetIterator<'a, D> {
        BitSetRef::new(&self.data).iter()
    }

    fn is_subset_of(&self, other: &Self) -> bool {
        BitSetRef::new(&self.data).is_subset_of(&BitSetRef::new(&other.data))
    }

    fn is_intersecting(&self, other: &Self) -> bool {
        BitSetRef::new(&self.data).is_intersecting(&BitSetRef::new(&other.data))
    }

    fn is_empty(&self) -> bool {
        BitSetRef::new(&self.data).is_empty()
    }

    fn len(&self) -> usize {
        BitSetRef::new(&self.data).len()
    }

    fn min_value(&self) -> Option<Self::Element> {
        BitSetRef::new(&self.data).min_value()
    }

    fn max_value(&self) -> Option<Self::Element> {
        BitSetRef::new(&self.data).max_value()
    }

    fn iter_decreasing<'b>(&'b self) -> Self::IterDecreasing<'b>
    where
        Self::Element: std::cmp::Ord,
    {
        DecreasingBitSetIterator::new(&self.data)
    }

    fn iter_increasing<'b>(&'b self) -> Self::IterIncreasing<'b>
    where
        Self::Element: std::cmp::Ord,
    {
        IncreasingBitSetIterator::new(&self.data)
    }

    fn into_iter_decreasing(self) -> Self::IntoIterDecreasing
    where
        Self::Element: std::cmp::Ord,
    {
        OwnedDecreasingBitSetIterator::new(self)
    }

    fn into_iter_increasing(self) -> Self::IntoIterIncreasing
    where
        Self::Element: std::cmp::Ord,
    {
        OwnedIncreasingBitSetIterator::new(self.data)
    }
}

impl<D> Clone for BitSet<D>
where
    D: BitStore,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

impl<D> BitSet<D>
where
    D: BitStore,
{
    pub fn new() -> Self {
        Self { data: vec![] }
    }

    pub fn fill(&mut self, n: usize) -> &mut Self {
        self.resize_to_fit(n);
        BitSetMut::new(&mut self.data).fill(n);
        self
    }

    pub fn opposite(&mut self, n: usize) -> &mut Self {
        self.resize_to_fit(n);
        BitSetMut::new(&mut self.data).opposite(n);
        self
    }

    pub fn resize_to_fit(&mut self, n: usize) {
        let new_len = (n + 2 * D::size() - 1) / D::size();
        if new_len > self.data.len() {
            self.data.resize(new_len, D::zero());
        }
    }

    pub fn shrink(&mut self) {
        while let Some(d) = self.data.last() {
            if *d != D::zero() {
                break;
            }
            self.data.pop();
        }
    }
}

pub struct IncreasingBitSetIterator<'a, D>
where
    D: BitStore,
{
    pub iter: std::slice::Iter<'a, D>,
    pub i: usize,
    pub j: usize,
    pub d: D,
}

impl<'a, D> IncreasingBitSetIterator<'a, D>
where
    D: BitStore,
{
    pub fn new(data: &'a [D]) -> Self {
        let mut iter = data.iter();
        let d = if let Some(e) = iter.next() {
            *e
        } else {
            D::zero()
        };
        Self {
            i: 0,
            j: 0,
            d,
            iter,
        }
    }

    pub fn new_from(data: &'a [D], p: usize) -> IncreasingBitSetIterator<'a, D> {
        let i = p / D::size();
        let j = p % D::size();
        let mut iter = data.iter();
        let mut d = D::zero();
        for _ in 0..(i + 1) {
            if let Some(e) = iter.next() {
                d = *e;
            } else {
                d = D::zero();
                break;
            }
        }
        IncreasingBitSetIterator::<'a, D> {
            i,
            j,
            d: d.shl(j),
            iter,
        }
    }
}

impl<'a, D> std::iter::Iterator for IncreasingBitSetIterator<'a, D>
where
    D: BitStore,
{
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        while self.d == D::zero() {
            self.i += 1;
            self.j = 0;
            self.d = *self.iter.next()?;
        }
        let leading_zeros = self.d.leading_zeros();
        self.j += leading_zeros + 1;
        self.d = self.d.shl(leading_zeros + 1);
        Some(self.i * D::size() + self.j - 1)
    }
}

pub struct OwnedIncreasingBitSetIterator<D>
where
    D: BitStore,
{
    iter: std::vec::IntoIter<D>,
    i: usize,
    j: usize,
    d: D,
}

impl<'a, D> OwnedIncreasingBitSetIterator<D>
where
    D: BitStore,
{
    pub fn new(data: Vec<D>) -> Self {
        let mut iter = data.into_iter();
        Self {
            i: 0,
            j: 0,
            d: if let Some(e) = iter.next() {
                e
            } else {
                D::zero()
            },
            iter,
        }
    }

    pub fn new_from(data: Vec<D>, p: usize) -> Self {
        let i = p / D::size();
        let j = p % D::size();
        let mut iter = data.into_iter();
        let mut d = D::zero();
        for _ in 0..(i + 1) {
            if let Some(e) = iter.next() {
                d = e;
            } else {
                d = D::zero();
                break;
            }
        }
        Self {
            i,
            j,
            d: d.shl(j),
            iter,
        }
    }
}

impl<'a, D> std::iter::Iterator for OwnedIncreasingBitSetIterator<D>
where
    D: BitStore,
{
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        while self.d == D::zero() {
            self.i += 1;
            self.j = 0;
            self.d = self.iter.next()?;
        }
        let leading_zeros = self.d.leading_zeros(); // = self.d & BitSet::<D, N>::leading_one() > D::zero();
        self.j += leading_zeros + 1;
        self.d = self.d.shl(leading_zeros + 1);
        Some(self.i * D::size() + self.j - 1)
    }
}

pub struct OwnedDecreasingBitSetIterator<D>
where
    D: BitStore,
{
    data: Vec<D>,
    i: usize,
    j: usize,
    d: D,
}

impl<D> OwnedDecreasingBitSetIterator<D>
where
    D: BitStore,
{
    fn new(bitset: BitSet<D>) -> Self {
        Self {
            i: bitset.data.len() - 1,
            j: D::size() - 1,
            d: bitset.data.last().map(|e| *e).unwrap_or_default(),
            data: bitset.data,
        }
    }
}

impl<D> std::iter::Iterator for OwnedDecreasingBitSetIterator<D>
where
    D: BitStore,
{
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        while self.d == D::zero() {
            if self.i == 0 {
                return None;
            }
            self.i -= 1;
            self.j = D::size() - 1;
            self.d = self.data[self.i];
        }

        let trailing_zeros = self.d.trailing_zeros();
        self.j = self.j.wrapping_sub(trailing_zeros + 1);
        self.d = self.d.shr(trailing_zeros + 1);
        Some(self.i * D::size() + self.j.wrapping_add(1))
    }
}

impl<D> IntoIterator for BitSet<D>
where
    D: BitStore,
{
    type Item = usize;
    type IntoIter = OwnedIncreasingBitSetIterator<D>;
    fn into_iter(self) -> Self::IntoIter {
        OwnedIncreasingBitSetIterator::new(self.data)
    }
}

pub struct BitSetRef<'a, D>
where
    D: BitStore,
{
    pub(crate) data: &'a [D],
}

impl<'a, D> Debug for BitSetRef<'a, D>
where
    D: BitStore,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.write_str(&format!(
            "{{{}}}",
            (0..self.data.len() * D::size())
                .filter(|i| self.contains(*i))
                .map(|i| i.to_string())
                .collect::<Vec<String>>()
                .join(",")
        ))?;
        Ok(())
    }
}

impl<'a, D> BitSetRef<'a, D>
where
    D: BitStore,
{
    pub fn new(data: &'a [D]) -> Self {
        Self { data }
    }

    pub fn contains(&self, i: usize) -> bool {
        if i / D::size() >= self.data.len() {
            return false;
        }
        self.data[i / D::size()] & leading_one::<D>().shr(i % D::size()) >= D::one()
    }

    pub fn is_subset_of(&self, other: &Self) -> bool {
        for i in 0..self.data.len() {
            if self.data[i] & !*(other.data.get(i).unwrap_or(&D::zero())) != D::zero() {
                return false;
            }
        }
        true
    }

    pub fn is_intersecting(&self, other: &Self) -> bool {
        self.data
            .iter()
            .zip(other.data.iter())
            .any(|(a, b)| (*a & *b) != D::zero())
    }

    pub fn is_empty(&self) -> bool {
        self.data.iter().all(|e| *e == D::zero())
    }

    pub fn len(&self) -> usize {
        self.data.iter().map(|d| d.count_ones()).sum::<usize>()
    }

    pub fn min_value(&self) -> Option<usize> {
        self.iter().nth(0)
    }

    pub fn max_value(&self) -> Option<usize> {
        let mut i = self.data.len().wrapping_sub(1);
        while i != usize::MAX && self.data[i] == D::zero() {
            if i == 0 {
                return None;
            }
            i = i.wrapping_sub(1);
        }
        if i == usize::MAX {
            return None;
        }
        let data = self.data[i];
        let j = D::size() - data.trailing_zeros() - 1;
        Some(i * D::size() + j)
    }

    pub fn iter(&self) -> IncreasingBitSetIterator<'a, D> {
        IncreasingBitSetIterator::new(self.data)
    }

    pub fn iter_from(&self, i: usize) -> IncreasingBitSetIterator<'a, D> {
        IncreasingBitSetIterator::new_from(self.data, i)
    }

    pub fn iter_decreasing<'b>(&'b self) -> DecreasingBitSetIterator<'b, D> {
        DecreasingBitSetIterator::new(&self.data)
    }

    pub fn iter_increasing<'b>(&'b self) -> IncreasingBitSetIterator<'b, D> {
        IncreasingBitSetIterator::new(&self.data)
    }
}

pub struct BitSetMut<'a, D> {
    pub(crate) data: &'a mut [D],
}

impl<'a, D> BitSetMut<'a, D>
where
    D: BitStore,
{
    pub fn new(data: &'a mut [D]) -> Self {
        Self { data }
    }

    pub fn contains(&self, i: usize) -> bool {
        BitSetRef::new(self.data).contains(i)
    }

    pub fn insert(&mut self, i: usize) -> &Self {
        debug_assert!(i / D::size() < self.data.len());
        self.data[i / D::size()] |= leading_one::<D>().shr(i % D::size());
        self
    }

    pub fn remove(&mut self, i: usize) -> &Self {
        if i / D::size() >= self.data.len() {
            return self;
        }
        self.data[i / D::size()] &= !(leading_one::<D>().shr(i % D::size()));
        self
    }

    fn border_mask_and_index(n: usize) -> (D, usize) {
        let mut result = !D::zero();
        result = result.shr(n % D::size());
        result = !result;
        (result, n / D::size())
    }

    pub fn fill(&mut self, n: usize) -> &Self {
        let (mask, m) = Self::border_mask_and_index(n);
        self.data[..m].iter_mut().for_each(|e| *e = !D::zero());
        self.data[m] = mask;
        self
    }

    pub fn opposite(&mut self, n: usize) -> &Self {
        let (mask, m) = Self::border_mask_and_index(n);
        self.data[..m].iter_mut().for_each(|e| *e = !*e);
        self.data[m] ^= mask;
        self
    }

    pub fn remove_set(&mut self, other: BitSetRef<'a, D>) -> &Self {
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(t, s)| *t &= !*s);
        self
    }

    pub fn insert_set(&mut self, other: BitSetRef<'a, D>) -> &Self {
        debug_assert!(self.data.len() == other.data.len());
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(t, s)| *t |= *s);
        self
    }

    pub fn intersect_set(&mut self, other: BitSetRef<'a, D>) -> &Self {
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(t, s)| *t &= *s);
        for i in other.data.len()..self.data.len() {
            self.data[i] = D::zero()
        }
        self
    }
}

pub struct BitSetArrayRefIterator<'a, D>
where
    D: BitStore,
{
    i: usize,
    bit_set_array: &'a BitSetArray<D>,
}

impl<'a, D> Iterator for BitSetArrayRefIterator<'a, D>
where
    D: BitStore,
{
    type Item = BitSetRef<'a, D>;
    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        self.i += 1;
        if i >= self.bit_set_array.len() {
            None?
        }
        Some(self.bit_set_array.get_ref(i))
    }
}

pub struct BitSetArrayMutIterator<'a, D>
where
    D: BitStore,
{
    iter: std::slice::ChunksMut<'a, D>,
}

impl<'a, D> Iterator for BitSetArrayMutIterator<'a, D>
where
    D: BitStore,
{
    type Item = BitSetMut<'a, D>;
    fn next(&mut self) -> Option<Self::Item> {
        let data = self.iter.next()?;
        Some(BitSetMut { data })
    }
}

#[derive(Debug, Clone)]
pub struct BitSetArray<D>
where
    D: BitStore,
{
    m: usize,
    data: Vec<D>,
}

impl<D> BitSetArray<D>
where
    D: BitStore,
{
    pub fn new(n: usize, m: usize) -> Self {
        Self {
            m,
            data: vec![D::zero(); n * m],
        }
    }

    pub fn get_m(&self) -> usize {
        self.m
    }

    pub fn iter(&self) -> BitSetArrayRefIterator<'_, D> {
        BitSetArrayRefIterator {
            i: 0,
            bit_set_array: self,
        }
    }

    pub fn iter_mut(&mut self) -> BitSetArrayMutIterator<'_, D> {
        BitSetArrayMutIterator {
            iter: self.data.chunks_mut(self.m),
        }
    }

    pub(crate) fn get(&self, i: usize) -> BitSet<D> {
        let mut j = self.m - 1;
        while j > 0 && self.data[(i * self.m + j)] == D::zero() {
            j -= 1;
        }
        let data = self.data[(i * self.m)..(i * self.m + j + 1)].to_vec();
        BitSet { data }
    }

    pub(crate) fn get_ref(&self, i: usize) -> BitSetRef<'_, D> {
        BitSetRef {
            data: &self.data[(i * self.m)..((i + 1) * self.m)],
        }
    }

    pub(crate) fn get_mut(&mut self, i: usize) -> BitSetMut<'_, D> {
        BitSetMut {
            data: &mut self.data[(i * self.m)..((i + 1) * self.m)],
        }
    }

    pub fn len(&self) -> usize {
        self.data.len() * self.m
    }

    pub fn reshape(&mut self, n: usize, m: usize) {
        let other = std::mem::replace(self, Self::new(n, m));

        self.iter_mut().zip(other.iter()).for_each(|(t, o)| {
            t.data
                .iter_mut()
                .zip(o.data.iter())
                .for_each(|(te, oe)| *te = *oe)
        })
    }
}

mod tests {
    #[test]
    fn test_bit_layout() {
        use super::*;
        let mut bit_array = BitSet::<u8>::new();
        bit_array.insert(0);
        bit_array.insert(1);
        bit_array.insert(2);
        bit_array.insert(3);
        bit_array.insert(14);
        assert_eq!(Some(0), bit_array.min_value());
        assert_eq!(Some(14), bit_array.max_value());
    }

    #[test]
    fn test_bit_set() {
        use super::*;
        assert_eq!(
            std::mem::align_of::<BitSet::<usize>>(),
            std::mem::align_of::<usize>()
        );
        assert_eq!(
            std::mem::size_of::<BitSet::<usize>>(),
            3 * std::mem::size_of::<usize>()
        );
    }

    #[test]
    fn test_bit_iterator() {
        use super::*;
        let mut bit_set = BitSet::<u8>::new();
        bit_set
            .insert(1)
            .insert(2)
            .insert(3)
            .insert(4)
            .insert(5)
            .insert(7)
            .insert(8)
            .insert(11)
            .insert(13)
            .insert(14)
            .insert(15);
        assert_eq!(
            "1,2,3,4,5,7,8,11,13,14,15",
            bit_set
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<String>>()
                .join(",")
        );

        assert_eq!(
            "3,4,5,7,8,11,13,14,15",
            bit_set
                .iter_from(3)
                .map(|i| i.to_string())
                .collect::<Vec<String>>()
                .join(",")
        );

        assert_eq!(
            "15,14,13,11,8,7,5,4,3,2,1",
            bit_set
                .iter_decreasing()
                .map(|i| i.to_string())
                .collect::<Vec<String>>()
                .join(",")
        );

        assert_eq!(
            "3,4,5,7,8,11,13,14,15",
            bit_set
                .clone()
                .into_iter_from(3)
                .map(|i| i.to_string())
                .collect::<Vec<String>>()
                .join(",")
        );

        assert_eq!(
            "1,2,3,4,5,7,8,11,13,14,15",
            bit_set
                .clone()
                .into_iter_increasing()
                .map(|i| i.to_string())
                .collect::<Vec<String>>()
                .join(",")
        );

        assert_eq!(
            "15,14,13,11,8,7,5,4,3,2,1",
            bit_set
                .clone()
                .into_iter_decreasing()
                .map(|i| i.to_string())
                .collect::<Vec<String>>()
                .join(",")
        );

        assert_eq!(
            "7",
            BitSet::<u8>::new()
                .insert(7)
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<String>>()
                .join(",")
        );
    }

    #[test]
    fn test_bit_fill() {
        use super::*;
        let mut bit_set = BitSet::<u32>::new();
        bit_set.fill(40);
        assert_eq!(
            (0..40).collect::<Vec<usize>>(),
            bit_set.iter().collect::<Vec<usize>>()
        );
    }

    #[test]
    fn test_opposite() {
        use super::*;
        let mut bit_set = BitSet::<u8>::new();
        bit_set.insert(2).insert(3).insert(4).opposite(5);
        assert_eq!(
            "0,1",
            bit_set
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<String>>()
                .join(",")
        );
    }

    #[test]
    fn test_set_operations() {
        use super::*;

        let bit_set1 = BitSet::<u8>::new().fill(16).clone();
        let bit_set2 = BitSet::<u8>::new().fill(8).opposite(24).clone();

        println!("{}, {}", bit_set1, bit_set2);

        assert_eq!(
            bit_set1.clone().insert_set(&bit_set2),
            BitSet::<u8>::new().fill(24)
        );
        assert_eq!(
            bit_set2.clone().insert_set(&bit_set1),
            BitSet::<u8>::new().fill(24)
        );
        assert_eq!(
            bit_set1.clone().intersect_set(&bit_set2),
            BitSet::<u8>::new().fill(8).opposite(16)
        );
        assert_eq!(
            bit_set2.clone().intersect_set(&bit_set1),
            BitSet::<u8>::new().fill(8).opposite(16)
        );
        assert_eq!(
            bit_set1.clone().remove_set(&bit_set2),
            BitSet::<u8>::new().fill(8)
        );
        assert_eq!(
            bit_set2.clone().remove_set(&bit_set1),
            BitSet::<u8>::new().fill(16).opposite(24)
        );
    }

    #[test]
    fn test_bit_set_array() {
        use super::*;
        let mut bit_set_array = BitSetArray::<u64>::new(3, 3);
        bit_set_array.get_mut(0).insert(64);
        let mut bit_set = bit_set_array.get(0);
        assert_eq!(bit_set.data.len(), 2);
        bit_set.remove(1);
        assert_eq!(bit_set.data.len(), 2);
        bit_set.remove(64);
        assert_eq!(bit_set.data.len(), 0);
        bit_set.insert(129);
        assert_eq!(bit_set.data.len(), 3);
        println!("{}", bit_set);
        assert_eq!("{129}", format!("{:?}", bit_set));
    }
}
