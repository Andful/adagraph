use std::collections::BTreeMap;

use adagraph::prelude::*;

use lazy_static::lazy_static;
use regex::Regex;
use serde::{
    ser::{SerializeMap, SerializeSeq},
    Serialize, Serializer,
};

/*pub fn parse_pace<G>(data: &str) -> G where G: ConstructableGraph + EdgeAddableGraph {

}*/

trait PaceGraphWrapper<G>
where
    G: Graph,
{
}
trait EdgeListGraphWrapper<G>
where
    G: Graph,
{
}
trait Graph6GraphWrapper<G>
where
    G: Graph,
{
}
pub struct JsonGraphWrapper<G>(pub G)
where
    G: Graph;

pub struct JsonSetWrapper<S>(pub S)
where
    S: Set;

#[derive(Serialize)]
struct Node<G>
where
    G: Graph,
    G::VertexIndex: Serialize,
{
    id: usize,
    v: G::VertexIndex,
}

#[derive(Serialize)]
struct Edge {
    source: usize,
    target: usize,
}

impl<G> Serialize for JsonGraphWrapper<G>
where
    G: GraphIterator,
    G::VertexIndex: Serialize + Ord,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut vertex_map = BTreeMap::<G::VertexIndex, usize>::new();
        let mut result = serializer.serialize_map(Some(2))?;
        result.serialize_entry(
            "nodes",
            &self
                .0
                .vertices()
                .enumerate()
                .map(|(id, v)| {
                    vertex_map.insert(v, id);
                    Node { id, v }
                })
                .collect::<Vec<Node<G>>>(),
        )?;

        result.serialize_entry(
            "links",
            &self
                .0
                .edges()
                .map(|e| {
                    let (v, w) = self.0.adjacent_vertives(e);
                    let source = *vertex_map.get(&v).unwrap();
                    let target = *vertex_map.get(&w).unwrap();
                    Edge { target, source }
                })
                .collect::<Vec<Edge>>(),
        )?;
        result.end()
    }
}

impl<S> Serialize for JsonSetWrapper<S>
where
    S: Set,
    S::Element: Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: Serializer,
    {
        let mut result = serializer.serialize_seq(None)?;
        for e in self.0.iter() {
            result.serialize_element(&e)?;
        }
        result.end()
    }
}

pub fn parse_pace<G>(s: &str) -> GraphEditingResult<G>
where
    G: BuildableGraph,
{
    lazy_static! {
        static ref RE_HEAD: Regex = Regex::new(r"^p \w+ (\d+) \d+$").unwrap();
        static ref RE_EDGE: Regex = Regex::new(r"^(\d+) (\d+)$").unwrap();
        static ref RE_COMMENT: Regex = Regex::new(r"^c.*$").unwrap();
    }
    let mut lines = s.split("\n");

    let n: usize = RE_HEAD
        .captures(
            lines
                .next()
                .ok_or(GraphEditingError::UnknownError("No Header".to_string()))?,
        )
        .ok_or(GraphEditingError::UnknownError(
            "header not matching".to_string(),
        ))?[1]
        .parse::<usize>()
        .expect("error parsing vertices");

    let mut builder = G::builder(n);
    let vertices = builder
        .vertex_set()
        .into_iter()
        .collect::<Vec<G::VertexIndex>>();

    if vertices.len() == 0 {
        builder.build().expect("Error building");
        panic!();
    }

    for l in lines.filter(|l| *l != "") {
        let captures = RE_EDGE.captures(l);
        let captures = if let Some(captures) = captures {
            captures
        } else {
            if RE_COMMENT.is_match(l) {
                continue;
            } else {
                panic!("{:?}", l);
            }
        };
        let v = captures[1].parse::<usize>().unwrap() - 1;
        let w = captures[2].parse::<usize>().unwrap() - 1;
        builder = builder.add_edge(vertices[v], vertices[w], ());
    }

    builder.build()
}

pub fn parse_dimancs<G>(s: &str) -> GraphEditingResult<G>
where
    G: BuildableGraph,
{
    //TODO check that the number of edges matches
    lazy_static! {
        static ref RE_COMMENT: Regex = Regex::new(r"c.*").unwrap();
        static ref RE_HEAD: Regex = Regex::new(r"p edge (\d+) \d+").unwrap();
        static ref RE_EDGE: Regex = Regex::new(r"e (\d+) (\d+)").unwrap();
    }
    let mut lines = s.split("\n").filter(|l| !RE_COMMENT.is_match(l));

    let n = RE_HEAD
        .captures(
            lines
                .next()
                .ok_or(GraphEditingError::UnknownError("No Header".to_string()))?,
        )
        .ok_or(GraphEditingError::UnknownError(
            "header not matching".to_string(),
        ))?[1]
        .parse::<usize>()
        .expect("error parsing vertices");

    let mut builder = G::builder(n);
    let vertices = builder
        .vertex_set()
        .into_iter()
        .collect::<Vec<G::VertexIndex>>();

    for l in lines.filter(|l| *l != "") {
        let captures = RE_EDGE.captures(l).unwrap();
        let v = captures[1].parse::<usize>().unwrap() - 1;
        let w = captures[2].parse::<usize>().unwrap() - 1;
        builder = builder.add_edge(vertices[v], vertices[w], ());
    }

    builder.build()
}

