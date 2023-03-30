#![feature(unboxed_closures)]
#![feature(fn_traits)]
#![feature(drain_filter)]

use std::fmt::Debug;
use std::{
    collections::{BTreeSet, VecDeque},
    iter::Iterator,
    path::PathBuf,
};

use cpu_time::ProcessTime;
use structopt::StructOpt;

use adagraph::prelude::*;
use adagraph_adjacency_graph::AdjGraph;
use adagraph_edge_addition_separators::enumerate_conditional_separators;
use adagraph_induced_subgraph::{AsInducedSubgraph, InducedSubgraph};
use adagraph_nauty::{BitSet, NautyGraph, NautyGraphVertexStoreWrapper};
use adagraph_separators::{CliqueMinimalSeparators, MinimalSeparators};
use adagraph_serde::{parse_dimancs, parse_pace};
use adagraph_takadas_recurrance::TakatasRecurrance;
use adagraph_triangulation::MaximumCardinalitySearchForMinimalTriangulation;

fn is_minimal_separator<G>(graph: &G, s: &G::VertexSet) -> bool
where
    G: AsInducedSubgraph,
{
    let mut components = graph.vertex_set();
    components.remove_set(s);
    graph
        .as_induced_subgraph(components)
        .connected_components()
        .filter(|c| graph.neighbor_set_exclusive(c) == *s)
        .nth(1)
        .is_some()
}

fn is_almost_clique<G>(graph: &G, vs: &G::VertexSet) -> bool
where
    G: AsInducedSubgraph,
{
    let target_degree = vs.len();

    let subgraph = graph.as_induced_subgraph(vs.clone());
    let mut count_n_minus_one = 0;
    let mut removed_vertex: Option<usize> = None;
    for v in vs.iter() {
        let degree = subgraph.vertex_neighbor_set(v).len();
        if degree + 2 == target_degree {
            count_n_minus_one += 1;
        } else if degree + 2 < target_degree {
            if removed_vertex.is_some() {
                return false;
            }
            removed_vertex = Some(degree);
        }
    }
    if let Some(removed_vertex_degree) = removed_vertex {
        count_n_minus_one <= (vs.len() - removed_vertex_degree - 1)
    } else {
        count_n_minus_one <= 2
    }
}

fn is_dense<G>(graph: &G, vs: &G::VertexSet) -> bool
where
    G: AsInducedSubgraph,
{
    let target_degree = vs.len();
    let subgraph = graph.as_induced_subgraph(vs.clone());
    vs.iter()
        .all(|v| subgraph.vertex_neighbor_set(v).len() >= target_degree - 2)
}

fn enumerate_edge_addition<G, S>(
    graph: &G,
    condition: &dyn Fn(&G::VertexSet) -> bool,
) -> Vec<G::VertexSet>
where
    G: AsInducedSubgraph,
    G::VertexSet: Ord,
    S: StoreWithKey<G::VertexIndex>,
{
    let mut visited = G::VertexSet::empty();
    let mut edges =
        MaximumCardinalitySearchForMinimalTriangulation::<G, S>::new(graph).flat_map(|v| {
            visited.insert(v);
            let mut result = graph.vertex_neighbor_set(v);
            result.intersect_set(&visited);
            result.into_iter().map(move |w| (v, w))
        });

    let s = enumerate_conditional_separators::<
        G::VertexIndex,
        G::VertexSet,
        S,
        &mut dyn Iterator<Item = (G::VertexIndex, G::VertexIndex)>,
        &dyn Fn(&G::VertexSet) -> bool,
    >(graph.vertex_set(), &mut edges, condition);
    s
}

#[derive(StructOpt)]
#[structopt(
    name = "enumerate-separator",
    about = "Enumerate minimal separators of a graph"
)]
struct Cli {
    #[structopt(help = "Output time in milliseconds", short = "t", long = "time")]
    get_time: bool,
    #[structopt(help = "Preprocessing", short = "p", long = "preprocess")]
    preprocess: bool,
    #[structopt(help = "Threading", short = "j", long = "threads", default_value = "1")]
    threads: usize,
    #[structopt(help = "Input file", parse(from_os_str))]
    input_graph_file: PathBuf,
    #[structopt(
        help = "Output file",
        short = "o",
        long = "output-file",
        parse(from_os_str)
    )]
    output_file: Option<PathBuf>,
    #[structopt(subcommand)]
    separator_type: SeparatorType,
}

#[derive(StructOpt, Copy, Clone, Eq, PartialEq)]
enum SeparatorEnumerationMethod {
    Standard,
    Filtering,
    Takatas,
    EdgeAddition,
}

#[derive(StructOpt, Copy, Clone, Eq, PartialEq)]
enum SeparatorEnumerationMethodMine {
    Mine,
    NotMine,
}

#[derive(StructOpt, Eq, PartialEq)]
enum SeparatorType {
    All {
        #[structopt(subcommand)]
        method: SeparatorEnumerationMethod,
    },
    AllAlmostCliques {
        #[structopt(subcommand)]
        method: SeparatorEnumerationMethod,
    },
    Clique,
    PairwiseParallelAlmostCliques {
        #[structopt(subcommand)]
        mine: SeparatorEnumerationMethodMine,
    },
    Dense {
        #[structopt(subcommand)]
        method: SeparatorEnumerationMethod,
    },
    OfSizeAtMost {
        #[structopt(short = "k")]
        k: usize,
        #[structopt(subcommand)]
        method: SeparatorEnumerationMethod,
    },
}

fn all_separators<G, S>(graph: G, method: SeparatorEnumerationMethod) -> BTreeSet<G::VertexSet>
where
    G: AsInducedSubgraph,
    G::OriginalGraph: BuildableGraph + EdgeAddableGraph + AsInducedSubgraph + Sized,
    G::VertexIndex: Ord,
    G::VertexSet: Ord,
    S: StoreWithKey<G::VertexIndex>,
{
    match method {
        SeparatorEnumerationMethod::Standard | SeparatorEnumerationMethod::Filtering => {
            let mut result = BTreeSet::new();
            MinimalSeparators::new(&graph, &mut result).for_each(drop);
            result
        }
        SeparatorEnumerationMethod::Takatas => {
            TakatasRecurrance::new(&graph, |_, _| true).collect()
        }
        SeparatorEnumerationMethod::EdgeAddition => {
            enumerate_edge_addition::<G, S>(&graph, &|_| true)
                .into_iter()
                .collect()
        }
    }
}

fn all_almost_clique_separators<G, S>(
    graph: G,
    method: SeparatorEnumerationMethod,
    preprocessing: bool,
) -> BTreeSet<G::VertexSet>
where
    G: AsInducedSubgraph + Send + Sync,
    G::OriginalGraph: BuildableGraph + EdgeAddableGraph + AsInducedSubgraph + Sized,
    G::VertexIndex: Ord + Send + Debug,
    G::VertexSet: Ord + Send + Debug,
    S: StoreWithKey<G::VertexIndex>,
{
    match method {
        SeparatorEnumerationMethod::Filtering => {
            MinimalSeparators::new(&graph, &mut BTreeSet::new())
                .filter(|vs| is_almost_clique(&graph, vs))
                .collect()
        }
        SeparatorEnumerationMethod::Standard => {
            if preprocessing {
                let mut separators = BTreeSet::new();
                let vs = graph.vertex_set();
                for v in graph.vertex_set().iter() {
                    let mut vs = vs.clone();
                    vs.remove(v);

                    separators.extend(
                        CliqueMinimalSeparators::<InducedSubgraph<G::OriginalGraph>, S>::new(
                            &graph.as_induced_subgraph(vs),
                            &mut BTreeSet::new(),
                        )
                        .map(|mut s| {
                            s.insert(v);
                            s
                        }),
                    );
                }
                separators
            } else {
                let mut separators = BTreeSet::new();
                let vs = graph.vertex_set();
                for v in graph.vertex_set().iter() {
                    let mut vs = vs.clone();
                    vs.remove(v);

                    let components: Vec<G::VertexSet> = graph
                        .as_induced_subgraph(vs)
                        .connected_components()
                        .collect();

                    if components.len() > 1 {
                        separators.insert(G::VertexSet::singleton(v));
                    }

                    components.into_iter().for_each(|c| {
                        separators.extend(
                            CliqueMinimalSeparators::<InducedSubgraph<G::OriginalGraph>, S>::new(
                                &graph.as_induced_subgraph(c),
                                &mut BTreeSet::new(),
                            )
                            .map(|mut s| {
                                s.insert(v);
                                s
                            })
                            .filter(|s| is_minimal_separator(&graph, s)),
                        );
                    });
                }
                separators
            }
        }
        SeparatorEnumerationMethod::Takatas => {
            TakatasRecurrance::new(&graph, |_, vs| is_almost_clique(&graph, vs)).collect()
        }
        SeparatorEnumerationMethod::EdgeAddition => {
            enumerate_edge_addition::<G, S>(&graph, &|vs| is_almost_clique(&graph, vs))
                .into_iter()
                .collect()
        }
    }
}

fn pairwise_parallel_almost_clique_separators<G, S>(
    graph: G,
    atoms: Vec<G::VertexSet>,
    mine: bool,
    preprocessed: bool,
) -> BTreeSet<G::VertexSet>
where
    G: AsInducedSubgraph + EdgeAddableGraph,
    G::OriginalGraph: Send,
    G::VertexIndex: Ord + Send + std::fmt::Debug,
    G::VertexSet: Ord + Send + Sync + std::fmt::Debug,
    S: StoreWithKey<G::VertexIndex>,
{
    match mine {
        true => {
            if !preprocessed {
                eprint!("Operation requires preprocessing");
            }
            let graph = &graph;
            let mut atoms: VecDeque<(
                AdjGraph<G::VertexIndex, G::VertexSet, S::Store<G::VertexSet>>,
                Option<G::VertexIndex>,
            )> = atoms
                .into_iter()
                .map(|a| (AdjGraph::new_from_graph(graph, a), None))
                .collect();
            let mut separators = Vec::new();
            while let Some((g, v)) = atoms.pop_back() {
                let mut ordered_vertices: VecDeque<G::VertexIndex> =
                    g.vertex_set().iter_increasing().collect();
                if let Some(v) = v {
                    while let Some(u) = ordered_vertices.pop_front() {
                        if v == u {
                            break;
                        }
                        ordered_vertices.push_back(u);
                    }
                }

                let result = ordered_vertices
                    .into_iter()
                    .filter_map(|v| {
                        let mut vs_without_v = g.vertex_set();
                        vs_without_v.remove(v);

                        let clique_separators = CliqueMinimalSeparators::<
                            InducedSubgraph<
                                AdjGraph<G::VertexIndex, G::VertexSet, S::Store<G::VertexSet>>,
                            >,
                            S,
                        >::new(
                            &g.as_induced_subgraph(vs_without_v),
                            &mut BTreeSet::new(),
                        )
                        .collect::<Vec<G::VertexSet>>();
                        if clique_separators.is_empty() {
                            None
                        } else {
                            Some((v, clique_separators))
                        }
                    })
                    .next();

                let (v, mut clique_separators) = if let Some((v, clique_separators)) = result {
                    (v, clique_separators)
                } else {
                    continue;
                };

                clique_separators.sort_by_key(|s| s.len());
                clique_separators.iter_mut().for_each(|s| {
                    s.insert(v);
                });

                let mut new_atoms = vec![g];

                for separator in clique_separators.iter() {
                    debug_assert!(
                        new_atoms
                            .iter()
                            .filter(|a| a.vertex_set().is_superset_of(&separator))
                            .count()
                            == 1
                    );
                    let atom = new_atoms
                        .drain_filter(|a| a.vertex_set().is_superset_of(&separator))
                        .next()
                        .unwrap();
                    let mut vs = atom.vertex_set();
                    vs.remove_set(&separator);

                    atom.clone()
                        .subgraph(&vs)
                        .connected_components()
                        .map(|mut c| {
                            c.insert_set(&separator);
                            c
                        })
                        .for_each(|c| {
                            let mut new_atom = atom.clone();
                            new_atom.subgraph(&c);
                            new_atom.saturate(separator.clone());
                            new_atoms.push(new_atom);
                        });
                }

                separators.extend(clique_separators);

                atoms.extend(new_atoms.into_iter().map(|a| (a, Some(v))));
            }
            separators.into_iter().collect()
        }
        false => {
            let mut graph = graph;
            let mut separators = BTreeSet::new();
            let mut consecutive = 1;
            let vs = graph.vertex_set();
            let n = vs.len();
            let mut graph = graph;
            for v in std::iter::repeat(()).flat_map(|_| vs.iter()) {
                if consecutive >= n + 1 {
                    return separators;
                }

                let mut without_v = vs.clone();
                without_v.remove(v);

                for almost_clique_separator in
                    CliqueMinimalSeparators::<InducedSubgraph<G::OriginalGraph>, S>::new(
                        &graph.as_induced_subgraph(without_v),
                        &mut BTreeSet::new(),
                    )
                    .map(|mut s| {
                        s.insert(v);
                        s
                    })
                    .collect::<Vec<G::VertexSet>>()
                    .into_iter()
                {
                    if is_minimal_separator(&graph, &almost_clique_separator) {
                        if separators.insert(almost_clique_separator.clone()) {
                            let mut visited = G::VertexSet::empty();
                            for v in almost_clique_separator.iter() {
                                for u in visited.iter() {
                                    if graph.get_edge(v, u).is_none() {
                                        graph.add_edge(v, u, ()).unwrap();
                                    }
                                }
                                visited.insert(v);
                            }
                            consecutive = 0;
                        }
                    }
                }
                consecutive += 1;
            }
            unreachable!();
        }
    }
}

fn dense_separators<G, S>(graph: G, method: SeparatorEnumerationMethod) -> BTreeSet<G::VertexSet>
where
    G: AsInducedSubgraph,
    G::OriginalGraph: BuildableGraph + EdgeAddableGraph + AsInducedSubgraph + Sized,
    G::VertexIndex: Ord,
    G::VertexSet: Ord,
    S: StoreWithKey<G::VertexIndex>,
{
    match method {
        SeparatorEnumerationMethod::Standard => {
            unimplemented!()
        }
        SeparatorEnumerationMethod::Filtering => {
            MinimalSeparators::new(&graph, &mut BTreeSet::new())
                .filter(|s| is_dense(&graph, s))
                .collect()
        }
        SeparatorEnumerationMethod::Takatas => {
            TakatasRecurrance::new(&graph, |_, s| is_dense(&graph, s))
                .filter(|s| is_dense(&graph, s))
                .collect()
        }
        SeparatorEnumerationMethod::EdgeAddition => {
            enumerate_edge_addition::<G, S>(&graph, &|s| is_dense(&graph, s))
                .into_iter()
                .collect()
        }
    }
}

fn k_separators<G, S>(
    graph: G,
    k: usize,
    method: SeparatorEnumerationMethod,
) -> BTreeSet<G::VertexSet>
where
    G: AsInducedSubgraph,
    G::OriginalGraph: BuildableGraph + EdgeAddableGraph + AsInducedSubgraph + Sized,
    G::VertexIndex: Ord,
    G::VertexSet: Ord,
    S: StoreWithKey<G::VertexIndex>,
{
    match method {
        SeparatorEnumerationMethod::Standard => {
            unimplemented!()
        }
        SeparatorEnumerationMethod::Filtering => {
            MinimalSeparators::new(&graph, &mut BTreeSet::new())
                .filter(|s: &G::VertexSet| s.len() <= k)
                .collect()
        }
        SeparatorEnumerationMethod::Takatas => {
            TakatasRecurrance::new(&graph, |_, s: &G::VertexSet| s.len() <= k)
                .filter(|s| s.len() <= k)
                .collect()
        }
        SeparatorEnumerationMethod::EdgeAddition => {
            enumerate_edge_addition::<G, S>(&graph, &|s: &G::VertexSet| s.len() <= k)
                .into_iter()
                .collect()
        }
    }
}

fn main() {
    let args = Cli::from_args();

    let input_graph = &args.input_graph_file;

    let content = match std::fs::read_to_string(input_graph) {
        Ok(content) => content,
        _ => {
            eprintln!("Error reading file {:?}", input_graph);
            return;
        }
    };

    let graph: NautyGraph = if input_graph.extension().map(|s| s.to_str()) == Some(Some("gr")) {
        parse_pace(&content).expect("Error parsing file")
    } else if input_graph.extension().map(|s| s.to_str()) == Some(Some("dimacs")) {
        parse_dimancs(&content).expect("Error parsing file")
    } else {
        eprintln!("Input file format unknown");
        return;
    };

    let start = ProcessTime::try_now().expect("Getting process time failed");

    let separators = if args.preprocess || SeparatorType::Clique == args.separator_type {
        let mut clique_separators: Vec<BitSet<usize>> = CliqueMinimalSeparators::<
            NautyGraph,
            NautyGraphVertexStoreWrapper,
        >::new(&graph, &mut BTreeSet::new())
        .collect();

        clique_separators.sort_by_key(|s| s.len());
        let mut atoms = vec![graph.vertex_set()];

        for s in clique_separators.iter() {
            debug_assert!(atoms.iter().filter(|a| a.is_superset_of(s)).count() == 1);
            let (atom_index, _) = atoms
                .iter()
                .enumerate()
                .filter(|(_, a)| a.is_superset_of(s))
                .next()
                .unwrap();
            let mut atom = atoms.remove(atom_index);
            atom.remove_set(s);
            debug_assert!(
                graph
                    .as_induced_subgraph(atom.clone())
                    .connected_components()
                    .count()
                    > 1
            );
            atoms.extend(
                graph
                    .as_induced_subgraph(atom)
                    .connected_components()
                    .map(|mut a| {
                        a.insert_set(s);
                        a
                    }),
            )
        }

        #[cfg(debug_assertions)]
        {
            println!("atoms {:?}", atoms);
        }

        debug_assert!(atoms.iter().all(|a| {
            CliqueMinimalSeparators::<
                    InducedSubgraph<NautyGraph>,
                    NautyGraphVertexStoreWrapper,
                >::new(&graph.as_induced_subgraph(a.clone()), &mut BTreeSet::new())
                .next()
                .is_none()
        }));

        let mut separators = match args.separator_type {
            SeparatorType::All { method } => atoms
                .into_iter()
                .flat_map(|a| {
                    all_separators::<InducedSubgraph<NautyGraph>, NautyGraphVertexStoreWrapper>(
                        graph.as_induced_subgraph(a),
                        method,
                    )
                })
                .collect(),
            SeparatorType::AllAlmostCliques { method } => atoms
                .into_iter()
                .flat_map(|a| {
                    all_almost_clique_separators::<
                        InducedSubgraph<NautyGraph>,
                        NautyGraphVertexStoreWrapper,
                    >(graph.as_induced_subgraph(a), method, true)
                })
                .collect(),
            SeparatorType::PairwiseParallelAlmostCliques { mine } => {
                pairwise_parallel_almost_clique_separators::<NautyGraph, NautyGraphVertexStoreWrapper>(
                    graph,
                    atoms,
                    mine == SeparatorEnumerationMethodMine::Mine,
                    true,
                )
            }
            SeparatorType::Dense { method } => atoms
                .into_iter()
                .flat_map(|a| {
                    dense_separators::<InducedSubgraph<NautyGraph>, NautyGraphVertexStoreWrapper>(
                        graph.as_induced_subgraph(a),
                        method,
                    )
                })
                .collect(),
            SeparatorType::OfSizeAtMost { k, method } => atoms
                .into_iter()
                .flat_map(|a| {
                    k_separators::<InducedSubgraph<NautyGraph>, NautyGraphVertexStoreWrapper>(
                        graph.as_induced_subgraph(a),
                        k,
                        method,
                    )
                })
                .collect(),
            SeparatorType::Clique => BTreeSet::new(),
        };
        separators.extend(clique_separators);
        separators
    } else {
        match args.separator_type {
            SeparatorType::All { method } => {
                all_separators::<NautyGraph, NautyGraphVertexStoreWrapper>(graph, method)
            }
            SeparatorType::AllAlmostCliques { method } => all_almost_clique_separators::<
                NautyGraph,
                NautyGraphVertexStoreWrapper,
            >(graph, method, false),
            SeparatorType::PairwiseParallelAlmostCliques { mine } => {
                let atoms = vec![graph.vertex_set()];
                pairwise_parallel_almost_clique_separators::<NautyGraph, NautyGraphVertexStoreWrapper>(
                    graph,
                    atoms,
                    mine == SeparatorEnumerationMethodMine::Mine,
                    false,
                )
            }
            SeparatorType::Dense { method } => {
                dense_separators::<NautyGraph, NautyGraphVertexStoreWrapper>(graph, method)
            }
            SeparatorType::OfSizeAtMost { k, method } => {
                k_separators::<NautyGraph, NautyGraphVertexStoreWrapper>(graph, k, method)
            }
            SeparatorType::Clique => unreachable!(),
        }
    };

    if args.get_time {
        let cpu_time = start
            .try_elapsed()
            .expect("Getting process time failed")
            .as_millis();
        println!(" {}", cpu_time);
    }

    if let Some(output_file) = args.output_file {
        std::fs::write(
            &output_file,
            format!(
                "{:?}\n",
                separators
                    .into_iter()
                    .map(|s| s.into_iter().collect())
                    .collect::<Vec<Vec<usize>>>()
            ),
        )
        .expect(&format!("File {:?}", output_file))
    }
}
