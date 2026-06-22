import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx

from onset.te.engine import (
    _all_pairs_paths as all_pairs_paths,
)
from onset.te.engine import (
    _boundary_capacity,
    _build_routing_usage,
    _ecmp_routes,
    _frt_decompose,
    _frt_paths,
    _load_topology,
    _normalize_scheme,
    _oblivious_paths_ft,
    _prune_scheme,
    _raecke_paths,
    evaluate,
)


class TeEngineTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.topology = self.root / "diamond.dot"
        self.topology.write_text(
            """digraph topology {
h1 [type=host]; h2 [type=host];
s1 [type=switch]; s2 [type=switch]; s3 [type=switch]; s4 [type=switch];
h1 -> s1 [cost=1, capacity="100Gbps"];
s1 -> s2 [cost=1, capacity="100Gbps"];
s1 -> s3 [cost=1, capacity="100Gbps"];
s2 -> s4 [cost=1, capacity="100Gbps"];
s3 -> s4 [cost=1, capacity="100Gbps"];
s4 -> h2 [cost=1, capacity="100Gbps"];
h2 -> s4 [cost=1, capacity="100Gbps"];
s4 -> s2 [cost=1, capacity="100Gbps"];
s4 -> s3 [cost=1, capacity="100Gbps"];
s2 -> s1 [cost=1, capacity="100Gbps"];
s3 -> s1 [cost=1, capacity="100Gbps"];
s1 -> h1 [cost=1, capacity="100Gbps"];
}
""",
            encoding="utf-8",
        )
        self.hosts = self.root / "diamond.hosts"
        self.hosts.write_text("h1\nh2\n", encoding="utf-8")
        self.matrix = self.root / "diamond.tm"
        self.matrix.write_text("0 60 60 0", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def evaluate(self, method):
        return evaluate(
            str(self.topology),
            str(self.matrix),
            str(self.hosts),
            method,
            str(self.root / "results"),
        )

    def test_ecmp_splits_equally_and_preserves_legacy_capacity_units(self):
        result = self.evaluate("-ecmp")

        self.assertEqual(result.num_paths, 4)
        self.assertAlmostEqual(result.max_congestion, 30 / (100 * 2**30))
        self.assertEqual(result.throughput, 1.0)
        self.assertTrue(
            (result.result_dir / "MaxExpCongestionVsIterations.dat").exists()
        )

    def test_mcf_uses_parallel_capacity_without_gurobi(self):
        result = self.evaluate("-mcf")

        self.assertAlmostEqual(result.max_congestion, 30 / (100 * 2**30))
        self.assertEqual(result.throughput, 1.0)

    def test_ecmp_reports_fair_share_loss_when_links_overload(self):
        self.matrix.write_text("0 300000000000 300000000000 0", encoding="utf-8")
        result = self.evaluate("-ecmp")
        expected_throughput = 2 * (100 * 2**30) / 300000000000

        self.assertGreater(result.max_congestion, 1.0)
        self.assertAlmostEqual(result.throughput, expected_throughput)
        self.assertAlmostEqual(result.congestion_loss, 1 - expected_throughput)

    def test_ecmp_distinguishes_unroutable_demand_from_congestion_loss(self):
        text = self.topology.read_text(encoding="utf-8")
        text = text.replace('s1 -> s3 [cost=1, capacity="100Gbps"];', "")
        text = text.replace('s1 -> s2 [cost=1, capacity="100Gbps"];', "")
        self.topology.write_text(text, encoding="utf-8")
        self.matrix.write_text("0 60 0 0", encoding="utf-8")

        result = self.evaluate("-ecmp")

        self.assertEqual(result.throughput, 0.0)
        self.assertEqual(result.congestion_loss, 0.0)
        self.assertEqual(result.failure_loss, 1.0)

    def test_ecmp_budget_is_deterministic(self):
        graph = _load_topology(str(self.topology))
        routes = _ecmp_routes(graph, {("h1", "h2"): 1.0}, budget=1)

        self.assertEqual(
            routes[("h1", "h2")],
            [(("h1", "s1", "s2", "s4", "h2"), 1.0)],
        )

    def test_semimcfraeke_finds_feasible_routes(self):
        result = self.evaluate("-semimcfraeke")

        self.assertAlmostEqual(result.max_congestion, 30 / (100 * 2**30))
        self.assertEqual(result.throughput, 1.0)
        self.assertGreater(result.num_paths, 0)

    def test_semimcfraekeft_finds_feasible_routes(self):
        result = self.evaluate("-semimcfraekeft")

        self.assertAlmostEqual(result.max_congestion, 30 / (100 * 2**30))
        self.assertEqual(result.throughput, 1.0)
        self.assertGreater(result.num_paths, 0)

    def test_semimcfraeke_handles_overload(self):
        self.matrix.write_text("0 300000000000 300000000000 0", encoding="utf-8")
        result = self.evaluate("-semimcfraeke")

        self.assertGreater(result.max_congestion, 1.0)
        self.assertGreater(result.throughput, 0.0)
        self.assertLess(result.throughput, 1.0)

    def test_semimcfraeke_distinguishes_unroutable_demand(self):
        text = self.topology.read_text(encoding="utf-8")
        text = text.replace('s1 -> s3 [cost=1, capacity="100Gbps"];', "")
        text = text.replace('s1 -> s2 [cost=1, capacity="100Gbps"];', "")
        self.topology.write_text(text, encoding="utf-8")
        self.matrix.write_text("0 60 0 0", encoding="utf-8")

        result = self.evaluate("-semimcfraeke")

        self.assertEqual(result.throughput, 0.0)
        self.assertEqual(result.congestion_loss, 0.0)
        self.assertEqual(result.failure_loss, 1.0)

    def test_semimcfraekeft_distinguishes_unroutable_demand(self):
        text = self.topology.read_text(encoding="utf-8")
        text = text.replace('s1 -> s3 [cost=1, capacity="100Gbps"];', "")
        text = text.replace('s1 -> s2 [cost=1, capacity="100Gbps"];', "")
        self.topology.write_text(text, encoding="utf-8")
        self.matrix.write_text("0 60 0 0", encoding="utf-8")

        result = self.evaluate("-semimcfraekeft")

        self.assertEqual(result.throughput, 0.0)
        self.assertEqual(result.congestion_loss, 0.0)
        self.assertEqual(result.failure_loss, 1.0)

    def test_frt_paths_non_root_lca(self):
        """_frt_paths must stop at the LCA, not traverse up to root."""
        tree = (
            "node",
            "r",
            {"a", "b", "c"},
            [
                ("leaf", "a", {"a"}),
                (
                    "node",
                    "x",
                    {"b", "c"},
                    [
                        ("leaf", "b", {"b"}),
                        ("leaf", "c", {"c"}),
                    ],
                ),
            ],
        )
        result = _frt_paths(tree, "b", "c")
        self.assertEqual(result, ["b", "x", "c"])

    def test_frt_paths_root_lca(self):
        tree = (
            "node",
            "r",
            {"a", "b"},
            [
                ("leaf", "a", {"a"}),
                ("leaf", "b", {"b"}),
            ],
        )
        result = _frt_paths(tree, "a", "b")
        self.assertEqual(result, ["a", "r", "b"])


class RaeckeTest(unittest.TestCase):
    """Tests for the Raecke/SMORE engine internals."""

    def _triangle_graph(self):
        """Bidirectional triangle s1↔s2↔s3↔s1 with two hosts h1→s1, h3→s3."""
        g = nx.DiGraph()
        for n in ("h1", "s1", "s2", "s3", "h3"):
            g.add_node(n, type="host" if n.startswith("h") else "switch")
        cap = float(2**30)  # 1 Gbps
        for a, b in (("s1", "s2"), ("s2", "s3"), ("s3", "s1")):
            g.add_edge(a, b, cost=1.0, capacity=cap)
            g.add_edge(b, a, cost=1.0, capacity=cap)
        for src, sw in (("h1", "s1"), ("s1", "h1"), ("h3", "s3"), ("s3", "h3")):
            g.add_edge(src, sw, cost=1.0, capacity=cap * 100)
        return g

    # -- boundary capacity -------------------------------------------------

    def test_boundary_capacity_single_node(self):
        g = nx.DiGraph()
        g.add_node("a", type="switch")
        g.add_node("b", type="switch")
        g.add_edge("a", "b", capacity=100.0)
        g.add_edge("b", "a", capacity=100.0)

        # Cluster {a}: only edge a→b crosses outward.
        self.assertEqual(_boundary_capacity(g, {"a"}), 100.0)
        # Cluster {b}: only edge b→a crosses outward.
        self.assertEqual(_boundary_capacity(g, {"b"}), 100.0)
        # Cluster {a,b}: all edges internal, boundary is zero.
        self.assertEqual(_boundary_capacity(g, {"a", "b"}), 0.0)

    def test_boundary_capacity_asymmetric_edges(self):
        g = nx.DiGraph()
        g.add_node("a", type="switch")
        g.add_node("b", type="switch")
        g.add_edge("a", "b", capacity=100.0)
        g.add_edge("b", "a", capacity=50.0)

        self.assertEqual(_boundary_capacity(g, {"a"}), 100.0)
        self.assertEqual(_boundary_capacity(g, {"b"}), 50.0)

    # -- building routing usage ---------------------------------------------

    def test_routing_usage_two_node_oracle(self):
        """Single bidirectional link: one child→parent path = exactly 1.0 both ways."""
        g = nx.DiGraph()
        g.add_node("a", type="switch")
        g.add_node("b", type="switch")
        g.add_edge("a", "b", cost=1.0, capacity=100.0)
        g.add_edge("b", "a", cost=1.0, capacity=100.0)
        phys = {("a", "b"): ["a", "b"], ("b", "a"): ["b", "a"]}
        frt = (
            "node",
            "b",
            {"a", "b"},
            [("leaf", "a", {"a"})],
        )
        usage, pruned, ptable = _build_routing_usage(g, frt, {"a", "b"}, phys)

        self.assertIsNotNone(pruned)
        self.assertEqual(usage[("a", "b")], 1.0)
        self.assertEqual(usage[("b", "a")], 1.0)

    def test_routing_usage_empty_child_skipped(self):
        """Endpoint-free FRT branch contributes no usage."""
        g = nx.DiGraph()
        g.add_node("a", type="switch")
        g.add_node("b", type="switch")
        g.add_node("c", type="switch")
        g.add_edge("a", "b", cost=1.0, capacity=100.0)
        g.add_edge("b", "a", cost=1.0, capacity=100.0)
        g.add_edge("b", "c", cost=1.0, capacity=100.0)
        g.add_edge("c", "b", cost=1.0, capacity=100.0)
        phys = {
            ("a", "b"): ["a", "b"],
            ("b", "a"): ["b", "a"],
            ("c", "b"): ["c", "b"],
            ("b", "c"): ["b", "c"],
        }
        frt = (
            "node",
            "b",
            {"a", "b", "c"},
            [
                ("leaf", "a", {"a"}),
                ("leaf", "c", set()),
            ],
        )
        usage, pruned, ptable = _build_routing_usage(g, frt, {"a"}, phys)

        self.assertIsNotNone(pruned)
        # Only a-b gets charged; c-b is skipped (no endpoints in c cluster).
        self.assertEqual(usage[("a", "b")], 1.0)
        self.assertEqual(usage[("b", "a")], 1.0)
        self.assertNotIn(("c", "b"), usage)

    def test_mw_edge_weights_sum_to_one(self):
        """The first and every later MW edge-weight vector sums to one."""
        g = self._triangle_graph()

        # Replicate MW iterations manually to check weight-vector sums.
        gc = g.copy()
        cumulative: dict[tuple[str, str], float] = {e: 0.0 for e in gc.edges()}
        epsilon = 0.1

        for iteration in range(5):
            dists, phys = all_pairs_paths(gc)
            frt = _frt_decompose(list(gc.nodes()), dists, rng=random.Random(43))
            usage, _pruned, _ptable = _build_routing_usage(g, frt, {"h1", "h3"}, phys)
            if not usage:
                break
            max_u = max(usage.values())
            if max_u <= 0:
                break
            for e, u in usage.items():
                cumulative[e] += u / max_u

            mw = {e: math.exp(epsilon * cumulative[e]) for e in gc.edges()}
            s = sum(mw.values())
            if s > 0:
                for e in gc.edges():
                    gc.edges[e]["cost"] = mw[e] / s

            # Assertion: edge costs must sum to one.
            cost_sum = sum(float(gc.edges[e]["cost"]) for e in gc.edges())
            self.assertAlmostEqual(
                cost_sum,
                1.0,
                places=6,
                msg=f"Iteration {iteration}: costs sum to {cost_sum}",
            )

    def test_routing_usage_dimensionless(self):
        """Usage values are non-negative, non-NaN (sanity check)."""
        g = self._triangle_graph()
        hosts = {"h1", "h3"}
        dists, phys = all_pairs_paths(g)
        frt = _frt_decompose(list(g.nodes()), dists, rng=random.Random(42))
        usage, pruned, ptable = _build_routing_usage(g, frt, hosts, phys)

        self.assertIsNotNone(pruned)
        self.assertGreater(len(usage), 0)
        for (_u, _v), val in usage.items():
            self.assertGreaterEqual(val, 0.0)
            self.assertFalse(math.isnan(val))

    # -- FRT tree invariants ------------------------------------------------

    def _all_frt_leaves(self, tree):
        """Recursively collect all leaves from an FRT tree."""
        leaves: list[tuple] = []
        if tree[0] == "leaf":
            leaves.append(tree)
        else:
            for child in tree[3]:
                leaves.extend(self._all_frt_leaves(child))
        return leaves

    def test_frt_every_leaf_is_singleton(self):
        """Every terminal leaf is centered on its sole vertex."""
        g = self._triangle_graph()
        dists, _phys = all_pairs_paths(g)
        for seed in (0, 1, 7, 42, 99):
            rng = random.Random(seed)
            frt = _frt_decompose(list(g.nodes()), dists, rng=rng)
            for leaf in self._all_frt_leaves(frt):
                cset = leaf[2]
                self.assertEqual(
                    len(cset),
                    1,
                    f"Seed {seed}: leaf {leaf[1]} has {len(cset)} vertices: {cset}",
                )
                self.assertEqual(leaf[1], next(iter(cset)))

    def test_frt_clustered_endpoints_separate(self):
        """Two endpoints clustered together at level zero are separated."""
        # Build a graph where two hosts are adjacent (distance 1) and
        # max_diameter is small, forcing initial_i = 0.  The FRT must still
        # decompose the two-host cluster into singletons.
        g = nx.DiGraph()
        g.add_node("h1", type="host")
        g.add_node("h2", type="host")
        g.add_edge("h1", "h2", cost=1.0, capacity=100.0)
        g.add_edge("h2", "h1", cost=1.0, capacity=100.0)
        dists, _phys = all_pairs_paths(g)
        frt = _frt_decompose(list(g.nodes()), dists, rng=random.Random(1))
        leaves = self._all_frt_leaves(frt)
        self.assertEqual(len(leaves), 2)
        for leaf in leaves:
            self.assertEqual(len(leaf[2]), 1)

    # -- MW convergence ------------------------------------------------------

    def test_mw_converges(self):
        """MW loop reaches total weight 1 via natural stopping condition."""
        g = self._triangle_graph()
        # The triangle with seed=42 should converge well within the safety
        # bound without hitting the RuntimeError guard.
        scheme = _raecke_paths(g, {"h1", "h3"}, seed=42)
        self.assertIn(("h1", "h3"), scheme)
        total = sum(scheme[("h1", "h3")].values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_raecke_returns_raw_complete_mass(self):
        """A converged triangle scheme needs no post-hoc normalization."""
        g = self._triangle_graph()
        scheme = _raecke_paths(g, {"h1", "h3"}, seed=42)

        for commodity in (("h1", "h3"), ("h3", "h1")):
            self.assertTrue(scheme[commodity])
            self.assertTrue(
                math.isclose(
                    sum(scheme[commodity].values()),
                    1.0,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
            )

    def test_raecke_disconnected_commodity_remains_empty(self):
        g = nx.DiGraph()
        g.add_node("h1", type="host")
        g.add_node("h2", type="host")

        scheme = _raecke_paths(g, {"h1", "h2"}, seed=42)

        self.assertEqual(scheme[("h1", "h2")], {})
        self.assertEqual(scheme[("h2", "h1")], {})

    # -- Raecke scheme invariants -------------------------------------------

    def test_raecke_produces_weighted_scheme(self):
        g = self._triangle_graph()
        scheme = _raecke_paths(g, {"h1", "h3"}, seed=42)

        self.assertIn(("h1", "h3"), scheme)
        self.assertGreater(len(scheme[("h1", "h3")]), 0)

    def test_raecke_probabilities_sum_to_one(self):
        g = self._triangle_graph()
        scheme = _raecke_paths(g, {"h1", "h3"}, seed=42)

        for paths in scheme.values():
            total = sum(paths.values())
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_raecke_paths_are_valid(self):
        """Every emitted path starts/ends at commodity hosts and uses real edges."""
        g = self._triangle_graph()
        scheme = _raecke_paths(g, {"h1", "h3"}, seed=42)

        for (src, dst), pps in scheme.items():
            for path in pps:
                self.assertEqual(path[0], src, f"Path starts at wrong node: {path}")
                self.assertEqual(path[-1], dst, f"Path ends at wrong node: {path}")
                for i in range(len(path) - 1):
                    self.assertTrue(
                        g.has_edge(path[i], path[i + 1]),
                        f"Missing edge {path[i]}→{path[i+1]} in path {path}",
                    )

    def test_raecke_deterministic_seed(self):
        """Same seed produces identical schemes."""
        g = self._triangle_graph()
        s1 = _raecke_paths(g, {"h1", "h3"}, seed=99)
        s2 = _raecke_paths(g, {"h1", "h3"}, seed=99)

        self.assertEqual(set(s1.keys()), set(s2.keys()))
        for comm in s1:
            self.assertEqual(
                {p: round(v, 10) for p, v in s1[comm].items()},
                {p: round(v, 10) for p, v in s2[comm].items()},
            )

    def test_raecke_multiple_seeds_produce_valid_paths(self):
        """Multiple different seeds all produce valid, non-empty path sets."""
        g = self._triangle_graph()
        for seed in (1, 2, 3, 7, 13):
            s = _raecke_paths(g, {"h1", "h3"}, seed=seed)
            pps = s[("h1", "h3")]
            self.assertGreater(len(pps), 0, f"No paths with seed {seed}")
            for path in pps:
                self.assertEqual(path[0], "h1")
                self.assertEqual(path[-1], "h3")
                for i in range(len(path) - 1):
                    self.assertTrue(g.has_edge(path[i], path[i + 1]))
            total = sum(pps.values())
            self.assertAlmostEqual(total, 1.0, places=6)

    # -- capacity scale invariance ------------------------------------------

    def test_raecke_scale_invariance(self):
        """Scaling all capacities by a constant should not change scheme."""

        def _make(cap_factor):
            g = nx.DiGraph()
            for n in ("h1", "s1", "s2", "h2"):
                g.add_node(n, type="host" if n.startswith("h") else "switch")
            base = 1000.0 * cap_factor
            g.add_edge("h1", "s1", cost=1.0, capacity=base * 100)
            g.add_edge("s1", "h1", cost=1.0, capacity=base * 100)
            g.add_edge("s1", "s2", cost=1.0, capacity=base)
            g.add_edge("s2", "s1", cost=1.0, capacity=base)
            g.add_edge("s2", "h2", cost=1.0, capacity=base * 100)
            g.add_edge("h2", "s2", cost=1.0, capacity=base * 100)
            return g

        g1 = _make(1.0)  # small units
        g2 = _make(1e9)  # large units
        s1 = _raecke_paths(g1, {"h1", "h2"}, seed=123)
        s2 = _raecke_paths(g2, {"h1", "h2"}, seed=123)

        self.assertEqual(set(s1[("h1", "h2")].keys()), set(s2[("h1", "h2")].keys()))
        for path in s1[("h1", "h2")]:
            self.assertAlmostEqual(
                s1[("h1", "h2")][path], s2[("h1", "h2")][path], places=10
            )

    # -- budget pruning -----------------------------------------------------

    def test_prune_scheme_respects_budget(self):
        scheme = {
            ("h1", "h2"): {
                ("h1", "s1", "s2", "h2"): 0.5,
                ("h1", "s1", "s3", "s4", "h2"): 0.3,
                ("h1", "s1", "s2", "s3", "s4", "h2"): 0.2,
            }
        }
        pruned = _prune_scheme(scheme, budget=2)
        self.assertEqual(len(pruned[("h1", "h2")]), 2)
        # Top two by probability should be the first two.
        self.assertEqual(pruned[("h1", "h2")][0], ("h1", "s1", "s2", "h2"))
        self.assertEqual(pruned[("h1", "h2")][1], ("h1", "s1", "s3", "s4", "h2"))

    def test_prune_scheme_keeps_all_when_budget_exceeds(self):
        scheme = {("h1", "h2"): {("h1", "s1", "h2"): 1.0}}
        pruned = _prune_scheme(scheme, budget=10)
        self.assertEqual(len(pruned[("h1", "h2")]), 1)

    # -- normalization ------------------------------------------------------

    def test_normalize_scheme_sums_to_one(self):
        scheme = {
            ("h1", "h2"): {("h1", "s1", "h2"): 0.2, ("h1", "s2", "h2"): 0.4},
        }
        norm = _normalize_scheme(scheme)
        self.assertAlmostEqual(sum(norm[("h1", "h2")].values()), 1.0, places=6)
        self.assertAlmostEqual(norm[("h1", "h2")][("h1", "s1", "h2")], 1 / 3, places=6)
        self.assertAlmostEqual(norm[("h1", "h2")][("h1", "s2", "h2")], 2 / 3, places=6)

    def test_normalize_scheme_zero_sum(self):
        scheme = {("h1", "h2"): {}}
        norm = _normalize_scheme(scheme)
        self.assertEqual(norm[("h1", "h2")], {})

    # -- FT fault-tolerance envelope ----------------------------------------

    def test_ft_envelope_no_baseline(self):
        """FT envelope merges only failure schemes, no separate baseline."""
        g = self._triangle_graph()
        ft = _oblivious_paths_ft(g, {"h1", "h3"})

        # FT should produce non-empty path sets.
        self.assertIn(("h1", "h3"), ft)
        self.assertGreater(len(ft[("h1", "h3")]), 0)

    def test_ft_envelope_probabilities_sum_to_one(self):
        g = self._triangle_graph()
        ft = _oblivious_paths_ft(g, {"h1", "h3"})

        for paths in ft.values():
            total = sum(paths.values())
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_ft_envelope_paths_are_valid(self):
        g = self._triangle_graph()
        ft = _oblivious_paths_ft(g, {"h1", "h3"})

        for (src, dst), pps in ft.items():
            for path in pps:
                self.assertEqual(path[0], src)
                self.assertEqual(path[-1], dst)
                for i in range(len(path) - 1):
                    self.assertTrue(g.has_edge(path[i], path[i + 1]))

    # -- FT envelope oracle tests -------------------------------------------
    #
    # Deterministic tests for _oblivious_paths_ft (YATES
    # all_failures_envelope semantics).  Monkeypatched _raecke_paths for
    # oracle assertions; one end-to-end regression runs the real solver.

    @staticmethod
    def _add_ss_edges(g, pairs, cap=float(2**30)):
        """Add bidirectional switch-switch edges."""
        for a, b in pairs:
            g.add_edge(a, b, cost=1.0, capacity=cap)
            g.add_edge(b, a, cost=1.0, capacity=cap)

    @staticmethod
    def _add_host_links(g, pairs, cap=None):
        """Add host-to-switch access links (both directions)."""
        if cap is None:
            cap = float(2**30) * 100.0
        for src, sw in pairs:
            g.add_edge(src, sw, cost=1.0, capacity=cap)
            g.add_edge(sw, src, cost=1.0, capacity=cap)

    def _empty_scheme(self, hosts):
        return {(s, t): {} for s in sorted(hosts) for t in sorted(hosts) if s != t}

    def test_ft_each_link_once(self):
        """(1) Each bidirectional switch-switch link yields exactly one call."""
        g = nx.DiGraph()
        for n in ("h1", "h2", "s1", "s2", "s3", "s4"):
            g.add_node(n, type="host" if n.startswith("h") else "switch")
        self._add_ss_edges(g, [("s1", "s2"), ("s1", "s3"), ("s2", "s4"), ("s3", "s4")])
        self._add_host_links(g, [("h1", "s1"), ("h2", "s4")])
        hosts = {"h1", "h2"}

        n = 0

        def mc(*a, **kw):
            nonlocal n
            n += 1
            return self._empty_scheme(hosts)

        with patch("onset.te.engine._raecke_paths", side_effect=mc):
            _oblivious_paths_ft(g, hosts)
        self.assertEqual(n, 4)

    def test_ft_filter_host_pair_reachability(self):
        """(2) Filter uses host-pair reachability, not full strong connectivity."""
        g = nx.DiGraph()
        for n in ("h1", "h3", "s1", "s2", "s3", "s4"):
            g.add_node(n, type="host" if n.startswith("h") else "switch")
        self._add_ss_edges(g, [("s1", "s2"), ("s2", "s3"), ("s3", "s4")])
        self._add_host_links(g, [("h1", "s1"), ("h3", "s3")])
        hosts = {"h1", "h3"}

        seen_fails = []

        def track(gf, *a, **kw):
            full_ss = {
                (u, v)
                for u, v in g.edges()
                if g.nodes[u].get("type") == "switch"
                and g.nodes[v].get("type") == "switch"
            }
            cur_ss = {
                (u, v)
                for u, v in gf.edges()
                if gf.nodes[u].get("type") == "switch"
                and gf.nodes[v].get("type") == "switch"
            }
            seen_fails.append(full_ss - cur_ss)
            return self._empty_scheme(hosts)

        with patch("onset.te.engine._raecke_paths", side_effect=track):
            _oblivious_paths_ft(g, hosts)

        s3_s4_gone = {("s3", "s4"), ("s4", "s3")}
        self.assertTrue(
            any(s3_s4_gone.issubset(m) for m in seen_fails),
            "s3-s4 failure (isolates switch-only s4) must be included",
        )

    def test_ft_disconnecting_failures_skipped(self):
        """(3) Failures that disconnect any host pair are skipped."""
        g = nx.DiGraph()
        for n in ("h1", "h2", "s1", "s2"):
            g.add_node(n, type="host" if n.startswith("h") else "switch")
        self._add_ss_edges(g, [("s1", "s2")])
        self._add_host_links(g, [("h1", "s1"), ("h2", "s2")])

        n = 0

        def mc(*a, **kw):
            nonlocal n
            n += 1
            return self._empty_scheme({"h1", "h2"})

        with patch("onset.te.engine._raecke_paths", side_effect=mc):
            _oblivious_paths_ft(g, {"h1", "h2"})
        self.assertEqual(n, 0)

    def test_ft_accumulate_then_normalize(self):
        """(4) Duplicate paths accumulate probs then normalise."""
        g = nx.DiGraph()
        for n in ("h1", "h2", "s1", "s2", "s3", "s4"):
            g.add_node(n, type="host" if n.startswith("h") else "switch")
        self._add_ss_edges(g, [("s1", "s2"), ("s1", "s3"), ("s2", "s4"), ("s3", "s4")])
        self._add_host_links(g, [("h1", "s1"), ("h2", "s4")])
        hosts = {"h1", "h2"}

        up = ("h1", "s1", "s2", "s4", "h2")
        lo = ("h1", "s1", "s3", "s4", "h2")
        up_r = ("h2", "s4", "s2", "s1", "h1")
        lo_r = ("h2", "s4", "s3", "s1", "h1")
        probs = iter([(0.8, 0.2), (0.3, 0.7), (0.5, 0.5), (0.6, 0.4)])

        def mk(*a, **kw):
            up_val, lo_val = next(probs)
            return {
                ("h1", "h2"): {up: up_val, lo: lo_val},
                ("h2", "h1"): {up_r: up_val, lo_r: lo_val},
            }

        with patch("onset.te.engine._raecke_paths", side_effect=mk):
            ft = _oblivious_paths_ft(g, hosts)

        self.assertAlmostEqual(ft[("h1", "h2")][up], 0.55, places=6)
        self.assertAlmostEqual(ft[("h1", "h2")][lo], 0.45, places=6)
        self.assertAlmostEqual(ft[("h2", "h1")][up_r], 0.55, places=6)
        self.assertAlmostEqual(ft[("h2", "h1")][lo_r], 0.45, places=6)

    def test_ft_no_baseline(self):
        """(5) No separate baseline (no-failure) scheme is merged."""
        g = nx.DiGraph()
        for n in ("h1", "h3", "s1", "s2", "s3"):
            g.add_node(n, type="host" if n.startswith("h") else "switch")
        self._add_ss_edges(g, [("s1", "s2"), ("s2", "s3"), ("s3", "s1")])
        self._add_host_links(g, [("h1", "s1"), ("h3", "s3")])
        hosts = {"h1", "h3"}
        full_edges = set(g.edges())

        with patch(
            "onset.te.engine._raecke_paths", return_value=self._empty_scheme(hosts)
        ) as m:
            _oblivious_paths_ft(g, hosts)

        self.assertEqual(m.call_count, 3)  # 3 SS pairs
        for ca in m.call_args_list:
            self.assertNotEqual(set(ca[0][0].edges()), full_edges)

    def test_ft_e2e_line_ghost(self):
        """End-to-end: _oblivious_paths_ft on line+ghost produces complete,
        valid scheme (regression: isolated switch-only vertex)."""
        g = nx.DiGraph()
        for n in ("h1", "h3", "s1", "s2", "s3", "s4"):
            g.add_node(n, type="host" if n.startswith("h") else "switch")
        self._add_ss_edges(g, [("s1", "s2"), ("s2", "s3"), ("s3", "s4")])
        self._add_host_links(g, [("h1", "s1"), ("h3", "s3")])

        ft = _oblivious_paths_ft(g, {"h1", "h3"})

        for comm in (("h1", "h3"), ("h3", "h1")):
            self.assertIn(comm, ft)
            pps = ft[comm]
            self.assertGreater(len(pps), 0)
            total = sum(pps.values())
            self.assertAlmostEqual(
                total, 1.0, places=6, msg=f"{comm} mass after FT envelope"
            )
            for path in pps:
                self.assertEqual(path[0], comm[0])
                self.assertEqual(path[-1], comm[1])
                for i in range(len(path) - 1):
                    self.assertTrue(g.has_edge(path[i], path[i + 1]))

    # -- end-to-end evaluation ----------------------------------------------

    def test_semimcfraeke_budget_caps_path_count(self):
        result = self._new_eval("-semimcfraeke", budget=1)
        # With budget=1, each commodity should have at most 1 path in the MCF.
        self.assertGreater(result.num_paths, 0)
        self.assertLessEqual(result.num_paths, 2)  # 2 commodities, 1 path each

    def test_semimcfraekeft_budget_caps_path_count(self):
        result = self._new_eval("-semimcfraekeft", budget=1)
        self.assertGreater(result.num_paths, 0)
        self.assertLessEqual(result.num_paths, 2)

    # -- helpers ------------------------------------------------------------

    def _new_eval(self, method, budget=3):
        """Run evaluate on a fresh temp topology."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            topo = root / "t.dot"
            hosts = root / "t.hosts"
            matrix = root / "t.tm"
            topo.write_text(
                """digraph topology {
h1 [type=host]; h2 [type=host];
s1 [type=switch]; s2 [type=switch]; s3 [type=switch]; s4 [type=switch];
h1 -> s1 [cost=1, capacity="100Gbps"];
s1 -> s2 [cost=1, capacity="100Gbps"];
s1 -> s3 [cost=1, capacity="100Gbps"];
s2 -> s4 [cost=1, capacity="100Gbps"];
s3 -> s4 [cost=1, capacity="100Gbps"];
s4 -> h2 [cost=1, capacity="100Gbps"];
h2 -> s4 [cost=1, capacity="100Gbps"];
s4 -> s2 [cost=1, capacity="100Gbps"];
s4 -> s3 [cost=1, capacity="100Gbps"];
s2 -> s1 [cost=1, capacity="100Gbps"];
s3 -> s1 [cost=1, capacity="100Gbps"];
s1 -> h1 [cost=1, capacity="100Gbps"];
}
""",
                encoding="utf-8",
            )
            hosts.write_text("h1\nh2\n", encoding="utf-8")
            matrix.write_text("0 60 60 0", encoding="utf-8")
            return evaluate(
                str(topo),
                str(matrix),
                str(hosts),
                method,
                str(root / "results"),
                budget=budget,
            )


if __name__ == "__main__":
    unittest.main()
