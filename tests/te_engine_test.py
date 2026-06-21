import tempfile
import unittest
from pathlib import Path

from onset.te.engine import _ecmp_routes, _load_topology, evaluate


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


if __name__ == "__main__":
    unittest.main()
