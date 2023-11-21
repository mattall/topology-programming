import unittest
from onset.simulator import Simulation


class sim_import_json_test(unittest.TestCase):
    def setUp(self) -> None:
        self.my_sim = Simulation(
            "sndlib_abilene",
            12,
            "py_test",
            start_clean=True,
            iterations=1,
            fallow_tx_allocation_strategy="read_capacity",
        )

    def test_import(self):
        self.my_sim = Simulation(
            "sndlib_abilene", 12, "py_test", start_clean=True
        )
        self.assertIsNotNone(self.my_sim)

    def test_run_sim(self):
        self.my_sim.perform_sim(self.demand_factor=100)
