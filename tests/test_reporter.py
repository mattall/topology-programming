"""Tests for the reporter module."""

import os
import tempfile
import pytest
from onset.reporter import write_optimization_reports
from onset.base_types import (
    OptimizationResult,
    TopologySolution,
    BackendProvenance,
    OptimizerStatus,
)


class TestWriteOptimizationReports:
    """Tests for write_optimization_reports."""

    def _make_result_with_solutions(self):
        """Create a minimal OptimizationResult with one solution."""
        sol = TopologySolution(
            selected_edges=frozenset({("a", "b")}),
            added=frozenset({("a", "b")}),
            dropped=frozenset(),
            commodity_flows=None,
            aggregate_edge_loads={("a", "b"): 0.5},
            solver_mlu=0.5,
            validated_mlu=0.5,
            change_count=1,
            objective_value=2.5,
            stable_topology_id="abc123",
            legacy_topology_id=1.0,
            provenance=BackendProvenance.OPEN,
            proven_optimal=True,
        )
        return OptimizationResult(
            solutions=(sol,),
            status=OptimizerStatus.TOP_K_REACHED,
            wall_time=0.1,
            backend=BackendProvenance.OPEN,
            solve_count=1,
            baseline_feasible=True,
            baseline_mlu=0.3,
        )

    def test_writes_all_report_files(self):
        result = self._make_result_with_solutions()
        return_data = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            write_optimization_reports(
                result=result,
                return_data=return_data,
                iteration_abs_path=tmpdir,
                iteration_id="test1",
                te_method="-mcf",
                opt_time=0.1,
                multi_sol_time="NaN",
                multi_sol_number_best_sol="NaN",
                multi_sol_best_mlu="NaN",
            )
            expected_files = [
                "TotalSolutions.dat", "OptTime.dat", "OptimalTopoID.dat",
                "CurrTopoID.dat", "DopplerMinMLU.dat", "DopplerMLU.dat",
            ]
            for fname in expected_files:
                fpath = os.path.join(tmpdir, fname)
                assert os.path.exists(fpath), f"Missing: {fname}"

    def test_none_result_writes_nan(self):
        return_data = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            write_optimization_reports(
                result=None,
                return_data=return_data,
                iteration_abs_path=tmpdir,
                iteration_id="test2",
                te_method="-mcf",
                opt_time="NaN",
                multi_sol_time="NaN",
                multi_sol_number_best_sol="NaN",
                multi_sol_best_mlu="NaN",
            )
            # TotalSolutions.dat should contain 0
            ts_path = os.path.join(tmpdir, "TotalSolutions.dat")
            assert os.path.exists(ts_path)
            with open(ts_path) as f:
                content = f.read()
                assert "0" in content

    def test_no_solutions_writes_nan(self):
        result = OptimizationResult(
            solutions=(),
            status=OptimizerStatus.INFEASIBLE,
            wall_time=0.0,
            backend=BackendProvenance.OPEN,
            solve_count=1,
        )
        return_data = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            write_optimization_reports(
                result=result,
                return_data=return_data,
                iteration_abs_path=tmpdir,
                iteration_id="test3",
                te_method="-mcf",
                opt_time=0.0,
                multi_sol_time="NaN",
                multi_sol_number_best_sol="NaN",
                multi_sol_best_mlu="NaN",
            )
            ts_path = os.path.join(tmpdir, "TotalSolutions.dat")
            assert os.path.exists(ts_path)
            with open(ts_path) as f:
                content = f.read()
                assert "0" in content

    def test_return_data_populated(self):
        result = self._make_result_with_solutions()
        return_data = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            write_optimization_reports(
                result=result,
                return_data=return_data,
                iteration_abs_path=tmpdir,
                iteration_id="test4",
                te_method="-mcf",
                opt_time=0.1,
                multi_sol_time="NaN",
                multi_sol_number_best_sol="NaN",
                multi_sol_best_mlu="NaN",
            )
            assert "Optimal Topology ID" in return_data
            assert "Doppler Min MLU" in return_data
