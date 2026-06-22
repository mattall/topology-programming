"""Unit tests for validation functions."""

from __future__ import annotations

import pytest

import onset.validation as validation


class TestVerifyTopo:
    def test_finds_gml_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(validation, "SCRIPT_HOME", str(tmp_path))
        gml_dir = tmp_path / "data" / "graphs" / "gml"
        gml_dir.mkdir(parents=True)
        (gml_dir / "testnet.gml").write_text("graph [ ]")
        result = validation.verify_topo("testnet", False, "baseline")
        assert result.endswith("testnet.gml")

    def test_finds_json_when_gml_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(validation, "SCRIPT_HOME", str(tmp_path))
        json_dir = tmp_path / "data" / "graphs" / "json"
        json_dir.mkdir(parents=True)
        (json_dir / "testnet.json").write_text("{}")
        result = validation.verify_topo("testnet", False, "baseline")
        assert result.endswith("testnet.json")

    def test_missing_topo_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(validation, "SCRIPT_HOME", str(tmp_path))
        with pytest.raises(SystemExit):
            validation.verify_topo("noexist", False, "baseline")


class TestCreateHostFile:
    def test_creates_host_file(self, tmp_path):
        hosts_path = str(tmp_path / "testnet.hosts")
        validation.create_host_file(hosts_path, 3)
        with open(hosts_path) as f:
            lines = f.readlines()
        assert lines == ["h1\n", "h2\n", "h3\n"]


class TestVerifyHosts:
    def test_uses_existing_host_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(validation, "SCRIPT_HOME", str(tmp_path))
        hosts_dir = tmp_path / "data" / "hosts"
        hosts_dir.mkdir(parents=True)
        (hosts_dir / "testnet.hosts").write_text("h1\nh2\n")
        result = validation.verify_hosts("testnet", False, 2)
        assert result.endswith("testnet.hosts")

    def test_creates_missing_host_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(validation, "SCRIPT_HOME", str(tmp_path))
        hosts_dir = tmp_path / "data" / "hosts"
        hosts_dir.mkdir(parents=True)
        result = validation.verify_hosts("testnet", False, 3)
        assert result.endswith("testnet.hosts")
        with open(result) as f:
            assert len(f.readlines()) == 3

    def test_wrong_line_count_recreates(self, tmp_path, monkeypatch):
        monkeypatch.setattr(validation, "SCRIPT_HOME", str(tmp_path))
        hosts_dir = tmp_path / "data" / "hosts"
        hosts_dir.mkdir(parents=True)
        (hosts_dir / "testnet.hosts").write_text("h1\n")
        result = validation.verify_hosts("testnet", False, 3)
        with open(result) as f:
            assert len(f.readlines()) == 3


class TestVerifyTraffic:
    def test_uses_existing_valid_traffic_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(validation, "SCRIPT_HOME", str(tmp_path))
        traffic_dir = tmp_path / "data" / "traffic"
        traffic_dir.mkdir(parents=True)
        (traffic_dir / "testnet.txt").write_text(
            "10 20 30 40\n10 20 30 40\n10 20 30 40\n"
        )
        result = validation.verify_traffic("testnet", "", False, 2, 2, 100)
        assert result.endswith("testnet.txt")

    def test_explicit_traffic_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(validation, "SCRIPT_HOME", str(tmp_path))
        tf = tmp_path / "custom.tm"
        tf.write_text("10 20 30 40\n10 20 30 40\n")
        result = validation.verify_traffic("testnet", str(tf), False, 2, 2, 100)
        assert result == str(tf)


class TestValidateSimulationInputs:
    def test_orchestrates_all_three(self, tmp_path, monkeypatch):
        monkeypatch.setattr(validation, "SCRIPT_HOME", str(tmp_path))
        gml_dir = tmp_path / "data" / "graphs" / "gml"
        hosts_dir = tmp_path / "data" / "hosts"
        traffic_dir = tmp_path / "data" / "traffic"
        for d in (gml_dir, hosts_dir, traffic_dir):
            d.mkdir(parents=True)
        (gml_dir / "testnet.gml").write_text("graph [ ]")
        (hosts_dir / "testnet.hosts").write_text("h1\nh2\n")
        (traffic_dir / "testnet.txt").write_text("10 20 30 40\n10 20 30 40\n")
        topo, hosts, traffic = validation.validate_simulation_inputs(
            network_name="testnet",
            shakeroute=False,
            topology_programming_method="baseline",
            traffic_file="",
            start_clean=False,
            num_hosts=2,
            iterations=2,
            magnitude=100,
        )
        assert topo.endswith("testnet.gml")
        assert hosts.endswith("testnet.hosts")
        assert traffic.endswith("testnet.txt")
