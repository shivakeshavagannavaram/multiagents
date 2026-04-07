"""Tests for the tool chain sampler."""

import random

import pytest

from saplvl.sampler.sampler import SamplingConstraints, ToolChainSampler


class TestToolChainSampler:
    def test_sample_chain_default(self, sample_graph, sample_registry, rng):
        sampler = ToolChainSampler(sample_graph, sample_registry, rng=rng)
        chain = sampler.sample_chain()
        assert len(chain) >= 2
        assert all(isinstance(node, tuple) and len(node) == 2 for node in chain)

    def test_sample_chain_min_tools(self, sample_graph, sample_registry, rng):
        sampler = ToolChainSampler(sample_graph, sample_registry, rng=rng)
        constraints = SamplingConstraints(min_tools=2, max_tools=4, min_steps=3, max_steps=5)
        chain = sampler.sample_chain(constraints)
        distinct_tools = {node[0] for node in chain}
        assert len(distinct_tools) >= 2

    def test_sample_chain_respects_max_steps(self, sample_graph, sample_registry, rng):
        sampler = ToolChainSampler(sample_graph, sample_registry, rng=rng)
        constraints = SamplingConstraints(min_steps=2, max_steps=3)
        chain = sampler.sample_chain(constraints)
        assert len(chain) <= 4  # max_steps + possible diversification

    def test_sample_chain_category_filter(self, sample_graph, sample_registry, rng):
        sampler = ToolChainSampler(sample_graph, sample_registry, rng=rng)
        constraints = SamplingConstraints(categories=["Travel"], min_steps=2, max_steps=3)
        chain = sampler.sample_chain(constraints)
        # Seed should be from Travel category
        assert chain[0][0] in ("HotelFinder", "FlightSearch")

    def test_sample_chain_exclude_tools(self, sample_graph, sample_registry, rng):
        sampler = ToolChainSampler(sample_graph, sample_registry, rng=rng)
        constraints = SamplingConstraints(
            exclude_tools=["HotelFinder", "FlightSearch"],
            min_tools=1, min_steps=1, max_steps=3,
        )
        chain = sampler.sample_chain(constraints)
        for node in chain:
            assert node[0] not in ("HotelFinder", "FlightSearch")

    def test_sample_chain_required_tools(self, sample_graph, sample_registry, rng):
        sampler = ToolChainSampler(sample_graph, sample_registry, rng=rng)
        constraints = SamplingConstraints(
            required_tools=["WeatherAPI"],
            min_tools=1, min_steps=1, max_steps=3,
        )
        chain = sampler.sample_chain(constraints)
        assert chain[0][0] == "WeatherAPI"

    def test_sample_chain_deterministic_with_seed(self, sample_graph, sample_registry):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        sampler1 = ToolChainSampler(sample_graph, sample_registry, rng=rng1)
        sampler2 = ToolChainSampler(sample_graph, sample_registry, rng=rng2)
        chain1 = sampler1.sample_chain()
        chain2 = sampler2.sample_chain()
        assert chain1 == chain2

    def test_sample_parallel_group(self, sample_graph, sample_registry, rng):
        sampler = ToolChainSampler(sample_graph, sample_registry, rng=rng)
        groups = sampler.sample_parallel_group()
        assert len(groups) >= 1
        for group in groups:
            assert len(group) >= 1

    def test_empty_graph(self, sample_registry, rng):
        import networkx as nx
        empty_graph = nx.DiGraph()
        sampler = ToolChainSampler(empty_graph, sample_registry, rng=rng)
        chain = sampler.sample_chain()
        assert chain == []
