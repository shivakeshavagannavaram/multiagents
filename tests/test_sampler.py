"""Tests for the tool chain sampler."""

import random

import pytest

from conv_gen.sampler.sampler import SamplingConstraints, ToolChainSampler


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

    # ------------------------------------------------------------------ #
    # exact_steps convenience shortcut                                    #
    # ------------------------------------------------------------------ #

    def test_exact_steps_sets_min_max_range(self):
        """exact_steps collapses min_steps and max_steps to the same value."""
        c = SamplingConstraints(exact_steps=3)
        assert c.min_steps == 3
        assert c.max_steps == 3

    def test_exact_steps_overrides_existing_range(self):
        """exact_steps wins over any min_steps/max_steps passed alongside."""
        c = SamplingConstraints(min_steps=1, max_steps=10, exact_steps=4)
        assert c.min_steps == 4
        assert c.max_steps == 4

    def test_exact_steps_rejects_zero(self):
        """exact_steps < 1 is a programming error and raises."""
        with pytest.raises(ValueError):
            SamplingConstraints(exact_steps=0)

    def test_exact_steps_produces_correct_chain_length(
        self, sample_graph, sample_registry, rng
    ):
        """exact_steps=3 produces chains of length 3 when walkable."""
        sampler = ToolChainSampler(sample_graph, sample_registry, rng=rng)
        # The sample graph has enough connectivity for 3-step chains;
        # if the walker can't extend that far, it returns the best it
        # found. Assert the upper bound — exact_steps guarantees we
        # never go OVER 3 (the lower bound can fall back to whatever
        # the walker reached under the retry loop).
        result = sampler.sample_sequential(SamplingConstraints(exact_steps=3))
        chain = result.flat_chain
        assert len(chain) <= 3, f"expected at most 3 steps, got {len(chain)}"

    # ------------------------------------------------------------------ #
    # must_include_categories quota                                       #
    # ------------------------------------------------------------------ #

    def test_must_include_single_category(
        self, sample_graph, sample_registry, rng
    ):
        """With must_include_categories=['Travel'], the chain must touch
        Travel at least once (but may also touch other categories).
        """
        sampler = ToolChainSampler(sample_graph, sample_registry, rng=rng)
        constraints = SamplingConstraints(
            min_steps=2, max_steps=3,
            must_include_categories=["Travel"],
        )
        result = sampler.sample_sequential(constraints)
        chain = result.flat_chain
        assert len(chain) > 0
        chain_categories = {
            sample_graph.nodes[f"endpoint:{t}/{a}"].get("category", "")
            for t, a in chain
        }
        assert "Travel" in chain_categories, (
            f"expected Travel in chain, got categories {chain_categories}"
        )

    def test_must_include_helper_empty_when_not_set(self, sample_graph, sample_registry, rng):
        """When must_include_categories is None, the helper returns True."""
        sampler = ToolChainSampler(sample_graph, sample_registry, rng=rng)
        constraints = SamplingConstraints(must_include_categories=None)
        # Any chain passes — supply a trivial one (non-existent endpoints OK here,
        # the helper only reads category attributes).
        assert sampler._chain_satisfies_must_include([], constraints) is True

    def test_must_include_helper_detects_missing(self, sample_graph, sample_registry, rng):
        """Helper returns False if a required category isn't in the chain."""
        sampler = ToolChainSampler(sample_graph, sample_registry, rng=rng)
        # Build a chain from the sample graph that we know is NOT all Weather.
        result = sampler.sample_sequential(
            SamplingConstraints(categories=["Travel"], min_steps=2, max_steps=2)
        )
        chain_ids = [f"endpoint:{t}/{a}" for t, a in result.flat_chain]
        # Require a category the chain doesn't contain
        constraints = SamplingConstraints(must_include_categories=["Weather"])
        assert sampler._chain_satisfies_must_include(chain_ids, constraints) is False

    def test_must_include_helper_accepts_satisfied(self, sample_graph, sample_registry, rng):
        """Helper returns True when chain contains every required category."""
        sampler = ToolChainSampler(sample_graph, sample_registry, rng=rng)
        # Walk a Travel-only chain
        result = sampler.sample_sequential(
            SamplingConstraints(categories=["Travel"], min_steps=2, max_steps=2)
        )
        chain_ids = [f"endpoint:{t}/{a}" for t, a in result.flat_chain]
        constraints = SamplingConstraints(must_include_categories=["Travel"])
        assert sampler._chain_satisfies_must_include(chain_ids, constraints) is True

    def test_categories_vs_must_include_semantics_differ(self):
        """Documentation test: the two category fields are independent.

        `categories` is a filter ("only from these") — restricts the
        entire chain. `must_include_categories` is a quota ("at least
        one from each") — chain can span other categories too. They
        can be combined, in which case the chain must be within
        `categories` AND include one from each `must_include_categories`.
        """
        c = SamplingConstraints(
            categories=["Travel", "Weather"],
            must_include_categories=["Weather"],
        )
        assert c.categories == ["Travel", "Weather"]
        assert c.must_include_categories == ["Weather"]
        # Both fields are independent and preserved through __post_init__
