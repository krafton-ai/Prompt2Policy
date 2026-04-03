"""Ensure contracts.py TypedDicts and schemas.py Pydantic models stay in sync."""

import pytest

from p2p.api.schemas import (
    AggregatedMetricsResponse,
    BenchmarkOptionsResponse,
    IterationRunEntry,
    MeanStdArray,
    RunMetricsResponse,
    SessionAnalysisResponse,
)
from p2p.contracts import (
    AggregatedMetrics,
    BenchmarkOptions,
    IterationRunInfo,
    MeanStdSeries,
    RunMetricsDetail,
    SessionAnalysis,
)

TYPE_PAIRS = [
    (IterationRunInfo, IterationRunEntry),
    (MeanStdSeries, MeanStdArray),
    (AggregatedMetrics, AggregatedMetricsResponse),
    (RunMetricsDetail, RunMetricsResponse),
    (SessionAnalysis, SessionAnalysisResponse),
    (BenchmarkOptions, BenchmarkOptionsResponse),
]


def _td_keys(td: type) -> set[str]:
    """Get all keys from a TypedDict."""
    return set(td.__required_keys__) | set(td.__optional_keys__)


@pytest.mark.parametrize(
    ("td", "pydantic"),
    TYPE_PAIRS,
    ids=[f"{td.__name__}-{p.__name__}" for td, p in TYPE_PAIRS],
)
def test_schema_fields_subset_of_contract(td, pydantic):
    """Pydantic model fields should be a subset of the TypedDict fields."""
    td_fields = _td_keys(td)
    schema_fields = set(pydantic.model_fields.keys())
    missing = schema_fields - td_fields
    assert not missing, (
        f"Schema {pydantic.__name__} has fields not in contract {td.__name__}: {missing}"
    )
