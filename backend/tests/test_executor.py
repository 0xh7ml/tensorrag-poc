import asyncio
import shutil
import tempfile

import pytest

from app.models.pipeline import EdgeConfig, NodeConfig, PipelineRequest
from app.controllers.executor import execute_pipeline
from app.services.storage import StorageService
from app.ws.status import WSManager


@pytest.fixture
def storage():
    tmp = tempfile.mkdtemp()
    svc = StorageService(base_dir=tmp)
    yield svc
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def ws():
    return WSManager()


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_full_pipeline(storage, ws):
    """Test a complete pipeline: load → split → define → train → evaluate."""
    pipeline = PipelineRequest(
        pipeline_id="full-test",
        nodes=[
            NodeConfig(
                id="load",
                type="data_load",
                config={"source": "sample", "sample_name": "california_housing"},
            ),
            NodeConfig(
                id="split",
                type="data_split",
                config={"train_ratio": 0.8, "test_ratio": 0.2, "random_seed": 42},
            ),
            NodeConfig(
                id="define",
                type="model_define",
                config={"model_type": "linear_regression"},
            ),
            NodeConfig(
                id="train",
                type="train",
                config={"target_column": "MedHouseVal"},
            ),
            NodeConfig(
                id="eval",
                type="evaluate",
                config={
                    "metrics": ["mse", "r2"],
                    "target_column": "MedHouseVal",
                },
            ),
        ],
        edges=[
            EdgeConfig(
                source="load", target="split",
                source_output="dataset", target_input="dataset",
            ),
            EdgeConfig(
                source="split", target="train",
                source_output="train_dataset", target_input="train_dataset",
            ),
            EdgeConfig(
                source="define", target="train",
                source_output="model_spec", target_input="model_spec",
            ),
            EdgeConfig(
                source="train", target="eval",
                source_output="trained_model", target_input="trained_model",
            ),
            EdgeConfig(
                source="split", target="eval",
                source_output="test_dataset", target_input="test_dataset",
            ),
        ],
    )

    state = _run(execute_pipeline(pipeline, storage, ws, None))

    assert state.status == "completed"
    assert all(s == "completed" for s in state.node_statuses.values())
    assert len(state.errors) == 0
    assert "dataset" in state.node_outputs["load"]
    assert "trained_model" in state.node_outputs["train"]
    assert "eval_report" in state.node_outputs["eval"]


def test_node_statuses_tracked(storage, ws):
    """Simple 2-node pipeline to verify status tracking."""
    pipeline = PipelineRequest(
        pipeline_id="status-test",
        nodes=[
            NodeConfig(
                id="load",
                type="data_load",
                config={"source": "sample", "sample_name": "california_housing"},
            ),
        ],
        edges=[],
    )

    state = _run(execute_pipeline(pipeline, storage, ws, None))
    assert state.status == "completed"
    assert state.node_statuses["load"] == "completed"


def test_failed_card_aborts_pipeline(storage, ws):
    """A card with bad config should fail and abort the pipeline."""
    pipeline = PipelineRequest(
        pipeline_id="fail-test",
        nodes=[
            NodeConfig(
                id="define",
                type="model_define",
                config={"model_type": "nonexistent_model"},
            ),
        ],
        edges=[],
    )

    state = _run(execute_pipeline(pipeline, storage, ws, None))
    assert state.status == "failed"
    assert state.node_statuses["define"] == "failed"
    assert "define" in state.errors
