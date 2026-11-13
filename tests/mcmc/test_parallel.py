from multiprocessing import active_children, get_context
from time import monotonic, sleep
from unittest.mock import Mock

import pytest
from numpy import asarray, isfinite

from inference.mcmc import GibbsChain, ParallelTempering
from inference.mcmc import parallel


def normal_posterior(theta):
    theta = asarray(theta)
    return -0.5 * float(theta @ theta)


class BlockingChain(GibbsChain):
    def __init__(self, ready):
        super().__init__(
            posterior=normal_posterior, start=[1.0], display_progress=False
        )
        self.ready = ready

    def take_step(self):
        self.ready.set()
        sleep(60)


@pytest.fixture
def spawn_context(monkeypatch):
    context = get_context("spawn")
    for name in ("Process", "Pipe", "Event"):
        monkeypatch.setattr(parallel, name, getattr(context, name))
    return context


def mock_resources(monkeypatch, processes):
    event = Mock()
    event.is_set.return_value = False
    pipes = [(Mock(), Mock()) for _ in processes]
    monkeypatch.setattr(parallel, "Event", Mock(return_value=event))
    monkeypatch.setattr(parallel, "Pipe", Mock(side_effect=pipes))
    monkeypatch.setattr(parallel, "Process", Mock(side_effect=processes))
    return event, pipes


def test_parallel_tempering_context(spawn_context):
    chains = [
        GibbsChain(
            posterior=normal_posterior,
            start=[1.0],
            temperature=temperature,
            display_progress=False,
        )
        for temperature in (1.0, 2.0)
    ]
    pt = ParallelTempering(chains)
    pids = {process.pid for process in pt.processes}
    try:
        with pt as sampler:
            assert sampler is pt
            sampler.take_steps(5)
            sampler.swap()
            recovered = sampler.return_chains()
        assert all(chain.chain_length == 6 for chain in recovered)
        assert all(isfinite(chain.get_sample()).all() for chain in recovered)
        assert all(connection.closed for connection in pt.connections)
        assert not any(process.pid in pids for process in active_children())
        pt.shutdown()
        for operation in (
            lambda: pt.take_steps(1),
            pt.swap,
            lambda: pt.advance(1),
            lambda: pt.run_for(minutes=0),
            pt.return_chains,
            pt.__enter__,
        ):
            with pytest.raises(RuntimeError, match="shut down"):
                operation()
    finally:
        pt.shutdown()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_parallel_tempering_context_exception(monkeypatch, error_type):
    process = Mock(pid=1)
    process.is_alive.return_value = False
    event, pipes = mock_resources(monkeypatch, [process])
    pt = ParallelTempering([Mock(inv_temp=1.0)])

    with pytest.raises(error_type, match="body failed"):
        with pt:
            raise error_type("body failed")

    event.set.assert_called_once_with()
    process.join.assert_called_once()
    process.close.assert_called_once_with()
    pipes[0][0].close.assert_called_once_with()
    pipes[0][1].close.assert_called_once_with()
    pt.shutdown()
    process.join.assert_called_once()


@pytest.mark.parametrize("failure", ["construct", "start"])
def test_parallel_tempering_startup_rollback(monkeypatch, failure):
    first = Mock(pid=1)
    first.is_alive.return_value = False
    second = Mock(pid=None)
    second.is_alive.return_value = False
    event, pipes = mock_resources(monkeypatch, [first, second])
    error = RuntimeError("startup failed")
    if failure == "construct":
        parallel.Process.side_effect = [first, error]
    else:
        second.start.side_effect = error

    with pytest.raises(RuntimeError, match="startup failed"):
        ParallelTempering([Mock(inv_temp=1.0), Mock(inv_temp=0.5)])

    first.start.assert_called_once_with()
    event.set.assert_called_once_with()
    first.join.assert_called_once()
    first.close.assert_called_once_with()
    second.join.assert_not_called()
    if failure == "start":
        second.close.assert_called_once_with()
    for parent, child in pipes:
        parent.close.assert_called_once_with()
        child.close.assert_called_once_with()


def test_parallel_tempering_shutdown_escalation(monkeypatch):
    process = Mock(pid=1)
    event, pipes = mock_resources(monkeypatch, [process])
    pt = ParallelTempering([Mock(inv_temp=1.0)])
    process.is_alive.side_effect = [True, True, False, False]

    with pytest.warns(RuntimeWarning, match="forcibly stopped"):
        pt.shutdown(timeout=0)

    event.set.assert_called_once_with()
    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.join.call_count == 3
    process.close.assert_called_once_with()
    pipes[0][0].close.assert_called_once_with()
    pt.shutdown()
    process.close.assert_called_once_with()


def test_parallel_tempering_stuck_worker(spawn_context):
    ready = spawn_context.Event()
    pt = ParallelTempering([BlockingChain(ready)])
    pids = {process.pid for process in pt.processes}
    try:
        pt.connections[0].send({"task": "advance", "advance_count": 1})
        assert ready.wait(timeout=20)
        start = monotonic()
        with pytest.warns(RuntimeWarning, match="forcibly stopped"):
            pt.shutdown(timeout=0.05)
        assert monotonic() - start < 5
        assert not any(process.pid in pids for process in active_children())
        assert all(connection.closed for connection in pt.connections)
    finally:
        pt.shutdown()


def test_parallel_tempering_shutdown_survivor(monkeypatch):
    process = Mock(pid=1)
    mock_resources(monkeypatch, [process])
    pt = ParallelTempering([Mock(inv_temp=1.0)])
    process.is_alive.return_value = True

    with pytest.raises(RuntimeError, match="Unable to stop.*1"):
        pt.shutdown(timeout=0)

    process.close.assert_not_called()
    process.is_alive.return_value = False
    pt.shutdown()
    process.close.assert_called_once_with()


def test_parallel_tempering_preserves_body_exception(monkeypatch):
    process = Mock(pid=1)
    process.is_alive.return_value = False
    mock_resources(monkeypatch, [process])
    pt = ParallelTempering([Mock(inv_temp=1.0)])
    shutdown = pt.shutdown
    monkeypatch.setattr(pt, "shutdown", Mock(side_effect=ValueError("cleanup failed")))
    try:
        with pytest.raises(RuntimeError, match="body failed") as caught:
            with pt:
                raise RuntimeError("body failed")
        assert isinstance(caught.value.__cause__, ValueError)
        assert str(caught.value.__cause__) == "cleanup failed"
    finally:
        shutdown()


def test_parallel_tempering_rejects_nested_context(monkeypatch):
    process = Mock(pid=1)
    process.is_alive.return_value = False
    mock_resources(monkeypatch, [process])
    pt = ParallelTempering([Mock(inv_temp=1.0)])
    with pt:
        with pytest.raises(RuntimeError, match="cannot be nested"):
            with pt:
                pass
        process.close.assert_not_called()
