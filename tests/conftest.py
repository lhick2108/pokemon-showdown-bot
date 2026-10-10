"""Pytest configuration for tests that need live services."""

import shutil
import uuid
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests marked network (disabled by default)",
    )


@pytest.fixture
def tmp_path(request):
    """Provide an isolated temp path under the workspace on restricted Windows."""
    parent = Path(__file__).parent / "fixtures" / ".pytest-tmp"
    path = parent / f"{request.node.name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        if path.parent == parent and path.exists() and not path.is_symlink():
            shutil.rmtree(path)
        try:
            parent.rmdir()
        except OSError:
            pass


def pytest_configure(config):
    config.addinivalue_line("markers", "network: test requiring internet access")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip_network = pytest.mark.skip(reason="pass --run-network to run network tests")
    for item in items:
        if item.get_closest_marker("network"):
            item.add_marker(skip_network)
