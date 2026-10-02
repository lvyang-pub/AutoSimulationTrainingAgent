import os
import sys

# Make `src` importable when pytest is invoked from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    # Registered here (not only in pytest.ini) so the marker survives even when
    # this file is loaded via the root bridge or an explicit -c.
    config.addinivalue_line(
        "markers",
        "slow: integration tests that run the MuJoCo env or PPO "
        '(deselect with -m "not slow")',
    )
