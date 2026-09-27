"""Tiny pytest-compatible runner for environments without pytest
(`python scripts/minitest.py tests/`). Supports: plain test functions,
`tmp_path`, session autouse fixtures without arguments, `pytest.raises`,
`pytest.approx`. Use real pytest when available: `python -m pytest -q`.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
import traceback
import types
from pathlib import Path

try:
    import pytest

    if __name__ == "__main__":
        sys.exit(pytest.main(["-q", *sys.argv[1:]]))
except ImportError:
    pass

class _Approx:
    def __init__(self, value, rel=1e-6, abs=1e-12):
        self.value, self.rel, self.abs = value, rel, abs

    def __eq__(self, other):
        return abs(other - self.value) <= max(self.rel * abs(self.value), self.abs)

class _Raises:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"expected {self.exc.__name__}")
        return issubclass(exc_type, self.exc)

def _build_shim() -> types.ModuleType:
    shim = types.ModuleType("pytest")
    fixtures: dict[str, tuple] = {}

    def fixture(func=None, scope="function", autouse=False):
        def deco(f):
            fixtures[f.__name__] = (f, scope, autouse)
            return f
        return deco(func) if func else deco

    shim.fixture = fixture
    shim.approx = _Approx
    shim.raises = _Raises
    shim.mark = types.SimpleNamespace(parametrize=lambda *a, **k: (lambda f: f), skipif=lambda *a, **k: (lambda f: f),
                                      skip=lambda *a, **k: (lambda f: f))
    shim.skip = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("skip"))
    shim._fixtures = fixtures
    return shim

def main(paths: list[str]) -> int:
    shim = _build_shim()
    sys.modules["pytest"] = shim
    files = []
    for p in paths or ["tests"]:
        path = Path(p)
        files += sorted(path.glob("test_*.py")) if path.is_dir() else [path]
    passed = failed = 0
    failures = []
    for file in files:
        spec = importlib.util.spec_from_file_location(file.stem, file)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        for name, (f, scope, autouse) in list(shim._fixtures.items()):
            if autouse and not inspect.signature(f).parameters:
                result = f()
                if inspect.isgenerator(result):
                    next(result, None)
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            kwargs = {}
            for param in inspect.signature(func).parameters:
                if param == "tmp_path":
                    kwargs[param] = Path(tempfile.mkdtemp())
                elif param in shim._fixtures:
                    kwargs[param] = shim._fixtures[param][0]()
            try:
                func(**kwargs)
                passed += 1
                print(f"PASS {file.name}::{name}")
            except Exception:
                failed += 1
                failures.append(f"{file.name}::{name}\n{traceback.format_exc()}")
                print(f"FAIL {file.name}::{name}")
        shim._fixtures.clear()
    for f in failures:
        print("\n" + "=" * 70 + "\n" + f)
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
