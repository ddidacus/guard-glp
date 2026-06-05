"""Smoke tests: the package and all its modules import cleanly (src-layout sanity)."""

import importlib

import pytest


def test_package_imports() -> None:
    import glp

    assert isinstance(glp.ROOT.as_posix(), str)


@pytest.mark.parametrize(
    "module",
    [
        "glp.denoiser",
        "glp.flow_matching",
        "glp.utils_acts",
        "glp.script_eval",
        "glp.script_probe",
        "glp.script_steer",
    ],
)
def test_submodule_imports(module: str) -> None:
    assert importlib.import_module(module) is not None


def test_public_symbols() -> None:
    from glp.denoiser import GLP, Denoiser, Normalizer, load_glp

    assert all(callable(obj) for obj in (GLP, Denoiser, Normalizer, load_glp))
