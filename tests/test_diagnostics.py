import pytest

from georag.diagnostics import collect_diagnostics, select_device


def test_cpu_diagnostic_executes_real_tensor_operation() -> None:
    result = collect_diagnostics("cpu", matrix_size=8)
    assert result["selected_device"] == "cpu"
    assert result["operation_shape"] == [8, 8]
    assert result["operation_finite"] is True


def test_unknown_device_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        select_device("quantum")
