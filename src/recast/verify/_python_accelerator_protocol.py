"""Strict, pickle-free values exchanged with accelerator verifier workers."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from typing import Any

REQUEST_SCHEMA = "recast.python-accelerator-worker-request.v1"
RESPONSE_SCHEMA = "recast.python-accelerator-worker-response.v1"

MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
MAX_ARRAY_BYTES = 16 * 1024 * 1024
MAX_CONTAINER_ITEMS = 100_000
MAX_DEPTH = 32


class ProtocolError(ValueError):
    """The isolated worker protocol is malformed or exceeds a bound."""


def canonical_bytes(value: object) -> bytes:
    """Return the sole accepted JSON encoding for worker requests/results."""

    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise ProtocolError("worker document is not canonical JSON") from error
    if len(encoded) > MAX_DOCUMENT_BYTES:
        raise ProtocolError("worker document exceeds the protocol bound")
    return encoded


def decode_document(payload: bytes) -> dict[str, Any]:
    """Strictly decode one canonical, bounded worker document."""

    if type(payload) is not bytes or len(payload) > MAX_DOCUMENT_BYTES:
        raise ProtocolError("worker document bytes are invalid")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError("worker document is not UTF-8 JSON") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ProtocolError("worker document must be a JSON object")
    if canonical_bytes(value) != payload:
        raise ProtocolError("worker document is not canonical")
    return value


def encode_value(np: Any, value: Any, *, _depth: int = 0) -> dict[str, Any]:
    """Encode numerical observations without importing classes from project code."""

    if _depth > MAX_DEPTH:
        raise ProtocolError("worker value nesting exceeds the protocol bound")
    if value is None:
        return {"kind": "none"}
    if type(value) is bool:
        return {"kind": "bool", "value": value}
    if type(value) is int:
        return {"kind": "int", "value": str(value)}
    if type(value) is float:
        return {"kind": "float", "value": value.hex()}
    if type(value) is complex:
        return {
            "kind": "complex",
            "real": value.real.hex(),
            "imag": value.imag.hex(),
        }
    if type(value) is str:
        return {"kind": "str", "value": value}
    if type(value) is bytes:
        return {"kind": "bytes", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, tuple):
        _bounded_container(value)
        return {
            "kind": "tuple",
            "items": [encode_value(np, item, _depth=_depth + 1) for item in value],
        }
    if isinstance(value, list):
        _bounded_container(value)
        return {
            "kind": "list",
            "items": [encode_value(np, item, _depth=_depth + 1) for item in value],
        }
    if isinstance(value, Mapping):
        if len(value) > MAX_CONTAINER_ITEMS or any(not isinstance(key, str) for key in value):
            raise ProtocolError("worker mapping is not a bounded string-keyed object")
        return {
            "kind": "dict",
            "items": [
                [key, encode_value(np, value[key], _depth=_depth + 1)]
                for key in sorted(value)
            ],
        }

    try:
        array = np.asarray(value)
    except Exception as error:
        raise ProtocolError("worker value has no safe array representation") from error
    if array.dtype.hasobject:
        raise ProtocolError("worker value requires an unsafe object array")
    contiguous = np.ascontiguousarray(array)
    payload = contiguous.tobytes(order="C")
    if len(payload) > MAX_ARRAY_BYTES:
        raise ProtocolError("worker array exceeds the protocol bound")
    return {
        "kind": "numpy_scalar" if isinstance(value, np.generic) else "ndarray",
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "data": base64.b64encode(payload).decode("ascii"),
    }


def decode_value(np: Any, document: object, *, _depth: int = 0) -> Any:
    """Decode only values produced by :func:`encode_value`."""

    if _depth > MAX_DEPTH or not isinstance(document, dict):
        raise ProtocolError("worker value is not a bounded tagged object")
    kind = document.get("kind")
    if kind == "none" and set(document) == {"kind"}:
        return None
    if kind == "bool" and set(document) == {"kind", "value"} and type(document["value"]) is bool:
        return document["value"]
    if kind == "int" and set(document) == {"kind", "value"}:
        value = document["value"]
        if not isinstance(value, str) or not value or len(value) > 10_000:
            raise ProtocolError("worker integer is invalid")
        try:
            return int(value)
        except ValueError as error:
            raise ProtocolError("worker integer is invalid") from error
    if kind == "float" and set(document) == {"kind", "value"}:
        return _hex_float(document["value"])
    if kind == "complex" and set(document) == {"kind", "real", "imag"}:
        return complex(_hex_float(document["real"]), _hex_float(document["imag"]))
    if kind == "str" and set(document) == {"kind", "value"}:
        value = document["value"]
        if not isinstance(value, str):
            raise ProtocolError("worker string is invalid")
        return value
    if kind == "bytes" and set(document) == {"kind", "data"}:
        return _base64(document["data"])
    if kind in {"tuple", "list"} and set(document) == {"kind", "items"}:
        items = document["items"]
        if not isinstance(items, list) or len(items) > MAX_CONTAINER_ITEMS:
            raise ProtocolError("worker sequence is invalid")
        decoded = [decode_value(np, item, _depth=_depth + 1) for item in items]
        return tuple(decoded) if kind == "tuple" else decoded
    if kind == "dict" and set(document) == {"kind", "items"}:
        items = document["items"]
        if not isinstance(items, list) or len(items) > MAX_CONTAINER_ITEMS:
            raise ProtocolError("worker mapping is invalid")
        result: dict[str, Any] = {}
        previous: str | None = None
        for pair in items:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or pair[0] in result
                or (previous is not None and pair[0] <= previous)
            ):
                raise ProtocolError("worker mapping entries are not canonical")
            result[pair[0]] = decode_value(np, pair[1], _depth=_depth + 1)
            previous = pair[0]
        return result
    if kind in {"ndarray", "numpy_scalar"} and set(document) == {
        "kind",
        "dtype",
        "shape",
        "data",
    }:
        dtype_name = document["dtype"]
        shape = document["shape"]
        if (
            not isinstance(dtype_name, str)
            or len(dtype_name) > 128
            or not isinstance(shape, list)
            or len(shape) > 32
            or any(
                type(size) is not int or size < 0 or size > MAX_CONTAINER_ITEMS
                for size in shape
            )
        ):
            raise ProtocolError("worker array metadata is invalid")
        try:
            dtype = np.dtype(dtype_name)
        except Exception as error:
            raise ProtocolError("worker array dtype is invalid") from error
        if dtype.hasobject:
            raise ProtocolError("worker object arrays are forbidden")
        data = _base64(document["data"])
        expected_items = 1
        for size in shape:
            expected_items *= size
            if expected_items * dtype.itemsize > MAX_ARRAY_BYTES:
                raise ProtocolError("worker array exceeds the protocol bound")
        if len(data) != expected_items * dtype.itemsize:
            raise ProtocolError("worker array byte count is invalid")
        array = np.frombuffer(data, dtype=dtype).copy().reshape(tuple(shape))
        if kind == "numpy_scalar":
            if array.shape != ():
                raise ProtocolError("worker NumPy scalar has a non-scalar shape")
            return array[()]
        return array
    raise ProtocolError("worker value tag or fields are invalid")


def _bounded_container(value: tuple[Any, ...] | list[Any]) -> None:
    if len(value) > MAX_CONTAINER_ITEMS:
        raise ProtocolError("worker sequence exceeds the protocol bound")


def _hex_float(value: object) -> float:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ProtocolError("worker hexadecimal float is invalid")
    try:
        return float.fromhex(value)
    except ValueError as error:
        raise ProtocolError("worker hexadecimal float is invalid") from error


def _base64(value: object) -> bytes:
    if not isinstance(value, str) or len(value) > (MAX_ARRAY_BYTES * 4 // 3 + 8):
        raise ProtocolError("worker base64 payload is invalid")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ProtocolError("worker base64 payload is invalid") from error
    if base64.b64encode(decoded).decode("ascii") != value:
        raise ProtocolError("worker base64 payload is not canonical")
    return decoded
