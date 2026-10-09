"""Explicit object graph (de)serialization support.

When object graph mode is enabled for an operation, every object serialized
through a `Schema` boundary is assigned an identifier the first time it is
encountered. Later encounters of the *same* object (identity, not equality)
are emitted as reference markers instead of being traversed again::

    {"$id": "0", "name": "root", "child": {"$id": "1", "parent": {"$ref": "0"}}}

On loading, the graph is rebuilt: identifiers and references share the very
same Python object, so circular references and shared objects round-trip.
Both forward references (reference appears before the matching identifier)
and backward references are supported.

The graph state is shared across all nested schemas participating in a single
operation through a `contextvars.ContextVar`. The state is only ever set when
object graph mode is explicitly enabled, so the default tree behavior is
unchanged.

This module is private API.
"""

from __future__ import annotations

import contextvars
import typing

from marshmallow.exceptions import SCHEMA

#: Key used for the identifier marker by default.
DEFAULT_ID_KEY = "$id"
#: Key used for the reference marker by default.
DEFAULT_REF_KEY = "$ref"

#: Sentinel meaning "object graph handling is suspended for this subtree"
#: (used by `Pluck`, which emits scalars and cannot carry graph markers).
_GRAPH_DISABLED: typing.Final = object()

_state: contextvars.ContextVar[typing.Any] = contextvars.ContextVar(
    "marshmallow_object_graph_state", default=None
)


# Error messages ------------------------------------------------------------

MSG_DUPLICATE_ID = "Duplicate object identifier: {id}."
MSG_DANGLING_REFERENCE = "Dangling reference to undefined object: {id}."
MSG_INVALID_ID = "Invalid object identifier: the identifier value must be a string."
MSG_INVALID_REFERENCE = (
    "Invalid object reference: the reference value must be a string and the "
    "marker may not contain any other field."
)
MSG_ID_AND_REFERENCE = "An object may not declare both an identifier and a reference."
MSG_ILLEGAL_POSITION = (
    "Object graph identifiers and references are only valid where a nested "
    "object is expected."
)


class _GraphReferenceError(Exception):
    """Internal control-flow signal, not a user-facing error.

    Raised while deserializing a mapping that consists solely of a reference
    marker. Carries the canonical (possibly still empty, in case of a forward
    reference) object so callers can short-circuit field processing without
    duplicating the shared instance.
    """

    def __init__(self, obj: typing.Any) -> None:
        self.obj = obj
        super().__init__("object graph reference marker")


class _GraphState:
    """State shared by every schema participating in one operation."""

    def __init__(self, *, id_key: str, ref_key: str) -> None:
        self.id_key = id_key
        self.ref_key = ref_key
        # Schema instances already checked for reserved-key collisions.
        self._checked_schemas: set[int] = set()
        # External (data_key) path currently being loaded. The path mirrors
        # how ValidationError messages are nested (field data keys and
        # collection indexes), so deferred errors can be reported at the
        # location of the offending reference.
        self.path: list[str | int] = []

    def push_path(self, segment: str | int) -> None:
        self.path.append(segment)

    def pop_path(self) -> None:
        self.path.pop()

    def check_schema(self, schema: typing.Any) -> None:
        """Fail atomically if ``schema`` exposes a field on the wire whose
        data key collides with one of the reserved graph keys."""
        schema_id = id(schema)
        if schema_id in self._checked_schemas:
            return
        self._checked_schemas.add(schema_id)
        reserved = {self.id_key, self.ref_key}
        external_keys: set[str] = set()
        for name, field_obj in schema.dump_fields.items():
            external_keys.add(
                field_obj.data_key if field_obj.data_key is not None else name
            )
        for name, field_obj in schema.load_fields.items():
            external_keys.add(
                field_obj.data_key if field_obj.data_key is not None else name
            )
        collisions = external_keys & reserved
        if collisions:
            raise ValueError(
                "Object graph mode reserves the field key(s) "
                f"{sorted(collisions)!r}, which collide with declared fields "
                f"of {schema!r}. Rename the conflicting fields or configure "
                "different object_graph_id_key/object_graph_ref_key keys."
            )


class _GraphDumpState(_GraphState):
    def __init__(self, *, id_key: str, ref_key: str) -> None:
        super().__init__(id_key=id_key, ref_key=ref_key)
        #: id() of serialized object -> graph identifier
        self.seen: dict[int, str] = {}
        self._counter = 0

    def seen_id(self, obj: typing.Any) -> str | None:
        """Return the existing identifier for ``obj`` or `None`."""
        return self.seen.get(id(obj))

    def register(self, obj: typing.Any) -> str:
        """Register ``obj`` before it is traversed and return its identifier.

        Registration happens before field serialization so that cycles
        terminate: when the same object is reached again from one of its own
        members, it is already known.
        """
        graph_id = str(self._counter)
        self._counter += 1
        self.seen[id(obj)] = graph_id
        return graph_id


class _GraphLoadState(_GraphState):
    def __init__(self, *, id_key: str, ref_key: str) -> None:
        super().__init__(id_key=id_key, ref_key=ref_key)
        #: graph identifier -> canonical object (shared identity)
        self.instances: dict[str, typing.Any] = {}
        #: identifiers that have an actual definition in the payload
        self.defined: set[str] = set()
        #: (identifier, path) for every reference that created a placeholder,
        #: used to report dangling references after the full traversal.
        self.reference_locations: set[tuple[str, tuple[str | int, ...]]] = set()

    def resolve(self, graph_id: str, factory: typing.Callable[[], typing.Any]):
        """Return the canonical object for ``graph_id``.

        For a forward reference whose definition has not been loaded yet a
        placeholder object is created immediately; it is populated in place
        once the matching identifier is encountered, so every reference stays
        connected to the same instance.
        """
        obj = self.instances.get(graph_id)
        if obj is None:
            obj = factory()
            self.instances[graph_id] = obj
            self.reference_locations.add((graph_id, tuple(self.path)))
        return obj

    def dangling_messages(self) -> typing.Iterator[dict]:
        """Yield ValidationError-style message dicts for references that
        never received a matching definition, keyed by reference location."""
        for graph_id, path in self.reference_locations:
            if graph_id in self.defined:
                continue
            node: typing.Any = {SCHEMA: [MSG_DANGLING_REFERENCE.format(id=graph_id)]}
            for segment in reversed(path):
                node = {segment: node}
            yield node


# Context management --------------------------------------------------------


def active() -> _GraphState | None:
    """Return the current graph state, or `None` outside graph operations
    (and inside suspended subtrees)."""
    current = _state.get()
    if current is _GRAPH_DISABLED:
        return None
    return current


def active_dump() -> _GraphDumpState | None:
    """Return the current state when serializing, else `None`."""
    current = active()
    return current if isinstance(current, _GraphDumpState) else None


def active_load() -> _GraphLoadState | None:
    """Return the current state when deserializing, else `None`."""
    current = active()
    return current if isinstance(current, _GraphLoadState) else None


@typing.overload
def enter(
    schema: typing.Any,
    *,
    object_graph: bool | None,
    for_load: typing.Literal[True],
) -> tuple[_GraphLoadState | None, contextvars.Token | None]: ...


@typing.overload
def enter(
    schema: typing.Any,
    *,
    object_graph: bool | None,
    for_load: typing.Literal[False] = False,
) -> tuple[_GraphDumpState | None, contextvars.Token | None]: ...


def enter(
    schema: typing.Any,
    *,
    object_graph: bool | None,
    for_load: bool = False,
) -> tuple[_GraphState | None, contextvars.Token | None]:
    """Enter a (de)serialization operation.

    Joins an in-progress graph operation when one exists, otherwise starts a
    new one when the mode is enabled. Returns ``(state, token)`` where
    ``token`` is only set for the operation that created the state; that
    operation owns finalization and context reset.
    """
    current = _state.get()
    if current is _GRAPH_DISABLED:
        return None, None
    if current is not None:
        current.check_schema(schema)
        return current, None
    enabled = schema.object_graph if object_graph is None else bool(object_graph)
    if not enabled:
        return None, None
    state_cls = _GraphLoadState if for_load else _GraphDumpState
    state = state_cls(
        id_key=schema.opts.object_graph_id_key,
        ref_key=schema.opts.object_graph_ref_key,
    )
    state.check_schema(schema)
    token = _state.set(state)
    return state, token


def exit_(token: contextvars.Token | None) -> None:
    if token is not None:
        _state.reset(token)


def suspend() -> contextvars.Token:
    """Suspend graph handling for a subtree (e.g. `Pluck` fields)."""
    return _state.set(_GRAPH_DISABLED)


def resume(token: contextvars.Token) -> None:
    _state.reset(token)
