"""Tests for the explicit object graph (de)serialization mode.

Object graph mode emits identifier/reference markers so that circular
references and shared objects round-trip through ``Nested`` fields. The
default tree behavior must remain unchanged when the mode is not enabled.
"""

import json
import typing

import pytest

from marshmallow import (
    EXCLUDE,
    INCLUDE,
    Schema,
    ValidationError,
    fields,
    post_dump,
    post_load,
    pre_dump,
    pre_load,
    validates,
    validates_schema,
)

ID = "$id"
REF = "$ref"
DANGLING = "Dangling reference to undefined object: {id}."
DUPLICATE = "Duplicate object identifier: {id}."
ILLEGAL_POSITION = (
    "Object graph identifiers and references are only valid where a nested "
    "object is expected."
)
INVALID_ID = "Invalid object identifier: the identifier value must be a string."
INVALID_REFERENCE = (
    "Invalid object reference: the reference value must be a string and the "
    "marker may not contain any other field."
)
ID_AND_REFERENCE = "An object may not declare both an identifier and a reference."


class NodeSchema(Schema):
    name = fields.Str()
    parent = fields.Nested(lambda: NodeSchema(), allow_none=True)
    friends = fields.List(fields.Nested(lambda: NodeSchema()))


def cyclic_graph() -> tuple[dict[str, typing.Any], dict[str, typing.Any]]:
    root: dict[str, typing.Any] = {"name": "root"}
    child: dict[str, typing.Any] = {"name": "child", "parent": root}
    root["parent"] = child
    root["friends"] = [child, root]
    return root, child


class TestObjectGraphDump:
    def test_self_cycle_emits_identifier_then_reference(self):
        root, _ = cyclic_graph()
        result = NodeSchema().dump(root, object_graph=True)
        assert result[ID] == "0"
        assert result["name"] == "root"
        assert result["parent"][ID] == "1"
        assert result["parent"]["name"] == "child"
        assert result["parent"]["parent"] == {REF: "0"}
        assert result["friends"] == [{REF: "1"}, {REF: "0"}]

    def test_identifier_is_first_key(self):
        root: dict[str, typing.Any] = {"name": "root"}
        root["parent"] = root
        result = NodeSchema().dump(root, object_graph=True)
        assert next(iter(result)) == ID

    def test_shared_object_is_referenced(self):
        shared: dict[str, typing.Any] = {
            "name": "shared",
            "parent": None,
            "friends": [],
        }
        root = {"name": "root", "parent": shared, "friends": [shared]}
        result = NodeSchema().dump(root, object_graph=True)
        assert result[ID] == "0"
        assert result["parent"] == {
            ID: "1",
            "name": "shared",
            "parent": None,
            "friends": [],
        }
        assert result["friends"][0] == {REF: "1"}

    def test_identity_not_equality(self):
        # Two equal, distinct objects must each receive their own identifier.
        first: dict[str, typing.Any] = {
            "name": "x",
            "parent": None,
            "friends": [],
        }
        second: dict[str, typing.Any] = {
            "name": "x",
            "parent": None,
            "friends": [],
        }
        root = {"name": "r", "parent": first, "friends": [second]}
        result = NodeSchema().dump(root, object_graph=True)
        assert ID in result["parent"]
        assert ID in result["friends"][0]
        assert result["parent"][ID] != result["friends"][0][ID]
        assert REF not in result["friends"][0]

    def test_many_top_level(self):
        a: dict[str, typing.Any] = {"name": "a", "parent": None, "friends": []}
        b: dict[str, typing.Any] = {"name": "b", "parent": a, "friends": [a]}
        result = NodeSchema().dump([a, b, a], many=True, object_graph=True)
        assert result[0][ID] == "0"
        assert result[1][ID] == "1"
        assert result[1]["parent"] == {REF: "0"}
        assert result[2] == {REF: "0"}

    def test_nested_many(self):
        a: dict[str, typing.Any] = {"name": "a", "parent": None, "friends": []}
        root = {"name": "r", "kids": [a, a]}

        class TreeSchema(Schema):
            name = fields.Str()
            kids = fields.Nested(NodeSchema, many=True)

        result = TreeSchema().dump(root, object_graph=True)
        assert result[ID] == "0"
        assert result["kids"][0][ID] == "1"
        assert result["kids"][1] == {REF: "1"}

    def test_shared_objects_in_dict_and_tuple(self):
        a: dict[str, typing.Any] = {"name": "a", "parent": None, "friends": []}
        b: dict[str, typing.Any] = {
            "name": "b",
            "parent": None,
            "friends": [a],
        }

        class Container(Schema):
            name = fields.Str()
            by_key = fields.Dict(keys=fields.Str(), values=fields.Nested(NodeSchema))
            pair = fields.Tuple((fields.Nested(NodeSchema), fields.Nested(NodeSchema)))

        obj = {"name": "c", "by_key": {"x": a, "y": a}, "pair": (a, b)}
        result = Container().dump(obj, object_graph=True)
        assert result[ID] == "0"
        assert result["by_key"]["x"][ID] == "1"
        assert result["by_key"]["y"] == {REF: "1"}
        assert result["pair"] == (
            {REF: "1"},
            {ID: "2", "name": "b", "parent": None, "friends": [{REF: "1"}]},
        )

    def test_data_key_is_used_for_fields_but_markers_are_reserved(self):
        class DataKeySchema(Schema):
            full_name = fields.Str(data_key="fullName")
            buddy = fields.Nested(lambda: DataKeySchema(), data_key="buddyKey")

        obj: dict[str, typing.Any] = {"full_name": "me"}
        obj["buddy"] = obj
        result = DataKeySchema().dump(obj, object_graph=True)
        assert result == {
            ID: "0",
            "fullName": "me",
            "buddyKey": {REF: "0"},
        }

    def test_only_and_exclude(self):
        class Limited(Schema):
            name = fields.Str()
            secret = fields.Str()
            me = fields.Nested(lambda: Limited())

        obj: dict[str, typing.Any] = {"name": "n", "secret": "s"}
        obj["me"] = obj
        only_result = Limited(only=("name", "me")).dump(obj, object_graph=True)
        assert only_result[ID] == "0"
        assert "secret" not in only_result
        assert only_result["me"] == {REF: "0"}

        exclude_result = Limited(exclude=("secret",)).dump(obj, object_graph=True)
        assert "secret" not in exclude_result
        assert exclude_result["me"] == {REF: "0"}

    def test_load_only_and_dump_only_fields(self):
        class LDSchema(Schema):
            name = fields.Str()
            secret = fields.Str(load_only=True)
            badge = fields.Str(dump_only=True)
            me = fields.Nested(lambda: LDSchema())

        obj: dict[str, typing.Any] = {
            "name": "a",
            "secret": "in",
            "badge": "out",
        }
        obj["me"] = obj
        result = LDSchema().dump(obj, object_graph=True)
        assert "secret" not in result
        assert result["badge"] == "out"
        assert result["me"] == {REF: "0"}

    def test_pluck_collapses_without_markers(self):
        class Artist(Schema):
            id = fields.Int()
            name = fields.Str()

        class Album(Schema):
            title = fields.Str()
            artist = fields.Pluck(Artist, "name")
            artists = fields.Pluck(Artist, "name", many=True)

        obj = {
            "title": "t",
            "artist": {"id": 1, "name": "A"},
            "artists": [
                {"id": 1, "name": "A"},
                {"id": 2, "name": "B"},
            ],
        }
        result = Album().dump(obj, object_graph=True)
        assert result["title"] == "t"
        assert result["artist"] == "A"
        assert result["artists"] == ["A", "B"]
        assert REF not in result["artist"]

    def test_json_dumps_roundtrip_structure(self):
        root, _ = cyclic_graph()
        encoded = NodeSchema().dumps(root, object_graph=True)
        decoded = json.loads(encoded)
        assert decoded[ID] == "0"
        assert decoded["parent"]["parent"] == {REF: "0"}

    def test_identifier_counter_resets_per_operation(self):
        schema = NodeSchema()
        first = schema.dump(
            {"name": "a", "parent": None, "friends": []}, object_graph=True
        )
        second = schema.dump(
            {"name": "b", "parent": None, "friends": []}, object_graph=True
        )
        assert first[ID] == "0"
        assert second[ID] == "0"

    def test_enabled_through_meta_and_constructor(self):
        class MetaSchema(Schema):
            class Meta:
                object_graph = True

            name = fields.Str()
            me = fields.Nested(lambda: MetaSchema())

        obj: dict[str, typing.Any] = {"name": "a"}
        obj["me"] = obj
        assert ID in MetaSchema().dump(obj)

        class PlainOuter(Schema):
            name = fields.Str()
            inner = fields.Nested(MetaSchema)

        inner = {"name": "in"}
        # The graph-enabled nested schema starts its own graph even though
        # the outer schema stays a tree.
        result = PlainOuter().dump({"name": "out", "inner": inner})
        assert result["inner"][ID] == "0"
        assert ID not in result

    def test_custom_marker_keys(self):
        class CustomSchema(Schema):
            class Meta:
                object_graph = True
                object_graph_id_key = "@id"
                object_graph_ref_key = "@ref"

            name = fields.Str()
            me = fields.Nested(lambda: CustomSchema())

        obj: dict[str, typing.Any] = {"name": "a"}
        obj["me"] = obj
        assert CustomSchema().dump(obj) == {
            "@id": "0",
            "name": "a",
            "me": {"@ref": "0"},
        }

    def test_none_nested_passes_through(self):
        result = NodeSchema().dump(
            {"name": "a", "parent": None, "friends": []}, object_graph=True
        )
        assert result["parent"] is None


class TestObjectGraphLoad:
    def test_rebuilds_self_cycle(self):
        payload = {
            ID: "0",
            "name": "root",
            "parent": {ID: "1", "name": "child", "parent": {REF: "0"}},
            "friends": [{REF: "1"}, {REF: "0"}],
        }
        result = NodeSchema().load(payload, object_graph=True)
        assert result["name"] == "root"
        assert result["parent"]["name"] == "child"
        assert result["parent"]["parent"] is result
        assert result["friends"][0] is result["parent"]
        assert result["friends"][1] is result

    def test_backward_reference_shares_identity(self):
        class CompanySchema(Schema):
            ceo = fields.Nested(NodeSchema)
            cto = fields.Nested(NodeSchema)

        payload = {
            "ceo": {
                ID: "p",
                "name": "ceo",
                "parent": {REF: "p"},
                "friends": [],
            },
            "cto": {REF: "p"},
        }
        result = CompanySchema().load(payload, object_graph=True)
        assert result["cto"] is result["ceo"]
        assert result["ceo"]["parent"] is result["ceo"]

    def test_forward_reference_is_resolved(self):
        class CompanySchema(Schema):
            cto = fields.Nested(NodeSchema)
            ceo = fields.Nested(NodeSchema)

        payload = {
            "cto": {REF: "p"},
            "ceo": {
                ID: "p",
                "name": "ceo",
                "parent": {REF: "p"},
                "friends": [],
            },
        }
        result = CompanySchema().load(payload, object_graph=True)
        assert result["cto"] is result["ceo"]
        assert result["cto"]["name"] == "ceo"
        assert result["ceo"]["parent"] is result["ceo"]

    def test_forward_reference_inside_nested_many(self):
        class TreeSchema(Schema):
            name = fields.Str()
            kids = fields.Nested(NodeSchema, many=True)

        payload = {
            ID: "root",
            "name": "root",
            "kids": [
                {REF: "k"},
                {
                    ID: "k",
                    "name": "kid",
                    "parent": {REF: "root"},
                    "friends": [],
                },
            ],
        }
        result = TreeSchema().load(payload, object_graph=True)
        assert result["kids"][0] is result["kids"][1]
        assert result["kids"][1]["parent"] is result

    def test_shared_objects_in_list_dict_and_tuple(self):
        class Container(Schema):
            by_key = fields.Dict(keys=fields.Str(), values=fields.Nested(NodeSchema))
            pair = fields.Tuple((fields.Nested(NodeSchema), fields.Nested(NodeSchema)))
            items = fields.List(fields.Nested(NodeSchema))

        payload = {
            "by_key": {
                "x": {ID: "1", "name": "a", "parent": None, "friends": []},
                "y": {REF: "1"},
            },
            "pair": [{REF: "1"}, {REF: "2"}],
            "items": [{ID: "2", "name": "b", "parent": {REF: "1"}, "friends": []}],
        }
        result = Container().load(payload, object_graph=True)
        assert result["by_key"]["x"] is result["by_key"]["y"]
        assert result["pair"][0] is result["by_key"]["x"]
        assert result["pair"][1] is result["items"][0]
        assert result["items"][0]["parent"] is result["by_key"]["x"]

    def test_many_top_level(self):
        payload: list[dict[str, typing.Any]] = [
            {ID: "0", "name": "a", "parent": None, "friends": []},
            {ID: "1", "name": "b", "parent": {REF: "0"}, "friends": []},
            {REF: "0"},
        ]
        result = NodeSchema().load(payload, many=True, object_graph=True)
        assert result[0] is result[2]
        assert result[1]["parent"] is result[0]

    def test_data_key_roundtrip(self):
        class DataKeySchema(Schema):
            full_name = fields.Str(data_key="fullName")
            buddy = fields.Nested(lambda: DataKeySchema(), data_key="buddyKey")

        payload = {
            ID: "0",
            "fullName": "me",
            "buddyKey": {REF: "0"},
        }
        result = DataKeySchema().load(payload, object_graph=True)
        assert result["full_name"] == "me"
        assert result["buddy"] is result

    def test_only_and_exclude_roundtrip(self):
        class Limited(Schema):
            name = fields.Str()
            secret = fields.Str()
            me = fields.Nested(lambda: Limited())

        payload = {ID: "0", "name": "n", "me": {REF: "0"}}
        result = Limited(only=("name", "me")).load(payload, object_graph=True)
        assert result["me"] is result
        assert "secret" not in result

    def test_unknown_exclude_and_include_strip_markers(self):
        payload = {
            ID: "7",
            "name": "a",
            "extra": 42,
            "parent": {REF: "7"},
            "friends": [],
        }
        excluded = NodeSchema(unknown=EXCLUDE).load(payload, object_graph=True)
        assert "extra" not in excluded
        assert ID not in excluded
        assert excluded["parent"] is excluded

        included = NodeSchema(unknown=INCLUDE).load(payload, object_graph=True)
        assert included["extra"] == 42
        assert ID not in included
        assert included["parent"] is included

    def test_unknown_raise_flags_non_marker_keys_only(self):
        payload = {
            ID: "0",
            "name": "a",
            "extra": 42,
            "parent": {REF: "0"},
            "friends": [],
        }
        with pytest.raises(ValidationError) as exc:
            NodeSchema().load(payload, object_graph=True)
        assert "extra" in exc.value.messages
        assert ID not in exc.value.messages
        assert REF not in exc.value.messages

    def test_markers_stripped_with_partial_loading(self):
        payload = {ID: "0", "parent": {REF: "0"}}
        result = NodeSchema().load(payload, partial=True, object_graph=True)
        assert result["parent"] is result
        assert ID not in result

    def test_custom_marker_keys_roundtrip(self):
        class CustomSchema(Schema):
            class Meta:
                object_graph = True
                object_graph_id_key = "@id"
                object_graph_ref_key = "@ref"

            name = fields.Str()
            me = fields.Nested(lambda: CustomSchema())

        result = CustomSchema().load({"@id": "9", "name": "a", "me": {"@ref": "9"}})
        assert result["me"] is result

    def test_loads_json_string(self):
        encoded = json.dumps(
            {ID: "0", "name": "a", "parent": {REF: "0"}, "friends": []}
        )
        result = NodeSchema().loads(encoded, object_graph=True)
        assert result["parent"] is result

    def test_separate_loads_build_independent_graphs(self):
        payload = {ID: "0", "name": "a", "parent": {REF: "0"}, "friends": []}
        first = NodeSchema().load(payload, object_graph=True)
        second = NodeSchema().load(payload, object_graph=True)
        assert first is not second
        assert first["parent"] is first
        assert second["parent"] is second


class TestObjectGraphFailures:
    def test_duplicate_identifier_fails_at_second_definition_path(self):
        payload = {
            ID: "1",
            "name": "a",
            "parent": {ID: "1", "name": "b"},
            "friends": [],
        }
        with pytest.raises(ValidationError) as exc:
            NodeSchema().load(payload, object_graph=True)
        assert exc.value.messages["parent"][ID] == [DUPLICATE.format(id="1")]

    def test_dangling_reference_at_nested_field(self):
        payload = {"name": "a", "parent": {REF: "missing"}, "friends": []}
        with pytest.raises(ValidationError) as exc:
            NodeSchema().load(payload, object_graph=True)
        assert exc.value.messages == {
            "parent": {"_schema": [DANGLING.format(id="missing")]}
        }

    def test_dangling_reference_path_through_list_indexes(self):
        payload: list[dict[str, typing.Any]] = [
            {"name": "a", "parent": None, "friends": []},
            {
                "name": "b",
                "parent": None,
                "friends": [
                    {
                        "name": "c",
                        "parent": None,
                        "friends": [
                            {"name": "d", "parent": {REF: "ghost"}, "friends": []}
                        ],
                    }
                ],
            },
        ]
        with pytest.raises(ValidationError) as exc:
            NodeSchema().load(payload, many=True, object_graph=True)
        assert exc.value.messages == {
            1: {
                "friends": {
                    0: {
                        "friends": {
                            0: {"parent": {"_schema": [DANGLING.format(id="ghost")]}}
                        }
                    }
                }
            }
        }

    def test_illegal_reference_position_at_scalar_field(self):
        with pytest.raises(ValidationError) as exc:
            NodeSchema().load({"name": {REF: "x"}}, object_graph=True)
        assert exc.value.messages["name"] == [ILLEGAL_POSITION]

    def test_illegal_reference_position_inside_scalar_list(self):
        class Tagged(Schema):
            name = fields.Str()
            tags = fields.List(fields.Str())

        with pytest.raises(ValidationError) as exc:
            Tagged().load(
                {"name": "a", "tags": ["ok", {REF: "x"}]},
                object_graph=True,
            )
        assert exc.value.messages["tags"][1] == [ILLEGAL_POSITION]

    def test_illegal_reference_position_for_raw_dict_field(self):
        class Holder(Schema):
            bag = fields.Dict()

        with pytest.raises(ValidationError) as exc:
            Holder().load({"bag": {REF: "x"}}, object_graph=True)
        assert exc.value.messages["bag"] == [ILLEGAL_POSITION]

    def test_illegal_identifier_marker_at_scalar_field(self):
        class Tagged(Schema):
            name = fields.Str()
            tags = fields.List(fields.Str())

        with pytest.raises(ValidationError) as exc:
            Tagged().load({"name": "a", "tags": [{ID: "z"}]}, object_graph=True)
        assert exc.value.messages["tags"][0] == [ILLEGAL_POSITION]

    def test_identifier_and_reference_in_same_object(self):
        with pytest.raises(ValidationError) as exc:
            NodeSchema().load({ID: "1", REF: "2", "name": "a"}, object_graph=True)
        assert exc.value.messages[REF] == [ID_AND_REFERENCE]

    def test_reference_marker_with_extra_fields(self):
        payload = {
            "name": "a",
            "parent": {REF: "x", "name": "b"},
            "friends": [],
        }
        with pytest.raises(ValidationError) as exc:
            NodeSchema().load(payload, object_graph=True)
        assert exc.value.messages["parent"][REF] == [INVALID_REFERENCE]

    def test_reference_marker_wrong_value_type(self):
        payload = {"name": "a", "parent": {REF: 9}, "friends": []}
        with pytest.raises(ValidationError) as exc:
            NodeSchema().load(payload, object_graph=True)
        assert exc.value.messages["parent"][REF] == [INVALID_REFERENCE]

    def test_identifier_wrong_value_type(self):
        with pytest.raises(ValidationError) as exc:
            NodeSchema().load({ID: 5, "name": "a"}, object_graph=True)
        assert exc.value.messages[ID] == [INVALID_ID]

    def test_multiple_errors_are_aggregated_and_load_fails(self):
        with pytest.raises(ValidationError) as exc:
            NodeSchema().load({ID: 5, "name": {REF: "x"}}, object_graph=True)
        assert ID in exc.value.messages
        assert "name" in exc.value.messages
        assert exc.value.valid_data is not None

    def test_validation_errors_and_graph_errors_coexist(self):
        class Strict(Schema):
            age = fields.Int(required=True)
            parent = fields.Nested(lambda: Strict())

        with pytest.raises(ValidationError) as exc:
            Strict().load({"age": "old", "parent": {REF: "gone"}}, object_graph=True)
        assert "age" in exc.value.messages
        assert exc.value.messages["parent"] == {"_schema": [DANGLING.format(id="gone")]}

    def test_reserved_field_conflict_raises_at_instantiation(self):
        class Conflict(Schema):
            class Meta:
                object_graph = True

            name = fields.Str(data_key=ID)

        with pytest.raises(ValueError, match="reserves the field key"):
            Conflict()

    def test_reserved_field_conflict_through_ref_key(self):
        class Conflict(Schema):
            class Meta:
                object_graph = True

            link = fields.Nested(lambda: Conflict(), data_key=REF)

        with pytest.raises(ValueError, match="reserves the field key"):
            Conflict()

    def test_reserved_field_conflict_detected_when_nested_joins_graph(self):
        # The nested schema itself is not graph-enabled, but it joins the
        # outer graph operation: its "$ref" data key still collides.
        class NestedConflict(Schema):
            name = fields.Str()
            link = fields.Str(data_key=REF)

        class Outer(Schema):
            child = fields.Nested(NestedConflict)

        with pytest.raises(ValueError, match="reserves the field key"):
            Outer().load({"child": {"name": "a", "link": "x"}}, object_graph=True)

    def test_reserved_field_allowed_when_mode_disabled(self):
        class NotGraph(Schema):
            name = fields.Str(data_key=ID)

        schema = NotGraph()
        assert schema.dump({"name": "a"}) == {ID: "a"}
        assert schema.load({ID: "a"}) == {"name": "a"}

    def test_invalid_meta_key_configuration(self):
        with pytest.raises(ValueError):

            class SameKeys(Schema):
                class Meta:
                    object_graph = True
                    object_graph_id_key = "x"
                    object_graph_ref_key = "x"

                name = fields.Str()

        with pytest.raises(ValueError):

            class EmptyKey(Schema):
                class Meta:
                    object_graph = True
                    object_graph_id_key = ""

                name = fields.Str()

    def test_validate_reports_graph_errors(self):
        errors = NodeSchema().validate(
            {"name": "a", "parent": {REF: "x"}}, object_graph=True
        )
        assert errors == {"parent": {"_schema": [DANGLING.format(id="x")]}}
        assert (
            NodeSchema().validate(
                {ID: "0", "name": "a", "parent": {REF: "0"}, "friends": []},
                object_graph=True,
            )
            == {}
        )

    def test_failed_operation_does_not_leak_graph_state(self):
        with pytest.raises(ValidationError):
            NodeSchema().load({"name": {REF: "x"}}, object_graph=True)
        # A subsequent tree dump/load behaves normally (no markers).
        assert NodeSchema().dump({"name": "a", "parent": None, "friends": []}) == {
            "name": "a",
            "parent": None,
            "friends": [],
        }
        assert NodeSchema().load({"name": "a"}) == {"name": "a"}


class TestHooksCooperation:
    def test_all_hooks_fire_and_reference_is_not_processed_twice(self):
        calls: list[typing.Any] = []

        class HookSchema(Schema):
            name = fields.Str()
            partner = fields.Nested(lambda: HookSchema())

            @pre_dump
            def _pre_dump(self, data, **kwargs):
                calls.append(("pre_dump", data.get("name")))
                return data

            @post_dump
            def _post_dump(self, data, **kwargs):
                calls.append(("post_dump", data.get(ID), data.get(REF)))
                return data

            @pre_load
            def _pre_load(self, data, **kwargs):
                calls.append(("pre_load", data.get(REF) or data.get("name")))
                return data

            @post_load
            def _post_load(self, data, **kwargs):
                calls.append(("post_load", data.get("name")))
                return data

            @validates("name")
            def _validates_name(self, value, **kwargs):
                calls.append(("validates", value))

            @validates_schema
            def _validates_schema(self, data, **kwargs):
                calls.append(("validates_schema", data.get("name")))

        obj: dict[str, typing.Any] = {"name": "x"}
        obj["partner"] = obj
        dumped = HookSchema().dump(obj, object_graph=True)
        assert dumped == {ID: "0", "name": "x", "partner": {REF: "0"}}

        loaded = HookSchema().load(
            {ID: "0", "name": "x", "partner": {REF: "0"}},
            object_graph=True,
        )
        assert loaded["partner"] is loaded

        # The definition object is validated/post-loaded exactly once; the
        # nested reference raw payload still passes through pre_load hooks.
        assert calls.count(("post_load", "x")) == 1
        assert calls.count(("validates", "x")) == 1
        assert calls.count(("validates_schema", "x")) == 1
        assert ("pre_load", "0") in calls
        # post_dump sees both the definition and the reference marker.
        assert ("post_dump", "0", None) in calls
        assert ("post_dump", None, "0") in calls
        assert ("pre_dump", "x") in calls

    def test_post_load_mutation_is_shared_through_references(self):
        class MutatingSchema(Schema):
            name = fields.Str()
            partner = fields.Nested(lambda: MutatingSchema())

            @post_load
            def _annotate(self, data, **kwargs):
                data["seen_by_post_load"] = True
                return data

        result = MutatingSchema().load(
            {ID: "0", "name": "x", "partner": {REF: "0"}},
            object_graph=True,
        )
        assert result["seen_by_post_load"] is True
        assert result["partner"] is result


class TestDefaultTreeBehavior:
    def test_dump_without_mode_emits_no_markers(self):
        # A shared, acyclic subobject is fully duplicated without markers (a
        # true cycle cannot be represented as a tree).
        child: dict[str, typing.Any] = {
            "name": "child",
            "parent": None,
            "friends": [],
        }
        root = {"name": "root", "parent": child, "friends": [child]}
        result = NodeSchema().dump(root)
        dumped = json.dumps(result)
        assert ID not in dumped
        assert REF not in dumped
        assert result["parent"]["name"] == "child"
        assert result["friends"][0] is not result["parent"]

    def test_equal_subobjects_are_duplicated_without_mode(self):
        shared: dict[str, typing.Any] = {
            "name": "s",
            "parent": None,
            "friends": [],
        }

        class Tree(Schema):
            name = fields.Str()
            first = fields.Nested(NodeSchema)
            second = fields.Nested(NodeSchema)

        result = Tree().dump({"name": "r", "first": shared, "second": shared})
        assert result["first"] == result["second"]
        assert result["first"] is not result["second"]
        assert ID not in result["first"]

    def test_marker_keys_are_unknown_fields_without_mode(self):
        with pytest.raises(ValidationError) as exc:
            NodeSchema().load({ID: "0", "name": "a"})
        assert ID in exc.value.messages

        class Excluding(Schema):
            name = fields.Str()
            parent = fields.Nested(lambda: Excluding())

        result = Excluding(unknown=EXCLUDE).load({ID: "0", "name": "a"})
        assert result == {"name": "a"}

    def test_graph_mode_can_be_toggled_per_call(self):
        schema = NodeSchema()
        obj: dict[str, typing.Any] = {
            "name": "a",
            "parent": None,
            "friends": [],
        }
        assert ID in schema.dump(obj, object_graph=True)
        assert ID not in schema.dump(obj)
        assert ID not in schema.dump(obj, object_graph=False)

        graph_schema = NodeSchema(object_graph=True)
        assert ID in graph_schema.dump(obj)
        assert ID not in graph_schema.dump(obj, object_graph=False)
