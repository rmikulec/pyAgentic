from deepdiff import DeepDiff

from objective_agents._base._params import Param, ParamInfo


def test_param_creation():

    class Test(Param):
        field: int

    type_, info = Test.__attributes__["field"]
    assert type_ == int, f"Typing is not being set properly, got: {type_}, expected: {int}"
    assert isinstance(
        info, ParamInfo
    ), "Default ParamInfo not being set when not supplied in definition"


def test_param_creation_with_info():

    class Test(Param):
        field: int = ParamInfo(description="This is a test")

    _, info = Test.__attributes__["field"]

    assert info.description == "This is a test", "Supplied ParamInfo is not being set when given"


def test_param_openai_export():

    class ExportTest(Param):
        field: int
        field_with_info: str = ParamInfo(description="This is a test")
        required_field: str = ParamInfo(required=True)
        field_with_default: str = ParamInfo(default="default")

    openai_param = ExportTest.to_openai()
    expected = {
        "type": "object",
        "properties": {
            "field": {"type": "integer"},
            "field_with_info": {"type": "string", "description": "This is a test"},
            "required_field": {"type": "string"},
            "field_with_default": {"type": "string"},
        },
        "required": ["required_field"],
    }

    diff = DeepDiff(openai_param, expected, ignore_order=True)

    assert not diff, f"OpenAI Export does not match expected: \n {diff.pretty()}"
