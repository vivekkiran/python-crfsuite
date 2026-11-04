import pytest

import pycrfsuite


def test_basic():
    seq = pycrfsuite.ItemSequence([])
    assert len(seq) == 0
    assert seq.items() == []


def test_lists():
    seq = pycrfsuite.ItemSequence([["foo", "bar"], ["bar", "baz"]])
    assert len(seq) == 2
    assert seq.items() == [{"foo": 1.0, "bar": 1.0}, {"bar": 1.0, "baz": 1.0}]
    assert pycrfsuite.ItemSequence(seq.items()).items() == seq.items()


def test_dicts():
    seq = pycrfsuite.ItemSequence(
        [
            {"foo": True, "bar": {"foo": -1, "baz": False}},
        ]
    )
    assert len(seq) == 1
    assert seq.items() == [{"foo": 1.0, "bar:foo": -1, "bar:baz": 0.0}]


def test_unicode():
    seq = pycrfsuite.ItemSequence(
        [
            {"foo": "привет", "ключ": 1.0, "привет": "мир"},
        ]
    )
    assert seq.items() == [{"foo:привет": 1.0, "ключ": 1.0, "привет:мир": 1.0}]


def test_bad():
    with pytest.raises(ValueError):
        seq = pycrfsuite.ItemSequence("foo")
        print(seq.items())

    with pytest.raises(ValueError):
        seq = pycrfsuite.ItemSequence([[{"foo": "bar"}]])
        print(seq.items())


@pytest.mark.parametrize(
    "bad_seq, message",
    [
        ("foo", "not a string"),
        (b"foo", "not a string"),
        (["foo"], "not a string"),
        ([b"foo"], "not a string"),
        ([[{"foo": "bar"}]], "got dict"),
        ([{1: 1.0}], "got int"),
        ([{"foo": {None: 1.0}}], "got NoneType"),
        ([{"foo": None}], "got NoneType for feature 'foo'"),
        ([{"foo": object()}], "got object for feature 'foo'"),
        ([{"ok": 1.0}, {"foo": {"bar": None}}], "got NoneType for feature 'bar'"),
    ],
)
def test_bad_feature_format_error(bad_seq, message):
    with pytest.raises(pycrfsuite.FeatureFormatError, match=message):
        pycrfsuite.ItemSequence(bad_seq)


def test_feature_format_error_is_value_and_type_error():
    assert issubclass(pycrfsuite.FeatureFormatError, ValueError)
    assert issubclass(pycrfsuite.FeatureFormatError, TypeError)
    with pytest.raises(TypeError):
        pycrfsuite.ItemSequence([{1: 1.0}])


def test_bad_input_does_not_leave_pending_exception():
    # Conversion errors used to be left pending and surface later, from an
    # unrelated call, as SystemError.
    for bad_seq in ([[{"foo": "bar"}]], [{"foo": None}], [{1: 1.0}]):
        with pytest.raises(pycrfsuite.FeatureFormatError):
            pycrfsuite.ItemSequence(bad_seq)
        assert len(pycrfsuite.ItemSequence([{"foo": 1.0}])) == 1


def test_bytes_keys():
    seq = pycrfsuite.ItemSequence([[b"foo", b"bar"], {b"baz": 0.5}])
    assert seq.items() == [{"foo": 1.0, "bar": 1.0}, {"baz": 0.5}]


def test_nested():
    seq = pycrfsuite.ItemSequence(
        [
            {
                "foo": {
                    "bar": "baz",
                    "spam": 0.5,
                    "egg": ["x", "y"],
                    "ham": {"x": -0.5, "y": -0.1},
                },
            },
            {
                "foo": {"bar": "ham", "spam": -0.5, "ham": {"x", "y"}},
            },
        ]
    )
    assert len(seq) == 2
    assert seq.items() == [
        {
            "foo:bar:baz": 1.0,
            "foo:spam": 0.5,
            "foo:egg:x": 1.0,
            "foo:egg:y": 1.0,
            "foo:ham:x": -0.5,
            "foo:ham:y": -0.1,
        },
        {
            "foo:bar:ham": 1.0,
            "foo:spam": -0.5,
            "foo:ham:x": 1.0,
            "foo:ham:y": 1.0,
        },
    ]
    assert pycrfsuite.ItemSequence(seq.items()).items() == seq.items()
