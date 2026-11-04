import struct

import pytest

from pycrfsuite import ItemSequence, Tagger, Trainer


def test_open_close_labels(model_filename, yseq):
    tagger = Tagger()

    with pytest.raises(ValueError):
        # tagger should be closed, so labels() method should fail here
        labels = tagger.labels()

    with tagger.open(model_filename):
        labels = tagger.labels()
    assert set(labels) == set(yseq)

    with pytest.raises(ValueError):
        # tagger should be closed, so labels() method should fail here
        labels = tagger.labels()


def test_open_non_existing():
    tagger = Tagger()
    with pytest.raises(IOError):
        tagger.open("foo")


def test_open_invalid():
    tagger = Tagger()
    with pytest.raises(ValueError):
        tagger.open(__file__)


def test_open_invalid_small(tmpdir):
    tmp = tmpdir.join("tmp.txt")
    tmp.write(b"foo")
    tagger = Tagger()
    with pytest.raises(ValueError):
        tagger.open(str(tmp))


def test_open_invalid_small_with_correct_signature(tmpdir):
    tmp = tmpdir.join("tmp.txt")
    tmp.write(b"lCRFfoo")
    tagger = Tagger()
    with pytest.raises(ValueError):
        tagger.open(str(tmp))


def test_open_invalid_with_correct_signature(tmpdir):
    tmp = tmpdir.join("tmp.txt")
    tmp.write(b"lCRFfoo" * 100)
    tagger = Tagger()
    with pytest.raises(ValueError):
        tagger.open(str(tmp))


def _corrupt(data, offset, new_bytes):
    return data[:offset] + new_bytes + data[offset + len(new_bytes) :]


def _header_field(data, index):
    # The 48-byte header is magic, size, type, version and 8 uint32 fields;
    # index counts uint32 fields after "type" (0=version ... 8=off_attrrefs).
    return struct.unpack_from("<I", data, 12 + 4 * index)[0]


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda d: d[: len(d) // 2], id="truncated"),
        pytest.param(lambda d: d[:48], id="header-only"),
        pytest.param(lambda d: _corrupt(d, 8, b"XXXX"), id="model-type"),
        pytest.param(
            lambda d: _corrupt(d, 4, struct.pack("<I", len(d) + 1)), id="size"
        ),
    ]
    + [
        pytest.param(
            lambda d, i=i: _corrupt(d, 12 + 4 * i, struct.pack("<I", 0xFFFFFF00)),
            id="offset-%s" % name,
        )
        for i, name in [
            (4, "features"),
            (5, "labels"),
            (6, "attrs"),
            (7, "labelrefs"),
            (8, "attrrefs"),
        ]
    ]
    + [
        pytest.param(
            lambda d, i=i: _corrupt(d, _header_field(d, i), b"\0\0\0\0"),
            id="chunk-tag-%s" % name,
        )
        for i, name in [
            (4, "features"),
            (5, "labels"),
            (6, "attrs"),
            (7, "labelrefs"),
            (8, "attrrefs"),
        ]
    ]
    + [
        pytest.param(
            lambda d, i=i: _corrupt(
                d, _header_field(d, i) + 4, struct.pack("<I", 0xFFFFFF00)
            ),
            id="chunk-size-%s" % name,
        )
        for i, name in [
            (4, "features"),
            (5, "labels"),
            (6, "attrs"),
            (7, "labelrefs"),
            (8, "attrrefs"),
        ]
    ],
)
def test_open_corrupted_model(tmpdir, model_bytes, corrupt):
    data = corrupt(model_bytes)
    assert data != model_bytes

    with pytest.raises(ValueError):
        Tagger().open_inmemory(data)

    tmp = tmpdir.join("corrupted.crfsuite")
    tmp.write(data, mode="wb")
    with pytest.raises(ValueError):
        Tagger().open(str(tmp))


def test_open_inmemory(model_bytes, xseq, yseq):
    with Tagger().open_inmemory(model_bytes) as tagger:
        assert tagger.tag(xseq) == yseq


def test_open_inmemory_invalid():
    tagger = Tagger()
    with pytest.raises(ValueError):
        tagger.open_inmemory(b"")

    with pytest.raises(ValueError):
        tagger.open_inmemory(b"lCRFabc")


_NOT_OPENED_CALLS = {
    "tag": lambda tagger, xseq: tagger.tag(xseq),
    "tag-current": lambda tagger, xseq: tagger.tag(),
    "set": lambda tagger, xseq: tagger.set(xseq),
    "labels": lambda tagger, xseq: tagger.labels(),
    "probability": lambda tagger, xseq: tagger.probability(["x"]),
    "marginal": lambda tagger, xseq: tagger.marginal("x", 0),
}


@pytest.mark.parametrize("call", _NOT_OPENED_CALLS.values(), ids=_NOT_OPENED_CALLS)
def test_tag_not_opened(xseq, call):
    # https://github.com/scrapinghub/python-crfsuite/issues/28
    tagger = Tagger()
    with pytest.raises(ValueError, match="^The tagger is not opened$"):
        call(tagger, xseq)


@pytest.mark.parametrize("call", _NOT_OPENED_CALLS.values(), ids=_NOT_OPENED_CALLS)
def test_tag_closed(model_filename, xseq, call):
    tagger = Tagger()
    tagger.open(model_filename)
    tagger.close()
    with pytest.raises(ValueError, match="^The tagger is not opened$"):
        call(tagger, xseq)


@pytest.mark.parametrize(
    "train_xseq, train_yseq",
    [
        pytest.param(None, None, id="no-data"),
        pytest.param([], [], id="empty-sequence"),
    ],
)
def test_tag_model_without_labels(tmpdir, xseq, train_xseq, train_yseq):
    # https://github.com/scrapinghub/python-crfsuite/issues/28: a model
    # trained without data has no labels, and tagging with it used to crash.
    model_filename = str(tmpdir.join("empty.crfsuite"))
    trainer = Trainer(verbose=False)
    if train_xseq is not None:
        trainer.append(train_xseq, train_yseq)
    trainer.train(model_filename)

    with Tagger().open(model_filename) as tagger:
        assert tagger.labels() == []
        assert tagger.tag([]) == []
        with pytest.raises(ValueError, match="no labels"):
            tagger.tag(xseq)
        with pytest.raises(ValueError, match="no labels"):
            tagger.tag()  # re-tag the sequence set by the failed call


def test_tag(model_filename, xseq, yseq):
    with Tagger().open(model_filename) as tagger:
        assert tagger.tag(xseq) == yseq


def test_tag_item_sequence(model_filename, xseq, yseq):
    with Tagger().open(model_filename) as tagger:
        assert tagger.tag(ItemSequence(xseq)) == yseq


def test_tag_string_lists(model_filename, xseq, yseq):
    with Tagger().open(model_filename) as tagger:
        # Working with lists is supported,
        # but if we discard weights the results become different
        data = [x.keys() for x in xseq]
        assert tagger.tag(data) != yseq


def test_tag_bools(model_filename, xseq, yseq):
    with Tagger().open(model_filename) as tagger:
        # Some values are bools:
        # True <=> 1.0; False <=> 0.0
        data = [
            {k: bool(v) if v == 0 or v == 1 else v for (k, v) in x.items()}
            for x in xseq
        ]
        assert tagger.tag(data) == yseq


def test_tag_formats(tmpdir, xseq, yseq):
    # make all coefficients 1 and check that results are the same
    model_filename = str(tmpdir.join("model.crfsuite"))
    xseq = [{key: 1 for key in x} for x in xseq]

    trainer = Trainer()
    trainer.set("c2", 1e-6)  # make sure model overfits
    trainer.append(xseq, yseq)
    trainer.train(model_filename)

    with Tagger().open(model_filename) as tagger:
        assert tagger.tag(xseq) == yseq

    # strings
    with Tagger().open(model_filename) as tagger:
        data = [x.keys() for x in xseq]
        assert tagger.tag(data) == yseq


@pytest.mark.parametrize(
    "bad_seq",
    [
        "foo",
        ["foo"],  # should be a list of lists of strings
        [[{"foo": 1.0}]],  # should be a list of dicts
    ],
)
def test_tag_invalid_feature_format(model_filename, bad_seq):
    with Tagger().open(model_filename) as tagger:
        with pytest.raises(ValueError):
            tagger.tag(bad_seq)


def test_tag_probability(model_filename, xseq, yseq):
    with Tagger().open(model_filename) as tagger:
        res = tagger.tag(xseq)
        prob = tagger.probability(res)
        prob2 = tagger.probability([yseq[0]] * len(yseq))
        assert prob > prob2
        assert 0 < prob < 1
        assert 0 < prob2 < 1


def test_dump(tmpdir, model_filename):
    with Tagger().open(model_filename) as tagger:
        dump_filename = str(tmpdir.join("dump.txt"))
        tagger.dump(dump_filename)

        with open(dump_filename, "rb") as f:
            res = f.read().decode("utf8")
            assert "LABELS = {" in res
            assert "солнце:не светит --> rainy:" in res

    # it shouldn't segfault on a closed tagger
    with pytest.raises(RuntimeError):
        tagger.dump(dump_filename)


def test_info(model_filename):
    with Tagger().open(model_filename) as tagger:
        res = tagger.info()

        assert res.transitions[("sunny", "sunny")] > res.transitions[("sunny", "rainy")]
        assert (
            res.state_features[("walk", "sunny")]
            > res.state_features[("walk", "rainy")]
        )
        assert ("солнце:не светит", "rainy") in res.state_features
        assert res.header["num_labels"] == "3"
        assert set(res.labels.keys()) == set(["sunny", "rainy", "好"])
        assert set(res.attributes.keys()) == set(
            ["shop", "walk", "clean", "солнце:не светит", "world"]
        )

    # it shouldn't segfault on a closed tagger
    with pytest.raises(RuntimeError):
        tagger.info()


def test_append_strstr_dicts(tmpdir):
    trainer = Trainer()
    trainer.append(
        [{"foo": "bar"}, {"baz": False}, {"foo": "bar", "baz": True}, {"baz": 0.2}],
        ["spam", "egg", "spam", "spam"],
    )
    model_filename = str(tmpdir.join("model.crfsuite"))
    trainer.train(model_filename)

    with Tagger().open(model_filename) as tagger:
        info = tagger.info()
        assert set(info.attributes.keys()) == {"foo:bar", "baz"}
        assert info.state_features[("foo:bar", "spam")] > 0


def test_append_nested_dicts(tmpdir):
    trainer = Trainer()
    trainer.append(
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
        ],
        ["first", "second"],
    )
    model_filename = str(tmpdir.join("model.crfsuite"))
    trainer.train(model_filename)

    with Tagger().open(model_filename) as tagger:
        info = tagger.info()
        assert set(info.attributes.keys()) == {
            "foo:bar:baz",
            "foo:spam",
            "foo:egg:x",
            "foo:egg:y",
            "foo:ham:x",
            "foo:ham:y",
            "foo:bar:ham",
        }

        for feat in ["foo:bar:baz", "foo:spam", "foo:egg:x", "foo:egg:y"]:
            assert info.state_features[(feat, "first")] > 0
            assert info.state_features.get((feat, "second"), 0) <= 0

        for feat in ["foo:bar:ham", "foo:ham:x", "foo:ham:y"]:
            assert info.state_features[(feat, "second")] > 0
            assert info.state_features.get((feat, "first"), 0) <= 0
