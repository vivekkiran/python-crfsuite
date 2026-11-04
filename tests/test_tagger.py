import gc
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

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


@pytest.mark.xfail(reason="see https://github.com/chokkan/crfsuite/pull/24", run=False)
def test_open_invalid_with_correct_signature(tmpdir):
    tmp = tmpdir.join("tmp.txt")
    tmp.write(b"lCRFfoo" * 100)
    tagger = Tagger()
    with pytest.raises(ValueError):
        tagger.open(str(tmp))


def test_open_inmemory(model_bytes, xseq, yseq):
    with Tagger().open_inmemory(model_bytes) as tagger:
        assert tagger.tag(xseq) == yseq


def test_open_inmemory_invalid():
    tagger = Tagger()
    with pytest.raises(ValueError):
        tagger.open_inmemory(b"")

    with pytest.raises(ValueError):
        tagger.open_inmemory(b"lCRFabc")


def _reuse_freed_memory(size):
    # Same-size allocations, so that a freed model buffer gets reused.
    return [b"\x5a" * size for _ in range(50)]


def test_open_inmemory_keeps_buffer_alive(model_bytes, xseq, yseq):
    # CRFsuite reads an in-memory model from the buffer passed to
    # open_inmemory() without copying it, so the Tagger must keep it alive.
    tagger = Tagger()
    data = bytes(bytearray(model_bytes))  # a buffer only the Tagger will hold
    refs = sys.getrefcount(data)
    tagger.open_inmemory(data)
    assert sys.getrefcount(data) == refs + 1

    del data
    gc.collect()
    junk = _reuse_freed_memory(len(model_bytes))
    assert set(tagger.labels()) == set(yseq)
    assert tagger.tag(xseq) == yseq
    del junk
    tagger.close()


@pytest.mark.parametrize("release", ["close", "open", "open_inmemory"])
def test_open_inmemory_releases_buffer(model_filename, model_bytes, release):
    tagger = Tagger()
    data = bytes(bytearray(model_bytes))
    refs = sys.getrefcount(data)
    tagger.open_inmemory(data)
    if release == "close":
        tagger.close()
    elif release == "open":
        tagger.open(model_filename)
    else:
        tagger.open_inmemory(model_bytes)
    assert sys.getrefcount(data) == refs


def test_failed_open_keeps_inmemory_buffer(model_filename, model_bytes):
    # An open() that fails before reaching CRFsuite must not release the
    # buffer of the in-memory model that is still open.
    tagger = Tagger()
    data = bytes(bytearray(model_bytes))
    tagger.open_inmemory(data)
    with pytest.raises(TypeError):  # paths must be str or bytes
        tagger.open(pathlib.Path(model_filename))
    held = sys.getrefcount(data)
    tagger.close()
    assert sys.getrefcount(data) == held - 1


@pytest.mark.parametrize("method", ["set", "tag"])
def test_tagger_not_opened(xseq, method):
    tagger = Tagger()
    with pytest.raises(ValueError, match="^The tagger is not opened$"):
        getattr(tagger, method)(xseq)


def test_tag(model_filename, xseq, yseq):
    with Tagger().open(model_filename) as tagger:
        assert tagger.tag(xseq) == yseq


def test_open_inmemory_while_tagging_in_threads(model_bytes, xseq, yseq):
    # Re-opening a shared tagger from fresh buffers while other threads tag
    # with it: the model/buffer swap happens under the tagger lock, so a
    # tagger never reads a buffer that has already been released.
    tagger = Tagger()
    tagger.open_inmemory(bytes(bytearray(model_bytes)))
    done = Event()

    def reopen():
        try:
            for _ in range(300):
                # The previous buffer is released as soon as this returns;
                # overwrite released memory so a dangling read gets junk.
                tagger.open_inmemory(bytes(bytearray(model_bytes)))
                junk = _reuse_freed_memory(len(model_bytes))
                del junk
        finally:
            done.set()

    def tag():
        while not done.is_set():
            assert tagger.tag(xseq) == yseq

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(tag) for _ in range(3)] + [pool.submit(reopen)]
        for future in futures:
            future.result()
    tagger.close()


def test_tag_shared_between_threads(model_filename, xseq):
    barrier = Barrier(4)
    with Tagger().open(model_filename) as tagger:
        sequences = [ItemSequence(xseq[:length]) for length in (1, 3, 7, 11)]
        expected = [tagger.tag(sequence) for sequence in sequences]

        def tag(index):
            barrier.wait(timeout=30)
            for _ in range(500):
                assert tagger.tag(sequences[index]) == expected[index]

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(tag, range(4)))


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


@pytest.mark.xfail()
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


def test_dump_overwrites_existing_file(tmpdir, model_filename):
    with Tagger().open(model_filename) as tagger:
        fresh = str(tmpdir.join("fresh.txt"))
        existing = str(tmpdir.join("existing.txt"))
        tagger.dump(fresh)
        with open(existing, "wb") as f:
            f.write(b"OLD CONTENT\n" * 1000)  # e.g. a dump of a bigger model

        tagger.dump(existing)

        with open(fresh, "rb") as f1, open(existing, "rb") as f2:
            assert f2.read() == f1.read()


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
