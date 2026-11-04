import os
import pathlib

import pytest

from pycrfsuite import Tagger, Trainer


def _train(path, xseq, yseq):
    trainer = Trainer(verbose=False)
    trainer.append(xseq, yseq)
    trainer.train(path)


def test_pathlike(tmpdir, xseq, yseq):
    model = pathlib.Path(str(tmpdir)) / "model.crfsuite"
    _train(model, xseq, yseq)
    assert model.is_file()

    with Tagger().open(model) as tagger:
        assert tagger.tag(xseq) == yseq
        dump = pathlib.Path(str(tmpdir)) / "dump.txt"
        tagger.dump(dump)
        assert dump.stat().st_size > 0


@pytest.mark.parametrize(
    "name",
    ["modèl_ñ.crfsuite", "модель.crfsuite", "模型.crfsuite"],
    ids=["latin", "cyrillic", "cjk"],
)
def test_non_ascii_model_path(tmpdir, xseq, yseq, name):
    # Python passes file names to CRFsuite as UTF-8; on Windows, CRFsuite's
    # fopen() used to interpret them in the ANSI code page and create a file
    # with a mangled name.
    directory = str(tmpdir)
    model = os.path.join(directory, name)
    _train(model, xseq, yseq)
    assert os.listdir(directory) == [name]

    with Tagger().open(model) as tagger:
        assert tagger.tag(xseq) == yseq
