"""The face adapter: lipread/adapt.py's data handling, and LiveModel loading an adapter over the base weights."""
import json

import numpy as np
import pytest

from lipread.adapt import main, norm, sentence_clips, split

from .conftest import needs_cuda, needs_weights


def test_norm_matches_how_wer_compares():
    assert norm("I'm a well-known guy, okay?") == "I'M A WELL KNOWN GUY OKAY"


def test_split_holds_out_every_nth_clip():
    rows = list(range(10))
    train, test = split(rows, 5)
    assert test == [4, 9] and train == [0, 1, 2, 3, 5, 6, 7, 8]
    assert split(rows, 0) == (rows, [])


def test_sentence_clips_skip_rest_clips_and_sort(tmp_path):
    d = tmp_path / 'session_01'
    d.mkdir()
    rows = [dict(id='s1_002', session='session_01', prompt='Second.'), dict(id='s1_rest_01', session='session_01',
            prompt=''), dict(id='s1_001', session='session_01', prompt='First.')]
    (d / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert [r['id'] for r in sentence_clips(['session_01'], root=str(tmp_path))] == ['s1_001', 's1_002']


def test_sentence_clips_read_from_the_folder_given(tmp_path):
    d = tmp_path / 'session_02_EMG-dropped-frames'
    d.mkdir()
    (d / 'manifest.jsonl').write_text(json.dumps(dict(id='s2_001', session='session_02', prompt='Hi.')) + '\n')
    assert [r['dir'] for r in sentence_clips([d.name], root=str(tmp_path))] == [d.name]


def test_never_trains_on_session_3():
    with pytest.raises(SystemExit):
        main(['--sessions', 'session_01', 'session_03'])


def test_a_session_is_not_both_trained_on_and_held_out():
    with pytest.raises(SystemExit):
        main(['--sessions', 'session_01', '--test-sessions', 'session_01'])


@pytest.fixture(scope='module')
def adapter_file(tmp_path_factory):
    """A face adapter with the right tensors and changed values (as lipread/adapt.py saves one)."""
    import torch
    from lipflow.vsr import LipReader
    from lipread.model import FACE_ADAPTER_PREFIXES
    reader = LipReader(device='cpu', use_lm=False, personal=False)
    g = torch.Generator().manual_seed(0)
    state = {k: v + 0.05 * torch.randn(v.shape, generator=g) for k, v in reader.model.state_dict().items()
             if k.startswith(FACE_ADAPTER_PREFIXES) and v.is_floating_point()}
    path = tmp_path_factory.mktemp('adapter') / 'face.pth'
    torch.save(state, path)
    return path, state


@needs_weights
@needs_cuda
def test_live_model_loads_the_adapter_and_reads_differently(adapter_file):
    import torch
    from lipread.model import LiveModel
    path, state = adapter_file
    base = LiveModel(warmup=())
    adapted = LiveModel(warmup=(), face_adapter=str(path))
    sd = adapted.reader.model.state_dict()
    assert all(torch.equal(sd[k].cpu(), v) for k, v in state.items())
    other = [k for k in sd if k.startswith('encoder.encoders.5.')]
    assert other and all(torch.equal(sd[k], base.reader.model.state_dict()[k]) for k in other)
    rois = np.random.default_rng(0).integers(0, 256, (30, 96, 96), dtype=np.uint8)
    a, b = base.read(rois), adapted.read(rois)
    assert a.topk_logp.shape == b.topk_logp.shape and not np.allclose(a.topk_logp, b.topk_logp)


@needs_weights
def test_a_file_with_other_tensors_is_refused(tmp_path, adapter_file):
    import torch
    from lipflow.vsr import LipReader
    from lipread.model import load_face_adapter
    _, state = adapter_file
    bad = dict(state)
    bad['decoder.embed.0.weight'] = torch.zeros(1)
    torch.save(bad, tmp_path / 'bad.pth')
    reader = LipReader(device='cpu', use_lm=False, personal=False)
    with pytest.raises(ValueError, match='not a face adapter'):
        load_face_adapter(reader.model, str(tmp_path / 'bad.pth'))


class FakeReader:
    """Enough of a LipReader for word_ends: 4 tokens, two words, 'A' = 1 token, 'BC' = 2 tokens."""
    token_list = ['<blank>', '▁A', '▁B', 'C']
    enc_device = 'cpu'

    def __init__(self):
        import torch
        self.encoded = 0
        outer = self

        class Enc:
            def __call__(self, x, _):
                outer.encoded += 1
                return x, None

        class Ctc:
            def log_softmax(self, enc):
                return enc.log_softmax(-1)

        self.model = type('M', (), dict(encoder=Enc(), ctc=Ctc()))()
        self._torch = torch

    def to_tensor(self, rois):
        t = self._torch.zeros(len(rois), 4)
        t[:, 0] = 5.0
        for frame, tok in ((2, 1), (4, 2), (6, 3)):
            t[frame, 0], t[frame, tok] = 0.0, 9.0
        return t


def test_word_ends_are_cached_and_a_hit_skips_the_encoder(tmp_path):
    from lipread.adapt import load_alignments, save_alignments, word_ends
    reader, cache = FakeReader(), {}
    rois = np.zeros((10, 96, 96), np.uint8)
    word_of, ends = word_ends(reader, rois, [1, 2, 3], 'clip_1', cache)
    assert word_of == [0, 1, 1] and ends == [3, 7] and reader.encoded == 1
    save_alignments(str(tmp_path / 'sub' / 'align.json'), cache)
    cache = load_alignments(str(tmp_path / 'sub' / 'align.json'))
    assert cache['clip_1']['words'] == ['A', 'BC'] and len(cache['clip_1']['starts']) == 2
    assert word_ends(reader, rois, [1, 2, 3], 'clip_1', cache) == (word_of, ends) and reader.encoded == 1
    word_ends(reader, rois, [1, 2, 3], 'clip_2', cache)
    assert reader.encoded == 2
    assert load_alignments(str(tmp_path / 'none.json')) == {}


def test_recipe_c_is_the_default_and_flags_override_it():
    from lipread.adapt import RECIPES, resolve_recipe
    assert resolve_recipe('c') == dict(cut_p=0.5, ctc_weight=0.5, train_ctc=True, extra_blocks=2)
    assert resolve_recipe('lipflow') == RECIPES['lipflow'] == dict(cut_p=0.0, ctc_weight=0.1, train_ctc=False,
                                                                    extra_blocks=0)
    assert resolve_recipe('c', cut_p=0.0, train_ctc=False)['train_ctc'] is False
    assert resolve_recipe('c', extra_blocks=0) == dict(cut_p=0.5, ctc_weight=0.5, train_ctc=True, extra_blocks=0)


@pytest.fixture
def fake_main(monkeypatch, tmp_path):
    """adapt.main with every heavy part replaced; log records the calls."""
    import sys
    import types
    import lipread.adapt as adapt
    log = dict(finetune=None, saved=[], beam=0, seen=[])
    rows = [dict(id=f's1_{i:03d}', prompt='Hi.', dir='session_01') for i in range(10)]
    monkeypatch.setattr(adapt, 'sentence_clips', lambda sessions: list(rows) if sessions else [])
    monkeypatch.setattr(adapt, 'crops', lambda r, face: np.zeros((5, 96, 96), np.uint8))
    monkeypatch.setattr(adapt, 'score', lambda *a: (0.5, 0.4, 0, 0))

    def finetune(clips, reader, **kw):
        log['finetune'] = kw
        return ('encoder.',)
    monkeypatch.setattr(adapt, 'finetune', finetune)
    monkeypatch.setattr(adapt, 'save', lambda reader, path, prefixes: (log['saved'].append(path),
                                                                      open(path, 'w').close(), 1)[2])
    monkeypatch.setattr(adapt, 'ALIGN_CACHE', str(tmp_path / 'align.json'))
    seen = iter([0.30, 0.35])
    monkeypatch.setattr(adapt, 'streaming_seen', lambda *a, **k: dict(seen_wer=next(seen)))
    monkeypatch.setitem(sys.modules, 'lipflow.vsr', types.SimpleNamespace(LipReader=lambda **k: object()))
    monkeypatch.setitem(sys.modules, 'lipflow.train_vsr', types.SimpleNamespace(DEFAULT_SCOPE='scope'))

    class Beam:
        def __init__(self, reader):
            log['beam'] += 1
    monkeypatch.setitem(sys.modules, 'lipread.final', types.SimpleNamespace(Beam=Beam))
    monkeypatch.setitem(sys.modules, 'lipread.model', types.SimpleNamespace(
        LiveModel=lambda **k: object(), FACE_ADAPTER_PREFIXES=adapt.FACE_ADAPTER_PREFIXES))
    return adapt, log


def test_main_trains_recipe_c_and_builds_no_beam_without_a_test_set(fake_main, tmp_path):
    adapt, log = fake_main
    assert adapt.main(['--out', str(tmp_path / 'a.pth')]) == 0
    assert log['beam'] == 0 and log['saved'] == [str(tmp_path / 'a.pth')]
    assert (log['finetune']['cut_p'], log['finetune']['ctc_weight'], log['finetune']['train_ctc'],
            log['finetune']['extra_blocks']) == (0.5, 0.5, True, 2)
    assert log['finetune']['align_cache'] == adapt.ALIGN_CACHE
    adapt.main(['--recipe', 'lipflow', '--no-train-ctc', '--out', str(tmp_path / 'b.pth')])
    assert log['finetune']['cut_p'] == 0.0 and log['finetune']['train_ctc'] is False


def test_main_gates_saving_on_the_seen_wer(fake_main, tmp_path):
    adapt, log = fake_main
    out = tmp_path / 'a.pth'
    assert adapt.main(['--test-every', '5', '--out', str(out)]) == 1
    assert log['beam'] == 1 and not out.exists() and not (tmp_path / 'a.pth.candidate').exists()
    assert log['saved'] == [str(out) + '.candidate']


def test_main_saves_a_worse_adapter_with_force_and_a_better_one_without(fake_main, tmp_path, monkeypatch):
    adapt, log = fake_main
    out = tmp_path / 'a.pth'
    assert adapt.main(['--test-every', '5', '--force', '--out', str(out)]) == 0 and out.exists()
    seen = iter([0.30, 0.25])
    monkeypatch.setattr(adapt, 'streaming_seen', lambda *a, **k: dict(seen_wer=next(seen)))
    out.unlink()
    assert adapt.main(['--test-every', '5', '--out', str(out)]) == 0 and out.exists()
