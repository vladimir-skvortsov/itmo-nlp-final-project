"""Unit tests for NEREL NER data loading and BIO label alignment."""

import pytest

from nerel_ner.data.dataset import (
    Entity,
    NERSentence,
    _align_labels,
    _split_sentences,
    build_label_maps,
    collect_entity_types,
    load_splits,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_entity_types():
    return ['ORG', 'PER']


@pytest.fixture
def label_maps(simple_entity_types):
    return build_label_maps(simple_entity_types)


# ---------------------------------------------------------------------------
# build_label_maps
# ---------------------------------------------------------------------------

def test_label_maps_structure(label_maps):
    label2id, _id2label = label_maps
    assert label2id['O'] == 0
    assert 'B-ORG' in label2id
    assert 'I-ORG' in label2id
    assert 'B-PER' in label2id
    assert 'I-PER' in label2id
    assert len(label2id) == 5  # O + 2*2


def test_label_maps_roundtrip(label_maps):
    label2id, id2label = label_maps
    for lbl, idx in label2id.items():
        assert id2label[idx] == lbl


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------

def test_split_sentences_basic():
    text = 'Иван Иванов работает.\nОН живёт в Москве.'
    entities = [
        Entity(id='T1', type='PER', start=0, end=11, text='Иван Иванов'),
        Entity(id='T2', type='LOC', start=33, end=39, text='Москве'),
    ]
    sents = _split_sentences('doc1', text, entities)
    assert len(sents) == 2
    assert sents[0].offset == 0
    assert sents[1].offset == len('Иван Иванов работает.\n')


def test_split_sentences_entity_assignment():
    text = 'Путин — президент.\nМедведев — премьер.'
    entities = [
        Entity(id='T1', type='PER', start=0, end=5, text='Путин'),
        Entity(id='T2', type='PER', start=19, end=27, text='Медведев'),
    ]
    sents = _split_sentences('doc1', text, entities)
    assert len(sents[0].entities) == 1
    assert sents[0].entities[0].text == 'Путин'
    assert len(sents[1].entities) == 1
    assert sents[1].entities[0].text == 'Медведев'


def test_split_sentences_empty_lines_skipped():
    text = 'Первое предложение.\n\nВторое предложение.'
    sents = _split_sentences('doc1', text, [])
    assert len(sents) == 2


def test_split_sentences_no_entities():
    text = 'Просто текст без сущностей.'
    sents = _split_sentences('doc1', text, [])
    assert len(sents) == 1
    assert sents[0].entities == []


# ---------------------------------------------------------------------------
# _align_labels
# ---------------------------------------------------------------------------

def test_align_labels_simple(label_maps):
    label2id, _ = label_maps
    # Simulate offset_mapping for: [CLS] "Иван" "##ов" [SEP] [PAD]
    # Entity PER: doc offset 0-7 ("Иванов"), sentence_offset=0
    offset_mapping = [(0, 0), (0, 4), (4, 6), (0, 0), (0, 0)]
    entities = [Entity(id='T1', type='PER', start=0, end=6, text='Иванов')]
    labels = _align_labels(offset_mapping, entities, label2id, sentence_offset=0)
    assert labels[0] == -100  # [CLS]
    assert labels[1] == label2id['B-PER']
    assert labels[2] == label2id['I-PER']
    assert labels[3] == -100  # [SEP]
    assert labels[4] == -100  # [PAD]


def test_align_labels_outside(label_maps):
    label2id, _ = label_maps
    offset_mapping = [(0, 0), (0, 5), (6, 10), (0, 0)]
    labels = _align_labels(offset_mapping, [], label2id, sentence_offset=0)
    assert labels[1] == label2id['O']
    assert labels[2] == label2id['O']


def test_align_labels_with_sentence_offset(label_maps):
    label2id, _ = label_maps
    # Sentence starts at doc offset 10; entity at doc offset 10-14.
    offset_mapping = [(0, 0), (0, 4), (0, 0)]
    entities = [Entity(id='T1', type='ORG', start=10, end=14, text='ТАСС')]
    labels = _align_labels(offset_mapping, entities, label2id, sentence_offset=10)
    assert labels[1] == label2id['B-ORG']


def test_align_labels_two_entities(label_maps):
    label2id, _ = label_maps
    # Two non-overlapping entities in the same sentence.
    offset_mapping = [(0, 0), (0, 3), (4, 7), (0, 0)]
    entities = [
        Entity(id='T1', type='PER', start=0, end=3, text='Иван'),
        Entity(id='T2', type='ORG', start=4, end=7, text='РИА'),
    ]
    labels = _align_labels(offset_mapping, entities, label2id, sentence_offset=0)
    assert labels[1] == label2id['B-PER']
    assert labels[2] == label2id['B-ORG']


# ---------------------------------------------------------------------------
# collect_entity_types
# ---------------------------------------------------------------------------

def test_collect_entity_types():
    sent1 = NERSentence(
        doc_id='d1', text='...', offset=0,
        entities=[Entity('T1', 'PER', 0, 5, 'Иван')]
    )
    sent2 = NERSentence(
        doc_id='d1', text='...', offset=10,
        entities=[Entity('T2', 'ORG', 10, 14, 'ТАСС')]
    )
    splits = {'train': [sent1, sent2]}
    types = collect_entity_types(splits)
    assert types == ['ORG', 'PER']


# ---------------------------------------------------------------------------
# load_splits (filesystem integration test — requires NEREL data)
# ---------------------------------------------------------------------------

def test_load_splits_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match='Split directory not found'):
        load_splits(tmp_path / 'nonexistent')


def test_load_splits_minimal(tmp_path):
    """Load a minimal BRAT-formatted dataset from a temp directory."""
    for split in ('train', 'dev', 'test'):
        split_dir = tmp_path / split
        split_dir.mkdir()
        (split_dir / 'doc1.txt').write_text('Иван Иванов работает в Москве.', encoding='utf-8')
        (split_dir / 'doc1.ann').write_text(
            'T1\tPER 0 11\tИван Иванов\nT2\tLOC 23 29\tМоскве\n',
            encoding='utf-8',
        )

    splits = load_splits(tmp_path)
    assert set(splits) == {'train', 'dev', 'test'}
    for split_sents in splits.values():
        assert len(split_sents) >= 1
        entity_types = {e.type for s in split_sents for e in s.entities}
        assert 'PER' in entity_types
        assert 'LOC' in entity_types
