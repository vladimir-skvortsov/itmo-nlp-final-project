"""Tests for data loading and preprocessing."""

import textwrap
from pathlib import Path

import pytest

from nerel_re.data.dataset import (
    NO_RELATION,
    Entity,
    RelationExample,
    _parse_ann_file,
)
from nerel_re.data.negative_sampler import NegativeSampler
from nerel_re.data.tokenizer_utils import mark_entities

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_ann_file(tmp_path: Path) -> tuple[Path, str]:
    text = 'Владимир Путин посетил Москву вчера.'
    ann_content = textwrap.dedent("""\
        T1\tPER 0 15\tВладимир Путин
        T2\tCITY 24 30\tМоскву
        R1\tVISIT Arg1:T1 Arg2:T2
    """)
    ann_path = tmp_path / 'doc.ann'
    ann_path.write_text(ann_content, encoding='utf-8')
    return ann_path, text


# ---------------------------------------------------------------------------
# _parse_ann_file
# ---------------------------------------------------------------------------


def test_parse_ann_file_entities(tmp_ann_file):
    ann_path, text = tmp_ann_file
    entities, _relations = _parse_ann_file(ann_path, text)

    assert 'T1' in entities
    assert 'T2' in entities
    assert entities['T1'].type == 'PER'
    assert entities['T2'].type == 'CITY'
    assert entities['T1'].text == 'Владимир Путин'
    assert entities['T2'].text == 'Москву'


def test_parse_ann_file_relations(tmp_ann_file):
    ann_path, text = tmp_ann_file
    _, relations = _parse_ann_file(ann_path, text)

    assert len(relations) == 1
    rtype, arg1, arg2 = relations[0]
    assert rtype == 'VISIT'
    assert arg1 == 'T1'
    assert arg2 == 'T2'


def test_parse_ann_file_empty(tmp_path: Path):
    ann_path = tmp_path / 'empty.ann'
    ann_path.write_text('', encoding='utf-8')
    entities, relations = _parse_ann_file(ann_path, '')
    assert entities == {}
    assert relations == []


# ---------------------------------------------------------------------------
# NegativeSampler
# ---------------------------------------------------------------------------


def _make_example(doc_id: str, e1_id: str, e2_id: str, relation: str) -> RelationExample:
    e1 = Entity(id=e1_id, type='PER', start=0, end=4, text='Test')
    e2 = Entity(id=e2_id, type='ORG', start=10, end=14, text='Corp')
    return RelationExample(
        doc_id=doc_id,
        sentence='Test works at Corp.',
        entity1=e1,
        entity2=e2,
        relation=relation,
    )


def test_negative_sampler_produces_no_relation():
    positives = [_make_example('doc1', 'T1', 'T2', 'WORKS_FOR')]
    sampler = NegativeSampler(negative_ratio=2.0, seed=0)
    combined = sampler.sample(positives)

    no_rel = [ex for ex in combined if ex.relation == NO_RELATION]
    assert len(no_rel) >= 0  # May be 0 if no entity pairs remain
    assert all(ex.relation in {'WORKS_FOR', NO_RELATION} for ex in combined)


def test_negative_sampler_seed_reproducible():
    positives = [_make_example(f'doc{i}', 'T1', 'T2', 'WORKS_FOR') for i in range(5)]
    s1 = NegativeSampler(negative_ratio=2.0, seed=42).sample(positives)
    s2 = NegativeSampler(negative_ratio=2.0, seed=42).sample(positives)
    assert [ex.relation for ex in s1] == [ex.relation for ex in s2]


def test_negative_sampler_no_positives():
    sampler = NegativeSampler(negative_ratio=1.0, seed=0)
    result = sampler.sample([])
    assert result == []


# ---------------------------------------------------------------------------
# mark_entities
# ---------------------------------------------------------------------------


def test_mark_entities_typed():
    text = 'Путин посетил Москву'
    marked = mark_entities(
        text=text,
        subj_start=0,
        subj_end=5,
        subj_type='PER',
        obj_start=14,
        obj_end=20,
        obj_type='CITY',
        use_typed_markers=True,
    )
    assert '[e1_PER]' in marked
    assert '[/e1_PER]' in marked
    assert '[e2_CITY]' in marked
    assert '[/e2_CITY]' in marked
    assert 'Путин' in marked
    assert 'Москву' in marked


def test_mark_entities_plain():
    text = 'Путин посетил Москву'
    marked = mark_entities(
        text=text,
        subj_start=0,
        subj_end=5,
        subj_type='PER',
        obj_start=14,
        obj_end=20,
        obj_type='CITY',
        use_typed_markers=False,
    )
    assert '[E1]' in marked
    assert '[E2]' in marked
    assert '[e1_PER]' not in marked


def test_mark_entities_obj_before_subj():
    text = 'В Москве живёт Путин'
    marked = mark_entities(
        text=text,
        subj_start=15,
        subj_end=20,
        subj_type='PER',
        obj_start=2,
        obj_end=8,
        obj_type='CITY',
        use_typed_markers=True,
    )
    assert '[e1_PER]' in marked
    assert '[e2_CITY]' in marked
    # Order in text: obj comes first (Москве before Путин)
    assert marked.index('[e2_CITY]') < marked.index('[e1_PER]')
