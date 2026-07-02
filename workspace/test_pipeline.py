"""
파이프라인 순수 로직 단위 테스트.
heavy 의존(GPU/모델 weight) 없이 신호처리·정렬·경로안전 로직을 검증한다.

실행:
    cd workspace && python -m pytest test_pipeline.py -v
    (pytest 없으면)  python test_pipeline.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_viewer
import postprocess


# ── make_viewer 순수 함수 ────────────────────────────────────────────────
def test_extract_ts_parses_and_no_dead_code():
    ts = make_viewer.extract_ts('sensor_data_20260421_063519.csv')
    assert ts is not None and ts > 0
    assert make_viewer.extract_ts('no_timestamp_here.csv') is None


def test_build_class_colors_unclassified_is_gray():
    colors = make_viewer.build_class_colors(['Walking', 'Lying', 'Unlabeled'])
    assert colors['Unlabeled'] == make_viewer.UNLABELED_COLOR
    assert colors['Walking'] != colors['Lying']


def test_build_shortcut_map_digits_for_positive():
    sc = make_viewer.build_shortcut_map(['Standing', 'Scratching', 'Unlabeled'])
    assert sc['1'] == 'Standing'           # positive → 숫자
    assert 'a' in sc and sc['a'] == 'Scratching'  # negative → 알파벳
    assert 'Unlabeled' in sc.values()


def test_make_segments_merges_short():
    # 0.1초짜리 짧은 B가 앞 A에 흡수되어야 함
    df = pd.DataFrame({
        'time_ms':   [0, 100, 200, 300, 400, 500, 5000, 6000],
        'pred_label':['A','A', 'A', 'B', 'A', 'A', 'C',  'C'],
        'confidence':[0.9]*8,
    })
    segs = make_viewer.make_segments(df, conf_threshold=0.7, min_seg_ms=500)
    labels = [s['label'] for s in segs]
    assert 'B' not in labels       # 짧은 B 흡수됨
    assert labels[-1] == 'C'


# ── postprocess 신호처리 ─────────────────────────────────────────────────
def test_segments_boundaries():
    labels = np.array(['a', 'a', 'b', 'b', 'b', 'a'])
    assert postprocess.segments(labels) == [(0, 2), (2, 5), (5, 6)]
    assert postprocess.segments(np.array([])) == []


def test_merge_short_absorbs_low_conf_blip():
    # 길이 1짜리 저신뢰 'X'는 이웃 'A'에 흡수
    labels = np.array(['A'] * 10 + ['X'] + ['A'] * 10)
    conf   = np.array([0.9] * 10 + [0.3] + [0.9] * 10)
    out = postprocess.merge_short(labels, conf, min_len=3, protect_conf=0.85)
    assert 'X' not in out


def test_merge_short_protects_high_conf_short():
    # 짧지만 고신뢰면 보존 (실제 짧은 행동)
    labels = np.array(['A'] * 10 + ['Barking'] + ['A'] * 10)
    conf   = np.array([0.9] * 10 + [0.95] + [0.9] * 10)
    out = postprocess.merge_short(labels, conf, min_len=3, protect_conf=0.85)
    assert 'Barking' in out


def test_mode_smooth_removes_isolated_flip():
    labels = np.array(['A', 'A', 'B', 'A', 'A'])
    conf   = np.ones(5)
    out = postprocess.mode_smooth(labels, conf, win=3)
    assert (out == 'A').all()


# ── 경로 안전 (회귀 방지: #4) ────────────────────────────────────────────
def test_safe_base_strips_traversal():
    import server
    assert server.safe_base('../../etc/passwd') == 'passwd'
    assert server.safe_base('normal_name') == 'normal_name'


def test_safe_under_blocks_escape():
    import server
    from pathlib import Path
    from fastapi import HTTPException
    # 정상: 디렉터리 안
    p = server.safe_under(server.TEST_DATA_DIR, 'clip.mp4')
    assert str(p).endswith('clip.mp4')
    # 탈출 시도는 basename만 남아 디렉터리 안에 머문다
    p2 = server.safe_under(server.TEST_DATA_DIR, '../../../etc/passwd')
    assert p2.parent == server.TEST_DATA_DIR.resolve()


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
