"""
오토라벨링 결과 후처리 필터 (보수적 모드)

프레임 단위 예측의 flicker(0.2초짜리 조각 세그먼트)를 정리한다.
핵심 원칙:
  - 짧은 구간이라도 신뢰도가 높으면 실제 짧은 행동(예: Barking)으로 보고 보존
  - 짧고 + 신뢰도 낮은 구간만 노이즈로 보고 이웃 라벨에 흡수
기존 _labeled.csv → _smoothed.csv 로 출력하며 컬럼 구성은 그대로 유지한다
(make_viewer.py 가 timestamp/센서/pred_label/confidence 를 그대로 사용).

사용:
  python postprocess.py --input outputs/<name>_labeled.csv \
      --min-duration 1.0 --smooth-window 0.5 --protect-conf 0.85
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append('/workspace')  # make_viewer.analyze_audio 임포트용 (cwd 무관)

UNCLASSIFIED = '미분류'
BARK_LABEL = 'Barking'
RED_DB_OFFSET = 15   # 뷰어 dbToColor 빨강 기준(AUDIO_THRESHOLD_DB + 15)과 동일


def detect_hz(df):
    iv = df['timestamp'].diff().dropna().median()
    return round(1000 / iv), iv


def segments(labels):
    """연속 동일 라벨 구간을 [(start, end)] (end exclusive) 로 반환"""
    n = len(labels)
    if n == 0:
        return []
    chg = np.where(labels[1:] != labels[:-1])[0] + 1
    bounds = np.concatenate([[0], chg, [n]])
    return list(zip(bounds[:-1], bounds[1:]))


def mode_smooth(labels, conf, win):
    """신뢰도 가중 다수결 슬라이딩 윈도우 스무딩.
    각 프레임을 중심으로 ±win//2 범위에서 라벨별 confidence 합이 가장 큰 라벨로 교체.
    confidence 로 가중하므로 확신이 강한 짧은 구간은 잘 지워지지 않는다."""
    if win <= 1:
        return labels.copy()
    n = len(labels)
    half = win // 2
    classes = list(np.unique(labels))
    cls_idx = {c: i for i, c in enumerate(classes)}
    # 프레임별 (라벨 one-hot * confidence) 를 누적합으로 윈도우 합산
    onehot = np.zeros((n, len(classes)), dtype=np.float64)
    onehot[np.arange(n), [cls_idx[l] for l in labels]] = conf
    csum = np.vstack([np.zeros(len(classes)), np.cumsum(onehot, axis=0)])
    out = labels.copy()
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        win_sum = csum[hi] - csum[lo]
        out[i] = classes[int(win_sum.argmax())]
    return out


def merge_short(labels, conf, min_len, protect_conf):
    """min_len(프레임) 미만 + 평균 confidence < protect_conf 인 구간을
    이웃(더 긴 쪽, 동률이면 평균 신뢰도 높은 쪽)에 흡수. 변화 없을 때까지 반복."""
    labels = labels.copy()
    while True:
        segs = segments(labels)
        if len(segs) <= 1:
            break
        # 길이 오름차순으로 가장 짧은 후보부터 처리.
        # seg의 위치(idx)를 함께 저장해 segs.index() 선형탐색(→전체 O(n²))을 제거.
        seg_info = []
        for idx, (s, e) in enumerate(segs):
            length = e - s
            avg_c = conf[s:e].mean()
            seg_info.append((length, avg_c, s, e, idx))
        seg_info.sort(key=lambda x: (x[0], x[1]))

        changed = False
        for length, avg_c, s, e, idx in seg_info:
            if length >= min_len:
                break  # 이보다 긴 구간은 더 볼 필요 없음
            if avg_c >= protect_conf:
                continue  # 짧지만 확신이 강함 → 실제 행동으로 보존
            left = segs[idx - 1] if idx > 0 else None
            right = segs[idx + 1] if idx < len(segs) - 1 else None
            cand = []
            if left:
                cand.append((left[1] - left[0], conf[left[0]:left[1]].mean(), labels[left[0]]))
            if right:
                cand.append((right[1] - right[0], conf[right[0]:right[1]].mean(), labels[right[0]]))
            if not cand:
                continue
            cand.sort(key=lambda x: (x[0], x[1]), reverse=True)
            labels[s:e] = cand[0][2]
            changed = True
            break  # 세그먼트 구조가 바뀌었으므로 재계산
        if not changed:
            break
    return labels


def apply_audio_barking(df, video_path, audio_offset_ms, threshold_db,
                        min_dur_ms, bark_label, db_margin):
    """영상 오디오에서 '소리가 큰 구간'을 찾아 해당 센서 프레임을 Barking으로 덮어쓴다.
    영상 상대시간(ms) == 센서 시작 기준 상대시간(ms) 으로 정렬.
    audio_offset_ms: 센서가 영상보다 X ms 늦으면 +X (미세조정용).
    db_margin: 적응형 임계값 마진(배경+N dB). 클수록 큰 소리만 Barking."""
    from make_viewer import analyze_audio  # 기존 적응형 dB 분석 재사용

    res = analyze_audio(video_path, threshold_db=threshold_db,
                        min_dur_ms=min_dur_ms, db_margin=db_margin)
    events = res.get('events', [])
    timeline = res.get('timeline', [])
    thr = res.get('threshold_db', -35)
    if not events and not timeline:
        print("  소리 구간 없음 → Barking 덮어쓰기 스킵")
        return df, 0

    t0 = int(df['timestamp'].iloc[0])
    rel = df['timestamp'].values - t0           # 센서 시작 기준 상대 ms
    vid_time = rel - audio_offset_ms            # 대응되는 영상 시간(ms)

    # 임계값 이상(뷰어에서 초록이 아닌 = 큰 소리) 버킷은 모두 Barking.
    # min_dur(연속 300ms) 필터를 거치는 '이벤트' 대신 dB 타임라인을 직접 사용해,
    # 짧고 강한 짖음 펄스(주황/빨강)도 빠짐없이 잡는다. 슬라이더 임계값이 경계.
    bucket_ms = (timeline[1]['t'] - timeline[0]['t']) if len(timeline) > 1 else 200
    mask = np.zeros(len(df), dtype=bool)
    n_loud = 0
    for pt in timeline:
        if pt['db'] >= thr:                         # 뷰어 dbToColor에서 초록 위(주황 이상)
            mask |= (vid_time >= pt['t']) & (vid_time < pt['t'] + bucket_ms)
            n_loud += 1
    # 보조: 감지된 연속 소리 구간도 포함(버킷 경계 사이 보정)
    for ev in events:
        mask |= (vid_time >= ev['start_ms']) & (vid_time <= ev['end_ms'])

    n = int(mask.sum())
    df.loc[mask, 'pred_label'] = bark_label
    print(f"  큰 소리 버킷 {n_loud}개(db≥{thr:.1f}) + 소리구간 {len(events)}개 "
          f"→ {n:,} 프레임 '{bark_label}' (offset {audio_offset_ms}ms)")
    return df, n


def postprocess(input_path, min_duration, smooth_window, protect_conf,
                video_path=None, audio_offset_ms=0, audio_threshold_db=None,
                audio_min_dur_ms=300, bark_label=BARK_LABEL, audio_db_margin=12,
                use_algo=True, algo_locomotion=False, algo_swap_posture=False):
    df = pd.read_csv(input_path)
    if 'pred_label' not in df.columns or 'confidence' not in df.columns:
        raise ValueError("입력에 pred_label / confidence 컬럼이 필요합니다.")

    hz, iv = detect_hz(df)
    labels = df['pred_label'].astype(object).values.copy()
    conf = df['confidence'].astype(float).values

    smooth_frames = int(round(smooth_window * hz))
    min_frames = int(round(min_duration * hz))

    before_segs = len(segments(labels))
    print(f"입력: {input_path}")
    print(f"  {hz}Hz, {len(df):,}행 ({len(df)/hz/60:.1f}분)")
    print(f"  세그먼트(전): {before_segs:,}")
    print(f"  스무딩 윈도우: {smooth_window}s ({smooth_frames} frames)")
    print(f"  최소 구간: {min_duration}s ({min_frames} frames), 보존 신뢰도: >= {protect_conf}")

    # ── 알고리즘 보정 (스무딩 전): 자세/이동/털기를 신호처리로 교정 ──
    if use_algo:
        print(f"\n[알고리즘 보정 — 하이브리드]")
        from algo_label import algorithmic_correct
        labels, _cal = algorithmic_correct(df.assign(pred_label=labels), hz,
                                           allow_locomotion=algo_locomotion,
                                           swap_posture=algo_swap_posture)

    smoothed = mode_smooth(labels, conf, smooth_frames)
    after_smooth = len(segments(smoothed))
    merged = merge_short(smoothed, conf, min_frames, protect_conf)
    after_merge = len(segments(merged))

    changed = int((merged != df['pred_label'].values).sum())
    print(f"  세그먼트(스무딩 후): {after_smooth:,}")
    print(f"  세그먼트(병합 후): {after_merge:,}")
    print(f"  라벨 변경 프레임: {changed:,} ({changed/len(df)*100:.1f}%)")

    df['pred_label'] = merged

    # ── 오디오 기반 Barking 덮어쓰기 (가장 마지막, 권위적) ──
    if video_path:
        print(f"\n[오디오 Barking] {video_path}")
        df, n_bark = apply_audio_barking(
            df, video_path, audio_offset_ms, audio_threshold_db,
            audio_min_dur_ms, bark_label, audio_db_margin)
        after_audio = len(segments(df['pred_label'].astype(object).values))
        print(f"  세그먼트(오디오 후): {after_audio:,}")

    base = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.dirname(input_path) or '.'
    out_path = os.path.join(out_dir, base + '_smoothed.csv')
    df.to_csv(out_path, index=False)

    print(f"\n클래스별 분포(후처리 후):")
    for cls, cnt in df['pred_label'].value_counts().items():
        print(f"  {cls:12s}: {cnt:>7,} ({cnt/len(df)*100:.1f}%)")
    print(f"\n저장: {out_path}")
    return out_path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='*_labeled.csv 경로')
    p.add_argument('--min-duration', type=float, default=1.0,
                   help='이보다 짧은(초) + 저신뢰 구간은 이웃에 병합 (기본 1.0)')
    p.add_argument('--smooth-window', type=float, default=0.5,
                   help='신뢰도 가중 다수결 스무딩 윈도우(초). 0이면 비활성 (기본 0.5)')
    p.add_argument('--protect-conf', type=float, default=0.85,
                   help='평균 신뢰도가 이 값 이상인 짧은 구간은 보존 (기본 0.85)')
    p.add_argument('--no-algo', action='store_true',
                   help='알고리즘 보정(자세 Lying/Standing) 비활성화 (기본: 활성)')
    p.add_argument('--algo-locomotion', action='store_true',
                   help='[실험] 알고리즘이 Walking/Running도 판정(신뢰도 낮음, 기본 off→DL)')
    p.add_argument('--swap-posture', action='store_true',
                   help='자세 군집 이름(Lying↔Standing)이 뒤집혔을 때 교정')
    # ── 오디오 기반 Barking 덮어쓰기 ──
    p.add_argument('--video', default=None,
                   help='영상 경로. 지정 시 소리 큰 구간을 Barking으로 덮어씀')
    p.add_argument('--audio-offset', type=int, default=0,
                   help='센서가 영상보다 X ms 늦으면 +X (정렬 미세조정, 기본 0)')
    p.add_argument('--audio-threshold-db', type=float, default=None,
                   help='소리 임계값 dB(절대값). 미지정 시 배경 노이즈 기준 적응형')
    p.add_argument('--audio-db-margin', type=float, default=12,
                   help='적응형 임계값 마진(배경+N dB). 클수록 큰 소리만 Barking (기본 12)')
    p.add_argument('--audio-min-dur', type=int, default=300,
                   help='이보다 짧은(ms) 소리 구간은 무시 (기본 300)')
    p.add_argument('--bark-label', default=BARK_LABEL,
                   help="덮어쓸 라벨명 (기본 'Barking')")
    args = p.parse_args()
    postprocess(args.input, args.min_duration, args.smooth_window, args.protect_conf,
                video_path=args.video, audio_offset_ms=args.audio_offset,
                audio_threshold_db=args.audio_threshold_db,
                audio_min_dur_ms=args.audio_min_dur, bark_label=args.bark_label,
                audio_db_margin=args.audio_db_margin, use_algo=not args.no_algo,
                algo_locomotion=args.algo_locomotion, algo_swap_posture=args.swap_posture)
