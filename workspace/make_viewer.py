"""
오토 레이블링 검수 뷰어 v6
- 클래스/색상/단축키/센서컬럼 CSV에서 자동 감지
- 다중 ID 지원 (id 컬럼 있으면 자동 tier 분리)
- 구간 분할/병합/되돌리기
- 검수 진행률/통계
- 도움말 모달
"""
import os, re, argparse, json
import numpy as np
import pandas as pd
from datetime import datetime

# sentiment 매핑
POSITIVE_CLASSES = {'Standing','Lying','Running','Walking','Sniffing','Eating','Drinking','Barking','Shaking'}
NEGATIVE_CLASSES = {'Scratching','Licking','Vomiting','Coughing'}

def get_sentiment(label):
    if label in POSITIVE_CLASSES: return 'Positive'
    if label in NEGATIVE_CLASSES: return 'Negative'
    if label == 'Unlabeled':          return 'Background'
    return 'Positive'

# 기본 색상 팔레트 (클래스 수에 따라 순환)
PALETTE = [
    '#378ADD','#1D9E75','#E24B4A','#EF9F27','#9B6DD6',
    '#D85A30','#42B0C4','#D4537E','#639922','#C4A020',
    '#20A4C4','#A43080','#4080C4','#80C440','#C44040',
    '#40C4A0','#A08040','#8040A0','#40A080','#A04040',
]
UNLABELED_COLOR = '#555555'

# 예약된 컬럼 (센서 컬럼 자동 감지 시 제외)
RESERVED_COLS = {'timestamp','time_ms','pred_label','confidence','id',
                 'ID','category','file_name','activity_id','final_activity'}

def build_class_colors(classes):
    """클래스 목록에서 자동으로 색상 할당"""
    colors = {}
    palette_classes = [c for c in classes if c != 'Unlabeled']
    for i, cls in enumerate(palette_classes):
        colors[cls] = PALETTE[i % len(PALETTE)]
    colors['Unlabeled'] = UNLABELED_COLOR
    return colors

def build_shortcut_map(classes):
    """클래스 목록에서 자동으로 단축키 할당
    긍정 행동: 1~9, 0
    부정 행동: a, b, c, d, ...
    Unlabeled: e (또는 부정행동 다음 알파벳)
    """
    num_keys  = list('1234567890')
    alpha_keys = list('abcdefghijklmnopqrstuvwxyz')
    sc = {}
    num_i   = 0
    alpha_i = 0
    for cls in classes:
        if cls == 'Unlabeled':
            continue
        if cls in POSITIVE_CLASSES:
            if num_i < len(num_keys):
                sc[num_keys[num_i]] = cls
                num_i += 1
        else:  # 부정행동 or 기타
            if alpha_i < len(alpha_keys):
                sc[alpha_keys[alpha_i]] = cls
                alpha_i += 1
    if 'Unlabeled' in classes:
        # 부정행동 다음 알파벳
        sc[alpha_keys[alpha_i]] = 'Unlabeled'
    return sc

def detect_sensor_cols(df):
    """센서 컬럼 자동 감지 (예약 컬럼 제외한 숫자형 컬럼)"""
    candidates = [c for c in df.columns if c not in RESERVED_COLS]
    sensor = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    return sensor

def get_video_codec(video_path):
    """영상 코덱 확인"""
    import subprocess
    r = subprocess.run(
        ['/usr/bin/ffprobe','-v','error','-select_streams','v:0',
         '-show_entries','stream=codec_name',
         '-of','default=noprint_wrappers=1:nokey=1', str(video_path)],
        capture_output=True, text=True)
    return r.stdout.strip()

def convert_to_h264(video_path):
    """hevc → h264 변환, 변환된 파일 경로 반환"""
    import subprocess
    video_path = str(video_path)
    out_path   = video_path.replace('.mp4', '_h264.mp4')
    print(f"hevc 감지 → H.264로 변환 중... ({os.path.getsize(video_path)/1024/1024:.1f}MB)")
    subprocess.run([
        '/usr/bin/ffmpeg', '-y', '-i', video_path,
        '-vcodec', 'libx264', '-crf', '23', '-preset', 'fast',
        '-acodec', 'aac', out_path
    ], check=True, capture_output=True)
    print(f"변환 완료: {out_path}")
    return out_path

def extract_ts(fname):
    m = re.search(r'(\d{8})_(\d{6})', fname)
    if m:
        return int(datetime.strptime(m.group(1)+m.group(2),'%Y%m%d%H%M%S').timestamp()*1000)
    return None

def make_segments(df, conf_threshold=0.7, min_seg_ms=500):
    """
    레이블 변화 기준으로 구간 생성
    min_seg_ms 이하 짧은 구간은 인접 구간으로 흡수
    """
    labels = df['pred_label'].values
    times  = df['time_ms'].values
    confs  = df['confidence'].values if 'confidence' in df.columns else np.ones(len(df))

    # 1차: 레이블 변화 기준으로 구간 생성
    raw_segs, i = [], 0
    while i < len(labels):
        j, cc = i+1, [confs[i]]
        while j < len(labels) and labels[j] == labels[i]:
            cc.append(confs[j]); j += 1
        avg = float(np.mean(cc))
        raw_segs.append({
            'label': labels[i], 'start_ms': int(times[i]), 'end_ms': int(times[j-1]),
            'conf': round(avg,3), 'low_conf': avg < conf_threshold,
            'start_idx': i, 'end_idx': j-1,
        })
        i = j

    # 2차: 짧은 구간 인접 구간으로 흡수 (앞 구간에 합침)
    merged = []
    for seg in raw_segs:
        dur = seg['end_ms'] - seg['start_ms']
        if merged and dur < min_seg_ms:
            # 앞 구간에 흡수
            prev = merged[-1]
            prev['end_ms']   = seg['end_ms']
            prev['end_idx']  = seg['end_idx']
            prev['conf']     = round((prev['conf'] + seg['conf']) / 2, 3)
            prev['low_conf'] = prev['conf'] < conf_threshold
        else:
            merged.append(seg)

    return merged

def analyze_audio(video_path, threshold_db=None, merge_gap_ms=1000, min_dur_ms=300,
                  db_margin=12):
    """
    영상에서 오디오 추출 후 dB 분석
    threshold_db=None이면 적응형 임계값 사용 (배경 노이즈 median + db_margin dB)
    db_margin: 클수록 더 큰 소리만 감지 (민감도 ↓)
    반환: {
      'events':   [{'start_ms': int, 'end_ms': int}, ...]   감지된 소리 구간
      'timeline': [{'t': int, 'db': float}, ...]            200ms 단위 dB 타임라인
    }
    """
    import subprocess, tempfile
    empty = {'events': [], 'timeline': [], 'threshold_db': -35}
    try:
        import librosa
        import numpy as _np
    except ImportError:
        print("librosa 미설치 → 오디오 분석 스킵")
        return empty

    wav_path = None
    try:
        # ffmpeg으로 오디오 WAV 추출 (mono, 22050Hz)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            wav_path = tmp.name

        subprocess.run([
            '/usr/bin/ffmpeg', '-y', '-i', str(video_path),
            '-ac', '1', '-ar', '22050', '-vn', wav_path
        ], capture_output=True, check=True)

        # librosa로 dB 분석
        y, sr = librosa.load(wav_path, sr=22050, mono=True)

        # 프레임 단위 RMS → dB 변환
        hop_length = 512  # ~23ms per frame
        rms   = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        db    = librosa.amplitude_to_db(rms, ref=_np.max)
        times = librosa.frames_to_time(range(len(db)), sr=sr, hop_length=hop_length)

        # ── 적응형 임계값: 배경 노이즈(median) 기준 +N dB ──────────────
        if threshold_db is None:
            noise_floor = float(_np.median(db))
            threshold_db = noise_floor + db_margin  # 배경보다 db_margin dB 이상 큰 소리만
            # 너무 낮거나 높지 않게 클램프
            threshold_db = max(-45, min(-10, threshold_db))
            print(f"적응형 임계값: 배경 {noise_floor:.1f}dB +{db_margin}dB → threshold {threshold_db:.1f}dB")

        # 임계값 이상 구간 추출
        loud_mask = db > threshold_db
        raw_events = []
        in_loud = False
        start_ms = 0
        for i, (t, loud) in enumerate(zip(times, loud_mask)):
            ms = int(t * 1000)
            if loud and not in_loud:
                in_loud = True
                start_ms = ms
            elif not loud and in_loud:
                in_loud = False
                dur = ms - start_ms
                if dur >= min_dur_ms:
                    raw_events.append({'start_ms': start_ms, 'end_ms': ms})
        if in_loud:
            ms = int(times[-1] * 1000)
            if ms - start_ms >= min_dur_ms:
                raw_events.append({'start_ms': start_ms, 'end_ms': ms})

        # 짧은 침묵 병합
        events = []
        if raw_events:
            merged = [dict(raw_events[0])]
            for ev in raw_events[1:]:
                if ev['start_ms'] - merged[-1]['end_ms'] <= merge_gap_ms:
                    merged[-1]['end_ms'] = ev['end_ms']
                else:
                    merged.append(dict(ev))
            events = merged

        # ── dB 타임라인 (200ms 단위 다운샘플링, train UI용) ──────────
        bucket_ms = 200
        total_ms  = int(times[-1] * 1000) if len(times) else 0
        timeline  = []
        if total_ms > 0:
            n_buckets = total_ms // bucket_ms + 1
            for b in range(n_buckets):
                t0, t1 = b * bucket_ms, (b + 1) * bucket_ms
                mask = (times * 1000 >= t0) & (times * 1000 < t1)
                if mask.any():
                    val = float(_np.max(db[mask]))
                else:
                    val = -60.0
                timeline.append({'t': int(t0), 'db': round(val, 1)})

        print(f"오디오 분석 완료: {len(events)}개 소리 구간, dB 타임라인 {len(timeline)}개, threshold={threshold_db:.1f}dB")
        return {'events': events, 'timeline': timeline, 'threshold_db': round(threshold_db, 1)}

    except Exception as e:
        print(f"오디오 분석 실패: {e}")
        return empty
    finally:
        # 임시 WAV 정리 (librosa.load 등에서 예외가 나도 누수 방지)
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass


def make_elan(sensor_path, label_path, video_path, output_html,
              conf_threshold=0.7, manual_offset=None,
              encoder_path=None, classes_override=None, extra_classes=None):

    import subprocess

    # ── 영상 코덱 확인 및 변환 ─────────────────────────────────────────────
    if video_path and os.path.exists(str(video_path)):
        codec = get_video_codec(video_path)
        print(f"영상 코덱: {codec}")
        if codec == 'hevc':
            video_path = convert_to_h264(video_path)
        video_filename = os.path.basename(video_path)
    else:
        video_filename = ''

    sensor = pd.read_csv(sensor_path)
    labels = pd.read_csv(label_path)

    # label CSV가 reviewed CSV인 경우 (pred_label 이미 있음) → 그대로 사용
    # label CSV가 labeled CSV인 경우 → sensor와 merge
    if 'pred_label' in labels.columns and 'accel_x' in labels.columns:
        # reviewed CSV를 label로 올린 경우: 센서값도 포함되어 있음
        df = labels.copy()
        # sensor 컬럼명 맞추기
        df['confidence'] = df['confidence'].fillna(0.0)
        df['pred_label'] = df['pred_label'].fillna('Unlabeled')
    elif 'pred_label' in labels.columns:
        # labeled CSV: timestamp 기준으로 merge
        merge_cols = ['timestamp','pred_label','confidence']
        merge_cols = [c for c in merge_cols if c in labels.columns]
        df = pd.merge(sensor, labels[merge_cols], on='timestamp', how='left')
        df['pred_label'] = df['pred_label'].fillna('Unlabeled')
        df['confidence'] = df['confidence'].fillna(0.0)
    else:
        # pred_label 없음 → 전부 Unlabeled
        print("⚠ label CSV에 pred_label 없음 → 전부 Unlabeled 처리")
        df = sensor.copy()
        df['pred_label'] = 'Unlabeled'
        df['confidence'] = 0.0

    # ── 센서 컬럼 자동 감지 ───────────────────────────────────────────────
    exist_sensor_cols = detect_sensor_cols(sensor)
    print(f"감지된 센서 컬럼: {exist_sensor_cols}")

    # ── 클래스 자동 감지 + 색상/단축키 자동 할당 ──────────────────────────
    # 우선순위: 1) --classes 직접 지정 2) label_encoder.pkl 3) CSV 감지
    if classes_override:
        classes = [c.strip() for c in classes_override.split(',')]
        print(f"클래스 직접 지정: {classes}")
    elif encoder_path and os.path.exists(encoder_path):
        import pickle
        with open(encoder_path, 'rb') as f:
            le = pickle.load(f)
        # numpy.str_ → 순수 str (NumPy 2.x는 repr이 np.str_('x') 라서 JS로 새면 문법오류)
        classes = [str(c) for c in le.classes_]
        print(f"label_encoder.pkl에서 로드: {classes}")
    else:
        classes = sorted(df['pred_label'].unique().tolist())
        print(f"CSV에서 감지: {classes}")

    # Unlabeled는 항상 맨 뒤에 추가
    classes = [c for c in classes if c != 'Unlabeled'] + ['Unlabeled']

    # extra_classes 추가 (Unlabeled 바로 앞에 삽입)
    if extra_classes:
        extras = [c.strip() for c in extra_classes.split(',') if c.strip() and c.strip() not in classes]
        classes = [c for c in classes if c != 'Unlabeled'] + extras + ['Unlabeled']
        print(f"추가 클래스: {extras}")
    # CSV에 있지만 클래스 목록에 없는 값 경고
    csv_classes = set(df['pred_label'].unique())
    missing = csv_classes - set(classes) - {'Unlabeled'}
    if missing:
        print(f"⚠ CSV에 있지만 클래스 목록에 없는 값: {missing} → Unlabeled로 처리")
        df.loc[df['pred_label'].isin(missing), 'pred_label'] = 'Unlabeled'

    CLASS_COLORS   = build_class_colors(classes)
    SHORTCUT_MAP   = build_shortcut_map(classes)
    print(f"색상 할당: {CLASS_COLORS}")
    print(f"단축키 할당: {SHORTCUT_MAP}")

    t0 = int(df['timestamp'].iloc[0])
    df['time_ms'] = (df['timestamp'] - t0).astype(int)
    imu_total_ms = int(df['time_ms'].iloc[-1])

    # 영상 길이 감지
    result = subprocess.run(
        ['/usr/bin/ffprobe','-v','error','-show_entries','format=duration',
         '-of','default=noprint_wrappers=1:nokey=1', str(video_path)],
        capture_output=True, text=True)
    try:
        video_dur_ms = int(float(result.stdout.strip()) * 1000)
    except:
        video_dur_ms = imu_total_ms
        print("영상 길이 감지 실패 → IMU 길이 사용")

    # 짧은 쪽에 맞춰 자르기
    total_ms = min(imu_total_ms, video_dur_ms)
    df = df[df['time_ms'] <= total_ms].reset_index(drop=True)

    if imu_total_ms > video_dur_ms:
        print(f"IMU({imu_total_ms}ms) > 영상({video_dur_ms}ms) → IMU 뒤 {imu_total_ms-video_dur_ms}ms 제거")
    elif video_dur_ms > imu_total_ms:
        print(f"영상({video_dur_ms}ms) > IMU({imu_total_ms}ms) → 영상 뒤 {video_dur_ms-imu_total_ms}ms 무시")
    else:
        print(f"영상/IMU 길이 일치: {total_ms}ms")
    print(f"최종 사용 길이: {total_ms}ms ({total_ms/1000:.3f}초), {len(df):,}행")

    # ── sentiment 컬럼 추가 (id 분기 전에 미리 추가) ─────────────────────
    df['sentiment'] = df['pred_label'].apply(get_sentiment)

    # ── id 컬럼 감지 ──────────────────────────────────────────────────────
    has_id = 'id' in df.columns
    if has_id:
        ids = sorted(df['id'].unique().tolist())
        print(f"ID 감지됨: {ids}")
        # id별 세그먼트 생성
        id_segs = {}
        id_low_conf = {}
        for id_val in ids:
            sub = df[df['id'] == id_val].reset_index(drop=True)
            segs_i = make_segments(sub, conf_threshold)
            id_segs[str(id_val)] = segs_i
            id_low_conf[str(id_val)] = sum(1 for s in segs_i if s['low_conf'])
        low_conf_count = sum(id_low_conf.values())
        total_seg_count = sum(len(v) for v in id_segs.values())
        # id별 CSV rows
        id_rows = {}
        for id_val in ids:
            sub = df[df['id'] == id_val].reset_index(drop=True)
            id_rows[str(id_val)] = sub[['timestamp','time_ms']+exist_sensor_cols+['pred_label','sentiment','confidence']].to_json(orient='records')
        id_segs_js   = str(id_segs).replace("'",'"').replace('True','true').replace('False','false')
        id_rows_js   = '{' + ','.join(f'"{k}":{v}' for k,v in id_rows.items()) + '}'
        ids_js       = str([str(i) for i in ids]).replace("'",'"')
    else:
        ids = ['default']
        segs = make_segments(df, conf_threshold)
        low_conf_count = sum(1 for s in segs if s['low_conf'])
        total_seg_count = len(segs)
        segs_str = str(segs).replace("'",'"').replace('True','true').replace('False','false')
        rows_str = df[['timestamp','time_ms']+exist_sensor_cols+['pred_label','sentiment','confidence']].to_json(orient='records')
        id_segs_js = '{"default":' + segs_str + '}'
        id_rows_js = '{"default":' + rows_str + '}'
        ids_js = '["default"]'

    video_filename = os.path.basename(video_path)

    save_cols    = ['timestamp'] + (['id'] if has_id else []) + exist_sensor_cols + ['pred_label','sentiment','confidence']
    colors_js    = str(CLASS_COLORS).replace("'",'"')
    classes_js   = str(list(CLASS_COLORS.keys())).replace("'",'"')
    sc_js        = str(SHORTCUT_MAP).replace("'",'"')
    save_cols_js = str(save_cols).replace("'",'"')
    out_filename  = os.path.splitext(os.path.basename(sensor_path))[0]+'_labeled_reviewed.csv'
    stat_filename = os.path.splitext(os.path.basename(sensor_path))[0]+'_review_stats.json'
    has_id_js     = 'true' if has_id else 'false'
    # 클래스 목록 JS 배열 (단축키 힌트용)

    csv_ts = extract_ts(os.path.basename(label_path))
    vid_ts = extract_ts(os.path.basename(video_path))
    offset_ms = manual_offset if manual_offset is not None else (csv_ts - vid_ts if csv_ts and vid_ts else 0)
    sync_info = f'오프셋 {offset_ms}ms'

    # ── 오디오 분석 (영상에서 소리 구간 + dB 타임라인 미리 추출) ──────────
    audio_result = {'events': [], 'timeline': [], 'threshold_db': -35}
    if video_path and os.path.exists(str(video_path)):
        print("오디오 분석 중...")
        audio_result = analyze_audio(video_path)

    audio_events_js     = json.dumps(audio_result['events'])
    audio_timeline_js   = json.dumps(audio_result['timeline'])
    audio_threshold_js  = audio_result['threshold_db']

    # 배속 버튼 미리 생성 (f-string 중첩 방지)
    speed_buttons = ''.join(
        f'<button class="btn speed-btn" data-speed="{sp}" onclick="setSpeed({sp})">{sp}x</button>'
        for sp in [0.25, 0.5, 1, 1.5, 2, 4]
    )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>오토 레이블링 검수 뷰어</title>
<style>
:root {{
  --bg:#0f0f0f; --bg2:#161616; --bg3:#1e1e1e;
  --border:#2a2a2a; --border2:#444;
  --text:#f0f0f0; --text2:#bbb; --text3:#888;
  --accent:#ff9944; --green:#1D9E75; --blue:#378ADD; --red:#E24B4A; --yellow:#EF9F27;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);display:flex;flex-direction:column;height:100vh;overflow:hidden;font-size:13px}}

/* 툴바 */
.toolbar{{background:var(--bg2);border-bottom:1px solid var(--border);padding:7px 14px;display:flex;align-items:center;gap:8px;flex-shrink:0;flex-wrap:wrap}}
.logo{{font-size:14px;font-weight:700;color:var(--text);letter-spacing:0.3px;margin-right:4px}}
.timecode{{font-family:'SF Mono',monospace;font-size:17px;font-weight:600;background:#000;padding:5px 14px;border-radius:6px;color:var(--accent);letter-spacing:1.5px;min-width:140px;text-align:center}}
.tb-sep{{width:1px;height:20px;background:var(--border2);margin:0 2px}}
.btn{{background:var(--bg3);border:1px solid var(--border2);color:var(--text2);padding:6px 12px;border-radius:6px;font-size:12px;cursor:pointer;transition:all 0.15s;white-space:nowrap;font-weight:500}}
.btn:hover{{background:#252525;border-color:#666;color:var(--text)}}
.btn-warn{{border-color:#7a5000;color:#ffaa33}}
.btn-warn:hover{{background:#2a1f00;border-color:#ffaa33;color:#ffcc66}}
.btn-audio{{border-color:#7a0030;color:#ff6b9d}}
.btn-audio:hover{{background:#2a001a;border-color:#ff6b9d;color:#ffaacc}}
.speed-btn{{padding:4px 12px;font-size:12px;min-width:42px;font-weight:600}}
.speed-btn.active{{background:#1a2a3a;border-color:var(--blue);color:var(--blue)}}
.audio-blk{{position:absolute;top:2px;height:calc(100% - 4px);border-radius:2px;pointer-events:all;cursor:pointer;opacity:0.85;transition:opacity 0.1s}}
.audio-blk:hover{{opacity:1;outline:1px solid #ff6b9d}}
.btn-save{{background:var(--green);border-color:var(--green);color:white;font-weight:600}}
.btn-save:hover{{background:#18856A}}
.btn-help{{background:#1a1a2e;border-color:#4455aa;color:#99aaff}}
.btn-help:hover{{background:#1e2040;border-color:#99aaff;color:#ccddff}}
.mod-counter{{font-size:12px;color:var(--text2);margin-left:auto;display:flex;align-items:center;gap:6px}}
.mod-badge{{background:#2a1a00;border:1px solid #7a5000;color:#ffaa33;border-radius:10px;padding:1px 8px;font-size:11px;font-weight:600}}
.mod-badge.zero{{background:var(--bg3);border-color:var(--border2);color:var(--text3)}}

/* 진행률 바 */
.progress-bar-wrap{{background:var(--bg);border-bottom:1px solid var(--border);padding:5px 14px;flex-shrink:0;display:flex;align-items:center;gap:10px}}
.progress-label{{font-size:11px;color:var(--text2);white-space:nowrap;min-width:180px}}
.progress-track{{flex:1;height:5px;background:#2a2a2a;border-radius:3px;overflow:hidden}}
.progress-fill{{height:100%;background:var(--green);border-radius:3px;transition:width 0.3s}}
.progress-pct{{font-size:12px;color:var(--green);font-weight:600;min-width:36px;text-align:right}}

/* 메인 */
.main{{display:flex;flex-direction:column;flex:1;min-height:0}}
.top-area{{display:flex;flex:1;min-height:0;border-bottom:1px solid var(--border)}}

/* 영상 — 너비 키움 */
.video-panel{{width:65%;flex-shrink:0;background:#000;display:flex;flex-direction:column;position:relative;overflow:hidden}}
.video-wrap{{flex:1;position:relative;overflow:hidden;cursor:grab;display:flex;align-items:center;justify-content:center}}
.video-wrap.grabbing{{cursor:grabbing}}
video{{max-width:100%;max-height:100%;object-fit:contain;transform-origin:center center;transition:none;user-select:none}}
.video-controls-overlay{{position:absolute;bottom:8px;right:8px;display:flex;gap:4px;z-index:20}}
.vid-btn{{background:rgba(0,0,0,0.6);border:1px solid rgba(255,255,255,0.2);color:#fff;border-radius:4px;padding:3px 8px;font-size:11px;cursor:pointer}}
.vid-btn:hover{{background:rgba(0,0,0,0.85)}}

/* 현재 프레임 레이블 바 (영상과 배속 선택 영역 사이, 불투명, 라벨 색으로 변함) */
.now-label-bar{{
  display:flex;align-items:baseline;gap:16px;
  padding:16px 22px;
  background:#161616;
  border-top:1px solid #222;
  flex-shrink:0;
  min-height:64px;
  transition:background 0.15s;
}}
.now-label{{font-size:30px;font-weight:800;letter-spacing:0.3px;line-height:1.1;color:#fff;text-shadow:0 1px 4px rgba(0,0,0,0.45)}}
.now-conf{{font-size:15px;color:rgba(255,255,255,0.85);text-shadow:0 1px 3px rgba(0,0,0,0.4)}}

/* 사이드 패널 — 키워서 잘 보이게 */
.side-panel{{flex:1;background:var(--bg2);border-left:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden}}
.side-section{{border-bottom:1px solid var(--border);padding:12px 16px;flex-shrink:0}}
.side-title{{font-size:11px;font-weight:700;color:var(--text2);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:10px}}

/* 입력 */
.tc-row{{display:flex;align-items:center;gap:6px;margin-bottom:9px}}
.tc-wrap{{flex:1;display:flex;flex-direction:column;gap:3px}}
.tc-wrap label{{font-size:10px;color:var(--text2);text-transform:uppercase;letter-spacing:0.5px;font-weight:600}}
.tc-input{{font-family:'SF Mono',monospace;background:#111;border:1px solid var(--border2);color:var(--accent);border-radius:6px;padding:7px 9px;font-size:14px;width:100%;letter-spacing:0.5px;transition:border-color 0.15s}}
.tc-input:focus{{outline:none;border-color:var(--accent);box-shadow:0 0 0 2px rgba(255,153,68,0.15)}}
.tc-input.error{{border-color:var(--red);color:var(--red)}}
.arrow{{color:var(--text2);font-size:16px;flex-shrink:0;margin-top:16px}}
.class-select{{width:100%;background:#111;border:1px solid var(--border2);color:var(--text);border-radius:6px;padding:7px 9px;font-size:13px;margin-bottom:9px;cursor:pointer;font-weight:500}}
.action-row{{display:flex;gap:6px;margin-bottom:6px}}
.apply-btn{{flex:1;padding:8px;background:var(--blue);border:none;color:white;border-radius:7px;font-size:13px;font-weight:600;cursor:pointer;transition:background 0.15s}}
.apply-btn:hover{{background:#2a6db5}}
.icon-btn{{padding:7px 10px;background:var(--bg3);border:1px solid var(--border2);color:var(--text2);border-radius:7px;font-size:12px;cursor:pointer;white-space:nowrap;transition:all 0.15s;font-weight:500}}
.icon-btn:hover{{background:#252525;color:var(--text);border-color:#666}}
.icon-btn.split{{border-color:#6a5a00;color:#ffcc33}}
.icon-btn.split:hover{{background:#2a2200;border-color:#ffcc33;color:#ffe066}}
.icon-btn.merge{{border-color:#1a3a6a;color:#66aaff}}
.icon-btn.merge:hover{{background:#0a1e30;border-color:#66aaff;color:#99ccff}}
.feedback{{font-size:12px;margin-top:6px;min-height:18px;line-height:1.5;color:var(--text)}}

/* 구간 목록 */
.seg-list-wrap{{flex:1;overflow-y:auto;padding:6px 8px}}
.seg-item{{display:flex;align-items:center;gap:7px;padding:7px 9px;border-radius:7px;cursor:pointer;border:1px solid transparent;margin-bottom:3px;transition:all 0.1s}}
.seg-item:hover{{background:var(--bg3)}}
.seg-item.active{{background:var(--bg3);border-color:var(--border2)}}
.seg-item.low-conf-item{{border-left:3px solid var(--red)!important}}
.seg-item.modified-item{{border-left:3px solid white!important}}
.seg-item.playing{{background:#13283d;box-shadow:inset 0 0 0 1px var(--blue)}}
.seg-item.playing .seg-name::before{{content:'▶ ';color:var(--blue);font-size:9px;vertical-align:middle}}
.seg-dot{{width:9px;height:9px;border-radius:3px;flex-shrink:0}}
.seg-info{{flex:1;min-width:0}}
.seg-name{{font-weight:600;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.seg-time{{font-size:11px;color:var(--text2);font-family:monospace;margin-top:2px}}
.seg-conf-badge{{font-size:10px;padding:2px 6px;border-radius:4px;flex-shrink:0;font-weight:600}}

/* 타임라인 */
.timeline-area{{background:var(--bg2);flex-shrink:0;height:300px;display:flex;flex-direction:column;border-top:1px solid var(--border)}}
.tl-toolbar{{padding:6px 12px;display:flex;align-items:center;gap:8px;border-bottom:1px solid var(--border);flex-shrink:0}}
.tl-title{{font-size:12px;font-weight:700;color:var(--text);text-transform:uppercase;letter-spacing:0.8px}}
.zoom-group{{display:flex;align-items:center;gap:4px;margin-left:auto}}
.zoom-btn{{background:var(--bg3);border:1px solid var(--border2);color:var(--text2);width:24px;height:24px;border-radius:4px;cursor:pointer;font-size:14px;display:flex;align-items:center;justify-content:center;transition:all 0.1s}}
.zoom-btn:hover{{background:#252525;color:var(--text)}}
.zoom-label{{font-size:12px;color:var(--text);min-width:32px;text-align:center;font-weight:600}}
.tl-hint{{font-size:11px;color:var(--text2)}}
.tl-body{{flex:1;display:flex;overflow:hidden}}
.tier-labels{{width:110px;flex-shrink:0;background:var(--bg);border-right:1px solid var(--border);display:flex;flex-direction:column}}
.tier-label-cell{{border-bottom:1px solid var(--border);display:flex;flex-direction:column;justify-content:center;padding:0 12px;flex:1}}
.tier-label-cell:last-child{{border-bottom:none}}
.tlc-name{{font-size:13px;font-weight:700;color:var(--text)}}
.tlc-sub{{font-size:10px;color:var(--text2);margin-top:2px}}
.tl-scroll{{flex:1;overflow-x:auto;overflow-y:hidden;position:relative}}
.tl-inner{{height:100%;position:relative;display:flex;flex-direction:column;min-width:100%}}
.ruler{{height:22px;background:#0d0d0d;border-bottom:1px solid var(--border);position:relative;flex-shrink:0;z-index:5}}
.r-tick{{position:absolute;top:0;height:100%;border-left:1px solid #2a2a2a}}
.r-tick-label{{position:absolute;top:3px;left:2px;font-size:9px;color:var(--text2);font-family:monospace;white-space:nowrap;background:#0d0d0d;padding:0 2px;pointer-events:none}}
.r-tick.major{{border-color:#3a3a3a}}
.r-tick.major .r-tick-label{{color:var(--text);font-weight:700}}
.tier-track{{flex:1;position:relative;border-bottom:1px solid var(--border);overflow:hidden}}
.tl-inner{{overflow:hidden}}
.tier-track:last-child{{border-bottom:none}}
.ann{{position:absolute;top:3px;height:calc(100% - 6px);border-radius:4px;overflow:hidden;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;color:white;cursor:pointer;user-select:none;transition:filter 0.1s}}
.ann:hover{{filter:brightness(1.15)}}
.ann.selected{{outline:2px solid white;outline-offset:1px;z-index:5;filter:brightness(1.1)}}
.ann.low-conf{{border-top:2px solid var(--red);border-bottom:2px solid var(--red)}}
.ann.modified{{border-top:2px solid white;border-bottom:2px solid white}}
.ann-text{{padding:0 8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;pointer-events:none;text-shadow:0 1px 3px rgba(0,0,0,0.5)}}
.conf-blk{{position:absolute;top:3px;height:calc(100% - 6px);border-radius:3px;pointer-events:none;display:flex;align-items:center}}
.conf-txt{{font-size:12px;color:rgba(255,255,255,0.9);padding:0 6px;overflow:hidden;white-space:nowrap;font-weight:700}}
.playhead{{position:absolute;top:0;width:2px;height:100%;background:var(--red);pointer-events:none;z-index:30}}
.ph-head{{position:absolute;top:-1px;left:-5px;width:12px;height:8px;background:var(--red);border-radius:2px 2px 0 0}}

/* 툴팁 */
.tooltip{{position:fixed;background:rgba(10,10,10,0.96);border:1px solid var(--border2);border-radius:7px;padding:8px 12px;font-size:12px;pointer-events:none;z-index:1000;display:none;max-width:220px;line-height:1.6}}
.tooltip.show{{display:block}}
.tt-label{{font-weight:700;margin-bottom:3px;font-size:13px}}
.tt-row{{font-size:11px;color:var(--text2)}}

/* 모달 공통 */
.modal-overlay{{position:fixed;inset:0;background:rgba(0,0,0,0.7);z-index:1500;display:none;align-items:center;justify-content:center}}
.modal-overlay.show{{display:flex}}
.modal{{background:#1a1a1a;border:1px solid #3a3a3a;border-radius:14px;padding:0;z-index:1600;max-height:85vh;overflow:hidden;display:flex;flex-direction:column;box-shadow:0 20px 60px rgba(0,0,0,0.7)}}
.modal-header{{padding:16px 20px;border-bottom:1px solid #2a2a2a;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}}
.modal-title{{font-size:15px;font-weight:600}}
.modal-close{{background:none;border:none;color:#666;font-size:18px;cursor:pointer;padding:0 4px;line-height:1}}
.modal-close:hover{{color:#ccc}}
.modal-body{{padding:20px;overflow-y:auto;flex:1}}

/* 도움말 모달 */
.help-modal{{width:600px}}
.help-section{{margin-bottom:20px}}
.help-section-title{{font-size:12px;font-weight:700;color:var(--accent);margin-bottom:10px;padding-bottom:5px;border-bottom:1px solid #2a2a2a;text-transform:uppercase;letter-spacing:0.5px}}
.help-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
.help-item{{background:#111;border:1px solid #2a2a2a;border-radius:8px;padding:10px 12px}}
.help-item-title{{font-size:11px;font-weight:600;color:var(--text);margin-bottom:4px;display:flex;align-items:center;gap:6px}}
.help-item-desc{{font-size:10px;color:var(--text3);line-height:1.5}}
.help-badge{{background:#1a2a1a;color:var(--green);border:1px solid #2a4a2a;border-radius:3px;padding:1px 5px;font-size:9px;font-weight:600}}
.help-badge.blue{{background:#1a1a2a;color:var(--blue);border-color:#2a2a4a}}
.help-badge.orange{{background:#2a1a00;color:var(--accent);border-color:#4a3a00}}
.help-badge.red{{background:#2a1a1a;color:var(--red);border-color:#4a2a2a}}
.shortcut-table{{width:100%;border-collapse:collapse}}
.shortcut-table td{{padding:4px 8px;font-size:10px;border-bottom:1px solid #1a1a1a}}
.shortcut-table td:first-child{{font-family:monospace;color:var(--accent);width:120px}}
.shortcut-table td:last-child{{color:var(--text2)}}
.workflow-steps{{display:flex;flex-direction:column;gap:6px}}
.workflow-step{{display:flex;align-items:flex-start;gap:10px;background:#111;border-radius:8px;padding:8px 12px}}
.step-num{{background:var(--blue);color:white;border-radius:50%;width:20px;height:20px;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;flex-shrink:0;margin-top:1px}}
.step-content{{flex:1}}
.step-title{{font-size:11px;font-weight:600;margin-bottom:2px}}
.step-desc{{font-size:10px;color:var(--text3);line-height:1.4}}

/* 저장 모달 */
.save-modal{{width:340px;text-align:center}}
.save-icon{{font-size:40px;margin-bottom:12px}}
.save-modal-title{{font-size:15px;font-weight:600;margin-bottom:6px}}
.save-modal-file{{font-size:11px;color:#888;margin-bottom:8px}}
.save-modal-desc{{font-size:11px;color:#555;margin-bottom:20px;line-height:1.5}}
.save-modal-btns{{display:flex;gap:8px;justify-content:center}}
.save-modal-btn{{padding:8px 20px;border-radius:6px;font-size:12px;cursor:pointer;font-weight:500}}

/* 스크롤바 */
.tl-scroll::-webkit-scrollbar{{height:14px}}
.tl-scroll::-webkit-scrollbar-track{{background:#0d0d0d;border-top:1px solid var(--border)}}
.tl-scroll::-webkit-scrollbar-thumb{{background:#5a5a5a;border-radius:7px;border:3px solid #0d0d0d;min-width:40px}}
.tl-scroll::-webkit-scrollbar-thumb:hover{{background:#787878}}
.tl-scroll{{scrollbar-width:auto;scrollbar-color:#5a5a5a #0d0d0d}}
.seg-list-wrap::-webkit-scrollbar{{width:10px}}
.seg-list-wrap::-webkit-scrollbar-track{{background:transparent}}
.seg-list-wrap::-webkit-scrollbar-thumb{{background:#4a4a4a;border-radius:5px;border:2px solid var(--bg2)}}
.seg-list-wrap::-webkit-scrollbar-thumb:hover{{background:#666}}
.modal-body::-webkit-scrollbar{{width:8px}}
.modal-body::-webkit-scrollbar-thumb{{background:#4a4a4a;border-radius:4px}}

.shortcut-bar{{background:var(--bg);border-top:1px solid var(--border);padding:6px 14px;font-size:11px;color:var(--text2);flex-shrink:0}}
</style>
</head>
<body>

<!-- 툴바 -->
<div class="toolbar">
  <span class="logo">오토 레이블링 검수 뷰어</span>
  <div class="timecode" id="tcDisp">00:00:00.000</div>
  <div class="tb-sep"></div>
  <button class="btn btn-warn" onclick="jumpLowConf()">⚠ 다음 검수필요 ({low_conf_count})</button>
  <button class="btn btn-audio" id="nextAudioBtn" onclick="jumpNextAudio()">🔊 다음 Barking 구간</button>
  <button class="btn" onclick="undoLast()">↩ 되돌리기</button>
  <button class="btn btn-help" onclick="showHelp()">❓ 도움말</button>
  <div class="mod-counter">
    수정됨 <span class="mod-badge zero" id="modBadge">0</span>
    <span style="color:var(--text3);font-size:10px">{sync_info}</span>
  </div>
  <label class="btn" style="cursor:pointer" title="이전 저장 CSV 불러오기">
    📂 불러오기
    <input type="file" id="loadCsvInp" accept=".csv" style="display:none" onchange="loadCsvBackup(this)">
  </label>
  <div style="display:flex;flex-direction:column;align-items:center;gap:1px">
    <span id="autosaveStatus" style="font-size:9px;color:var(--text3)">자동저장 대기 중</span>
  </div>
  <button class="btn btn-save" onclick="downloadCSV()">💾 CSV 저장</button>
</div>

<!-- 진행률 바 -->
<div class="progress-bar-wrap">
  <div class="progress-label" id="progressLabel">검수 진행률 계산 중...</div>
  <div class="progress-track"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
  <div class="progress-pct" id="progressPct">0%</div>
</div>

<!-- 메인 -->
<div class="main">
  <div class="top-area">
    <div class="video-panel">
      <div id="videoPickWrap" style="display:flex;flex-direction:column;align-items:center;justify-content:center;flex:1;gap:14px;background:#0a0a0a;">
        <div style="font-size:14px;color:var(--text2);">영상 파일을 선택하세요</div>
        <div style="font-size:12px;color:var(--text3);">📁 {video_filename}</div>
        <label style="cursor:pointer;background:var(--blue);color:white;padding:9px 22px;border-radius:7px;font-size:13px;font-weight:600;">
          영상 선택
          <input type="file" id="videoPicker" accept="video/*" style="display:none">
        </label>
      </div>
      <div class="video-wrap" id="videoWrap">
        <video id="vid" controls style="display:none"></video>
        <div class="video-controls-overlay">
          <button class="vid-btn" onclick="vidZoom(1.25)">＋</button>
          <button class="vid-btn" onclick="vidZoom(0.8)">－</button>
          <button class="vid-btn" onclick="vidFit()">Fit</button>
          <button class="vid-btn" onclick="vidFull()">⛶</button>
        </div>
      </div>
      <!-- 현재 프레임 레이블: 영상과 배속 선택 영역 사이 고정 바 -->
      <div class="now-label-bar" id="nowLabelBar">
        <span class="now-label" id="nowLabel">—</span>
        <span class="now-conf" id="nowConf"></span>
      </div>
      <div style="display:flex;align-items:center;gap:6px;padding:7px 10px;background:#0a0a0a;border-top:1px solid #222">
        <span style="font-size:11px;color:#555;margin-right:4px">배속</span>
        {speed_buttons}
      </div>
      <!-- Audio 트랙: 영상 아래 독립 영역, 기차처럼 흘러가는 UI -->
      <div style="background:#0d0d0d;border-top:1px solid #222;padding:6px 10px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
          <span style="font-size:10px;color:#888;font-weight:600;min-width:36px">AUDIO</span>
          <canvas id="globalLevelMeter" width="120" height="18" style="border-radius:3px;flex-shrink:0"></canvas>
          <span id="audioSelectedRange" onclick="applyAudioRangeToInputs()" style="font-size:10px;color:#fff;background:#1a1a1a;padding:2px 8px;border-radius:3px;margin-left:4px;display:none;cursor:pointer;border:1px solid #444" title="클릭하면 시작/종료 시간에 채워짐"></span>
          <span style="font-size:9px;color:#555;margin-left:auto">← 지나감&nbsp;&nbsp;|&nbsp;&nbsp;현재&nbsp;&nbsp;|&nbsp;&nbsp;다가옴 →</span>
        </div>
        <!-- 기차 UI: 현재 위치 중앙 고정, 좌우로 흘러감 -->
        <div id="audioTrainWrap" style="position:relative;height:36px;background:#0a0a0a;border-radius:4px;overflow:hidden;cursor:pointer">
          <canvas id="audioTrainCanvas" style="position:absolute;top:0;left:0;width:100%;height:100%"></canvas>
          <div id="audioCenterLine" style="position:absolute;top:0;left:50%;width:2px;height:100%;background:#fff;opacity:0.5;pointer-events:none;z-index:10"></div>
        </div>
      </div>
    </div>


    <div class="side-panel">
      <div class="side-section">
        <div class="side-title">구간 레이블 수정</div>
        <div class="tc-row">
          <div class="tc-wrap">
            <label>시작 시간</label>
            <input class="tc-input" id="startInp" placeholder="00:00:00.000" maxlength="12">
          </div>
          <div class="arrow">→</div>
          <div class="tc-wrap">
            <label>종료 시간</label>
            <input class="tc-input" id="endInp" placeholder="00:00:00.000" maxlength="12">
          </div>
        </div>
        <select class="class-select" id="labelSel">
          {''.join(f'<option value="{c}">{list(SHORTCUT_MAP.keys())[i] if i < len(SHORTCUT_MAP) else ""}. {c}</option>' for i,c in enumerate(classes))}
        </select>
        <div class="action-row">
          <button class="apply-btn" onclick="applyEdit()">✓ 적용 (Enter)</button>
          <button class="icon-btn" onclick="fillFromVideo()">▶ 현재시간</button>
        </div>
        <div class="action-row">
          <button class="icon-btn split" onclick="splitAtCurrent()" style="flex:1">✂ 현재위치에서 분할</button>
          <button class="icon-btn merge" onclick="mergeSelected()" style="flex:1">⊕ 인접 구간 병합</button>
        </div>
        <div class="feedback" id="feedback"></div>
      </div>

      <div class="side-section" style="padding:7px 14px 5px">
        <div class="side-title">구간 목록</div>
      </div>
      <div class="seg-list-wrap" id="segListWrap"></div>
    </div>
  </div>

  <!-- 타임라인 -->
  <div class="timeline-area">
    <div class="tl-toolbar">
      <span class="tl-title">타임라인</span>
      <span class="tl-hint">클릭=이동 &nbsp;|&nbsp; 구간클릭=선택+이동</span>
      <div id="idTabs" style="display:flex;gap:4px;margin-left:12px"></div>
      <div class="zoom-group">
        <button class="zoom-btn" onclick="zoom(-1)">−</button>
        <span class="zoom-label" id="zoomLabel">5×</span>
        <button class="zoom-btn" onclick="zoom(1)">+</button>
        <button class="btn" style="padding:3px 8px;font-size:10px" onclick="resetZoom()">리셋</button>
      </div>
    </div>
    <div class="tl-body">
      <div class="tier-labels" id="tierLabels"></div>
      <div class="tl-scroll" id="tlScroll">
        <div class="tl-inner" id="tlInner">
          <div class="ruler" id="ruler"></div>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="shortcut-bar">
  단축키: {'  '.join(f'{k}={v}' for k,v in list(SHORTCUT_MAP.items()))} &nbsp;|&nbsp; Space=재생/정지 &nbsp;|&nbsp; ←→=1초 &nbsp;|&nbsp; Shift+←→=프레임(40ms) &nbsp;|&nbsp; Ctrl+Z=되돌리기 &nbsp;|&nbsp; +/-=줌
</div>

<!-- 툴팁 -->
<div class="tooltip" id="tooltip">
  <div class="tt-label" id="ttLabel"></div>
  <div class="tt-row" id="ttTime"></div>
  <div class="tt-row" id="ttConf"></div>
</div>

<!-- 도움말 모달 -->
<div class="modal-overlay" id="helpOverlay" onclick="if(event.target===this)hideHelp()">
  <div class="modal help-modal">
    <div class="modal-header">
      <div class="modal-title">❓ 오토 레이블링 검수 뷰어 사용법</div>
      <button class="modal-close" onclick="hideHelp()">✕</button>
    </div>
    <div class="modal-body">

      <div class="help-section">
        <div class="help-section-title">🚀 기본 워크플로우</div>
        <div class="workflow-steps">
          <div class="workflow-step">
            <div class="step-num">1</div>
            <div class="step-content">
              <div class="step-title">영상 재생하면서 확인</div>
              <div class="step-desc">Space 키로 재생/정지, ← → 키로 1초씩 이동합니다. 행동이 바뀌는 구간을 찾으세요.</div>
            </div>
          </div>
          <div class="workflow-step">
            <div class="step-num">2</div>
            <div class="step-content">
              <div class="step-title">수정할 구간 선택</div>
              <div class="step-desc">타임라인의 구간을 클릭하거나, ⚠ 다음 검수필요 버튼으로 신뢰도 낮은 구간으로 이동합니다.</div>
            </div>
          </div>
          <div class="workflow-step">
            <div class="step-num">3</div>
            <div class="step-content">
              <div class="step-title">레이블 수정</div>
              <div class="step-desc">시작/종료 시간을 입력하고 레이블을 선택 후 Enter 또는 적용 버튼을 누릅니다. 숫자키 1~9로 빠르게 선택할 수 있습니다.</div>
            </div>
          </div>
          <div class="workflow-step">
            <div class="step-num">4</div>
            <div class="step-content">
              <div class="step-title">CSV 저장</div>
              <div class="step-desc">검수가 끝나면 우측 상단 💾 CSV 저장 버튼을 눌러 결과를 저장합니다. 센서값과 레이블이 함께 저장됩니다.</div>
            </div>
          </div>
        </div>
      </div>

      <div class="help-section">
        <div class="help-section-title">🎯 주요 기능</div>
        <div class="help-grid">
          <div class="help-item">
            <div class="help-item-title"><span class="help-badge orange">3개 Tier</span> 타임라인</div>
            <div class="help-item-desc">Model Pred.(원본), Reviewed(수정), Confidence(신뢰도) 3개 줄로 비교하며 볼 수 있습니다.</div>
          </div>
          <div class="help-item">
            <div class="help-item-title"><span class="help-badge red">⚠</span> 검수 필요 구간</div>
            <div class="help-item-desc">빨간 테두리 구간은 AI 신뢰도가 낮아 검수가 필요합니다. 버튼으로 순서대로 이동할 수 있습니다.</div>
          </div>
          <div class="help-item">
            <div class="help-item-title"><span class="help-badge">✂ 분할</span> 구간 나누기</div>
            <div class="help-item-desc">영상 재생 중 행동이 바뀌는 순간에서 ✂ 버튼을 누르면 현재 위치에서 구간을 둘로 나눕니다.</div>
          </div>
          <div class="help-item">
            <div class="help-item-title"><span class="help-badge blue">⊕ 병합</span> 구간 합치기</div>
            <div class="help-item-desc">선택한 구간을 기준으로 좌우로 이어진 같은 레이블 구간을 개수 제한 없이 한 번에 합칩니다.</div>
          </div>
          <div class="help-item">
            <div class="help-item-title"><span class="help-badge">줌</span> 타임라인 확대</div>
            <div class="help-item-desc">+ / - 키 또는 버튼으로 타임라인을 확대/축소합니다. 짧은 구간을 정밀하게 볼 때 유용합니다.</div>
          </div>
          <div class="help-item">
            <div class="help-item-title"><span class="help-badge orange">↩</span> 되돌리기</div>
            <div class="help-item-desc">실수로 잘못 수정했을 때 Ctrl+Z 또는 버튼으로 최대 50단계까지 되돌릴 수 있습니다.</div>
          </div>
        </div>
      </div>

      <div class="help-section">
        <div class="help-section-title">⌨ 키보드 단축키</div>
        <table class="shortcut-table">
          <tr><td>1 ~ 9, 0</td><td>레이블 선택 (Standing, Lying, Running...)</td></tr>
          <tr><td>Enter</td><td>구간 수정 적용</td></tr>
          <tr><td>Space</td><td>재생 / 정지</td></tr>
          <tr><td>← →</td><td>1초 앞뒤로 이동</td></tr>
          <tr><td>Shift + ← →</td><td>0.04초(1프레임) 앞뒤로 이동</td></tr>
          <tr><td>Ctrl + Z</td><td>되돌리기</td></tr>
          <tr><td>+ / -</td><td>타임라인 줌인 / 줌아웃</td></tr>
        </table>
      </div>

      <div class="help-section">
        <div class="help-section-title">💡 시간 입력 방법</div>
        <div class="help-item" style="margin-bottom:0">
          <div class="help-item-desc" style="font-size:11px;line-height:1.7">
            숫자만 입력해도 자동으로 형식이 맞춰집니다.<br>
            예) <span style="color:var(--accent);font-family:monospace">0305</span> → <span style="color:var(--green);font-family:monospace">00:03:05.000</span><br>
            예) <span style="color:var(--accent);font-family:monospace">030512</span> → <span style="color:var(--green);font-family:monospace">00:03:05.120</span><br>
            입력창을 벗어나면 정확한 형식으로 자동 완성됩니다.
          </div>
        </div>
      </div>

    </div>
  </div>
</div>

<!-- 저장 모달 -->
<div class="modal-overlay" id="saveOverlay">
  <div class="modal save-modal">
    <div class="modal-body" style="text-align:center;padding:28px 32px">
      <div class="save-icon">💾</div>
      <div class="save-modal-title">CSV 파일명 설정</div>
      <div style="margin:14px 0 8px;text-align:left">
        <label style="font-size:11px;color:var(--text2);font-weight:600">파일명 (확장자 제외)</label>
        <div style="display:flex;align-items:center;gap:6px;margin-top:5px">
          <input id="saveFileNameInp" type="text"
            style="flex:1;background:#111;border:1px solid var(--border2);color:var(--accent);
                   border-radius:6px;padding:8px 10px;font-size:13px;font-family:monospace;outline:none"
            placeholder="파일명 입력">
          <span style="color:var(--text3);font-size:12px;white-space:nowrap">.csv</span>
        </div>
      </div>
      <div class="save-modal-desc" id="saveDesc" style="margin-top:10px"></div>
      <div class="save-modal-btns">
        <button class="save-modal-btn" onclick="hideSaveModal()"
          style="background:#2a2a2a;border:1px solid #444;color:#ccc">취소</button>
        <button class="save-modal-btn" onclick="confirmSave()"
          style="background:var(--green);border:none;color:white">저장</button>
      </div>
    </div>
  </div>
</div>

<script>
const COLORS      = __COLORS_JS__;
const CLASSES     = __CLASSES_JS__;
const SHORTCUTS   = __SC_JS__;
const TOTAL_MS    = {total_ms};
const TOTAL_SEC   = TOTAL_MS/1000;
const SAVE_COLS   = {save_cols_js};
const OUT_NAME    = '{out_filename}';
const STAT_NAME   = '{stat_filename}';
const CONF_THRESH = {conf_threshold};
const IMU_STEP    = 10;
const TOTAL_SEGS  = {total_seg_count};
const LOW_CONF_TOTAL = {low_conf_count};
const HAS_ID      = {has_id_js};
const IDS         = {ids_js};
const ID_SEGS_ORIG= __ID_SEGS_ORIG__;
const ID_ROWS     = __ID_ROWS__;
const START_TIME  = Date.now();

// 현재 선택된 ID (단일 동물이면 'default')
let currentId = IDS[0];

// ID별 상태
const state = {{}};
IDS.forEach(id => {{
  state[id] = {{
    tier1Segs: JSON.parse(JSON.stringify(ID_SEGS_ORIG[id])),
    tier2Segs: JSON.parse(JSON.stringify(ID_SEGS_ORIG[id])),
    history:   [],
    selectedIdx: -1,
    lastSegIdx:  -1,
  }};
}});

// 현재 ID의 상태 접근 편의 함수
function cur() {{ return state[currentId]; }}
// 하위 호환: 기존 코드가 tier2Segs 직접 참조하는 부분을 위한 proxy
function getTier2Segs() {{ return cur().tier2Segs; }}

const vid      = document.getElementById('vid');
const tcDisp   = document.getElementById('tcDisp');
const startI   = document.getElementById('startInp');
const endI     = document.getElementById('endInp');
const labelS   = document.getElementById('labelSel');
const feedbk   = document.getElementById('feedback');
const tooltip  = document.getElementById('tooltip');
const modBadge = document.getElementById('modBadge');

// 하위 호환 (단일 ID 코드들이 직접 참조)
Object.defineProperty(window, 'tier2Segs', {{
  get: () => cur().tier2Segs,
  set: (v) => {{ cur().tier2Segs = v; }}
}});
Object.defineProperty(window, 'selectedIdx', {{
  get: () => cur().selectedIdx,
  set: (v) => {{ cur().selectedIdx = v; }}
}});
Object.defineProperty(window, 'lastSegIdx', {{
  get: () => cur().lastSegIdx,
  set: (v) => {{ cur().lastSegIdx = v; }}
}});
Object.defineProperty(window, 'history', {{
  get: () => cur().history,
  set: (v) => {{ cur().history = v; }}
}});

let zoomIdx    = 6;
let zoomLevel  = 5;

const ZOOM_LEVELS = [0.5,0.75,1,1.5,2,3,5,8,12,16,24,32];

// ── sentiment 폰트 색상 ───────────────────────────────────────────────────
const G_POSITIVE = new Set(['Standing','Lying','Running','Walking','Sniffing','Eating','Drinking','Barking','Shaking']);
const G_NEGATIVE = new Set(['Scratching','Licking','Vomiting','Coughing']);
function getSentiment(label) {{
  if(G_POSITIVE.has(label)) return 'Positive';
  if(G_NEGATIVE.has(label)) return 'Negative';
  if(label==='Unlabeled') return 'Background';
  return 'Positive';
}}
function getLabelTextColor(label) {{
  if(G_NEGATIVE.has(label)) return '#ff6b6b';  // 빨간색 (부정)
  if(label==='Unlabeled')       return '#888888';  // 회색 (Unlabeled)
  return '#ffffff';                              // 흰색 (긍정)
}}
// 배경색 밝기에 따라 검정/흰색 글자 자동 선택 (색칠된 구간 막대 위 가독성 확보)
function getContrastText(hex) {{
  if(!hex) return '#fff';
  const c = hex.replace('#','');
  if(c.length < 6) return '#fff';
  const r = parseInt(c.substr(0,2),16),
        g = parseInt(c.substr(2,2),16),
        b = parseInt(c.substr(4,2),16);
  // 상대 휘도 (0~1). 밝은 배경(노랑/주황 등)이면 검정 글자
  const lum = (0.299*r + 0.587*g + 0.114*b) / 255;
  return lum > 0.6 ? '#111' : '#fff';
}}

// ── ID 탭 + 동적 Tier 생성 ───────────────────────────────────────────────
function buildTiers() {{
  const tlInner = document.getElementById('tlInner');
  const labelsEl = document.getElementById('tierLabels');

  // 기존 tier 요소 제거 (ruler 제외)
  Array.from(tlInner.querySelectorAll('.tier-track')).forEach(e=>e.remove());
  labelsEl.innerHTML = '';

  // ruler용 빈 레이블 셀
  const lruler = document.createElement('div');
  lruler.style.cssText = 'height:22px;flex:none;background:#080808;border-bottom:1px solid var(--border);';
  labelsEl.appendChild(lruler);

  IDS.forEach((id, idIdx) => {{
    const prefix = HAS_ID ? id+'_' : '';

    // 다중 ID일 때 구분 헤더
    if (HAS_ID) {{
      const isActive = id === currentId;
      // label 헤더
      const lhdr = document.createElement('div');
      lhdr.style.cssText = `background:#0a0a0a;border-bottom:1px solid var(--border);${{idIdx>0?'border-top:2px solid #444;':''}}padding:0 8px;display:flex;align-items:center;height:18px;flex:none;`;
      lhdr.innerHTML = `<span style="font-size:9px;font-weight:700;color:${{isActive?'var(--accent)':'var(--text3)'}};text-transform:uppercase;letter-spacing:0.8px">${{id}}</span>`;
      labelsEl.appendChild(lhdr);
      // track 헤더
      const thdr = document.createElement('div');
      thdr.style.cssText = `background:#0a0a0a;border-bottom:1px solid var(--border);${{idIdx>0?'border-top:2px solid #444;':''}}height:18px;flex:none;`;
      tlInner.appendChild(thdr);
    }}

    // tier1
    const l1=document.createElement('div'); l1.className='tier-label-cell';
    l1.innerHTML='<div class="tlc-name">Model Pred.</div><div class="tlc-sub">읽기 전용</div>';
    labelsEl.appendChild(l1);
    const t1=document.createElement('div'); t1.className='tier-track'; t1.id=prefix+'tier1'; t1.style.flex='1';
    tlInner.appendChild(t1);

    // tier2
    const l2=document.createElement('div'); l2.className='tier-label-cell';
    l2.innerHTML='<div class="tlc-name">Reviewed</div><div class="tlc-sub" style="color:var(--green)">수정 가능</div>';
    labelsEl.appendChild(l2);
    const t2=document.createElement('div'); t2.className='tier-track'; t2.id=prefix+'tier2'; t2.style.flex='1.4';
    tlInner.appendChild(t2);

    // tier3
    const l3=document.createElement('div'); l3.className='tier-label-cell';
    l3.innerHTML='<div class="tlc-name">Confidence</div><div class="tlc-sub">참조용</div>';
    labelsEl.appendChild(l3);
    const t3=document.createElement('div'); t3.className='tier-track'; t3.id=prefix+'tier3'; t3.style.flex='1';
    tlInner.appendChild(t3);

    // tier4 (Audio) 제거 - 영상 아래 별도 트랙으로 이동
  }});
}}

// ── ID 탭 렌더링 ──────────────────────────────────────────────────────────
function buildIdTabs() {{
  const tabsEl = document.getElementById('idTabs');
  tabsEl.innerHTML = '';
  if (!HAS_ID) return;
  IDS.forEach(id => {{
    const btn = document.createElement('button');
    btn.textContent = id;
    btn.style.cssText = `padding:3px 10px;border-radius:4px;font-size:11px;font-weight:600;cursor:pointer;border:1px solid;transition:all 0.15s;`;
    const isActive = id === currentId;
    btn.style.background = isActive ? 'var(--accent)' : 'var(--bg3)';
    btn.style.borderColor = isActive ? 'var(--accent)' : 'var(--border2)';
    btn.style.color       = isActive ? '#000' : 'var(--text2)';
    btn.onclick = () => {{ currentId = id; buildIdTabs(); renderAll(); renderSegList(); }};
    tabsEl.appendChild(btn);
  }});
}}

// ── tier 요소 ID 헬퍼 ────────────────────────────────────────────────────
function tierId(n) {{
  return HAS_ID ? currentId+'_tier'+n : 'tier'+n;
}}

// ── 영상 줌/패닝 ─────────────────────────────────────────────────────────
let vidScale = 1, vidTX = 0, vidTY = 0;
let isPanning = false, panStartX = 0, panStartY = 0;

function applyVidTransform() {{
  vid.style.transform = `translate(${{vidTX}}px, ${{vidTY}}px) scale(${{vidScale}})`;
}}

function vidZoom(factor) {{
  vidScale = Math.min(Math.max(vidScale * factor, 0.3), 8);
  applyVidTransform();
}}

function vidFit() {{
  vidScale = 1; vidTX = 0; vidTY = 0;
  applyVidTransform();
}}

function vidFull() {{
  if(vid.requestFullscreen) vid.requestFullscreen();
}}

const videoWrap = document.getElementById('videoWrap');
if(videoWrap) {{
  // 마우스 휠 줌
  videoWrap.addEventListener('wheel', e => {{
    e.preventDefault();
    vidZoom(e.deltaY < 0 ? 1.1 : 0.9);
  }}, {{passive: false}});

  // 드래그 패닝
  videoWrap.addEventListener('mousedown', e => {{
    if(e.button !== 0) return;
    isPanning = true;
    panStartX = e.clientX - vidTX;
    panStartY = e.clientY - vidTY;
    videoWrap.classList.add('grabbing');
  }});
  window.addEventListener('mousemove', e => {{
    if(!isPanning) return;
    vidTX = e.clientX - panStartX;
    vidTY = e.clientY - panStartY;
    applyVidTransform();
  }});
  window.addEventListener('mouseup', () => {{
    isPanning = false;
    videoWrap.classList.remove('grabbing');
  }});

  // 더블클릭 fit
  videoWrap.addEventListener('dblclick', vidFit);
}}

// ── 영상 파일 선택 ───────────────────────────────────────────────────────
document.getElementById('videoPicker').addEventListener('change', e => {{
  const file = e.target.files[0];
  if (!file) return;
  vid.src = URL.createObjectURL(file);
  vid.style.display = 'block';
  document.getElementById('videoPickWrap').style.display = 'none';
  vid.load();
}});

// ── 배속 제어 ────────────────────────────────────────────────────────────
function setSpeed(s) {{
  vid.playbackRate = s;
  document.querySelectorAll('.speed-btn').forEach(b => {{
    b.classList.toggle('active', parseFloat(b.dataset.speed) === s);
  }});
}}
document.addEventListener('DOMContentLoaded', () => {{
  setSpeed(1);
  // 서버 분석 dB 타임라인/소리 구간이 있으면 바로 렌더링
  if (audioTimeline.length > 0 || audioEvents.length > 0) {{
    renderAudioTier();
    console.log('오디오 분석 로드 완료: 구간', audioEvents.length, '개, 타임라인', audioTimeline.length, '개');
  }}
}});

// ── Web Audio 소리 감지 ──────────────────────────────────────────────────
let audioCtx = null, analyser = null, audioSource = null;
let audioEvents   = {audio_events_js};    // 서버에서 미리 분석한 소리 구간
let audioTimeline = {audio_timeline_js};  // 200ms 단위 dB 타임라인 (train UI용)
let currentAudioIdx = -1;

const AUDIO_THRESHOLD_DB = {audio_threshold_js};  // 서버에서 적응형으로 계산된 임계값
const AUDIO_MERGE_GAP_MS = 1000;
const AUDIO_MIN_DUR_MS   = 300;

let _isLoud = false, _loudStart = 0, _rawEvents = [];
let _meterHistory = [];
const METER_HISTORY = 60;

function initAudio() {{
  if (!vid.src || audioCtx) return;
  try {{
    audioCtx    = new (window.AudioContext || window.webkitAudioContext)();
    analyser    = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    audioSource = audioCtx.createMediaElementSource(vid);
    audioSource.connect(analyser);
    analyser.connect(audioCtx.destination);
  }} catch(e) {{ console.warn('Web Audio 초기화 실패:', e); }}
}}

const _audioData = new Uint8Array(128);
function getVolumeDB() {{
  if (!analyser) return -100;
  analyser.getByteFrequencyData(_audioData);
  const avg = _audioData.reduce((a,b)=>a+b,0) / _audioData.length;
  return avg === 0 ? -100 : 20 * Math.log10(avg / 255);
}}

function mergeAndRenderAudio() {{
  if (_rawEvents.length === 0) return;
  const merged = [{{..._rawEvents[0]}}];
  for (let i = 1; i < _rawEvents.length; i++) {{
    const last = merged[merged.length-1];
    if (_rawEvents[i].start_ms - last.end_ms <= AUDIO_MERGE_GAP_MS)
      last.end_ms = _rawEvents[i].end_ms;
    else merged.push({{..._rawEvents[i]}});
  }}
  audioEvents = merged;
  renderAudioTier();
}}

// ── 기차(train) UI: 현재 위치 중앙 고정, dB 타임라인이 좌우로 흘러감 ──────
const TRAIN_WINDOW_MS = 12000;  // 화면에 보이는 좌우 범위 (현재 ±6초)

function dbToColor(db) {{
  // AUDIO_THRESHOLD_DB(배경 노이즈 기준)을 경계로 색상 구분
  if (db < AUDIO_THRESHOLD_DB - 6)  return '#2d8f5c';  // 조용함 (어두운 초록)
  if (db < AUDIO_THRESHOLD_DB)      return '#44dd88';  // 보통 (초록)
  if (db < AUDIO_THRESHOLD_DB + 8)  return '#ffcc00';  // 큼 (노랑)
  if (db < AUDIO_THRESHOLD_DB + 15) return '#ff6600';  // 매우 큼 (주황)
  return '#ff2222';                                     // 최대 (빨강)
}}

function renderAudioTier() {{
  drawAudioTrain();

  // 캔버스 클릭 → 클릭 위치의 시간으로 이동 + 해당 막대 구간 표시
  const wrap = document.getElementById('audioTrainWrap');
  if (wrap && !wrap._clickBound) {{
    wrap._clickBound = true;
    wrap.addEventListener('click', e => {{
      const rect = wrap.getBoundingClientRect();
      const relX = e.clientX - rect.left;       // 0 ~ width
      const centerX = rect.width / 2;
      const msPerPx = TRAIN_WINDOW_MS / rect.width;
      const msOffset = (relX - centerX) * msPerPx;
      const curMs = Math.round(vid.currentTime * 1000);
      const targetMs = Math.max(0, Math.min(TOTAL_MS, curMs + msOffset));

      // 클릭한 dB 막대(bucket)의 시작/끝 시간 계산
      const bucketMs = (audioTimeline.length > 1) ? (audioTimeline[1].t - audioTimeline[0].t) : 200;
      const bucketStart = Math.floor(targetMs / bucketMs) * bucketMs;
      const bucketEnd   = bucketStart + bucketMs;

      // 막대 구간 표시
      const rangeEl = document.getElementById('audioSelectedRange');
      if (rangeEl) {{
        rangeEl.textContent = msToTC(bucketStart) + ' ~ ' + msToTC(bucketEnd);
        rangeEl.style.display = 'inline-block';
      }}

      vid.currentTime = targetMs / 1000;

      // 클릭 위치 근처 소리 구간 있으면 해당 구간 선택 + 시작/종료 시간에 소리 구간 timestamp 채움
      const nearEv = audioEvents.find(ev => targetMs >= ev.start_ms - 500 && targetMs <= ev.end_ms + 500);
      if (nearEv) {{
        currentAudioIdx = audioEvents.indexOf(nearEv);
        vid.currentTime = nearEv.start_ms / 1000;
        const si = tier2Segs.findIndex(s => nearEv.start_ms >= s.start_ms && nearEv.start_ms < s.end_ms);
        if (si >= 0) selectSeg(si);
        // 소리 구간 timestamp를 시작/종료 시간 입력칸에 직접 채움
        startI.value = msToTC(nearEv.start_ms);
        endI.value   = msToTC(nearEv.end_ms);
        if (rangeEl) rangeEl.textContent = msToTC(nearEv.start_ms) + ' ~ ' + msToTC(nearEv.end_ms);
      }} else {{
        // 소리 구간이 아니면 클릭한 dB 막대 구간을 시작/종료 시간에 채움
        startI.value = msToTC(bucketStart);
        endI.value   = msToTC(bucketEnd);
      }}
    }});
  }}
}}

function drawAudioTrain() {{
  const canvas = document.getElementById('audioTrainCanvas');
  if (!canvas) return;
  const wrap = canvas.parentElement;
  const W = wrap.clientWidth, H = wrap.clientHeight;
  if (W === 0 || H === 0) return;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  const curMs = Math.round(vid.currentTime * 1000);
  const msPerPx = TRAIN_WINDOW_MS / W;

  // ── dB 타임라인 막대 (200ms 단위) ─────────────────────────────
  if (audioTimeline.length > 0) {{
    const bucketMs = audioTimeline[1] ? (audioTimeline[1].t - audioTimeline[0].t) : 200;
    const barPxW = Math.max(1, bucketMs / msPerPx);
    audioTimeline.forEach(pt => {{
      const dt = pt.t - curMs;                  // 현재 대비 시간차
      if (dt < -TRAIN_WINDOW_MS/2 || dt > TRAIN_WINDOW_MS/2) return;
      const x = W/2 + dt / msPerPx;
      const norm  = Math.max(0, Math.min(1, (pt.db + 60) / 60));
      const barH  = Math.max(2, norm * H);
      // 지나간 구간은 어둡게, 다가오는 구간은 밝게
      const past  = dt < 0;
      ctx.fillStyle = dbToColor(pt.db);
      ctx.globalAlpha = past ? 0.35 : 0.9;
      ctx.fillRect(x, H - barH, barPxW, barH);
    }});
    ctx.globalAlpha = 1;
  }}

  // ── 구간 범위(회색 직사각형 틀) ────────────────────────────────
  // 재생 중: 중앙바(현재 재생 위치)가 지나는 구간 / 정지 중: 선택한 구간
  let boxIdx;
  if (vid.paused) {{
    boxIdx = selectedIdx;
  }} else {{
    boxIdx = tier2Segs.findIndex(s => curMs >= s.start_ms && curMs <= s.end_ms);
  }}
  if (boxIdx >= 0 && tier2Segs[boxIdx]) {{
    const sel = tier2Segs[boxIdx];
    const dtStart = sel.start_ms - curMs;
    const dtEnd   = sel.end_ms   - curMs;
    if (!(dtEnd < -TRAIN_WINDOW_MS/2 || dtStart > TRAIN_WINDOW_MS/2)) {{
      const x1 = Math.max(0, W/2 + dtStart / msPerPx);
      const x2 = Math.min(W, W/2 + dtEnd / msPerPx);
      ctx.save();
      ctx.strokeStyle = '#cccccc';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(x1, 2, Math.max(1, x2 - x1), H - 4);
      ctx.restore();
    }}
  }}
}}

// 재생 위치 업데이트 → train UI 다시 그림
function updateAudioPlayhead() {{
  drawAudioTrain();
  updateAudioCenterLabel();
}}

// 현재 재생 위치(중앙선)가 지나는 dB 구간/소리 구간의 timestamp 표시
let _lastAudioLabelKey = null;
let _curAudioRangeStart = null, _curAudioRangeEnd = null;

function updateAudioCenterLabel() {{
  const rangeEl = document.getElementById('audioSelectedRange');
  if (!rangeEl) return;
  const curMs = Math.round(vid.currentTime * 1000);

  // 1) 감지된 소리 구간 위에 있으면 그 구간 표시
  const ev = audioEvents.find(e => curMs >= e.start_ms && curMs <= e.end_ms);
  if (ev) {{
    const key = 'ev_' + ev.start_ms;
    _curAudioRangeStart = ev.start_ms;
    _curAudioRangeEnd   = ev.end_ms;
    if (key !== _lastAudioLabelKey) {{
      rangeEl.textContent = msToTC(ev.start_ms) + ' ~ ' + msToTC(ev.end_ms);
      rangeEl.style.display = 'inline-block';
      rangeEl.style.color = '#ff9d3a';
      rangeEl.style.borderColor = '#ff9d3a';
      _lastAudioLabelKey = key;
    }}
    return;
  }}

  // 2) 일반 dB 막대(bucket) 구간 표시
  if (audioTimeline.length > 1) {{
    const bucketMs = audioTimeline[1].t - audioTimeline[0].t;
    const bucketStart = Math.floor(curMs / bucketMs) * bucketMs;
    const bucketEnd   = bucketStart + bucketMs;
    _curAudioRangeStart = bucketStart;
    _curAudioRangeEnd   = bucketEnd;
    const key = 'bk_' + bucketStart;
    if (key !== _lastAudioLabelKey) {{
      rangeEl.textContent = msToTC(bucketStart) + ' ~ ' + msToTC(bucketEnd);
      rangeEl.style.display = 'inline-block';
      rangeEl.style.color = '#fff';
      rangeEl.style.borderColor = '#444';
      _lastAudioLabelKey = key;
    }}
  }}
}}

// AUDIO 줄의 timestamp 표시를 클릭하면 시작/종료 시간 입력칸에 채움
function applyAudioRangeToInputs() {{
  if (_curAudioRangeStart === null) return;
  startI.value = msToTC(_curAudioRangeStart);
  endI.value   = msToTC(_curAudioRangeEnd);
  setFeedback(`⏱ 시간 입력됨: ${{msToTC(_curAudioRangeStart)}} ~ ${{msToTC(_curAudioRangeEnd)}}`, 'var(--blue)');
}}

function jumpNextAudio() {{
  const curMs = Math.round(vid.currentTime * 1000);
  const idx = tier2Segs.findIndex(s => s.label === 'Barking' && s.start_ms > curMs);
  if (idx >= 0) {{
    const next = tier2Segs[idx];
    vid.currentTime = next.start_ms / 1000;
    selectSeg(idx);
    setFeedback(`🔊 다음 Barking 구간 (${{msToTC(next.start_ms)}})`, '#ff6b9d');
  }} else {{
    setFeedback('더 이상 Barking 구간이 없습니다.', 'var(--text2)');
  }}
}}

// vid 로드 후 Audio 초기화
vid.addEventListener('play', () => {{ if (!audioCtx) initAudio(); }});

// requestAnimationFrame 기반 실시간 분석
let _rafId = null;
function _audioLoop() {{
  updateAudioDetection();
  _rafId = requestAnimationFrame(_audioLoop);
}}
vid.addEventListener('play',  () => {{ if (audioCtx && !_rafId) _rafId = requestAnimationFrame(_audioLoop); }});
vid.addEventListener('pause', () => {{ if (_rafId) {{ cancelAnimationFrame(_rafId); _rafId = null; }} updateAudioPlayhead(); }});
// 탐색(시점 이동) 시에도 train UI 다시 그림
vid.addEventListener('seeked',  updateAudioPlayhead);
vid.addEventListener('seeking', updateAudioPlayhead);
vid.addEventListener('loadedmetadata', updateAudioPlayhead);
vid.addEventListener('ended', () => {{ if (_rafId) {{ cancelAnimationFrame(_rafId); _rafId = null; }} }});

// OBS 스타일 레벨 미터
let _meterCanvas = null;
function ensureMeter() {{}}

function drawMeter(db) {{
  IDS.forEach(id => {{
    const prefix = HAS_ID ? id+'_' : '';
    const canvas = document.getElementById(prefix+'levelMeter');
    if (!canvas) return;
    const W = canvas.width, H = canvas.height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    // 배경
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, W, H);

    // dB → 0~1 정규화 (-60dB ~ 0dB)
    const norm = Math.max(0, Math.min(1, (db + 60) / 60));
    const fillW = norm * (W - 20);

    // 색상 그라디언트
    const grad = ctx.createLinearGradient(0, 0, W-20, 0);
    grad.addColorStop(0,    '#44dd88');
    grad.addColorStop(0.6,  '#ffcc00');
    grad.addColorStop(0.85, '#ff6600');
    grad.addColorStop(1.0,  '#ff2222');
    ctx.fillStyle = grad;
    ctx.fillRect(1, 3, fillW, H-6);

    // 격자선
    [0.25,0.5,0.75,0.9].forEach(p => {{
      ctx.fillStyle = 'rgba(0,0,0,0.4)';
      ctx.fillRect(p*(W-20), 0, 1, H);
    }});

    // dB 텍스트
    ctx.fillStyle = db > AUDIO_THRESHOLD_DB ? '#fff' : '#666';
    ctx.font = 'bold 9px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(db.toFixed(0)+'dB', W-2, H-3);
  }});
}}

function updateAudioDetection() {{
  if (!analyser) return;
  const db   = getVolumeDB();
  const ms   = Math.round(vid.currentTime * 1000);
  const loud = db > AUDIO_THRESHOLD_DB;

  drawMeter(db);

  if (loud && !_isLoud)  {{ _isLoud = true;  _loudStart = ms; }}
  else if (!loud && _isLoud) {{
    _isLoud = false;
    if (ms - _loudStart >= AUDIO_MIN_DUR_MS) {{
      _rawEvents.push({{start_ms: _loudStart, end_ms: ms}});
      mergeAndRenderAudio();
    }}
  }}
}}

function msToTC(ms){{
  ms=Math.max(0,Math.round(ms));
  const h=Math.floor(ms/3600000),m=Math.floor((ms%3600000)/60000),
        s=Math.floor((ms%60000)/1000),mss=ms%1000;
  return `${{p2(h)}}:${{p2(m)}}:${{p2(s)}}.${{p3(mss)}}`;
}}
function tcToMs(tc){{
  const m=tc.trim().match(/^(\d{{1,2}}):(\d{{1,2}}):(\d{{1,2}})\.(\d{{1,3}})$/);
  if(!m)return null;
  return parseInt(m[1])*3600000+parseInt(m[2])*60000+parseInt(m[3])*1000+parseInt(m[4].padEnd(3,'0'));
}}
function p2(n){{return String(n).padStart(2,'0');}}
function p3(n){{return String(n).padStart(3,'0');}}
function snapMs(ms,dir){{return dir==='floor'?Math.floor(ms/IMU_STEP)*IMU_STEP:Math.ceil(ms/IMU_STEP)*IMU_STEP;}}

// ── 자동 포맷팅 ───────────────────────────────────────────────────────────
function autoFormatTC(input){{
  input.addEventListener('input',e=>{{
    let v=e.target.value.replace(/[^0-9]/g,'');
    if(v.length>9)v=v.slice(0,9);
    let out='';
    if(v.length>6)out=v.slice(0,2)+':'+v.slice(2,4)+':'+v.slice(4,6)+'.'+v.slice(6);
    else if(v.length>4)out=v.slice(0,2)+':'+v.slice(2,4)+':'+v.slice(4);
    else if(v.length>2)out=v.slice(0,2)+':'+v.slice(2);
    else out=v;
    e.target.value=out;
  }});
  input.addEventListener('blur',e=>{{
    const ms=tcToMs(e.target.value);
    if(ms!==null)e.target.value=msToTC(ms);
    else if(e.target.value)e.target.classList.add('error');
  }});
  input.addEventListener('focus',e=>e.target.classList.remove('error'));
}}

// ── 줌 ───────────────────────────────────────────────────────────────────
function zoom(dir){{
  zoomIdx=Math.max(0,Math.min(ZOOM_LEVELS.length-1,zoomIdx+dir));
  zoomLevel=ZOOM_LEVELS[zoomIdx];
  document.getElementById('zoomLabel').textContent=zoomLevel+'×';
  applyZoom();
}}
function resetZoom(){{zoomIdx=2;zoomLevel=1;document.getElementById('zoomLabel').textContent='1×';applyZoom();}}
function applyZoom(){{
  const inner=document.getElementById('tlInner');
  inner.style.minWidth=(zoomLevel*100)+'%';
  renderRuler();renderAll();
  const scroll=document.getElementById('tlScroll');
  const pct=vid.currentTime/TOTAL_SEC;
  scroll.scrollLeft=Math.max(0,pct*inner.offsetWidth-scroll.offsetWidth/2);
}}

// ── 눈금 ─────────────────────────────────────────────────────────────────
function renderRuler(){{
  const ruler=document.getElementById('ruler');ruler.innerHTML='';
  const visMs=TOTAL_MS/zoomLevel;
  const step=visMs<=10000?500:visMs<=30000?1000:visMs<=120000?5000:10000;
  const majorStep=step*5;
  for(let t=0;t<=TOTAL_MS;t+=step){{
    const d=document.createElement('div');
    const isMajor=(t%majorStep===0);
    d.className='r-tick'+(isMajor?' major':'');
    d.style.left=(t/TOTAL_MS*100)+'%';
    if(isMajor||step>=1000){{
      const lbl=document.createElement('div');lbl.className='r-tick-label';lbl.textContent=msToTC(t);d.appendChild(lbl);
    }}
    ruler.appendChild(d);
  }}
}}

// ── 플레이헤드 ───────────────────────────────────────────────────────────
function updatePlayheads(){{
  const pct=(vid.currentTime*1000/TOTAL_MS*100)+'%';
  IDS.forEach(id=>{{
    [1,2,3].forEach(n=>{{
      const tId = HAS_ID ? id+'_tier'+n : 'tier'+n;
      const el=document.getElementById(tId);
      if(!el)return;
      let ph=document.getElementById('ph_'+tId);
      if(!ph){{ph=document.createElement('div');ph.className='playhead';ph.id='ph_'+tId;ph.innerHTML='<div class="ph-head"></div>';el.appendChild(ph);}}
      ph.style.left=pct;
    }});
  }});
  // Audio 플레이헤드
  updateAudioPlayhead();
  updateAudioDetection();
  tcDisp.textContent=msToTC(Math.round(vid.currentTime*1000));
  const ms=Math.round(vid.currentTime*1000);
  const idx=tier2Segs.findIndex(s=>ms>=s.start_ms&&ms<=s.end_ms);
  if(idx!==lastSegIdx){{
    document.getElementById('srow_'+lastSegIdx)?.classList.remove('playing');
    lastSegIdx=idx;
    if(idx>=0){{
      const s=tier2Segs[idx];
      document.getElementById('nowLabel').textContent=s.label;
      document.getElementById('nowLabel').style.color='#fff';
      document.getElementById('nowConf').textContent='conf: '+s.conf.toFixed(3)+(s.low_conf?' ⚠':'');
      document.getElementById('nowLabelBar').style.background=COLORS[s.label]||'#161616';
      const row=document.getElementById('srow_'+idx);
      if(row){{row.classList.add('playing');row.scrollIntoView({{block:'nearest'}});}}
    }} else {{
      document.getElementById('nowLabel').textContent='—';
      document.getElementById('nowConf').textContent='';
      document.getElementById('nowLabelBar').style.background='#161616';
    }}
  }}
  if(!vid.paused){{
    const scroll=document.getElementById('tlScroll');
    const inner=document.getElementById('tlInner');
    const phX=(vid.currentTime/TOTAL_SEC)*inner.offsetWidth;
    const visW=scroll.offsetWidth;
    if(phX>scroll.scrollLeft+visW*0.8)scroll.scrollLeft=phX-visW*0.3;
  }}
}}

// ── Tier 렌더링 ───────────────────────────────────────────────────────────
function renderTier1(){{
  const el=document.getElementById(tierId(1));
  if(!el)return;
  Array.from(el.querySelectorAll('.ann')).forEach(e=>e.remove());
  state[currentId].tier1Segs.forEach(s=>{{
    const d=document.createElement('div');
    d.className='ann'+(s.low_conf?' low-conf':'');
    d.style.left=(s.start_ms/TOTAL_MS*100)+'%';
    d.style.width=Math.max(0.1,(s.end_ms-s.start_ms)/TOTAL_MS*100)+'%';
    d.style.background=(COLORS[s.label]||'#999')+'99';
    d.style.border='none';
    d.innerHTML=`<span class="ann-text" style="color:${{getContrastText(COLORS[s.label]||'#999')}}">${{s.label}}</span>`;
    addTooltip(d,s);el.appendChild(d);
  }});
}}

function renderTier2(){{
  const el=document.getElementById(tierId(2));
  if(!el)return;
  Array.from(el.querySelectorAll('.ann')).forEach(e=>e.remove());
  tier2Segs.forEach((s,i)=>{{
    const d=document.createElement('div');
    let cls='ann'+(s.low_conf?' low-conf':'')+(s.modified?' modified':'')+(i===selectedIdx?' selected':'');
    d.className=cls;
    d.style.left=(s.start_ms/TOTAL_MS*100)+'%';
    d.style.width=Math.max(0.1,(s.end_ms-s.start_ms)/TOTAL_MS*100)+'%';
    d.style.background=COLORS[s.label]||'#999';
    d.innerHTML=`<span class="ann-text" style="color:${{getContrastText(COLORS[s.label]||'#999')}}">${{s.label}}</span>`;
    d.addEventListener('click',e=>{{
      const inner=document.getElementById('tlInner');
      const rect=inner.getBoundingClientRect();
      const clickMs=(e.clientX-rect.left)/inner.offsetWidth*TOTAL_MS;
      vid.currentTime=Math.max(s.start_ms,Math.min(clickMs,s.end_ms))/1000;
      selectSeg(i);
    }});
    addTooltip(d,s);el.appendChild(d);
  }});
}}

function renderTier3(){{
  const el=document.getElementById(tierId(3));
  if(!el)return;
  Array.from(el.querySelectorAll('.conf-blk')).forEach(e=>e.remove());
  tier2Segs.forEach((s,i)=>{{
    const d=document.createElement('div');d.className='conf-blk';
    d.style.left=(s.start_ms/TOTAL_MS*100)+'%';
    d.style.width=Math.max(0.1,(s.end_ms-s.start_ms)/TOTAL_MS*100)+'%';
    const a=Math.max(0.15,s.conf);
    d.style.background=s.low_conf?`rgba(226,75,74,${{a}})`:`rgba(29,158,117,${{a}})`;
    d.style.cursor='pointer';
    d.style.pointerEvents='auto';
    if((s.end_ms-s.start_ms)/TOTAL_MS*zoomLevel>0.04){{
      const sp=document.createElement('span');sp.className='conf-txt';sp.textContent=s.conf.toFixed(2);d.appendChild(sp);
    }}
    // ── Confidence 구간 클릭 → 입력창 자동 채움 + 영상 이동 ──────────
    d.addEventListener('click',e=>{{
      e.stopPropagation();
      const inner=document.getElementById('tlInner');
      const rect=inner.getBoundingClientRect();
      const clickMs=(e.clientX-rect.left)/inner.offsetWidth*TOTAL_MS;
      vid.currentTime=Math.max(s.start_ms,Math.min(clickMs,s.end_ms))/1000;
      startI.value=msToTC(s.start_ms);
      endI.value=msToTC(s.end_ms);
      labelS.value=s.label;
      selectSeg(i);
      setFeedback(
        `${{s.low_conf?'⚠ 낮은신뢰도':'✓'}} ${{s.label}} | ${{msToTC(s.start_ms)}}~${{msToTC(s.end_ms)}} | conf:${{s.conf}}`,
        s.low_conf?'var(--red)':COLORS[s.label]||'var(--green)'
      );
    }});
    addTooltip(d,s);
    el.appendChild(d);
  }});
  el.onclick=e=>{{
    if(e.target!==el)return;
    const inner=document.getElementById('tlInner');
    vid.currentTime=((e.clientX-inner.getBoundingClientRect().left)/inner.offsetWidth)*TOTAL_SEC;
  }};
}}

function renderAll(){{renderTier1();renderTier2();renderTier3();}}

function addTooltip(el,s){{
  el.addEventListener('mouseenter',e=>{{
    tooltip.classList.add('show');
    document.getElementById('ttLabel').textContent=s.label;
    document.getElementById('ttLabel').style.color=COLORS[s.label]||'#fff';
    document.getElementById('ttTime').textContent=msToTC(s.start_ms)+' → '+msToTC(s.end_ms);
    document.getElementById('ttConf').textContent='confidence: '+s.conf.toFixed(3)+(s.low_conf?' ⚠ 낮음':'');
    posTooltip(e);
  }});
  el.addEventListener('mousemove',posTooltip);
  el.addEventListener('mouseleave',()=>tooltip.classList.remove('show'));
}}
function posTooltip(e){{tooltip.style.left=(e.clientX+12)+'px';tooltip.style.top=(e.clientY-10)+'px';}}

// ── 구간 선택 ────────────────────────────────────────────────────────────
function selectSeg(idx){{
  if(selectedIdx>=0)document.getElementById('srow_'+selectedIdx)?.classList.remove('active');
  selectedIdx=idx;
  const s=tier2Segs[idx];
  startI.value=msToTC(s.start_ms);endI.value=msToTC(s.end_ms);labelS.value=s.label;
  setFeedback(`선택: ${{s.label}} | ${{msToTC(s.start_ms)}} ~ ${{msToTC(s.end_ms)}} | conf:${{s.conf}}`,'#888');
  document.getElementById('srow_'+idx)?.classList.add('active');
  document.getElementById('srow_'+idx)?.scrollIntoView({{block:'nearest'}});
  renderTier2();
  drawAudioTrain();   // 선택 구간 범위를 오디오 영역에 회색 직사각형으로 즉시 표시
}}

// ── 구간 목록 ────────────────────────────────────────────────────────────
function renderSegList(){{
  const wrap=document.getElementById('segListWrap');wrap.innerHTML='';
  tier2Segs.forEach((s,i)=>{{
    const item=document.createElement('div');
    let cls='seg-item'+(s.low_conf?' low-conf-item':'')+(s.modified?' modified-item':'')+(i===selectedIdx?' active':'')+(i===lastSegIdx?' playing':'');
    item.className=cls;item.id='srow_'+i;
    const confColor=s.low_conf?'var(--red)':s.conf>0.85?'var(--green)':'#888';
    item.innerHTML=`
      <div class="seg-dot" style="background:${{COLORS[s.label]||'#999'}}"></div>
      <div class="seg-info">
        <div class="seg-name" style="color:${{getLabelTextColor(s.label)}}">${{s.label}}</div>
        <div class="seg-time">${{msToTC(s.start_ms)}} ~ ${{msToTC(s.end_ms)}}</div>
      </div>
      <div class="seg-conf-badge" style="color:${{confColor}};background:${{confColor}}22">${{s.conf.toFixed(2)}}</div>
    `;
    item.addEventListener('click',()=>{{vid.currentTime=s.start_ms/1000;selectSeg(i);}});
    wrap.appendChild(item);
  }});
}}

// ── 진행률 ───────────────────────────────────────────────────────────────
function updateProgress(){{
  const total    = tier2Segs.length;
  const modified = tier2Segs.filter(s=>s.modified).length;
  const lowRemain= tier2Segs.filter(s=>s.low_conf).length;
  const pct      = total>0 ? Math.round(modified/total*100) : 0;
  document.getElementById('progressFill').style.width=pct+'%';
  document.getElementById('progressPct').textContent=pct+'%';
  document.getElementById('progressLabel').textContent=
    `수정됨 ${{modified}}/${{total}}구간 | 낮은신뢰도 ${{lowRemain}}개 남음`;
}}

// ── 수정 카운터 ──────────────────────────────────────────────────────────
function updateModCounter(){{
  const cnt=tier2Segs.filter(s=>s.modified).length;
  modBadge.textContent=cnt;
  modBadge.className='mod-badge'+(cnt===0?' zero':'');
  updateProgress();
}}

// ── 구간 수정 ────────────────────────────────────────────────────────────
function applyEdit(){{
  const s0=tcToMs(startI.value),e0=tcToMs(endI.value),newLabel=labelS.value;
  if(s0===null||e0===null){{setFeedback('시간 형식 오류: 00:00:00.000','var(--red)');startI.classList.add('error');endI.classList.add('error');return;}}
  startI.classList.remove('error');endI.classList.remove('error');
  const startMs=snapMs(s0,'floor'),endMs=snapMs(e0,'ceil');
  if(startMs>=endMs){{setFeedback('종료시간 > 시작시간 이어야 합니다','var(--red)');return;}}
  if(endMs>TOTAL_MS){{setFeedback('종료시간이 전체 길이를 초과합니다','var(--red)');return;}}
  history.push(JSON.parse(JSON.stringify(tier2Segs)));if(history.length>50)history.shift();
  const result=[];let inserted=false;
  for(const s of tier2Segs){{
    if(s.end_ms<=startMs){{result.push(s);continue;}}
    if(s.start_ms>=endMs){{if(!inserted){{result.push(makeNewSeg(startMs,endMs,newLabel));inserted=true;}}result.push(s);continue;}}
    if(s.start_ms<startMs)result.push({{...s,end_ms:startMs,end_idx:findIdx(startMs-IMU_STEP)}});
    if(!inserted){{result.push(makeNewSeg(startMs,endMs,newLabel));inserted=true;}}
    if(s.end_ms>endMs)result.push({{...s,start_ms:endMs,start_idx:findIdx(endMs)}});
  }}
  if(!inserted)result.push(makeNewSeg(startMs,endMs,newLabel));
  tier2Segs=result.sort((a,b)=>a.start_ms-b.start_ms);
  ID_ROWS[currentId].forEach(r=>{{if(r.time_ms>=startMs&&r.time_ms<endMs)r.pred_label=newLabel;}});
  const snapped=s0!==startMs||e0!==endMs;
  setFeedback(`✓ ${{newLabel}} | ${{msToTC(startMs)}} ~ ${{msToTC(endMs)}}${{snapped?' [스냅]':''}}`,COLORS[newLabel]||'var(--green)');
  renderAll();renderSegList();updateModCounter();

  // ── 적용 후 자동 다음 구간으로 이동 ──────────────────────────────────
  // 종료시간 → 다음 구간 시작시간으로 자동 설정
  const newIdx=tier2Segs.findIndex(s=>s.start_ms===startMs&&s.end_ms===endMs);
  const nextIdx=newIdx+1;
  if(nextIdx<tier2Segs.length){{
    const next=tier2Segs[nextIdx];
    // 시작시간 = 현재 종료시간, 종료시간 = 다음 구간 종료시간
    startI.value=msToTC(endMs);
    endI.value=msToTC(next.end_ms);
    // 다음 구간의 모델 예측 레이블 자동 세팅
    labelS.value=next.label;
    vid.currentTime=endMs/1000;
    setFeedback(
      `✓ 적용됨 → 다음구간: ${{next.label}} (${{msToTC(endMs)}}~${{msToTC(next.end_ms)}}) conf:${{next.conf}} — 맞으면 Enter, 틀리면 레이블 바꿔서 Enter`,
      COLORS[next.label]||'var(--green)'
    );
    selectSeg(nextIdx);
  }} else {{
    setFeedback('✓ 마지막 구간 적용 완료!','var(--green)');
    if(newIdx>=0)selectSeg(newIdx);
  }}
}}

// ── 구간 분할 ────────────────────────────────────────────────────────────
function splitAtCurrent(){{
  const curMs=snapMs(Math.round(vid.currentTime*1000),'floor');
  const idx=tier2Segs.findIndex(s=>curMs>s.start_ms&&curMs<s.end_ms);
  if(idx<0){{setFeedback('분할할 구간이 없습니다. 구간 중간에서 실행하세요.','#888');return;}}
  history.push(JSON.parse(JSON.stringify(tier2Segs)));if(history.length>50)history.shift();
  const s=tier2Segs[idx];
  const left ={{...s,end_ms:curMs,end_idx:findIdx(curMs-IMU_STEP),modified:true}};
  const right={{...s,start_ms:curMs,start_idx:findIdx(curMs),modified:true}};
  tier2Segs.splice(idx,1,left,right);
  ID_ROWS[currentId].forEach(r=>{{if(r.time_ms>=curMs&&r.time_ms<s.end_ms)r.pred_label=right.label;}});
  renderAll();renderSegList();updateModCounter();
  setFeedback(`✂ 분할 완료: ${{msToTC(curMs)}}에서 나눔`,COLORS[s.label]||'var(--green)');
  selectSeg(idx);
}}

// ── 구간 병합 ────────────────────────────────────────────────────────────
function mergeSelected(){{
  if(selectedIdx<0){{setFeedback('병합할 구간을 먼저 선택하세요.','#888');return;}}
  const label=tier2Segs[selectedIdx].label;
  // 선택 구간을 기준으로 좌우로 연속된 같은 레이블 구간을 모두 포함 (개수 무관)
  let lo=selectedIdx, hi=selectedIdx;
  while(lo>0 && tier2Segs[lo-1].label===label) lo--;
  while(hi<tier2Segs.length-1 && tier2Segs[hi+1].label===label) hi++;
  if(lo===hi){{setFeedback('인접한 같은 레이블 구간이 없습니다.','#888');return;}}
  history.push(JSON.parse(JSON.stringify(tier2Segs)));if(history.length>50)history.shift();
  const run=tier2Segs.slice(lo,hi+1);
  const first=run[0], last=run[run.length-1];
  const avgConf=run.reduce((s,x)=>s+x.conf,0)/run.length;
  const merged={{...first,end_ms:last.end_ms,end_idx:last.end_idx,modified:true,
    conf:Math.round(avgConf*1000)/1000,low_conf:run.some(x=>x.low_conf)}};
  tier2Segs.splice(lo,run.length,merged);
  renderAll();renderSegList();updateModCounter();
  setFeedback(`⊕ ${{run.length}}개 구간 병합 완료: ${{label}} ${{msToTC(first.start_ms)}}~${{msToTC(last.end_ms)}}`,COLORS[label]||'var(--green)');
  selectSeg(lo);
}}

function makeNewSeg(startMs,endMs,label){{
  const rows=ID_ROWS[currentId].filter(r=>r.time_ms>=startMs&&r.time_ms<endMs);
  const avg=rows.length?rows.reduce((a,b)=>a+(b.confidence||1),0)/rows.length:1;
  return{{label,start_ms:startMs,end_ms:endMs,conf:Math.round(avg*1000)/1000,
          low_conf:avg<CONF_THRESH,start_idx:findIdx(startMs),end_idx:findIdx(endMs-IMU_STEP),modified:true}};
}}
function findIdx(ms){{
  let lo=0,hi=ID_ROWS[currentId].length-1;
  while(lo<hi){{const mid=Math.floor((lo+hi)/2);if(ID_ROWS[currentId][mid].time_ms<ms)lo=mid+1;else hi=mid;}}
  return lo;
}}

function fillFromVideo(){{startI.value=msToTC(snapMs(Math.round(vid.currentTime*1000),'floor'));endI.focus();}}
function undoLast(){{
  if(!history.length){{setFeedback('되돌릴 내용 없음','#888');return;}}
  tier2Segs=history.pop();
  tier2Segs.forEach(s=>{{ID_ROWS[currentId].forEach(r=>{{if(r.time_ms>=s.start_ms&&r.time_ms<s.end_ms)r.pred_label=s.label;}});}});
  renderAll();renderSegList();updateModCounter();setFeedback('↩ 되돌렸습니다','var(--yellow)');
}}
function jumpLowConf(){{
  const cur=vid.currentTime*1000;
  const next=tier2Segs.find(s=>s.low_conf&&s.start_ms>cur+100);
  if(next){{vid.currentTime=next.start_ms/1000;selectSeg(tier2Segs.indexOf(next));setFeedback(`⚠ ${{next.label}} conf:${{next.conf}}`,'var(--red)');}}
  else setFeedback('검수 필요 구간 없음 ✓','var(--green)');
}}
function setFeedback(msg,color){{feedbk.textContent=msg;feedbk.style.color=color;}}

// ── 도움말 모달 ──────────────────────────────────────────────────────────
function showHelp(){{document.getElementById('helpOverlay').classList.add('show');}}
function hideHelp(){{document.getElementById('helpOverlay').classList.remove('show');}}

// ── 저장 모달 ────────────────────────────────────────────────────────────
function hideSaveModal(){{document.getElementById('saveOverlay').classList.remove('show');}}

// ── CSV + 통계 저장 ───────────────────────────────────────────────────────
function downloadCSV(){{
  // 저장 모달 먼저 띄우기 (파일명 입력)
  const inp = document.getElementById('saveFileNameInp');
  inp.value = OUT_NAME.replace('.csv','');
  document.getElementById('saveDesc').textContent = '저장할 파일명을 확인하거나 변경하세요.';
  document.getElementById('saveOverlay').classList.add('show');
  setTimeout(()=>{{inp.focus();inp.select();}}, 100);
}}

function confirmSave(){{
  // 파일명 확정 후 실제 저장
  let fname = document.getElementById('saveFileNameInp').value.trim();
  if(!fname) fname = OUT_NAME.replace('.csv','');
  // 확장자 중복 방지
  if(fname.endsWith('.csv')) fname = fname.slice(0,-4);
  const finalName = fname + '.csv';
  const statName  = fname + '_stats.json';

  // tier2 → ID_ROWS 동기화 + sentiment 업데이트
  IDS.forEach(id=>{{
    state[id].tier2Segs.forEach(s=>{{
      ID_ROWS[id].forEach(r=>{{
        if(r.time_ms>=s.start_ms&&r.time_ms<s.end_ms){{
          r.pred_label=s.label;
          r.sentiment=getSentiment(s.label);
        }}
      }});
    }});
  }});
  const allRows=[];
  IDS.forEach(id=>{{
    ID_ROWS[id].forEach(r=>{{const row={{...r}};if(HAS_ID)row.id=id;allRows.push(row);}});
  }});
  allRows.sort((a,b)=>a.timestamp-b.timestamp||(HAS_ID?String(a.id).localeCompare(String(b.id)):0));

  const lines=[SAVE_COLS.join(',')];
  allRows.forEach(r=>lines.push(SAVE_COLS.map(c=>r[c]!==undefined?r[c]:'').join(',')));
  const blob=new Blob([lines.join('\\n')],{{type:'text/csv'}});
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a');a.href=url;a.download=finalName;a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);

  // 통계 저장
  const elapsed=Math.round((Date.now()-START_TIME)/1000);
  const modCount=IDS.reduce((acc,id)=>acc+state[id].tier2Segs.filter(s=>s.modified).length,0);
  const stats={{file:finalName,review_time_sec:elapsed,
    review_time_min:Math.round(elapsed/60*10)/10,
    ids:IDS,modified_segments:modCount,timestamp:new Date().toISOString()}};
  const sa=document.createElement('a');
  sa.href=URL.createObjectURL(new Blob([JSON.stringify(stats,null,2)],{{type:'application/json'}}));
  sa.download=statName;sa.click();

  hideSaveModal();
  setFeedback(`✓ 저장 완료: ${{finalName}}`,'var(--green)');
}}

// ── 키보드 ───────────────────────────────────────────────────────────────
document.addEventListener('keydown',e=>{{
  const tag=e.target.tagName;
  if(tag==='INPUT'||tag==='SELECT'){{if(e.key==='Enter'){{applyEdit();e.preventDefault();}}return;}}
  if(e.key==='Escape'){{hideHelp();hideSaveModal();return;}}
  // 저장 모달 열려있을 때 Enter → confirmSave
  if(e.key==='Enter'&&document.getElementById('saveOverlay').classList.contains('show')){{
    confirmSave();e.preventDefault();return;
  }}
  if(SHORTCUTS[e.key]){{labelS.value=SHORTCUTS[e.key];return;}}
  if(e.key===' '){{vid.paused?vid.play():vid.pause();e.preventDefault();}}
  if(e.key==='ArrowRight'){{vid.currentTime+=e.shiftKey?0.04:1;e.preventDefault();}}
  if(e.key==='ArrowLeft'){{vid.currentTime-=e.shiftKey?0.04:1;e.preventDefault();}}
  if(e.key==='z'&&(e.ctrlKey||e.metaKey)){{undoLast();e.preventDefault();}}
  if(e.key==='+'||e.key==='=')zoom(1);
  if(e.key==='-')zoom(-1);
}});

vid.addEventListener('timeupdate',updatePlayheads);

// ── CSV 불러오기 (백업 복구) ─────────────────────────────────────────────
function loadCsvBackup(input) {{
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {{
    try {{
      const lines = e.target.result.trim().split('\\n');
      const headers = lines[0].split(',');
      const labelIdx = headers.indexOf('pred_label');
      const startIdx = headers.indexOf('timestamp');
      if (labelIdx < 0) {{ setFeedback('❌ pred_label 컬럼을 찾을 수 없습니다', 'var(--red)'); return; }}

      // CSV rows → timestamp별 레이블 맵
      const labelMap = {{}};
      for (let i = 1; i < lines.length; i++) {{
        const cols = lines[i].split(',');
        if (cols.length < headers.length) continue;
        const ts = parseFloat(cols[startIdx]);
        const label = cols[labelIdx];
        if (!isNaN(ts) && label) labelMap[ts] = label;
      }}

      // tier2Segs 레이블 업데이트
      let updated = 0;
      tier2Segs.forEach(s => {{
        const rows = ID_ROWS[currentId].filter(r => r.time_ms >= s.start_ms && r.time_ms < s.end_ms);
        if (rows.length === 0) return;
        const ts = rows[0].timestamp;
        if (labelMap[ts] !== undefined && labelMap[ts] !== s.label) {{
          s.label = labelMap[ts];
          s.modified = true;
          updated++;
        }}
      }});

      renderAll(); renderSegList(); updateProgress();
      setFeedback(`✓ CSV 불러오기 완료 — ${{updated}}개 구간 복구됨`, 'var(--green)');
      input.value = '';
    }} catch(err) {{
      setFeedback('❌ CSV 파싱 오류: ' + err.message, 'var(--red)');
    }}
  }};
  reader.readAsText(file);
}}

// ── 현재 검수 상태 → reviewed CSV 문자열 ─────────────────────────────────
function buildReviewedCsv() {{
  // tier2Segs(수정본) → ID_ROWS 동기화
  IDS.forEach(id => {{
    state[id].tier2Segs.forEach(s => {{
      ID_ROWS[id].forEach(r => {{
        if (r.time_ms >= s.start_ms && r.time_ms < s.end_ms) {{
          r.pred_label = s.label;
          r.sentiment  = getSentiment(s.label);
        }}
      }});
    }});
  }});
  const allRows = [];
  IDS.forEach(id => {{
    ID_ROWS[id].forEach(r => {{ const row = {{...r}}; if(HAS_ID) row.id = id; allRows.push(row); }});
  }});
  allRows.sort((a,b) => a.timestamp - b.timestamp);
  const lines = [SAVE_COLS.join(',')];
  allRows.forEach(r => lines.push(SAVE_COLS.map(c => r[c] !== undefined ? r[c] : '').join(',')));
  return lines.join('\\n');
}}

// ── 저장된 reviewed CSV → 현재 상태 복원 ─────────────────────────────────
function rebuildSegsFromRows(id) {{
  const rows = ID_ROWS[id];
  const origSegs = ID_SEGS_ORIG[id] || [];
  function origLabelAt(ms) {{
    for (const s of origSegs) {{ if (ms >= s.start_ms && ms <= s.end_ms) return s.label; }}
    return null;
  }}
  const segs = [];
  let i = 0;
  while (i < rows.length) {{
    let j = i + 1;
    const cc = [rows[i].confidence != null ? rows[i].confidence : 1];
    let mod = rows[i].pred_label !== origLabelAt(rows[i].time_ms);
    while (j < rows.length && rows[j].pred_label === rows[i].pred_label) {{
      cc.push(rows[j].confidence != null ? rows[j].confidence : 1);
      if (rows[j].pred_label !== origLabelAt(rows[j].time_ms)) mod = true;
      j++;
    }}
    const avg = cc.reduce((a,b)=>a+b,0) / cc.length;
    segs.push({{
      label: rows[i].pred_label, start_ms: rows[i].time_ms, end_ms: rows[j-1].time_ms,
      conf: Math.round(avg*1000)/1000, low_conf: avg < CONF_THRESH,
      start_idx: i, end_idx: j-1, modified: mod
    }});
    i = j;
  }}
  return segs;
}}

function applyReviewedCsv(text) {{
  const lines = text.trim().split('\\n');
  if (lines.length < 2) return 0;
  const headers = lines[0].split(',');
  const tsIdx = headers.indexOf('timestamp');
  const labelIdx = headers.indexOf('pred_label');
  const idIdx = headers.indexOf('id');
  if (tsIdx < 0 || labelIdx < 0) return 0;
  const map = {{}};
  for (let i = 1; i < lines.length; i++) {{
    const c = lines[i].split(',');
    if (c.length < headers.length) continue;
    const id = (HAS_ID && idIdx >= 0) ? c[idIdx] : 'default';
    map[id + '|' + c[tsIdx]] = c[labelIdx];
  }}
  let changed = 0;
  IDS.forEach(id => {{
    ID_ROWS[id].forEach(r => {{
      const lbl = map[id + '|' + String(r.timestamp)];
      if (lbl !== undefined && lbl !== '') {{
        if (r.pred_label !== lbl) changed++;
        r.pred_label = lbl;
        r.sentiment = getSentiment(lbl);
      }}
    }});
    state[id].tier2Segs = rebuildSegsFromRows(id);
    state[id].history = [];
  }});
  return changed;
}}

// 서버에 저장된 reviewed CSV가 있으면 자동 복원 (CLI 방식은 fetch 실패 → 무시)
function autoRestore() {{
  const base = OUT_NAME.replace('_labeled_reviewed.csv', '');
  fetch('/load/' + base)
    .then(r => r.json())
    .then(d => {{
      if (!d || !d.exists || !d.csv) return;
      const changed = applyReviewedCsv(d.csv);
      renderAll(); renderSegList(); updateModCounter(); updateProgress();
      const when = d.updated ? new Date(d.updated * 1000).toLocaleString('ko-KR') : '';
      setFeedback(`↩ 이전 검수 내역 복원됨 (${{changed}}개 구간 수정) ${{when}}`, 'var(--green)');
    }})
    .catch(() => {{}});  // 서버 없으면 무시 (CLI 방식)
}}

// ── 자동저장 (30초마다) ──────────────────────────────────────────────────
function autoSave() {{
  const csv = buildReviewedCsv();
  // 서버에 autosave 요청 (서버 방식일 때만)
  fetch('/autosave/' + OUT_NAME.replace('_labeled_reviewed.csv',''), {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{csv}})
  }}).then(() => {{
    const t = new Date().toLocaleTimeString('ko-KR', {{hour:'2-digit',minute:'2-digit',second:'2-digit'}});
    document.getElementById('autosaveStatus').textContent = '저장 ' + t;
    document.getElementById('autosaveStatus').style.color = 'var(--green)';
    setTimeout(() => {{
      document.getElementById('autosaveStatus').style.color = 'var(--text3)';
    }}, 3000);
  }}).catch(() => {{
    // 서버 없으면 무시 (CLI 방식)
  }});
}}

// 자동저장 30초마다 실행
setInterval(autoSave, 30000);

// 페이지 닫기/새로고침 직전 마지막 저장 보장 (30초 주기 사이 유실 방지)
window.addEventListener('beforeunload', () => {{
  try {{
    const csv = buildReviewedCsv();
    const base = OUT_NAME.replace('_labeled_reviewed.csv', '');
    const blob = new Blob([JSON.stringify({{csv}})], {{type: 'application/json'}});
    navigator.sendBeacon('/autosave/' + base, blob);
  }} catch(e) {{}}
}});

autoFormatTC(startI);autoFormatTC(endI);
buildTiers();
buildIdTabs();
renderRuler();renderAll();renderSegList();updateProgress();applyZoom();
autoRestore();  // 서버에 저장된 이전 검수 내역이 있으면 복원
window.addEventListener('resize',()=>{{renderRuler();renderAll();drawAudioTrain();}});

// ── 타임라인 드래그 스크롤 ────────────────────────────────────────────────
// 타임라인을 마우스로 끌어 좌우 이동. 드래그한 경우 클릭(이동/선택)은 무시.
(function(){{
  const tl = document.getElementById('tlScroll');
  if (!tl) return;
  let isDragging = false, startX = 0, scrollLeft = 0, moved = false;
  tl.style.cursor = 'grab';
  tl.addEventListener('mousedown', e => {{
    if (e.button !== 0) return;
    isDragging = true;
    moved = false;
    startX = e.pageX - tl.offsetLeft;
    scrollLeft = tl.scrollLeft;
    tl.style.cursor = 'grabbing';
  }});
  window.addEventListener('mouseup', () => {{
    isDragging = false;
    tl.style.cursor = 'grab';
  }});
  tl.addEventListener('mousemove', e => {{
    if (!isDragging) return;
    const x = e.pageX - tl.offsetLeft;
    const walk = (x - startX) * 1.5;
    if (Math.abs(walk) > 3) {{
      moved = true;
      e.stopPropagation();
      tl.scrollLeft = scrollLeft - walk;
    }}
  }});
  // 드래그 중이면 클릭 이벤트 차단 (capture 단계에서 가로채 seek/select 방지)
  tl.addEventListener('click', e => {{
    if (moved) {{ e.stopPropagation(); moved = false; }}
  }}, true);
}})();
</script>
</body>
</html>"""

    with open(output_html,'w',encoding='utf-8') as f:
        html_out = html
        html_out = html_out.replace('__ID_SEGS_ORIG__', id_segs_js)
        html_out = html_out.replace('__ID_ROWS__',      id_rows_js)
        html_out = html_out.replace('__COLORS_JS__',    colors_js)
        html_out = html_out.replace('__CLASSES_JS__',   classes_js)
        html_out = html_out.replace('__SC_JS__',        sc_js)
        f.write(html_out)
    print(f"저장 완료: {output_html} ({os.path.getsize(output_html)/1024/1024:.1f}MB)")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--sensor',    required=True)
    parser.add_argument('--label',     required=True)
    parser.add_argument('--video',     default=None)
    parser.add_argument('--output',    default=None)
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--offset',    type=int,   default=None)
    parser.add_argument('--encoder',   type=str,   default='/workspace/preprocessed/label_encoder.pkl',
                        help='label_encoder.pkl 경로 (전체 클래스 목록 로드)')
    parser.add_argument('--classes',   type=str,   default=None,
                        help='클래스 직접 지정 (쉼표 구분, 예: Standing,Lying,Running)')
    parser.add_argument('--extra-classes', type=str, default=None,
                        help='추가 클래스 (쉼표 구분, 예: Scratching,Licking,Vomiting,Coughing)')
    args=parser.parse_args()
    out=args.output or (args.video.replace('.mp4','_viewer.html') if args.video else
                        args.sensor.replace('.csv','_viewer.html'))
    make_elan(args.sensor,args.label,args.video,out,args.threshold,args.offset,
              encoder_path=args.encoder, classes_override=args.classes,
              extra_classes=args.extra_classes)
