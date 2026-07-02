"""
오토 레이블링 검수 서버
FastAPI 기반 내부망 검수 플랫폼

설치: pip install fastapi uvicorn python-multipart
실행: python /workspace/server.py
접속: http://서버IP:8888
"""
import os
import sys
import json
import subprocess
from datetime import datetime
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.append('/workspace')
# 코덱 확인/변환 로직은 make_viewer와 공유 (중복 제거)
from make_viewer import get_video_codec, convert_to_h264

app = FastAPI(title="오토 레이블링 검수 서버")
# 내부망 도구지만 쓰기 API가 있으므로 CORS는 동일 출처만 허용(필요 시 환경변수로 확장)
_ALLOWED = [o for o in os.environ.get('ALLOW_ORIGINS', '').split(',') if o]
app.add_middleware(CORSMiddleware, allow_origins=_ALLOWED, allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 캐시된 viewer HTML 삭제 → /viewer 요청 시 항상 최신 make_viewer.py로 재생성.
    목록(get_file_list)은 영속되는 라벨 CSV 기준이라 삭제해도 비지 않는다."""
    deleted = 0
    for f in OUTPUTS_DIR.glob('*_viewer.html'):
        f.unlink(missing_ok=True)
        deleted += 1
    if deleted:
        print(f"[서버 시작] 캐시 viewer HTML {deleted}개 삭제 → 열람 시 최신 버전으로 재생성")

WORKSPACE     = Path('/workspace')
OUTPUTS_DIR   = WORKSPACE / 'outputs'
TEST_DATA_DIR = WORKSPACE / 'test_data'
VIDEO_MAP_FILE = OUTPUTS_DIR / '.video_map.json'   # 재시작 후에도 센서↔영상 매핑 유지

# 파이프라인 실행 상태 추적
pipeline_status = {}      # {job_id: {status, message, viewer_url}}  (휘발성)
MAX_STATUS_ENTRIES = 200  # 무한 증가 방지


def _prune_status():
    """오래된 job 상태를 제거해 메모리 무한 증가 방지."""
    if len(pipeline_status) > MAX_STATUS_ENTRIES:
        for k in list(pipeline_status)[:-MAX_STATUS_ENTRIES]:
            pipeline_status.pop(k, None)


def _load_video_map():
    try:
        if VIDEO_MAP_FILE.exists():
            return json.loads(VIDEO_MAP_FILE.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"[video_map 로드 실패] {e}")
    return {}


def _save_video_map():
    try:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        VIDEO_MAP_FILE.write_text(json.dumps(video_map, ensure_ascii=False), encoding='utf-8')
    except Exception as e:
        print(f"[video_map 저장 실패] {e}")


video_map = _load_video_map()   # {sensor_base: video_filename} — 디스크에서 복원


def safe_under(base_dir: Path, name: str) -> Path:
    """사용자 입력 파일명을 base_dir 안으로 강제(경로 탈출 방지).
    디렉터리 성분을 제거하고, resolve 결과가 base_dir 내부인지 검증."""
    candidate = (base_dir / Path(name).name).resolve()
    base = base_dir.resolve()
    if base not in candidate.parents and candidate != base:
        raise HTTPException(400, '잘못된 파일명')
    return candidate


def safe_base(name: str) -> str:
    """f-string 보간용 base 식별자에서 경로 구분자/.. 제거."""
    clean = Path(name).name
    if not clean or clean in ('.', '..'):
        raise HTTPException(400, '잘못된 이름')
    return clean


def safe_rel(sub: str) -> str:
    """outputs 기준 하위폴더 상대경로 검증(경로 탈출 방지). 빈 값이면 ''."""
    if not sub:
        return ''
    sub = sub.replace('\\', '/').strip('/')
    if not sub:
        return ''
    target = (OUTPUTS_DIR / sub).resolve()
    base = OUTPUTS_DIR.resolve()
    if base != target and base not in target.parents:
        raise HTTPException(400, '잘못된 하위폴더')
    return sub


def reviewed_paths(base: str, sub: str = ''):
    """검수결과(csv)/통계(json) 저장 경로를 하위폴더까지 반영해 반환."""
    base = safe_base(base)
    rel  = safe_rel(sub)
    d    = (OUTPUTS_DIR / rel) if rel else OUTPUTS_DIR
    return d, d / f'{base}_labeled_reviewed.csv', d / f'{base}_review_stats.json'


def get_file_list():
    # viewer HTML은 시작 시 삭제되므로 라벨 CSV 기준으로 목록을 구성하되,
    # outputs 하위폴더(예: 260521/B)까지 재귀 탐색해 (sub, base) 단위로 모은다.
    found = {}   # (rel, base) -> 대표 mtime

    def _add(path, suffix):
        rel = str(path.parent.relative_to(OUTPUTS_DIR)).replace('\\', '/')
        rel = '' if rel == '.' else rel
        base = path.name[:-len(suffix)]
        key = (rel, base)
        found[key] = max(found.get(key, 0), path.stat().st_mtime)

    for p in OUTPUTS_DIR.rglob('*_labeled.csv'):
        _add(p, '_labeled.csv')
    for p in OUTPUTS_DIR.rglob('*_labeled_smoothed.csv'):
        _add(p, '_labeled_smoothed.csv')

    items = []
    for (rel, base), mtime in sorted(found.items(), key=lambda kv: kv[1], reverse=True):
        rdir     = (OUTPUTS_DIR / rel) if rel else OUTPUTS_DIR
        reviewed = rdir / f'{base}_labeled_reviewed.csv'
        items.append({
            'name':     (rel + '/' if rel else '') + base,
            'base':     base,
            'sub':      rel,
            'html':     f'{base}_viewer.html',
            'status':   'completed' if reviewed.exists() else 'pending',
            'updated':  datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M'),
        })
    return items


def run_pipeline(job_id: str, sensor_path: Path, video_path: Path, label_path: Path = None,
                 threshold: float = 0.7, audio_db_margin: float = 12, out_subdir: str = ''):
    """백그라운드에서 파이프라인 실행 (out_subdir: outputs 하위폴더 → 입력 구조 유지)"""
    try:
        base       = sensor_path.stem
        out_dir    = (OUTPUTS_DIR / out_subdir) if out_subdir else OUTPUTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        viewer_out = out_dir / f'{base}_viewer.html'

        if label_path:
            # 이미 레이블 파일 있으면 오토레이블링/후처리 스킵 (검수된 데이터 보존)
            pipeline_status[job_id] = {'status': 'running', 'message': '[1/1] 검수 뷰어 생성 중...', 'viewer_url': None}
            label_out = label_path
        else:
            # 1단계: 오토 레이블링
            pipeline_status[job_id] = {'status': 'running', 'message': '[1/3] 오토 레이블링 중...', 'viewer_url': None}
            r1 = subprocess.run(
                ['python', '/workspace/auto_label.py', '--input', str(sensor_path),
                 '--threshold', str(threshold), '--output-dir', str(out_dir)],
                capture_output=True, text=True
            )
            if r1.returncode != 0:
                pipeline_status[job_id] = {'status': 'error', 'message': r1.stderr[-500:], 'viewer_url': None}
                return
            label_out = out_dir / f'{base}_labeled.csv'

            # 2단계: 필터링(flicker 정리) + 소리 큰 구간 Barking 처리
            pipeline_status[job_id]['message'] = '[2/3] 필터링 + 소리 처리 중...'
            r_pp = subprocess.run([
                'python', '/workspace/postprocess.py',
                '--input',         str(label_out),
                '--min-duration',   '2.0',
                '--smooth-window',  '1.0',
                '--protect-conf',   '0.85',
                '--video',          str(video_path),
                '--audio-db-margin', str(audio_db_margin),
            ], capture_output=True, text=True)
            smoothed_out = out_dir / f'{base}_labeled_smoothed.csv'
            if r_pp.returncode == 0 and smoothed_out.exists():
                label_out = smoothed_out          # 후처리 결과를 뷰어 입력으로 사용
            else:
                # 후처리 실패 시 원본 라벨로 폴백 (버튼이 죽지 않게)
                print('[postprocess 실패 → 원본 라벨 사용]', r_pp.stderr[-500:])

            pipeline_status[job_id]['message'] = '[3/3] 검수 뷰어 생성 중...'

        r2 = subprocess.run([
            'python', '/workspace/make_viewer.py',
            '--sensor',        str(sensor_path),
            '--label',         str(label_out),
            '--video',         str(video_path),
            '--output',        str(viewer_out),
            '--threshold',     str(threshold),
            '--encoder',       '/workspace/preprocessed/label_encoder.pkl',
            '--extra-classes', 'Scratching,Licking,Vomiting,Coughing',
            '--out-subdir',    out_subdir,
        ], capture_output=True, text=True)

        if r2.returncode != 0:
            pipeline_status[job_id] = {'status': 'error', 'message': r2.stderr[-500:], 'viewer_url': None}
            return

        pipeline_status[job_id] = {
            'status':     'done',
            'message':    '완료!',
            'viewer_url': f'/viewer/{viewer_out.name}' + (f'?sub={out_subdir}' if out_subdir else ''),
        }

    except Exception as e:
        pipeline_status[job_id] = {'status': 'error', 'message': str(e), 'viewer_url': None}


# ── 라우트 ────────────────────────────────────────────────────────────────

@app.get('/', response_class=HTMLResponse)
async def index():
    items   = get_file_list()
    pending = [i for i in items if i['status'] == 'pending']
    done    = [i for i in items if i['status'] == 'completed']

    def make_card(item):
        color = '#1D9E75' if item['status'] == 'completed' else '#EF9F27'
        label = '✅ 완료' if item['status'] == 'completed' else '⏳ 대기'
        sub_q = f'?sub={item["sub"]}' if item["sub"] else ''
        return f'''
        <div class="card" onclick="location.href='/viewer/{item["html"]}{sub_q}'">
          <div class="card-top">
            <div class="card-name">{item["name"]}</div>
            <div class="badge" style="background:{color}22;color:{color}">{label}</div>
          </div>
          <div class="card-meta">📅 {item["updated"]}</div>
        </div>'''

    cards = ''.join(make_card(i) for i in items)

    return f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>오토 레이블링 검수 서버</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,sans-serif;background:#0f0f0f;color:#eee;min-height:100vh}}
.header{{background:#161616;border-bottom:1px solid #2a2a2a;padding:16px 32px;display:flex;align-items:center;justify-content:space-between}}
.logo{{font-size:18px;font-weight:700}}
.sub{{font-size:12px;color:#555;margin-top:3px}}
.container{{max-width:960px;margin:0 auto;padding:32px 24px}}

/* 업로드 섹션 */
.upload-box{{background:#161616;border:2px dashed #333;border-radius:12px;padding:28px;margin-bottom:32px;transition:border-color 0.15s,background 0.15s}}
.upload-box.dragover{{border-color:#378ADD;background:#10243a}}
.upload-title{{font-size:14px;font-weight:700;margin-bottom:16px;color:#eee}}
.upload-row{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px}}
.file-label{{display:flex;flex-direction:column;gap:6px}}
.file-label span{{font-size:11px;color:#888;font-weight:600;text-transform:uppercase;letter-spacing:0.5px}}
.file-input{{background:#111;border:1px solid #333;border-radius:7px;padding:10px 12px;color:#eee;font-size:12px;cursor:pointer;width:100%}}
.file-input::-webkit-file-upload-button{{background:#2a2a2a;border:1px solid #444;color:#ccc;padding:4px 10px;border-radius:4px;cursor:pointer;font-size:11px;margin-right:10px}}
.run-btn{{width:100%;padding:12px;background:#378ADD;border:none;color:white;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;transition:background 0.15s}}
.run-btn:hover{{background:#2a6db5}}
.run-btn:disabled{{background:#333;color:#666;cursor:not-allowed}}
.progress{{display:none;margin-top:14px;padding:12px 16px;background:#111;border-radius:8px;font-size:12px;color:#aaa;border-left:3px solid #378ADD}}
.progress.error{{border-color:#E24B4A;color:#ff6b6b}}
.progress.done{{border-color:#1D9E75;color:#1D9E75}}

/* 파일 목록 */
.section-title{{font-size:11px;font-weight:700;color:#555;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:14px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:10px}}
.card{{background:#161616;border:1px solid #2a2a2a;border-radius:10px;padding:14px 18px;cursor:pointer;transition:all 0.15s}}
.card:hover{{border-color:#444;background:#1a1a1a}}
.card-top{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}}
.card-name{{font-size:12px;font-weight:600;color:#eee;word-break:break-all}}
.badge{{font-size:11px;font-weight:600;padding:2px 8px;border-radius:10px;white-space:nowrap;margin-left:8px}}
.card-meta{{font-size:11px;color:#555}}
.empty{{text-align:center;padding:40px;color:#444;font-size:13px}}
</style>
</head>
<body>
<div class="header">
  <div>
    <div class="logo">🐾 오토 레이블링 검수 서버</div>
    <div class="sub">반려동물 행동 분류 검수 플랫폼</div>
  </div>
  <div style="font-size:12px;color:#555">{len(items)}개 파일 | 대기 {len(pending)} / 완료 {len(done)}</div>
</div>

<div class="container">

  <!-- 파이프라인 실행 -->
  <div class="upload-box" id="uploadBox">
    <div class="upload-title">새 파일 검수 시작 <span style="font-size:11px;color:#555;font-weight:400">— 파일을 끌어다 놓아도 됩니다</span></div>
    <div class="upload-row">
      <div class="file-label">
        <span>센서 CSV</span>
        <input class="file-input" type="file" id="sensorFile" accept=".csv">
      </div>
      <div class="file-label">
        <span>영상 MP4</span>
        <input class="file-input" type="file" id="videoFile" accept=".mp4,video/*">
      </div>
    </div>
    <div class="file-label" style="margin-bottom:14px;max-width:360px">
      <span>하위폴더 (선택) — outputs/&lt;여기&gt;/ 아래에 저장, 비우면 최상위</span>
      <input class="file-input" type="text" id="subdirInput" placeholder="예: 260521/B">
    </div>
    <div class="file-label" style="margin-bottom:14px;max-width:240px">
      <span>신뢰도 임계값 (이 값 미만 → Unlabeled)</span>
      <input class="file-input" type="number" id="thresholdInput" value="0.7" min="0" max="1" step="0.05">
    </div>
    <div class="file-label" style="margin-bottom:14px;max-width:360px">
      <span>소리 임계값 — 배경 소음 +<b id="dbMarginVal">12</b> dB (클수록 큰 소리만 Barking)</span>
      <input type="range" id="audioDbMargin" min="6" max="24" step="1" value="12"
             style="width:100%;accent-color:#378ADD"
             oninput="document.getElementById('dbMarginVal').textContent=this.value">
      <span style="font-size:10px;color:#555">← 민감 (작은 소리도 Barking)　|　둔감 (큰 소리만) →</span>
    </div>
    <button class="run-btn" id="runBtn" onclick="runPipeline()">▶ 오토 레이블링 + 뷰어 생성</button>
    <div class="progress" id="progress"></div>
  </div>

  <!-- 파일 목록 -->
  <div class="section-title">검수 파일 목록</div>
  <div class="cards" id="fileList">
    {cards if cards else '<div class="empty">아직 파일이 없습니다.<br>위에서 센서 CSV와 영상을 업로드하세요.</div>'}
  </div>
</div>

<script>
async function runPipeline() {{
  const sensor = document.getElementById('sensorFile').files[0];
  const video  = document.getElementById('videoFile').files[0];
  const threshold = document.getElementById('thresholdInput').value || '0.7';
  const audioDbMargin = document.getElementById('audioDbMargin').value || '12';
  const subdir = document.getElementById('subdirInput').value.trim();
  if (!sensor || !video) {{ alert('센서 CSV와 영상 파일을 모두 선택하세요.'); return; }}

  const btn  = document.getElementById('runBtn');
  const prog = document.getElementById('progress');
  btn.disabled = true;
  btn.textContent = '업로드 중...';
  prog.style.display = 'block';
  prog.className = 'progress';
  prog.textContent = '파일 업로드 중...';

  const fd = new FormData();
  fd.append('sensor', sensor);
  fd.append('video',  video);
  fd.append('threshold', threshold);
  fd.append('audio_db_margin', audioDbMargin);
  fd.append('subdir', subdir);

  try {{
    const r    = await fetch('/upload', {{method:'POST', body:fd}});
    const data = await r.json();
    if(data.error) throw new Error(data.error);
    btn.textContent = '처리 중...';
    poll(data.job_id, prog, btn);
  }} catch(e) {{
    prog.className = 'progress error';
    prog.textContent = '업로드 실패: ' + e.message;
    btn.disabled = false;
    btn.textContent = '▶ 오토 레이블링 + 뷰어 생성';
  }}
}}

function poll(job_id, prog, btn) {{
  const iv = setInterval(async () => {{
    const r = await fetch('/status/' + job_id);
    const s = await r.json();
    prog.textContent = s.message;
    if (s.status === 'done') {{
      clearInterval(iv);
      prog.className = 'progress done';
      prog.innerHTML = '✅ 완료! <a href="' + s.viewer_url + '" '
        + 'style="display:inline-block;margin-left:8px;padding:6px 14px;background:#1D9E75;color:#fff;'
        + 'border-radius:6px;font-weight:600;text-decoration:none">뷰어 열기 →</a>';
      btn.disabled = false;
      btn.textContent = '▶ 오토 레이블링 + 뷰어 생성';
      refreshFileList();   // 페이지 새로고침 없이 목록만 갱신 → 완료 링크 유지
    }} else if (s.status === 'error') {{
      clearInterval(iv);
      prog.className = 'progress error';
      prog.textContent = '❌ 오류: ' + s.message;
      btn.disabled = false;
      btn.textContent = '▶ 오토 레이블링 + 뷰어 생성';
    }}
  }}, 2000);
}}

// 페이지 전체 새로고침 없이 파일 목록만 다시 그림 (완료 링크 보존)
async function refreshFileList() {{
  try {{
    const r = await fetch('/');
    const t = await r.text();
    const doc = new DOMParser().parseFromString(t, 'text/html');
    const fresh = doc.getElementById('fileList');
    if (fresh) document.getElementById('fileList').innerHTML = fresh.innerHTML;
  }} catch(e) {{}}
}}

// ── 드래그&드롭 업로드 ────────────────────────────────────────────────────
// 떨어뜨린 파일을 확장자로 분류해 센서(.csv)/영상(나머지) input에 채운다.
function assignDropped(files) {{
  let sensorSet = false, videoSet = false;
  for (const f of files) {{
    const isCsv = f.name.toLowerCase().endsWith('.csv');
    const target = isCsv ? 'sensorFile' : 'videoFile';
    if (isCsv ? sensorSet : videoSet) continue;
    const dt = new DataTransfer();
    dt.items.add(f);
    document.getElementById(target).files = dt.files;
    if (isCsv) sensorSet = true; else videoSet = true;
    // 폴더째 드래그해서 상대경로가 있으면 하위폴더 자동 채움 (best-effort)
    const rp = f.webkitRelativePath || '';
    if (isCsv && rp.includes('/')) {{
      const sd = rp.slice(0, rp.lastIndexOf('/'));
      const inp = document.getElementById('subdirInput');
      if (inp && !inp.value.trim()) inp.value = sd;
    }}
  }}
  const prog = document.getElementById('progress');
  if (sensorSet || videoSet) {{
    prog.style.display = 'block';
    prog.className = 'progress';
    prog.textContent = '선택됨: ' +
      [sensorSet ? '센서 CSV' : null, videoSet ? '영상' : null].filter(Boolean).join(', ') +
      ' — ▶ 버튼을 누르세요.';
  }}
}}

const _uploadBox = document.getElementById('uploadBox');
['dragenter','dragover'].forEach(ev => _uploadBox.addEventListener(ev, e => {{
  e.preventDefault(); e.stopPropagation();
  _uploadBox.classList.add('dragover');
}}));
['dragleave','drop'].forEach(ev => _uploadBox.addEventListener(ev, e => {{
  e.preventDefault(); e.stopPropagation();
  if (ev === 'dragleave' && _uploadBox.contains(e.relatedTarget)) return;
  _uploadBox.classList.remove('dragover');
}}));
_uploadBox.addEventListener('drop', e => {{
  if (e.dataTransfer && e.dataTransfer.files.length) assignDropped(e.dataTransfer.files);
}});
// 페이지 전체로 떨어뜨렸을 때 브라우저가 파일을 열어버리는 기본 동작 방지
['dragover','drop'].forEach(ev => window.addEventListener(ev, e => {{
  if (!_uploadBox.contains(e.target)) e.preventDefault();
}}));
</script>
</body>
</html>'''


@app.post('/upload')
async def upload(background_tasks: BackgroundTasks,
                 sensor: UploadFile = File(...),
                 video:  UploadFile = File(...),
                 threshold: float = Form(0.7),
                 audio_db_margin: float = Form(12),
                 subdir: str = Form('')):
    """파일 업로드 + 파이프라인 백그라운드 실행 (subdir: outputs/test_data 하위폴더)"""
    try:
        # 하위폴더(경로 탈출 방지) → test_data/<subdir>/ 아래에 저장
        rel = safe_rel(subdir)
        save_dir = (TEST_DATA_DIR / rel) if rel else TEST_DATA_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        sensor_path = safe_under(save_dir, sensor.filename or 'sensor.csv')
        video_path  = safe_under(save_dir, video.filename or 'video.mp4')

        with open(sensor_path, 'wb') as f:
            f.write(await sensor.read())
        with open(video_path, 'wb') as f:
            f.write(await video.read())

        # 업로드 센서 CSV 컬럼 검증 (백그라운드 깊은 곳에서 터지지 않게 조기 차단)
        import pandas as _pd
        try:
            df_check = _pd.read_csv(sensor_path, nrows=1)
        except Exception as e:
            return JSONResponse({'error': f'CSV 읽기 실패: {e}'}, status_code=400)
        cols = set(df_check.columns)
        if 'timestamp' not in cols:
            return JSONResponse({'error': "CSV에 'timestamp' 컬럼이 없습니다."}, status_code=400)
        is_reviewed = 'pred_label' in cols
        if not is_reviewed:
            accel_ok = {'accel_x', 'accel_y', 'accel_z'} <= cols or {'acc_x', 'acc_y', 'acc_z'} <= cols
            gyro_ok  = {'gyro_x', 'gyro_y', 'gyro_z'} <= cols
            if not (accel_ok and gyro_ok):
                return JSONResponse(
                    {'error': '센서 CSV에 accel_x/y/z, gyro_x/y/z 컬럼이 필요합니다. '
                              f'현재: {sorted(cols)}'}, status_code=400)

        # 코덱 확인 및 H.264 변환 (make_viewer와 공유 로직)
        if get_video_codec(video_path) == 'hevc':
            orig_video = video_path                    # 변환 후 삭제할 원본 hevc
            video_path = Path(convert_to_h264(video_path))
            if video_path != orig_video:
                orig_video.unlink(missing_ok=True)     # 원본 제거(디스크 절약)
            print(f"hevc → H.264 변환 완료: {video_path.name} (원본 삭제)")

        # 센서-영상 매핑 저장 (디스크 영속화 → 재시작 후에도 영상 탐색 가능)
        sensor_base = sensor_path.stem
        video_map[sensor_base] = video_path.name
        _save_video_map()

        # 파일 종류 자동 감지 (reviewed CSV인지 원본 센서 CSV인지)
        label_path = None
        try:
            if is_reviewed:
                # reviewed CSV를 올린 경우 → label로 사용, 오토레이블링 스킵
                label_path = sensor_path
                ts_match = re.search(r'(\d{8}_\d{6})', sensor_path.stem)
                if ts_match:
                    ts = ts_match.group(1)
                    orig = next((f for f in TEST_DATA_DIR.rglob(f'*{ts}*.csv')
                                 if 'labeled' not in f.stem), None)
                    if orig:
                        sensor_path = orig
                        video_map[orig.stem] = video_path.name
                        _save_video_map()
                print(f"레이블 CSV 감지: {label_path.name} → 오토레이블링 스킵")
        except Exception:
            pass

        # 파이프라인 백그라운드 실행
        job_id = f"{safe_base(sensor.filename or 'job')}_{datetime.now().strftime('%H%M%S')}"
        _prune_status()
        pipeline_status[job_id] = {'status': 'running', 'message': '시작 중...', 'viewer_url': None}
        background_tasks.add_task(run_pipeline, job_id, sensor_path, video_path, label_path, threshold, audio_db_margin, rel)

        return JSONResponse({'job_id': job_id})

    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)


@app.get('/status/{job_id}')
async def status(job_id: str):
    """파이프라인 실행 상태 조회"""
    s = pipeline_status.get(job_id, {'status': 'unknown', 'message': '알 수 없음', 'viewer_url': None})
    return JSONResponse(s)


@app.get('/viewer/{filename}', response_class=HTMLResponse)
async def viewer(filename: str, sub: str = ''):
    """뷰어 요청 시 make_viewer.py로 항상 재생성 → 캐시 없이 최신 UI/라벨 반영.
    (검수 결과는 생성된 HTML의 autoRestore가 /load로 복원하므로 손실 없음)
    sub: outputs 기준 하위폴더(예: 260521/B) — 검수결과를 같은 폴더에 저장하기 위함."""
    filename = Path(filename).name
    if not filename.endswith('_viewer.html'):
        raise HTTPException(400, '잘못된 파일명')
    rel         = safe_rel(sub)
    label_dir   = (OUTPUTS_DIR / rel) if rel else OUTPUTS_DIR
    viewer_out  = safe_under(label_dir, filename) if rel else safe_under(OUTPUTS_DIR, filename)
    base        = filename.replace('_viewer.html', '')
    sensor_base = base.replace('_labeled', '')

    # 영상 파일 찾기 (매핑 우선, 없으면 test_data 전체에서 재귀 탐색)
    video = None
    if sensor_base in video_map:
        video = TEST_DATA_DIR / Path(video_map[sensor_base]).name
        if not video.exists():
            video = None
    if video is None:
        ts_match = re.search(r'(\d{8}_\d{6})', sensor_base)
        pat = f'*{ts_match.group(1)}*.mp4' if ts_match else f'*{sensor_base}*.mp4'
        video = next(TEST_DATA_DIR.rglob(pat), None)

    # 센서/라벨 파일 찾기 (센서는 test_data 재귀, 라벨은 해당 하위폴더에서 smoothed > labeled 순)
    sensor = next(TEST_DATA_DIR.rglob(f'*{sensor_base}*.csv'), None)
    label = label_dir / f'{sensor_base}_labeled_smoothed.csv'
    if not label.exists():
        label = label_dir / f'{sensor_base}_labeled.csv'
    if not label.exists():
        raise HTTPException(404, f'레이블 파일 없음: {sensor_base}')
    if not sensor:
        raise HTTPException(404, f'센서 파일 없음: {sensor_base}')

    # make_viewer.py로 재생성 (항상 최신 코드/라벨 반영, 검수결과 저장 하위폴더 전달)
    r = subprocess.run([
        'python', '/workspace/make_viewer.py',
        '--sensor',        str(sensor),
        '--label',         str(label),
        '--video',         str(video) if video else '',
        '--output',        str(viewer_out),
        '--encoder',       '/workspace/preprocessed/label_encoder.pkl',
        '--extra-classes', 'Scratching,Licking,Vomiting,Coughing',
        '--out-subdir',    rel,
    ], capture_output=True, text=True)
    if r.returncode != 0:
        raise HTTPException(500, f'뷰어 생성 실패: {r.stderr[-300:]}')

    html = viewer_out.read_text(encoding='utf-8')

    if video:
        video_url = f'/video/{video.name}'
        html = html.replace('<div id="videoPickWrap"', '<div id="videoPickWrap" style="display:none!important"', 1)
        html = html.replace(
            '<video id="vid" controls style="display:none">',
            f'<video id="vid" controls src="{video_url}" style="display:block">'
        )

    # CSV 저장을 서버로 (stats 선언 이후 교체)
    base = safe_base(base)
    sub_q      = f'?sub={rel}' if rel else ''        # 하위폴더를 저장 요청에 실어보냄
    saved_name = (f'{rel}/' if rel else '') + f'{base}_labeled_reviewed.csv'
    old_save = "hideSaveModal();\n  setFeedback(`\u2713 \uc800\uc7a5 \uc644\ub8cc: ${finalName}`,'var(--green)');\n}"
    new_save = (
        f"fetch('/save/{base}{sub_q}', {{\n"
        f"    method:'POST',\n"
        f"    headers:{{'Content-Type':'application/json'}},\n"
        f"    body:JSON.stringify({{csv:lines.join('\\n'),stats:stats}})\n"
        f"  }}).then(r=>r.json()).then(()=>{{\n"
        f"    hideSaveModal();\n"
        f"    setFeedback('\u2713 \uc800\uc7a5 \uc644\ub8cc: {saved_name}','var(--green)');\n"
        f"  }}).catch(()=>{{\n"
        f"    const a2=document.createElement('a');\n"
        f"    a2.href=URL.createObjectURL(blob);\n"
        f"    a2.download=finalName;\n"
        f"    a2.click();\n"
        f"    hideSaveModal();\n"
        f"    setFeedback('\u2713 \ub85c\ucec8 \uc800\uc7a5 \uc644\ub8cc','var(--yellow)');\n"
        f"  }});\n"
        f"}}"
    )
    if old_save in html:
        html = html.replace(old_save, new_save)
    return HTMLResponse(content=html)


@app.get('/video/{filename}')
async def stream_video(filename: str, request: Request):
    """영상 스트리밍 (Range 요청 지원). test_data 하위폴더까지 파일명으로 재귀 탐색."""
    path = safe_under(TEST_DATA_DIR, filename)
    if not path.exists():
        # 하위폴더(예: test_data/260521/B/)에 있는 경우 파일명으로 재귀 탐색
        name = Path(filename).name
        found = next(TEST_DATA_DIR.rglob(name), None)
        if found and found.resolve().is_relative_to(TEST_DATA_DIR.resolve()):
            path = found
    if not path.exists():
        raise HTTPException(404, '영상 없음')

    file_size    = path.stat().st_size
    range_header = request.headers.get('range')

    if range_header:
        start, end = range_header.replace('bytes=', '').split('-')
        start = int(start)
        end   = int(end) if end else file_size - 1
        chunk = end - start + 1

        def iter_file():
            with open(path, 'rb') as f:
                f.seek(start)
                remaining = chunk
                while remaining:
                    data = f.read(min(65536, remaining))
                    if not data: break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(iter_file(), status_code=206, media_type='video/mp4',
            headers={
                'Content-Range':  f'bytes {start}-{end}/{file_size}',
                'Accept-Ranges':  'bytes',
                'Content-Length': str(chunk),
            })

    def iter_full():
        with open(path, 'rb') as f:
            while chunk := f.read(65536):
                yield chunk

    return StreamingResponse(iter_full(), media_type='video/mp4',
        headers={
            'Accept-Ranges':  'bytes',
            'Content-Length': str(file_size),
        })


@app.post('/autosave/{base}')
async def autosave(base: str, request: Request, sub: str = ''):
    """30초마다 자동저장 — reviewed CSV 덮어씀 (sub=하위폴더)"""
    try:
        d, reviewed, _ = reviewed_paths(base, sub)
        data = await request.json()
        d.mkdir(parents=True, exist_ok=True)
        reviewed.write_text(data.get('csv', ''), encoding='utf-8')
        return JSONResponse({'ok': True})
    except Exception as e:
        return JSONResponse({'ok': False, 'error': str(e)})


@app.get('/save/{base}')
async def get_save(base: str, sub: str = ''):
    """저장된 reviewed CSV 존재 여부 확인 (sub=하위폴더)"""
    _, reviewed, _ = reviewed_paths(base, sub)
    if reviewed.exists():
        return JSONResponse({'exists': True, 'size': reviewed.stat().st_size,
                             'updated': reviewed.stat().st_mtime})
    return JSONResponse({'exists': False})


@app.get('/load/{base}')
async def load_reviewed(base: str, sub: str = ''):
    """저장된 reviewed CSV 내용 반환 — 뷰어 재진입 시 자동 복원용 (sub=하위폴더)"""
    _, reviewed, _ = reviewed_paths(base, sub)
    if reviewed.exists():
        return JSONResponse({'exists': True,
                             'csv': reviewed.read_text(encoding='utf-8'),
                             'updated': reviewed.stat().st_mtime})
    return JSONResponse({'exists': False})


@app.post('/save/{base}')
async def save_result(base: str, request: Request, sub: str = ''):
    """검수 결과 서버에 저장 (sub=하위폴더 → 입력 구조와 동일하게 저장)"""
    try:
        d, reviewed, stats = reviewed_paths(base, sub)
        data = await request.json()
        d.mkdir(parents=True, exist_ok=True)
        reviewed.write_text(data.get('csv', ''), encoding='utf-8')
        stats.write_text(
            json.dumps(data.get('stats', {}), ensure_ascii=False, indent=2), encoding='utf-8')
        return JSONResponse({'ok': True})
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == '__main__':
    print("==============================")
    print(" 오토 레이블링 검수 서버")
    print(" http://localhost:8888")
    print(" 브라우저에서 위 주소로 접속하세요")
    print("==============================")
    uvicorn.run(app, host='0.0.0.0', port=8888)
