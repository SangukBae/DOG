"""
오토 레이블링 검수 서버
FastAPI 기반 내부망 검수 플랫폼

설치: pip install fastapi uvicorn python-multipart
실행: python /workspace/server.py
접속: http://서버IP:8888
"""
import os
import json
import subprocess
from datetime import datetime
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="오토 레이블링 검수 서버")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WORKSPACE     = Path('/workspace')
OUTPUTS_DIR   = WORKSPACE / 'outputs'
TEST_DATA_DIR = WORKSPACE / 'test_data'

# 파이프라인 실행 상태 추적
pipeline_status = {}  # {job_id: {status, message, viewer_url}}
video_map = {}        # {sensor_base: video_filename}


def get_file_list():
    items = []
    htmls = sorted(OUTPUTS_DIR.glob('*_viewer.html'), key=lambda x: x.stat().st_mtime, reverse=True)
    for html in htmls:
        base     = html.stem.replace('_viewer', '')
        reviewed = OUTPUTS_DIR / f'{base}_labeled_reviewed.csv'
        stat     = 'completed' if reviewed.exists() else 'pending'
        items.append({
            'name':     base,
            'html':     html.name,
            'status':   stat,
            'updated':  datetime.fromtimestamp(html.stat().st_mtime).strftime('%Y-%m-%d %H:%M'),
        })
    return items


def run_pipeline(job_id: str, sensor_path: Path, video_path: Path, label_path: Path = None,
                 threshold: float = 0.7, audio_db_margin: float = 12):
    """백그라운드에서 파이프라인 실행"""
    try:
        base       = sensor_path.stem
        viewer_out = OUTPUTS_DIR / f'{base}_viewer.html'

        if label_path:
            # 이미 레이블 파일 있으면 오토레이블링/후처리 스킵 (검수된 데이터 보존)
            pipeline_status[job_id] = {'status': 'running', 'message': '[1/1] 검수 뷰어 생성 중...', 'viewer_url': None}
            label_out = label_path
        else:
            # 1단계: 오토 레이블링
            pipeline_status[job_id] = {'status': 'running', 'message': '[1/3] 오토 레이블링 중...', 'viewer_url': None}
            r1 = subprocess.run(
                ['python', '/workspace/auto_label.py', '--input', str(sensor_path),
                 '--threshold', str(threshold)],
                capture_output=True, text=True
            )
            if r1.returncode != 0:
                pipeline_status[job_id] = {'status': 'error', 'message': r1.stderr[-500:], 'viewer_url': None}
                return
            label_out = OUTPUTS_DIR / f'{base}_labeled.csv'

            # 2단계: 필터링(flicker 정리) + 소리 큰 구간 Barking 처리
            pipeline_status[job_id]['message'] = '[2/3] 필터링 + 소리 처리 중...'
            r_pp = subprocess.run([
                'python', '/workspace/postprocess.py',
                '--input',         str(label_out),
                '--min-duration',   '1.0',
                '--smooth-window',  '0.5',
                '--protect-conf',   '0.85',
                '--video',          str(video_path),
                '--audio-db-margin', str(audio_db_margin),
            ], capture_output=True, text=True)
            smoothed_out = OUTPUTS_DIR / f'{base}_labeled_smoothed.csv'
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
        ], capture_output=True, text=True)

        if r2.returncode != 0:
            pipeline_status[job_id] = {'status': 'error', 'message': r2.stderr[-500:], 'viewer_url': None}
            return

        pipeline_status[job_id] = {
            'status':     'done',
            'message':    '완료!',
            'viewer_url': f'/viewer/{viewer_out.name}',
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
        return f'''
        <div class="card" onclick="location.href='/viewer/{item["html"]}'">
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
.upload-box{{background:#161616;border:2px dashed #333;border-radius:12px;padding:28px;margin-bottom:32px}}
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
  <div class="upload-box">
    <div class="upload-title">새 파일 검수 시작</div>
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
    <div class="file-label" style="margin-bottom:14px;max-width:240px">
      <span>신뢰도 임계값 (이 값 미만 → 미분류)</span>
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
</script>
</body>
</html>'''


@app.post('/upload')
async def upload(background_tasks: BackgroundTasks,
                 sensor: UploadFile = File(...),
                 video:  UploadFile = File(...),
                 threshold: float = Form(0.7),
                 audio_db_margin: float = Form(12)):
    """파일 업로드 + 파이프라인 백그라운드 실행"""
    try:
        # 파일 저장
        sensor_path = TEST_DATA_DIR / sensor.filename
        video_path  = TEST_DATA_DIR / video.filename
        TEST_DATA_DIR.mkdir(exist_ok=True)

        with open(sensor_path, 'wb') as f:
            f.write(await sensor.read())
        with open(video_path, 'wb') as f:
            f.write(await video.read())

        # 코덱 확인 및 H.264 변환
        codec = subprocess.run(
            ['/usr/bin/ffprobe','-v','error','-select_streams','v:0',
             '-show_entries','stream=codec_name',
             '-of','default=noprint_wrappers=1:nokey=1', str(video_path)],
            capture_output=True, text=True).stdout.strip()

        if codec == 'hevc':
            h264_path = TEST_DATA_DIR / video.filename.replace('.mp4', '_h264.mp4')
            subprocess.run([
                '/usr/bin/ffmpeg', '-y', '-i', str(video_path),
                '-vcodec', 'libx264', '-crf', '23', '-preset', 'fast',
                '-acodec', 'aac', str(h264_path)
            ], check=True, capture_output=True)
            video_path = h264_path
            print(f"hevc → H.264 변환 완료: {h264_path.name}")

        # 센서-영상 매핑 저장
        sensor_base = sensor_path.stem
        video_map[sensor_base] = video_path.name

        # 파일 종류 자동 감지 (reviewed CSV인지 원본 센서 CSV인지)
        import pandas as _pd
        label_path = None
        try:
            df_check = _pd.read_csv(sensor_path, nrows=1)
            if 'pred_label' in df_check.columns:
                # reviewed CSV를 올린 경우 → label로 사용, 오토레이블링 스킵
                label_path = sensor_path
                ts_match = re.search(r'(\d{8}_\d{6})', sensor_path.stem)
                if ts_match:
                    ts = ts_match.group(1)
                    orig = next((f for f in TEST_DATA_DIR.glob(f'*{ts}*.csv')
                                 if 'labeled' not in f.stem), None)
                    if orig:
                        sensor_path = orig
                        video_map[orig.stem] = video_path.name
                print(f"레이블 CSV 감지: {label_path.name} → 오토레이블링 스킵")
        except Exception:
            pass

        # 파이프라인 백그라운드 실행
        job_id = f"{sensor.filename}_{datetime.now().strftime('%H%M%S')}"
        pipeline_status[job_id] = {'status': 'running', 'message': '시작 중...', 'viewer_url': None}
        background_tasks.add_task(run_pipeline, job_id, sensor_path, video_path, label_path, threshold, audio_db_margin)

        return JSONResponse({'job_id': job_id})

    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=500)


@app.get('/status/{job_id}')
async def status(job_id: str):
    """파이프라인 실행 상태 조회"""
    s = pipeline_status.get(job_id, {'status': 'unknown', 'message': '알 수 없음', 'viewer_url': None})
    return JSONResponse(s)


@app.get('/viewer/{filename}', response_class=HTMLResponse)
async def viewer(filename: str):
    """뷰어 HTML 반환 (영상 스트리밍 URL로 교체)"""
    path = OUTPUTS_DIR / filename
    if not path.exists():
        raise HTTPException(404, '파일 없음')

    html = path.read_text(encoding='utf-8')

    # 영상 파일 찾기 (매핑 우선, 없으면 날짜 패턴으로 찾기)
    sensor_base = filename.replace('_viewer.html', '').replace('_labeled', '')
    if sensor_base in video_map:
        video = TEST_DATA_DIR / video_map[sensor_base]
        if not video.exists():
            video = None
    else:
        ts_match = re.search(r'(\d{8}_\d{6})', sensor_base)
        if ts_match:
            ts = ts_match.group(1)
            video = next(TEST_DATA_DIR.glob(f'*{ts}*.mp4'), None)
        else:
            video = next(TEST_DATA_DIR.glob(f'*{sensor_base}*.mp4'), None)

    if video:
        video_url = f'/video/{video.name}'
        html = html.replace('<div id="videoPickWrap"', '<div id="videoPickWrap" style="display:none!important"', 1)
        html = html.replace(
            '<video id="vid" controls style="display:none">',
            f'<video id="vid" controls src="{video_url}" style="display:block">'
        )

    # CSV 저장을 서버로 (stats 선언 이후 교체)
    base = filename.replace('_viewer.html', '')
    old_save = "hideSaveModal();\n  setFeedback(`\u2713 \uc800\uc7a5 \uc644\ub8cc: ${finalName}`,'var(--green)');\n}"
    new_save = (
        f"fetch('/save/{base}', {{\n"
        f"    method:'POST',\n"
        f"    headers:{{'Content-Type':'application/json'}},\n"
        f"    body:JSON.stringify({{csv:lines.join('\\n'),stats:stats}})\n"
        f"  }}).then(r=>r.json()).then(()=>{{\n"
        f"    hideSaveModal();\n"
        f"    setFeedback('\u2713 \uc11c\ubc84 \uc800\uc7a5 \uc644\ub8cc','var(--green)');\n"
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
    """영상 스트리밍 (Range 요청 지원)"""
    path = TEST_DATA_DIR / filename
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
async def autosave(base: str, request: Request):
    """30초마다 자동저장 — reviewed CSV 덮어씀"""
    try:
        data = await request.json()
        csv  = data.get('csv', '')
        path = OUTPUTS_DIR / f'{base}_labeled_reviewed.csv'
        path.write_text(csv, encoding='utf-8')
        return JSONResponse({'ok': True})
    except Exception as e:
        return JSONResponse({'ok': False, 'error': str(e)})


@app.get('/save/{base}')
async def get_save(base: str):
    """저장된 reviewed CSV 존재 여부 확인"""
    path = OUTPUTS_DIR / f'{base}_labeled_reviewed.csv'
    if path.exists():
        return JSONResponse({'exists': True, 'size': path.stat().st_size,
                             'updated': path.stat().st_mtime})
    return JSONResponse({'exists': False})


@app.post('/save/{base}')
async def save_result(base: str, request: Request):
    """검수 결과 서버에 저장"""
    try:
        data = await request.json()
        (OUTPUTS_DIR / f'{base}_labeled_reviewed.csv').write_text(data.get('csv',''), encoding='utf-8')
        (OUTPUTS_DIR / f'{base}_review_stats.json').write_text(
            json.dumps(data.get('stats',{}), ensure_ascii=False, indent=2), encoding='utf-8')
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
