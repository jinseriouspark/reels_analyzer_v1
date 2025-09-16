"""
🎬 릴스 분석 Streamlit 앱 (OCR 제거 + 빠른모드 + 지연임포트)
"""

import streamlit as st
import asyncio
import os
import re
import time
from datetime import datetime
from io import BytesIO
from typing import Optional

from PIL import Image, ImageDraw, ImageFont  # 콜라주용


# =========================
# 유틸: 점수 표준화 & 비동기 실행
# =========================
def _is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False


def normalize_scores(scores):
    if scores is None:
        return {}
    if isinstance(scores, dict):
        return {str(k): float(v) for k, v in scores.items() if _is_number(v)}
    if isinstance(scores, list):
        tmp = {}
        for item in scores:
            if isinstance(item, dict):
                key = item.get("category") or item.get("name") or item.get("label")
                val = item.get("score") or item.get("value") or item.get("val")
                if key is not None and _is_number(val):
                    tmp[str(key)] = float(val)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                key, val = item[0], item[1]
                if key is not None and _is_number(val):
                    tmp[str(key)] = float(val)
        return tmp
    return {}


def run_sync(coro):
    """Streamlit 환경에서 안전하게 코루틴 실행"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
    else:
        return asyncio.run(coro)


# =========================
# API 키 핸들링 (세션 입력용)
# =========================
def _apply_api_keys(openai_key: str = "", gemini_key: str = "", claude_key: str = ""):
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    else:
        os.environ.pop("OPENAI_API_KEY", None)
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
    else:
        os.environ.pop("GEMINI_API_KEY", None)
    if claude_key:
        os.environ["CLAUDE_API_KEY"] = claude_key
    else:
        os.environ.pop("CLAUDE_API_KEY", None)


def _load_default_keys_from_env_and_secrets():
    def _get(name, env_name):
        try:
            return st.secrets.get(name, "")
        except Exception:
            return os.environ.get(env_name, "")
    return {
        "openai": _get("OPENAI_API_KEY", "OPENAI_API_KEY"),
        "gemini": _get("GEMINI_API_KEY", "GEMINI_API_KEY"),
        "claude": _get("CLAUDE_API_KEY", "CLAUDE_API_KEY"),
    }


# =========================
# contact_sheet: 샘플링 + PNG/PDF
# =========================
def sample_evenly(paths, limit: int):
    if not paths:
        return []
    if limit <= 0 or limit >= len(paths):
        return paths
    step = max(1, len(paths) // limit)
    return paths[::step][:limit]


def make_contact_sheet(
    image_paths,
    cols: int = 5,
    thumb_width: int = 300,
    pad: int = 8,
    bg=(255, 255, 255),
    annotate: bool = False,
    font_path: Optional[str] = None,
    font_size: int = 16,
    draw_border: bool = False,
):
    if not image_paths:
        raise ValueError("image_paths is empty")

    thumbs = []
    for p in image_paths:
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert("RGB")
        w, h = im.size
        new_h = max(1, int(h * (thumb_width / float(w))))
        im = im.resize((thumb_width, new_h), Image.LANCZOS)
        thumbs.append((os.path.basename(p), im))
    if not thumbs:
        raise ValueError("no readable images")

    cols = max(1, int(cols))
    rows = (len(thumbs) + cols - 1) // cols
    row_heights = []
    for r in range(rows):
        row_imgs = [im for _, im in thumbs[r * cols:(r + 1) * cols]]
        row_heights.append(max(img.size[1] for img in row_imgs))

    sheet_w = cols * thumb_width + (cols + 1) * pad
    sheet_h = sum(row_heights) + (rows + 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), bg)
    draw = ImageDraw.Draw(sheet)

    if annotate:
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
    else:
        font = None

    y = pad
    idx = 0
    border_color = (200, 200, 200)
    label_bg = (255, 255, 255)
    label_fg = (0, 0, 0)

    for r in range(rows):
        x = pad
        for c in range(cols):
            if idx >= len(thumbs):
                break
            name, im = thumbs[idx]
            sheet.paste(im, (x, y))
            if draw_border:
                draw.rectangle([x, y, x + im.size[0] - 1, y + im.size[1] - 1], outline=border_color, width=1)
            if annotate and font is not None:
                lh = font_size + 6
                ly1 = y + im.size[1] - lh
                ly2 = y + im.size[1]
                draw.rectangle([x, ly1, x + im.size[0], ly2], fill=label_bg)
                draw.text((x + 4, ly1 + 3), name, fill=label_fg, font=font)
            x += thumb_width + pad
            idx += 1
        y += row_heights[r] + pad
    return sheet


def build_contact_sheet_bytes(
    image_paths,
    cols: int = 5,
    thumb_width: int = 300,
    pad: int = 8,
    bg=(255, 255, 255),
    annotate: bool = False,
    font_path: Optional[str] = None,
    font_size: int = 16,
    draw_border: bool = False,
    pdf_resolution: float = 150.0,
):
    sheet = make_contact_sheet(
        image_paths, cols=cols, thumb_width=thumb_width, pad=pad, bg=bg,
        annotate=annotate, font_path=font_path, font_size=font_size, draw_border=draw_border
    )
    png_buf = BytesIO()
    sheet.save(png_buf, format="PNG")
    png_bytes = png_buf.getvalue()

    pdf_buf = BytesIO()
    sheet.convert("RGB").save(pdf_buf, format="PDF", resolution=pdf_resolution)
    pdf_bytes = pdf_buf.getvalue()
    return png_bytes, pdf_bytes


# =========================
# 페이지 설정 & 스타일
# =========================
st.set_page_config(
    page_title="🎬 릴스 분석기",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #FF6B6B;
    font-size: 3rem;
    margin-bottom: 2rem;
}
.metric-container {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.feedback-box {
    background: #e8f4fd;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)


# =========================
# 세션 상태
# =========================
def init_session_state():
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    if "current_analysis" not in st.session_state:
        st.session_state.current_analysis = None
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = _load_default_keys_from_env_and_secrets()
        _apply_api_keys(
            st.session_state.api_keys.get("openai", ""),
            st.session_state.api_keys.get("gemini", ""),
            st.session_state.api_keys.get("claude", ""),
        )


@st.cache_resource
def get_analyzer():
    """지연 임포트로 UI 안전 부팅"""
    try:
        from analyzer import ReelAnalyzer  # 여기서만 임포트
        return ReelAnalyzer()
    except Exception as e:
        st.warning(f"Analyzer 로드 실패(임시 더미 사용): {e}")

        class _Dummy:
            async def analyze_video(self, **kwargs):
                return {
                    "overall_score": 7,
                    "scores": {
                        "visual_appeal": 7, "content_structure": 7,
                        "trend_fit": 6, "emotional_impact": 7, "viral_potential": 6
                    },
                    "final_feedback_text": "더미 분석 결과입니다. 의존 패키지 설치 후 재시도하세요.",
                    "specific_improvements": ["인트로를 더 짧게", "CTA 자막 추가"],
                    "next_steps": ["씬 전환 실험", "후킹 문구 테스트"],
                    "frame_paths": [], "asr_text": "", "asr_segments": []
                }

        return _Dummy()


# =========================
# 파일 저장
# =========================
def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r'[^A-Za-z0-9._-]+', '_', name)
    return name or "video"


def save_uploaded_file(uploaded_file):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    base, ext = os.path.splitext(_sanitize_filename(uploaded_file.name))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{base}_{ts}{ext or '.mp4'}"
    file_path = os.path.join(upload_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


# =========================
# 메인
# =========================
def main():
    init_session_state()

    # 헤더
    st.markdown('<h1 class="main-header">🎬 릴스 분석기</h1>', unsafe_allow_html=True)
    st.markdown("**AI가 당신의 릴스를 분석하고 개선 방법을 제안합니다**")

    # 사이드바: 키 입력 + 분석/추출 옵션 + 속도
    with st.sidebar:
        st.header("🔐 API 키 입력 (세션에만 저장)")
        #openai_key = st.text_input("OpenAI API Key", value=st.session_state.api_keys.get("openai", ""), type="password", placeholder="sk-...")
        gemini_key = st.text_input("Gemini API Key", value=st.session_state.api_keys.get("gemini", ""), type="password", placeholder="AIza...")
        #claude_key = st.text_input("Claude API Key", value=st.session_state.api_keys.get("claude", ""), type="password")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("키 적용", use_container_width=True):
                st.session_state.api_keys = {
                    #"openai": openai_key.strip(),
                    "gemini": gemini_key.strip(),
                   # "claude": claude_key.strip(),
                }
                _apply_api_keys(**{
                   # "openai_key": st.session_state.api_keys["openai"],
                    "gemini_key": st.session_state.api_keys["gemini"],
                   # "claude_key": st.session_state.api_keys["claude"],
                })
                st.cache_resource.clear()  # analyzer 재초기화
                st.success("✅ 키 적용 완료")
        with col_b:
            if st.button("키 삭제", use_container_width=True):
                st.session_state.api_keys = {"openai": "", "gemini": "", "claude": ""}
                _apply_api_keys("", "", "")
                st.cache_resource.clear()
                st.warning("🔒 키 제거됨")
        st.caption("※ 키는 이 세션 메모리에만 저장됩니다. 새로고침하면 초기화돼요.")

        st.header("⚙️ 기본 설정")
        teacher_options = {
            "정선생님": "teacher_jung",
            "김선생님": "teacher_kim",
            "이선생님": "teacher_lee",
            "박선생님": "teacher_park",
        }
        selected_teacher = st.selectbox("👨‍🏫 선생님 선택", options=list(teacher_options.keys()))
        teacher_id = teacher_options[selected_teacher]

        model_options = ["gemini", "openai"]
        selected_model = st.selectbox("🤖 AI 모델", options=model_options)

        st.subheader("🎯 분석 옵션")
        detailed_analysis = st.checkbox("상세 분석", value=True)
        include_suggestions = st.checkbox("개선 제안", value=True)

        st.subheader("🧰 추출 옵션")
        do_asr = st.checkbox("오디오 → 텍스트(ASR) 실행", value=True)
        asr_model_size = st.selectbox("ASR 모델 크기", ["tiny", "base", "small", "medium", "large-v3"], index=2)
        do_frames = st.checkbox("프레임 캡쳐 실행", value=True)
        frame_interval = st.slider("프레임 간격(초)", 0.5, 5.0, 2.0, 0.5)

        st.subheader("⚡ 속도")
        fast_mode = st.checkbox("빠른 모드(품질↓, 속도↑)", value=True)
        if fast_mode:
            # 빠른 프리셋: ASR 생략 + 프레임 간격 늘림
            do_asr = False
            frame_interval = max(frame_interval, 3.0)

        with st.expander("🩺 진단 (Diagnostics)"):
            import shutil, sys
            st.write("Python:", sys.version.split()[0])
            st.write("ffmpeg:", "OK" if shutil.which("ffmpeg") else "NOT FOUND")
            for lib in ["opencv_python_headless", "faster_whisper", "google.generativeai"]:
                try:
                    __import__(lib)
                    st.write(f"{lib}: OK")
                except Exception as e:
                    st.write(f"{lib}: ❌ {e.__class__.__name__}")

    # 메인 영역
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 비디오 업로드")
        uploaded_file = st.file_uploader(
            "릴스 비디오를 업로드하세요",
            type=["mp4", "avi", "mov", "mkv"],
            help="최대 100MB, 5분 이내 영상"
        )

        if uploaded_file:
            st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}")
            st.video(uploaded_file)

            if st.button("🚀 분석 시작", type="primary", use_container_width=True, key="btn_analyze"):
                analyze_video(
                    uploaded_file=uploaded_file,
                    teacher_id=teacher_id,
                    model_name=selected_model,
                    detailed_analysis=detailed_analysis,
                    include_suggestions=include_suggestions,
                    do_asr=do_asr,
                    asr_model_size=asr_model_size,
                    do_frames=do_frames,
                    frame_interval=frame_interval,
                    fast_mode=fast_mode,
                )

    with col2:
        st.header("📊 분석 결과")
        if st.session_state.current_analysis:
            display_analysis_results(st.session_state.current_analysis)
        else:
            st.info("👈 비디오를 업로드하고 분석을 시작하세요!")

    # 하단 - 분석 이력
    st.header("📈 분석 이력")
    display_analysis_history()


# =========================
# 동작 루틴
# =========================
def analyze_video(
    uploaded_file,
    teacher_id: str,
    model_name: str,
    detailed_analysis: bool,
    include_suggestions: bool,
    do_asr: bool,
    asr_model_size: str,
    do_frames: bool,
    frame_interval: float,
    fast_mode: bool = False,
):
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # 1) 파일 저장
        status_text.text("📁 파일 저장 중...")
        progress_bar.progress(10)
        file_path = save_uploaded_file(uploaded_file)

        # 2) 캡쳐/ASR 모듈 지연 임포트
        try:
            from capture import capture_frames, asr_transcribe
        except Exception as e:
            st.error(
                "⚠️ 캡쳐 모듈(capture.py) 또는 의존 패키지 로드 실패.\n"
                f"사유: {e}\n\n"
                "필요 패키지: opencv-python-headless, faster-whisper, 그리고 시스템 ffmpeg 또는 imageio-ffmpeg"
            )
            progress_bar.empty()
            status_text.empty()
            return

        # 3) 오디오/프레임 추출
        extracted = {"asr_text": "", "asr_segments": [], "frame_paths": [], "fast_mode": fast_mode}

        # 프레임
        frames_dir = os.path.join("captures", os.path.splitext(os.path.basename(file_path))[0])
        if do_frames:
            status_text.text("🎞 프레임 캡쳐 중...")
            progress_bar.progress(35)
            frame_paths = capture_frames(
                video_path=file_path, interval_sec=frame_interval, out_dir=frames_dir
            )
            extracted["frame_paths"] = frame_paths
        else:
            frame_paths = []

        # ASR
        if do_asr:
            status_text.text("🎵 오디오 → 텍스트(ASR) 중...")
            progress_bar.progress(55)
            try:
                # 새 capture.py(권장) 시그니처
                asr = asr_transcribe(
                    file_path,
                    engine="faster-whisper",
                    model_size=asr_model_size,
                    beam_size=1 if fast_mode else 5,   # 빠른 모드에서 속도↑
                )
            except TypeError:
                # 구 시그니처 호환
                asr = asr_transcribe(file_path, engine="faster-whisper", model_size=asr_model_size)
            extracted["asr_text"] = asr.get("text", "")
            extracted["asr_segments"] = asr.get("segments", [])

        # 4) 분석기 실행
        status_text.text("🤖 AI 분석 중...")
        progress_bar.progress(75)
        analyzer = get_analyzer()

        try:
            # 최신 analyzer는 fast_mode & pre_context 지원
            result = run_sync(
            analyzer.analyze_video(
                file_path=file_path,
                teacher_id=teacher_id,
                model_name=model_name,
                detailed=detailed_analysis,
                include_suggestions=include_suggestions,
                pre_context=extracted,   # extracted 안에 extracted["fast_mode"]=fast_mode 이미 있음
            )
        )
        except TypeError:
            # 구버전 호환(해당 인자 제거)
            result = run_sync(
                analyzer.analyze_video(
                    file_path=file_path,
                    teacher_id=teacher_id,
                    model_name=model_name,
                    detailed=detailed_analysis,
                    include_suggestions=include_suggestions,
                )
            )

        progress_bar.progress(90)
        status_text.text("💾 결과 저장 중...")

        # 추출 결과 병합(화면 표시/콜라주 용)
        if isinstance(result, dict):
            result.setdefault("frame_paths", extracted["frame_paths"])
            result.setdefault("asr_text", extracted["asr_text"])
            result.setdefault("asr_segments", extracted["asr_segments"])

        # 세션 저장 & 이력 기록
        st.session_state.current_analysis = result
        st.session_state.analysis_history.append({
            "timestamp": datetime.now().isoformat(),
            "filename": uploaded_file.name,
            "teacher": teacher_id,
            "model": model_name,
            "score": (result or {}).get("overall_score", 0),
            "frames": extracted["frame_paths"],
        })

        progress_bar.progress(100)
        status_text.text("✅ 분석 완료!")
        time.sleep(0.6)
        st.rerun()

    except Exception as e:
        st.error(f"❌ 분석 중 오류 발생: {str(e)}")
        progress_bar.empty()
        status_text.empty()


# =========================
# 화면 표시
# =========================
def display_analysis_results(result):
    if not isinstance(result, dict):
        st.error("❌ 분석 결과 형식이 올바르지 않습니다.")
        return

    if result.get("error"):
        st.error(f"❌ 분석 실패: {result.get('error_message') or result['error']}")
        return

    # 전체 점수
    overall_score = result.get("overall_score", 0)
    st.metric(
        label="🏆 전체 점수",
        value=f"{overall_score}/10",
        delta=("훌륭해요!" if _is_number(overall_score) and float(overall_score) >= 8
               else "좋아요!" if _is_number(overall_score) and float(overall_score) >= 6
               else "개선 필요")
    )

    # 세부 점수
    raw_scores = result.get("scores", {})
    scores = normalize_scores(raw_scores)
    if scores:
        st.subheader("📊 세부 점수")
        num = len(scores)
        cols_metrics = st.columns(min(4, max(1, num)))
        for i, (category, score) in enumerate(scores.items()):
            with cols_metrics[i % len(cols_metrics)]:
                st.metric(
                    label=str(category).replace('_', ' ').title(),
                    value=f"{score}/10"
                )
    else:
        st.info("세부 점수가 제공되지 않았습니다.")

    # 피드백
    if result.get("final_feedback_text"):
        st.markdown('<div class="feedback-box">', unsafe_allow_html=True)
        st.subheader("💬 선생님 피드백")
        st.write(result["final_feedback_text"])
        st.markdown("</div>", unsafe_allow_html=True)

    if result.get("specific_improvements"):
        st.subheader("🎯 구체적 개선사항")
        for i, improvement in enumerate(result["specific_improvements"], 1):
            st.write(f"{i}. {improvement}")

    if result.get("next_steps"):
        st.subheader("🚀 다음에 시도해보세요")
        for step in result["next_steps"]:
            st.write(f"• {step}")

    # ASR 표시
    with st.expander("🔊 ASR 텍스트 보기"):
        asr_text = result.get("asr_text") or ""
        if asr_text:
            st.code(asr_text[:5000])
        else:
            st.caption("ASR 결과가 없습니다.")

    # 프레임 콜라주
    frames = result.get("frame_paths") or (
        st.session_state.analysis_history[-1].get("frames", [])
        if st.session_state.analysis_history else []
    )
    if frames:
        st.subheader("🎞️ 프레임 콜라주")
        max_n = max(1, min(100, len(frames)))   # 최대 100 또는 전체 수
        min_n = 1 if len(frames) < 20 else 20
        default_n = min(50, max_n)
        if default_n < min_n:
            default_n = min_n
        limit = st.slider(
            "포함할 프레임 수",
            min_value=min_n,
            max_value=max_n,
            value=default_n,
            step=1 if max_n < 20 else 10,
        )
        n_cols = st.slider("열 개수", 3, 8, 5, 1)
        thumb_w = st.slider("썸네일 가로(px)", 160, 480, 280, 20)
        annotate = st.checkbox("파일명 표시", value=False)
        border = st.checkbox("테두리", value=True)

        sampled = sample_evenly(frames, limit)

        if st.button("콜라주 PNG/PDF 생성"):
            png_bytes, pdf_bytes = build_contact_sheet_bytes(
                sampled, cols=n_cols, thumb_width=thumb_w, annotate=annotate, draw_border=border
            )
            st.image(png_bytes, caption="Contact Sheet Preview", use_container_width=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button("⬇️ PNG 다운로드", data=png_bytes, file_name=f"contact_sheet_{ts}.png", mime="image/png")
            st.download_button("⬇️ PDF 다운로드", data=pdf_bytes, file_name=f"contact_sheet_{ts}.pdf", mime="application/pdf")


def display_analysis_history():
    if not st.session_state.analysis_history:
        st.info("아직 분석 이력이 없습니다.")
        return

    recent = st.session_state.analysis_history[-5:]
    for item in reversed(recent):
        with st.expander(f"📹 {item['filename']} (점수: {item['score']}/10)"):
            col1, col2, col3 = st.columns(3)
            col1.write(f"**일시:** {item['timestamp'][:19]}")
            col2.write(f"**선생님:** {item['teacher']}")
            col3.write(f"**모델:** {item['model']}")


# =========================
if __name__ == "__main__":
    main()
