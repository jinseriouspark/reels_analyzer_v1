# capture.py
import os
import subprocess
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

# 음성 인식
try:
    from faster_whisper import WhisperModel
    _HAS_FWHISPER = True
except Exception:
    _HAS_FWHISPER = False

# OCR
import pytesseract


# ===============================
# 🎞 프레임 캡쳐
# ===============================
def capture_frames(
    video_path: str,
    interval_sec: float = 2.0,
    out_dir: str = "frames",
    limit: Optional[int] = None,
    resize_width: Optional[int] = None,
    jpeg_quality: int = 85,
) -> List[str]:
    """
    일정 간격(초)으로 프레임 캡쳐 (경과시간 기반, float 지원)
    - interval_sec: 0.5 같은 소수 가능
    - limit: 최대 저장 프레임 수(메모리/디스크 보호)
    - resize_width: 저장 전에 가로 리사이즈(px). None이면 원본 크기.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    saved: List[str] = []
    next_capture_t = 0.0  # 다음 캡쳐 목표 시간(초)
    idx = 0

    # 읽으면서 경과시간으로 샘플
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 현재 프레임의 시간(초)
        t = (idx / fps) if fps > 0 else (len(saved) * interval_sec)

        if t + 1e-6 >= next_capture_t:
            # 리사이즈(선택)
            if resize_width and frame.shape[1] > resize_width:
                h = int(frame.shape[0] * (resize_width / frame.shape[1]))
                frame = cv2.resize(frame, (resize_width, h), interpolation=cv2.INTER_AREA)

            # 저장
            fname = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
            cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            saved.append(fname)

            # 다음 캡쳐 시간 갱신
            next_capture_t += float(interval_sec)

            # limit 초과 방지
            if limit and len(saved) >= limit:
                break

        idx += 1

    cap.release()
    return saved


# ===============================
# 🎵 오디오 추출 & ASR
# ===============================
def _check_ffmpeg_available() -> bool:
    try:
        p = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return p.returncode == 0
    except Exception:
        return False

def extract_audio(video_path: str, audio_path: str = "temp_audio.wav") -> str:
    try:
        import imageio_ffmpeg
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()  # ✅ 바이너리 경로 확보
    except Exception:
        ffmpeg_bin = "ffmpeg"  # 로컬/환경에 설치되어 있으면 PATH 사용

    cmd = [
        ffmpeg_bin, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg 오디오 추출 실패: {p.stderr.decode(errors='ignore')[:300]}")
    return audio_path

def asr_transcribe(
    video_path: str,
    engine: str = "faster-whisper",
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
    keep_audio: bool = False,
) -> Dict[str, Any]:
    """
    영상에서 음성 인식 (ASR)
    - engine: "faster-whisper"만 지원
    - model_size: "tiny"|"base"|"small"|"medium"|"large-v3" 등
    - device: "cpu"|"cuda"
    - compute_type: "int8"|"float16"|"default" 등
    """
    audio_path = extract_audio(video_path)

    try:
        if engine != "faster-whisper":
            raise ValueError(f"Unsupported ASR engine: {engine}")
        if not _HAS_FWHISPER:
            raise RuntimeError("faster-whisper 모듈이 설치되어 있지 않습니다. pip install faster-whisper")

        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, info = model.transcribe(audio_path, beam_size=beam_size)
        texts = []
        seg_list = []
        for seg in segments:
            texts.append(seg.text)
            seg_list.append({"start": seg.start, "end": seg.end, "text": seg.text})

        return {
            "language": getattr(info, "language", ""),
            "text": " ".join(texts).strip(),
            "segments": seg_list,
        }
    finally:
        # 임시 오디오 파일 정리
        if not keep_audio and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass


# ===============================
# 📝 OCR (상/중/하 영역, 전처리 포함)
# ===============================
def _preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """OCR 전처리: gray→CLAHE→median blur→adaptive thres→morph open"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 대비 향상
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # 노이즈 감소
    gray = cv2.medianBlur(gray, 3)
    # 가변 임계
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 21, 10)
    # 얇은 노이즈 제거
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    return th

def _read_image_any(img_path: str) -> Optional[np.ndarray]:
    try:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(img_path)  # fallback
        return img
    except Exception:
        return None

def ocr_images_multi_band(
    image_paths: List[str],
    lang: str = "kor+eng",
    psm: int = 6,   # 6: Assume a single uniform block of text
    oem: int = 3,   # 3: Default, based on what is available
    scale_up: float = 1.5,   # 가독성을 위해 확대
) -> Dict[str, List[str]]:
    """
    프레임 이미지에서 OCR 텍스트 추출 (상/중/하 3분할)
    - lang: "kor+eng" 권장
    - psm/oem: tesseract config
    - scale_up: 전처리 이미지 확대 비율
    """
    results = {"top": [], "middle": [], "bottom": []}
    config = f"--oem {int(oem)} --psm {int(psm)}"

    for img_path in image_paths:
        img_bgr = _read_image_any(img_path)
        if img_bgr is None:
            for k in results.keys():
                results[k].append("")
            continue

        h, w = img_bgr.shape[:2]
        bands = {
            "top":    img_bgr[0:int(h/3), :],
            "middle": img_bgr[int(h/3):int(2*h/3), :],
            "bottom": img_bgr[int(2*h/3):, :],
        }

        for band_name, band_bgr in bands.items():
            proc = _preprocess_for_ocr(band_bgr)
            if scale_up and scale_up != 1.0:
                proc = cv2.resize(proc, (int(proc.shape[1]*scale_up), int(proc.shape[0]*scale_up)),
                                  interpolation=cv2.INTER_CUBIC)
            # pytesseract는 RGB/그레이 둘 다 가능하나, 여기선 단일 채널 이미지 전달
            try:
                txt = pytesseract.image_to_string(proc, lang=lang, config=config)
            except Exception as e:
                txt = ""
            results[band_name].append((txt or "").strip())

    return results
