"""
⚙️ 설정 관리 (Streamlit용, Google 연동 제거 버전)
- .env 값이 없어도 앱이 동작하도록 '관대한' 기본값을 둡니다.
- 디렉토리 자동 생성, 선택적 유효성 점검 지원.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """앱 전역 설정"""

    # ===== API Keys (옵션) =====
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")   # 없으면 사이드바에서 모델 선택만, 키는 사용 안 함
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")

    # ===== 파일/디렉토리 =====
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
    CAPTURE_ROOT = os.getenv("CAPTURE_ROOT", "captures")
    ALLOWED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(100 * 1024 * 1024)))  # 100MB

    # ===== 비디오/프레임 캡쳐 기본 =====
    FRAME_INTERVAL_SEC = float(os.getenv("FRAME_INTERVAL_SEC", "2.0"))
    FRAME_RESIZE_WIDTH = int(os.getenv("FRAME_RESIZE_WIDTH", "720"))  # capture.py에서 사용할 수 있음
    MAX_FRAMES_FOR_ANALYSIS = int(os.getenv("MAX_FRAMES_FOR_ANALYSIS", "80"))
    FRAME_QUALITY = int(os.getenv("FRAME_QUALITY", "85"))  # (선택) JPEG 저장 품질

    # ===== ASR 기본 =====
    ASR_ENGINE = os.getenv("ASR_ENGINE", "faster-whisper")
    ASR_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "small")
    ASR_DEVICE = os.getenv("ASR_DEVICE", "cpu")           # "cpu" | "cuda"
    ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8")  # "default" | "int8" | "float16" 등

    # ===== OCR 기본 =====
    OCR_LANG = os.getenv("OCR_LANG", "kor+eng")
    OCR_SAMPLES = int(os.getenv("OCR_SAMPLES", "20"))

    # ===== 분석 기본 =====
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai")  # UI에서 덮어쓸 수 있음
    ANALYSIS_TIMEOUT = int(os.getenv("ANALYSIS_TIMEOUT", "120"))

    # 엄격 검증 여부(선택): "1"이면 강제 검증
    STRICT_VALIDATION = os.getenv("STRICT_VALIDATION", "0") == "1"

    @classmethod
    def ensure_dirs(cls):
        """필요한 디렉토리 생성"""
        os.makedirs(cls.UPLOAD_DIR, exist_ok=True)
        os.makedirs(cls.CAPTURE_ROOT, exist_ok=True)

    @classmethod
    def validate(cls):
        """
        가벼운 유효성 점검.
        - 기본적으로 에러를 일으키지 않음(STRICT_VALIDATION=1일 때만 엄격)
        """
        errors = []

        # 확장자 목록 체크
        if not cls.ALLOWED_VIDEO_FORMATS:
            errors.append("ALLOWED_VIDEO_FORMATS가 비어 있습니다.")

        # 엄격 모드에서만 키 검증
        if cls.STRICT_VALIDATION:
            if cls.DEFAULT_MODEL == "openai" and not cls.OPENAI_API_KEY:
                errors.append("STRICT 모드: OPENAI_API_KEY가 필요합니다.")
            if cls.DEFAULT_MODEL == "gemini" and not cls.GEMINI_API_KEY:
                errors.append("STRICT 모드: GEMINI_API_KEY가 필요합니다.")
            if cls.DEFAULT_MODEL == "claude" and not cls.CLAUDE_API_KEY:
                errors.append("STRICT 모드: CLAUDE_API_KEY가 필요합니다.")

        # 디렉토리 생성 시도
        cls.ensure_dirs()

        if errors:
            raise ValueError("설정 오류:\n" + "\n".join(errors))

    @classmethod
    def get_available_models(cls):
        """
        사용 가능한 모델 후보를 반환.
        키가 없어도 UI에서 선택은 가능하므로, 여기서는 단순 목록만.
        """
        models = []
        # 키가 있으면 우선 노출
        if cls.OPENAI_API_KEY:
            models.append("openai")
        if cls.GEMINI_API_KEY:
            models.append("gemini")
        if cls.CLAUDE_API_KEY:
            models.append("claude")

        # 키가 없어도 선택 가능하도록 보조로 채워줌(중복 방지)
        for m in ["openai", "gemini", "claude"]:
            if m not in models:
                models.append(m)
        return models


# 전역 설정 사용 예
config = Config()

try:
    config.validate()
    print("✅ 설정 검증 완료 (STRICT:", Config.STRICT_VALIDATION, ")")
except ValueError as e:
    # STRICT 모드에서만 주로 발생. 기본 모드에서는 대부분 통과.
    print(f"❌ 설정 오류: {e}")
