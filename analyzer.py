"""
🤖 릴스 분석 엔진 (Streamlit-독립형)
- Gemini(OpenAI 선택적)로 프레임 이미지+컨텍스트를 분석
- fast_mode 지원: 프레임 1장, 다운스케일, 토큰 축소로 속도 개선
"""

import os
import io
import re
import json
import base64
from datetime import datetime
from typing import Dict, List, Optional

import cv2
from PIL import Image

# ===== Optional: Gemini / OpenAI =====
_HAS_GEMINI = False
try:
    import google.generativeai as genai
    _HAS_GEMINI = True
except Exception:
    genai = None

try:
    import openai  # 현재는 더미 경로만
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False


class ReelAnalyzer:
    """릴스(짧은 영상) 분석기"""

    def __init__(self):
        self.models: Dict[str, object] = {}
        self._init_models()

    # ---------- 모델 초기화 ----------
    def _init_models(self):
        """API 키가 있는 모델만 초기화"""
        if _HAS_GEMINI and os.getenv("GEMINI_API_KEY"):
            try:
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                # 빠른 응답 위주 모델
                self.models["gemini"] = genai.GenerativeModel("gemini-1.5-flash")
            except Exception:
                pass

        # 필요 시 확장
        if _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
            # openai.api_key = os.getenv("OPENAI_API_KEY")
            self.models["openai"] = "gpt-4o-mini"  # placeholder (아래 _analyze_with_openai에서 더미 응답)

    # ---------- 퍼블릭 API ----------
    async def analyze_video(
        self,
        file_path: str,
        teacher_id: str,
        model_name: str = "gemini",
        detailed: bool = True,
        include_suggestions: bool = True,
        pre_context: Optional[dict] = None,
        fast_mode: bool = False,
    ) -> Dict:
        """
        비디오 분석 메인 함수
        """
        try:
            # 1) 비디오 전처리 (프레임 추출/압축)
            video_data = self._process_video(file_path, fast_mode=fast_mode)

            # 2) 선생님 프로필(톤/스타일)
            teacher_data = self._get_teacher_profile(teacher_id)

            # 3) 분석 프롬프트 구성(ASR 텍스트 등 pre_context 포함)
            prompt = self._create_analysis_prompt(
                video_data=video_data,
                teacher_data=teacher_data,
                detailed=detailed,
                include_suggestions=include_suggestions,
                pre_context=pre_context or {},
            )

            # 4) 모델 호출
            raw = await self._analyze_with_ai(
                prompt=prompt,
                frames=video_data["frames"],
                model_name=model_name,
                fast_mode=fast_mode,
            )

            # 5) 결과 가공
            result = self._process_analysis_result(
                raw_result=raw,
                video_data=video_data,
                teacher_id=teacher_id,
                model_name=model_name,
            )
            return result

        except Exception as e:
            return {
                "error": True,
                "error_message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    # ---------- 비디오 처리 ----------
    def _process_video(self, file_path: str, fast_mode: bool = False) -> Dict:
        """
        OpenCV로 비디오를 열고 프레임을 일정 간격으로 추출하여 JPEG base64로 담는다.
        fast_mode: 프레임 1장, 640px, JPEG 70 / 기본: 프레임 3장, 960px, JPEG 85
        """
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError("비디오 파일을 열 수 없습니다")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (frame_count / fps) if fps > 0 else 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        frames: List[Dict] = []
        max_frames = 1 if fast_mode else 3          # ✅ 속도 우선
        step = max(1, int(fps)) if fps > 0 else 1   # 초당 1장
        end = min(frame_count, max_frames * step)

        target_w = 640 if fast_mode else 960        # ✅ 다운스케일
        jpeg_q = 70 if fast_mode else 85            # ✅ 용량 축소

        for i in range(0, end, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            # 다운스케일
            if frame.shape[1] > target_w:
                h = int(frame.shape[0] * (target_w / frame.shape[1]))
                frame = cv2.resize(frame, (target_w, h), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_q)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            frames.append({
                "timestamp": float(i / fps) if fps > 0 else 0.0,
                "image_base64": b64
            })
            if len(frames) >= max_frames:
                break

        cap.release()

        return {
            "filename": os.path.basename(file_path),
            "metadata": {
                "duration": round(duration, 2),
                "fps": fps,
                "resolution": f"{width}x{height}",
                "is_vertical": bool(height > width),
                "frame_count": frame_count,
            },
            "frames": frames,
            # 아래 두 키는 app에서 setdefault로 합쳐줌(여기선 빈 값만 유지)
            "audio_data": {"has_audio": True},
            "text_data": {},
        }

    # ---------- 교사 프로필 ----------
    def _get_teacher_profile(self, teacher_id: str) -> Dict:
        """
        간단한 하드코딩 프로필 (DB 제거 버전).
        필요 시 확장/외부화 가능.
        """
        profiles = {
            "teacher_jung": {
                "name": "정선생님",
                "teaching_style": "친근하지만 핵심을 콕 짚어주는",
                "tone": "따뜻하고 실용적인 말투로, 명확한 근거와 함께 제안합니다."
            },
            "teacher_kim": {
                "name": "김선생님",
                "teaching_style": "분석적이고 데이터 지향적인",
                "tone": "냉철하고 깔끔한 말투로, 수치와 사례를 듭니다."
            },
            "teacher_lee": {
                "name": "이선생님",
                "teaching_style": "유머러스하지만 할 말은 하는",
                "tone": "가볍게 시작해도 끝은 뾰족하게 정리합니다."
            },
            "teacher_park": {
                "name": "박선생님",
                "teaching_style": "감성적이고 스토리텔링 중심의",
                "tone": "정서적 공감을 전제로, 스토리 흐름을 강조합니다."
            },
        }
        return profiles.get(teacher_id, {
            "name": "선생님",
            "teaching_style": "친근한",
            "tone": "부드럽지만 명료한 말투로 조언합니다."
        })

    # ---------- 프롬프트 ----------
    def _create_analysis_prompt(
        self,
        video_data: Dict,
        teacher_data: Dict,
        detailed: bool,
        include_suggestions: bool,
        pre_context: Dict,
    ) -> str:
        teacher_name = teacher_data.get("name", "선생님")
        teaching_style = teacher_data.get("teaching_style", "친근한")
        teacher_tone = teacher_data.get("tone", "명확하고 친절한 말투")

        meta = video_data.get("metadata", {})
        frames = video_data.get("frames", [])

        asr_text = (pre_context or {}).get("asr_text", "")
        fast_mode = (pre_context or {}).get("fast_mode", False)

        # 요청 JSON 스펙(모델이 그대로 내도록 강제)
        json_spec = """
{
  "scores": {
    "visual_appeal": <1-10 number>,
    "content_structure": <1-10 number>,
    "trend_fit": <1-10 number>,
    "emotional_impact": <1-10 number>,
    "viral_potential": <1-10 number>
  },
  "main_feedback": "<3-5문장 핵심 피드백>",
  "specific_improvements": ["<구체적 개선1>", "<구체적 개선2>", "<구체적 개선3>"],
  "encouragement": "<격려 한 문단>",
  "next_steps": ["<다음에 시도1>", "<다음에 시도2>"]
}
""".strip()

        prompt = f"""
당신은 '{teacher_name}'입니다. 말투는 다음을 따릅니다: {teacher_tone}
지도 스타일: {teaching_style}

# 영상 메타데이터
- 파일명: {video_data.get('filename')}
- 길이(초): {meta.get('duration')}
- FPS: {meta.get('fps')}
- 해상도: {meta.get('resolution')}
- 세로영상: {'예' if meta.get('is_vertical') else '아니오'}
- 샘플 프레임 수: {len(frames)}

# 분석 요청
이 릴스(Short-form) 영상을 평가하고, {teacher_name}의 말투/스타일로 피드백을 작성하세요.
평가 기준:
1) 시각적 매력(Visual Appeal)
2) 컨텐츠 구성(Content Structure)
3) 트렌드 적합성(Trend Fit)
4) 감정적 임팩트(Emotional Impact)
5) 바이럴 가능성(Viral Potential)

{('※ 빠른모드: 간결하게 요점만 제시하세요.' if fast_mode else '')}
{('※ 개선 제안을 반드시 포함하세요.' if include_suggestions else '※ 개선 제안은 생략해도 됩니다.')}

# 음성 텍스트(있다면 요약에 참고):
{asr_text[:1200] if asr_text else '(없음)'}

# 출력 형식(반드시 유효한 JSON으로만 응답):
{json_spec}
""".strip()

        return prompt

    # ---------- 모델 디스패치 ----------
    async def _analyze_with_ai(
        self,
        prompt: str,
        frames: List[Dict],
        model_name: str,
        fast_mode: bool = False,
    ) -> str:
        model_name = (model_name or "").lower()
        if model_name == "gemini":
            return await self._analyze_with_gemini(prompt, frames, fast_mode=fast_mode)
        elif model_name == "openai":
            return await self._analyze_with_openai(prompt, frames, fast_mode=fast_mode)
        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")

    # ---------- Gemini ----------
    async def _analyze_with_gemini(
        self,
        prompt: str,
        frames: List[Dict],
        fast_mode: bool = False,
    ) -> str:
        if "gemini" not in self.models:
            raise RuntimeError("Gemini 모델이 초기화되지 않았습니다. GEMINI_API_KEY를 확인하세요.")

        model = self.models["gemini"]
        parts: List[object] = [prompt]

        # 빠른모드: 1장, 아니면 최대 3장
        use_n = 1 if fast_mode else min(3, len(frames))
        for f in frames[:use_n]:
            try:
                img_bytes = base64.b64decode(f["image_base64"])
                # PIL 이미지를 넘겨도 되고 inline_data dict로 넘겨도 됨
                parts.append({"mime_type": "image/jpeg", "data": img_bytes})
            except Exception:
                continue

        # 토큰/온도 축소
        cfg = None
        if _HAS_GEMINI:
            cfg = genai.types.GenerationConfig(
                temperature=0.3 if fast_mode else 0.7,
                max_output_tokens=250 if fast_mode else 900,
                response_mime_type="application/json",
            )

        resp = model.generate_content(parts, generation_config=cfg)
        # 대부분 JSON 문자열을 반환(응답이 비어있을 대비)
        return (resp.text or "").strip()

    # ---------- OpenAI (더미) ----------
    async def _analyze_with_openai(
        self,
        prompt: str,
        frames: List[Dict],
        fast_mode: bool = False,
    ) -> str:
        """
        필요 시 구현. 현재는 JSON 더미를 반환하여 전체 플로우를 유지.
        """
        dummy = {
            "scores": {
                "visual_appeal": 7,
                "content_structure": 7,
                "trend_fit": 6,
                "emotional_impact": 7,
                "viral_potential": 6,
            },
            "main_feedback": "OpenAI 더미 응답: 전반적으로 구조가 명확하고 시각적 임팩트가 준수합니다. "
                             "초반 2초의 후킹을 더 강하게 만들고, 자막 대비를 높이면 도달률 향상을 기대할 수 있습니다.",
            "specific_improvements": ["인트로 1.0→0.6초", "CTA 자막 대비 20%↑", "BGM 드롭으로 전환 강조"],
            "encouragement": "좋은 시도입니다. 작은 개선이 누적되면 도달·완시율이 올라갑니다!",
            "next_steps": ["후킹 3안 A/B 테스트", "씬 전환 리듬 다양화"],
        }
        return json.dumps(dummy, ensure_ascii=False)

    # ---------- 결과 가공 ----------
    def _process_analysis_result(
        self,
        raw_result: str,
        video_data: Dict,
        teacher_id: str,
        model_name: str,
    ) -> Dict:
        """
        모델 원문(raw_result)을 JSON으로 파싱하고, 앱에서 쓰기 좋은 형태로 변환
        """
        try:
            data = None

            # 1) 우선 JSON으로 직파싱
            try:
                data = json.loads(raw_result)
            except Exception:
                pass

            # 2) 실패하면 문자열 안의 JSON blob 추출
            if data is None:
                m = re.search(r'\{[\s\S]*\}', raw_result)
                if m:
                    try:
                        data = json.loads(m.group(0))
                    except Exception:
                        pass

            # 3) 그래도 실패 → 최소 구조 확보
            if data is None or not isinstance(data, dict):
                data = {
                    "scores": {
                        "visual_appeal": 7,
                        "content_structure": 7,
                        "trend_fit": 6,
                        "emotional_impact": 7,
                        "viral_potential": 6,
                    },
                    "main_feedback": (raw_result[:400] if isinstance(raw_result, str) else "분석 결과(원문) 없음"),
                    "specific_improvements": ["JSON 파싱 실패로 기본 제안 제공"],
                    "encouragement": "다시 시도해보면 더 정교한 응답을 받을 수 있어요.",
                    "next_steps": ["인트로 후킹 강화", "CTA 자막 대비 조정"],
                }

            scores = data.get("scores", {})
            overall = self._calculate_overall_score(scores)

            # 최종 피드백 텍스트(섹션 합치기)
            final_feedback = self._generate_final_feedback(data)

            # 앱에서 바로 쓰도록 구성
            result = {
                "analysis_id": f"analysis_{int(datetime.now().timestamp())}",
                "video_id": f"video_{int(datetime.now().timestamp())}",
                "teacher_id": teacher_id,
                "model_name": model_name,
                "scores": scores,  # dict 그대로 (app.py의 normalize_scores가 처리)
                "final_feedback_text": final_feedback,
                "overall_score": overall,
                "specific_improvements": data.get("specific_improvements", []),
                "next_steps": data.get("next_steps", []),
                "timestamp": datetime.now().isoformat(),
                "raw_response": (raw_result or "")[:2000],
                # 표시용 원문 필드(선택)
                "main_feedback": data.get("main_feedback", ""),
                "encouragement": data.get("encouragement", ""),
            }
            return result

        except Exception as e:
            return {
                "analysis_id": f"error_{int(datetime.now().timestamp())}",
                "error": True,
                "error_message": f"결과 처리 오류: {str(e)}",
                "raw_response": raw_result,
                "timestamp": datetime.now().isoformat(),
            }

    def _generate_final_feedback(self, data: Dict) -> str:
        """피드백/개선/다음단계를 하나의 텍스트로 합치기"""
        parts = []
        if data.get("main_feedback"):
            parts.append(data["main_feedback"])

        if data.get("specific_improvements"):
            parts.append("\n📍 구체적 개선사항:")
            for i, imp in enumerate(data["specific_improvements"], 1):
                parts.append(f"{i}. {imp}")

        if data.get("next_steps"):
            parts.append("\n🚀 다음에 시도해보세요:")
            for step in data["next_steps"]:
                parts.append(f"• {step}")

        if data.get("encouragement"):
            parts.append(f"\n💪 {data['encouragement']}")

        return "\n".join(parts).strip()

    def _calculate_overall_score(self, scores: Dict) -> float:
        """숫자 점수 평균"""
        if not scores or not isinstance(scores, dict):
            return 7.0
        vals = []
        for v in scores.values():
            try:
                vals.append(float(v))
            except Exception:
                continue
        if not vals:
            return 7.0
        return round(sum(vals) / len(vals), 1)
