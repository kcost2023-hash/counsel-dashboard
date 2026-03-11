"""
timblo_api.py
─────────────────────────────────────────────────────────────
팀블로(Timblo) AI 회의록 API 클라이언트
PDF 문서: AI 회의록 API 인터페이스 v0.0.7 기준

사용하는 API 엔드포인트
  - POST /external/upload             → 음성 파일 업로드 (STT + AI 회의록 생성 요청)
  - GET  /external/list               → 상담 목록 조회
  - GET  /external/{contentId}        → 상담 상세 조회 (tab=segments,speakerInfo 등)

[설계 원칙]
  화자 역할(상담원/고객)을 코드에서 강제 지정하지 않습니다.
  대신 전체 대화를 중립 레이블로 AI에게 전달하고,
  AI가 발화 패턴을 스스로 분석하여 역할을 판별한 뒤 평가합니다.

주요 함수/메서드
  [독립 함수]
  - format_upload_filename()   파일명을 팀블로 시작시간 인식 형식으로 변환
  - build_conversation_text()  segments → 중립 레이블 대화 텍스트 조합
  - compute_speech_stats()     화자별 발화량·비중 계산 (경청 지표용)
  - build_analysis_prompt()    AI 역할 판별 + 15항목 평가 프롬프트 생성 ★

  [TimbloClient 메서드]
  - upload_content()           녹음 파일 업로드 → STT + AI 회의록 생성 요청
  - get_content_list()         완료된 상담 목록 가져오기
  - get_content_detail()       상담 상세 + segments 원본 가져오기
  - get_transcript_text()      화자명 붙인 단순 텍스트 반환
  - get_conversation()         대화 텍스트 + 발화통계 + AI 분석 프롬프트 한 번에 반환 ★
  - fetch_all_transcripts()    전체 목록 일괄 수집 (대시보드 연동용)
"""

import os
import requests
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ──────────────────────────────────────────────────────────
# 기본 평가 항목 및 기준 (app.py와 동일 — AI 프롬프트 생성에 사용)
# ──────────────────────────────────────────────────────────

DEFAULT_EVAL_WEIGHTS: dict[str, int] = {
    "오프닝": 5, "흐름": 5, "경청": 5, "목적파악": 10,
    "현실조건": 10, "망설임요인": 10, "직무이해": 8, "로드맵": 8,
    "교육과정": 7, "장비가치": 7, "맞춤제안": 10, "반론대응": 5,
    "동기부여": 4, "전환유도": 4, "후속연결": 2,
}

DEFAULT_EVAL_CRITERIA: dict[str, str] = {
    "오프닝":     "밝고 친근하게 인사하며, 상담 목적을 명확히 안내하고 라포를 자연스럽게 형성한다.",
    "흐름":       "상담 전체가 논리적 순서로 진행되며, 주제 간 연결이 자연스럽고 대화가 끊기지 않는다.",
    "경청":       "고객의 말에 집중하고 공감 반응을 보이며, 말을 중간에 끊지 않고 충분히 경청한다.",
    "목적파악":   "고객이 왜 상담을 받으러 왔는지, 진짜 원하는 것이 무엇인지를 구체적으로 파악한다.",
    "현실조건":   "고객의 시간, 비용, 거리, 가족 상황 등 현실적인 제약 조건을 빠짐없이 파악한다.",
    "망설임요인": "고객이 결정을 망설이는 이유를 구체적으로 찾아내고, 이를 논리적으로 해소해준다.",
    "직무이해":   "고객이 희망하는 직무에 대해 정확하고 풍부한 정보(취업률, 급여, 전망 등)를 제공한다.",
    "로드맵":     "수강 완료 후 취업까지의 구체적인 단계와 예상 기간을 시각적으로 설명한다.",
    "교육과정":   "교육 내용, 커리큘럼, 수업 방식, 수료 후 자격증 취득 여부를 알기 쉽게 안내한다.",
    "장비가치":   "교육에 사용하는 장비나 실습 환경의 품질과 현업 수준을 구체적으로 설명한다.",
    "맞춤제안":   "고객의 개인 상황(나이, 경력, 목표)에 딱 맞는 과정을 이유와 근거를 들어 추천한다.",
    "반론대응":   "고객의 부정적 반응이나 반론에 당황하지 않고 침착하게 논리적으로 대응한다.",
    "동기부여":   "고객이 변화를 결심할 수 있도록 성공 사례와 가능성을 들어 동기와 의지를 높인다.",
    "전환유도":   "자연스러운 대화 흐름 속에서 등록, 예약, 다음 단계로 부드럽게 이끈다.",
    "후속연결":   "다음 상담 일정이나 추가 안내를 명확히 약속하며 긍정적으로 마무리한다.",
}

# 업로드 지원 파일 확장자
SUPPORTED_AUDIO_EXTENSIONS = {
    ".flac", ".mp3", ".m4a", ".wav", ".aac",
    ".ogg", ".wma", ".mp4", ".mov", ".webm",
}


# ──────────────────────────────────────────────────────────
# 에러 코드 → 사람이 읽을 수 있는 메시지 매핑 (PDF 문서 기준)
# ──────────────────────────────────────────────────────────
_ERROR_MESSAGES = {
    400: "요청 파라미터 오류 (tab 값 또는 필수 파라미터를 확인하세요)",
    401: "인증 실패 — API 키 또는 이메일을 확인하세요",
    403: "읽기 권한이 없습니다",
    413: "휴지통으로 이동된 콘텐츠입니다. 복원 후 조회해 주세요",
    421: "전사 처리가 실패하여 조회할 수 없는 콘텐츠입니다",
    423: "완전히 삭제되어 조회할 수 없는 콘텐츠입니다",
}


# ──────────────────────────────────────────────────────────
# 데이터 구조 (응답 필드를 타입으로 정의)
# ──────────────────────────────────────────────────────────

@dataclass
class Segment:
    """음성 기록 원본 데이터 1건 (segments 배열의 원소)"""
    segment_id: str
    speaker_id: int
    text: str
    start_time: float   # 초 단위
    end_time: float
    duration: float

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(
            segment_id=str(d.get("segmentId", "")),
            speaker_id=int(d.get("speakerId", 0)),
            text=str(d.get("text", "")),
            start_time=float(d.get("startTime", 0)),
            end_time=float(d.get("endTime", 0)),
            duration=float(d.get("duration", 0)),
        )


@dataclass
class Speaker:
    """참석자(화자) 정보"""
    speaker_id: int
    name: str                       # 기본 이름 (예: "참석자 1")
    display_name: Optional[str]     # 사용자가 수정한 이름

    @classmethod
    def from_dict(cls, d: dict) -> "Speaker":
        return cls(
            speaker_id=int(d.get("speakerId", 0)),
            name=str(d.get("name", "")),
            display_name=d.get("displayName"),
        )

    @property
    def label(self) -> str:
        """표시 이름 우선, 없으면 기본 이름"""
        return self.display_name if self.display_name else self.name


@dataclass
class ContentMeta:
    """회의록 메타 정보"""
    content_id: str
    title: str
    meeting_start_time: Optional[str]
    meeting_end_time: Optional[str]
    record_type: str    # AUDIO / VIDEO / RECORD / CALL

    @classmethod
    def from_dict(cls, d: dict) -> "ContentMeta":
        return cls(
            content_id=str(d.get("contentId", "")),
            title=str(d.get("editedTitle") or d.get("title", "")),
            meeting_start_time=d.get("meetingStartTime"),
            meeting_end_time=d.get("meetingEndTime"),
            record_type=str(d.get("type", "")),
        )


@dataclass
class ContentDetail:
    """상세 조회 결과 전체"""
    meta: ContentMeta
    speakers: list[Speaker]         = field(default_factory=list)
    segments: list[Segment]         = field(default_factory=list)
    merged_segments: list[Segment]  = field(default_factory=list)
    status: str                     = ""


@dataclass
class UploadResult:
    """
    POST /external/upload 성공 응답
    STT 처리는 비동기이므로 status는 초기값 "WAITING" 또는 "ERROR"
    """
    content_id: str
    title: str
    record_type: str
    meeting_start_time: Optional[str]
    meeting_end_time: Optional[str]
    file_id: str
    transcribe_status: str          # "WAITING" | "ERROR"

    @classmethod
    def from_dict(cls, d: dict) -> "UploadResult":
        content   = d.get("content", {})
        transcribe = d.get("transcribe", {})
        return cls(
            content_id         = str(content.get("contentId", "")),
            title              = str(content.get("title", "")),
            record_type        = str(content.get("type", "")),
            meeting_start_time = content.get("meetingStartTime"),
            meeting_end_time   = content.get("meetingEndTime"),
            file_id            = str(transcribe.get("fileId", "")),
            transcribe_status  = str(transcribe.get("status", "WAITING")),
        )


# ──────────────────────────────────────────────────────────
# 예외 클래스
# ──────────────────────────────────────────────────────────

class TimbloAPIError(Exception):
    """API 호출 실패 시 발생하는 예외"""
    def __init__(self, http_code: int, message: str):
        self.http_code = http_code
        friendly = _ERROR_MESSAGES.get(http_code, "")
        detail = f" ({friendly})" if friendly else ""
        super().__init__(f"[{http_code}] {message}{detail}")


# ──────────────────────────────────────────────────────────
# 독립 유틸리티 함수
# ──────────────────────────────────────────────────────────

def format_upload_filename(original_name: str, start_dt: datetime | None = None) -> str:
    """
    팀블로가 회의 시작 시간을 자동 인식하는 파일명 형식으로 변환합니다.

    팀블로 규칙:
      파일명: A.Biz_w_rec_{YYYYMMDD_HHmmss}.{확장자}
      type:   RECORD
      isRecord: TRUE
      → meetingStartTime이 파일명의 타임스탬프로 자동 설정됩니다.

    Parameters
    ----------
    original_name : 원본 파일명 (확장자 포함, 예: "상담_2026-01-15.flac")
    start_dt      : 회의 시작 시각 (None이면 현재 시각 사용)

    Returns
    -------
    str  예시) "A.Biz_w_rec_20260115_140000.flac"
    """
    _, ext = os.path.splitext(original_name)
    if not ext:
        ext = ".bin"
    dt = start_dt or datetime.now()
    ts = dt.strftime("%Y%m%d_%H%M%S")
    return f"A.Biz_w_rec_{ts}{ext}"


# ──────────────────────────────────────────────────────────
# 독립 함수 1: 중립 대화 텍스트 조합
# ──────────────────────────────────────────────────────────

def build_conversation_text(
    detail: "ContentDetail",
    use_merged: bool = True,
) -> dict:
    """
    segments를 시간 순서대로 읽어 중립 레이블('참석자 N')로 대화 텍스트를 조합합니다.
    화자 역할(상담원/고객)은 코드에서 지정하지 않습니다 — AI가 직접 판별합니다.

    Returns
    -------
    dict:
        "text"       : str   "참석자 1: 안녕하세요\\n참석자 2: 네, IT 쪽에..."
        "plain_text" : str   레이블 없이 발화 텍스트만 이어붙인 버전
        "lines"      : list  발화 단위 상세 정보 리스트
    """
    segs = detail.merged_segments if use_merged else detail.segments
    if not segs:
        segs = detail.segments

    name_map: dict[int, str] = {s.speaker_id: s.label for s in detail.speakers}

    lines       = []
    text_parts  = []
    plain_parts = []

    for seg in segs:
        sid   = seg.speaker_id
        label = name_map.get(sid, f"참석자 {sid}")

        lines.append({
            "speaker_id": sid,
            "label":      label,
            "text":       seg.text,
            "start_time": seg.start_time,
            "end_time":   seg.end_time,
            "duration":   seg.duration,
        })
        text_parts.append(f"{label}: {seg.text}")
        plain_parts.append(seg.text)

    return {
        "text":       "\n".join(text_parts),
        "plain_text": " ".join(plain_parts),
        "lines":      lines,
    }


# ──────────────────────────────────────────────────────────
# 독립 함수 2: 화자별 발화량·비중 계산
# ──────────────────────────────────────────────────────────

def compute_speech_stats(
    detail: "ContentDetail",
    use_merged: bool = True,
) -> dict:
    """
    화자별 발화 통계를 계산합니다.
    AI가 '경청' 항목을 평가할 때 참고 데이터로 활용됩니다.

    Returns
    -------
    dict  {speakerId: {...}} 형태, 각 항목:
        "name"              : str
        "utterance_count"   : int
        "total_chars"       : int
        "total_duration"    : float
        "avg_chars"         : float
        "talk_ratio"        : float  (%)
    """
    segs = detail.merged_segments if use_merged else detail.segments
    if not segs:
        segs = detail.segments

    name_map: dict[int, str] = {s.speaker_id: s.label for s in detail.speakers}
    stats: dict[int, dict] = {}

    for seg in segs:
        sid = seg.speaker_id
        if sid not in stats:
            stats[sid] = {
                "name":            name_map.get(sid, f"참석자 {sid}"),
                "utterance_count": 0,
                "total_chars":     0,
                "total_duration":  0.0,
            }
        stats[sid]["utterance_count"] += 1
        stats[sid]["total_chars"]     += len(seg.text)
        stats[sid]["total_duration"]  += seg.duration

    total_duration = sum(v["total_duration"] for v in stats.values()) or 1.0
    for sid in stats:
        cnt = stats[sid]["utterance_count"] or 1
        stats[sid]["avg_chars"]  = round(stats[sid]["total_chars"] / cnt, 1)
        stats[sid]["talk_ratio"] = round(stats[sid]["total_duration"] / total_duration * 100, 1)

    return stats


# ──────────────────────────────────────────────────────────
# 독립 함수 3: AI 역할 판별 + 15항목 평가 프롬프트 생성 ★
# ──────────────────────────────────────────────────────────

def build_analysis_prompt(
    conversation_text: str,
    speech_stats: dict,
    criteria: dict | None = None,
    weights: dict | None = None,
) -> str:
    """
    AI에게 전달할 분석 프롬프트를 생성합니다.

    [동작 원리]
    1단계 — AI가 발화 패턴을 읽고 스스로 상담원/고객 역할을 판별
    2단계 — 상담원의 발언을 집중 분석하여 15개 항목 점수 산출
    ※ 발화 비중 통계를 함께 제공하여 '경청' 항목 자동 반영
    """
    if criteria is None:
        criteria = DEFAULT_EVAL_CRITERIA
    if weights is None:
        weights = DEFAULT_EVAL_WEIGHTS

    stats_lines = []
    for sid, s in speech_stats.items():
        stats_lines.append(
            f"  - {s['name']}: 발화 {s['utterance_count']}회 | "
            f"총 {s['total_chars']}자 | "
            f"평균 {s['avg_chars']}자/발화 | "
            f"발화 비중 {s['talk_ratio']}%"
        )
    stats_block = "\n".join(stats_lines) if stats_lines else "  (통계 없음)"

    criteria_lines = [
        f"- {cat} (배점 {weights.get(cat, 0)}점): {criteria.get(cat, '')}"
        for cat in weights
    ]
    criteria_block = "\n".join(criteria_lines)
    scores_example = ", ".join(f'"{cat}": 0' for cat in weights)

    prompt = f"""당신은 직업훈련 상담 품질 평가 전문가입니다.
아래 상담 대화를 읽고 2단계로 분석을 진행하세요.

{'═' * 54}
[화자별 발화 통계]
{stats_block}

※ 발화 비중이 높을수록 주도적으로 말한 사람입니다.
   반드시 발화 내용의 성격(설명/질문/고민)까지 함께 보고 역할을 최종 판단하세요.
{'═' * 54}

[전체 대화 내용 — 시간 순서]
{conversation_text}

{'═' * 54}
[분석 지시]

▶ 1단계: 역할 판별
아래 기준을 참고해 각 화자가 상담원인지 고객인지 스스로 판단하세요.

  - 상담원: 직업, 자격증, 장비, 교육 과정을 전문적으로 설명하고
           질문을 주도하며 상담 흐름을 이끄는 사람
  - 고객:  주로 질문을 하거나 짧게 답변하며,
           나이·시간·비용 등 자신의 상황과 고민을 이야기하는 사람

▶ 2단계: 상담원 발언 분석 및 15개 항목 평가
역할을 확정한 뒤, 상담원의 발언만 집중 분석하여
아래 항목을 각각 1~5점으로 평가하세요.

[평가 항목 및 기준]
{criteria_block}

⚠️ 경청 및 반응 적절성 자동 반영 규칙:
  ① 상담원의 발화 비중이 75% 이상이거나,
     고객 발화 횟수가 전체의 20% 미만인 경우
     → '경청' 항목을 1~2점으로 평가하고, 개선점에 구체적 사유를 명시하세요.
  ② 고객이 자신의 상황을 충분히 말할 기회를 받지 못한 경우
     → '목적파악', '현실조건', '망설임요인' 항목 점수도 낮게 반영하세요.
  ③ 상담원이 일방적으로 설명만 하고 고객 반응을 확인하지 않는 경우
     → '흐름' 및 '동기부여' 항목에도 패널티를 부여하세요.

{'═' * 54}
[출력 형식]
반드시 아래 JSON 형식만 반환하세요. 앞뒤에 다른 텍스트를 포함하지 마세요.

{{
  "identified_counselor": "참석자 N",
  "identified_customer":  "참석자 N",
  "role_reason": "역할 판별 근거를 1~2문장으로 설명",
  "scores": {{{scores_example}}},
  "total": 0.0,
  "summary": "상담 전반에 대한 요약 (2~3문장)",
  "strengths": "잘한 점을 구체적으로",
  "improvements": "개선이 필요한 점을 구체적으로 (경청 문제 포함)",
  "coaching": "다음 상담을 위한 실천 가능한 코칭 멘트",
  "needs_review": false,
  "excellent": false
}}"""

    return prompt


# ──────────────────────────────────────────────────────────
# 이메일 정제 유틸리티
# ──────────────────────────────────────────────────────────

def _sanitize_email(email: str) -> str:
    """
    이메일 주소에서 흔한 오염 문자를 자동으로 제거합니다.

    처리 항목:
      - 앞뒤 공백 제거 (strip)
      - 'mailto:' 접두어 제거 (링크 복사 시 붙는 경우)
      - 양쪽 꺾쇠 '<...>' 제거 (RFC 5322 형식 복사 시 붙는 경우)
    """
    email = email.strip()
    if email.lower().startswith("mailto:"):
        email = email[len("mailto:"):]
    email = email.strip("<>").strip()
    return email


# ──────────────────────────────────────────────────────────
# 메인 클라이언트 클래스
# ──────────────────────────────────────────────────────────

class TimbloClient:
    """
    팀블로 API 클라이언트

    사용 예시
    ─────────
    client = TimbloClient(
        api_base="https://api.timblo.io",
        api_key="YOUR_API_KEY",
        email="your@email.com",
    )

    # 음성 파일 업로드
    with open("상담.flac", "rb") as f:
        result = client.upload_content(f.read(), "상담.flac")
    print(result.content_id, result.transcribe_status)

    # 완료된 상담 목록 가져오기
    contents = client.get_content_list(status_filter="DONE")
    """

    def __init__(self, api_base: str, api_key: str, email: str, timeout: int = 60):
        """
        Parameters
        ----------
        api_base : 팀블로 API 기본 URL (예: "https://api.timblo.io")
        api_key  : Bearer 인증 키
        email    : 회원 식별용 이메일 (대소문자 구분 없음)
        timeout  : HTTP 요청 타임아웃 (초) — 업로드 시 크게 잡을 것
        """
        self.api_base = api_base.rstrip("/")
        self.email    = _sanitize_email(email)
        self.timeout  = timeout
        self._auth_header = f"Bearer {api_key}"

    # ── 내부 공통 메서드 ──────────────────────────────────

    def _get(self, path: str, params: dict) -> dict:
        """GET 요청 + 공통 에러 처리"""
        url = f"{self.api_base}{path}"
        response = requests.get(
            url,
            headers={
                "Authorization": self._auth_header,
                "Content-Type":  "application/json",
            },
            params=params,
            timeout=self.timeout,
        )
        body      = response.json()
        http_code = body.get("httpCode", response.status_code)

        if http_code != 200:
            raise TimbloAPIError(http_code, body.get("message", "알 수 없는 오류"))

        return body.get("data", {})

    def _post_multipart(
        self,
        path: str,
        params: dict,
        files: dict,
        data: dict,
    ) -> dict:
        """
        POST multipart/form-data 요청 + 공통 에러 처리

        ※ Content-Type 헤더는 requests가 multipart 경계값과 함께 자동 설정합니다.
           Authorization 헤더만 수동으로 지정합니다.
        """
        url = f"{self.api_base}{path}"
        response = requests.post(
            url,
            headers={"Authorization": self._auth_header},
            params=params,
            files=files,
            data=data,
            timeout=self.timeout,
        )
        body      = response.json()
        http_code = body.get("httpCode", response.status_code)

        if http_code != 200:
            raise TimbloAPIError(http_code, body.get("message", "알 수 없는 오류"))

        return body.get("data", {})

    # ── 공개 메서드 ───────────────────────────────────────

    def upload_content(
        self,
        file_bytes: bytes,
        original_filename: str,
        file_type: str = "RECORD",
        lang: str = "ko",
        attendee_num: int | None = None,
        is_record: bool = True,
        summary: str = "large",
        start_dt: datetime | None = None,
    ) -> UploadResult:
        """
        음성 파일을 팀블로에 업로드하여 STT + AI 회의록 생성을 요청합니다.

        업로드 직후 transcribe_status = "WAITING" 상태이며,
        처리 완료까지 수 초~수 분 소요됩니다.
        완료 여부는 get_content_detail() 의 status 필드로 확인합니다.

        파일명은 팀블로 시작시간 인식 형식으로 자동 변환됩니다:
          A.Biz_w_rec_{YYYYMMDD_HHmmss}.{확장자}

        Parameters
        ----------
        file_bytes        : 파일 바이너리 (open(path, "rb").read() 또는 st.file_uploader 결과)
        original_filename : 원본 파일명 (확장자 포함)
        file_type         : "AUDIO" | "VIDEO" | "RECORD" | "CALL" (기본: "RECORD")
        lang              : "ko" | "en" (기본: "ko")
        attendee_num      : 화자 수 — None이면 자동 감지(최대 99)
        is_record         : 녹음 파일 여부 (기본: True)
        summary           : "large" | "medium" (기본: "large")
        start_dt          : 회의 시작 시각 — None이면 현재 시각 사용

        Returns
        -------
        UploadResult  (content_id, title, transcribe_status 등 포함)

        Raises
        ------
        TimbloAPIError  : API 인증 실패, 파라미터 오류 등
        """
        formatted_name = format_upload_filename(original_filename, start_dt)

        params = {"email": self.email}
        files  = {"file": (formatted_name, file_bytes)}
        data   = {
            "type":     file_type,
            "lang":     lang,
            "isRecord": str(is_record).lower(),
            "summary":  summary,
        }
        if attendee_num is not None:
            data["attendeeNum"] = str(attendee_num)

        raw = self._post_multipart(
            "/external/upload",
            params=params,
            files=files,
            data=data,
        )
        return UploadResult.from_dict(raw)

    def get_content_list(self, status_filter: str = "DONE") -> list[dict]:
        """
        완료된 상담(회의록) 목록 조회

        Parameters
        ----------
        status_filter : "DONE" (요약 완료) | "NOTERROR" (진행중 포함)

        Returns
        -------
        list of dict  (contentId, title, meetingStartTime, speakerInfo, ...)
        """
        params = {"email": self.email, "statusFilter": status_filter}
        data   = self._get("/external/list", params)
        return data if isinstance(data, list) else []

    def get_content_detail(
        self,
        content_id: str,
        tabs: list[str] | None = None,
    ) -> ContentDetail:
        """
        상담 상세 데이터 조회

        Parameters
        ----------
        content_id : 회의록 고유 ID (contentId)
        tabs       : 가져올 데이터 탭 목록
                     가능한 값: segments, mergedSegments, speakerInfo,
                                aiResult, summaryTime, bookmarks, file
                     None이면 segments + speakerInfo 기본 조회

        Returns
        -------
        ContentDetail 객체
        """
        if tabs is None:
            tabs = ["segments", "speakerInfo"]

        params = {
            "email": self.email,
            "tab":   ",".join(tabs),
        }
        data = self._get(f"/external/{content_id}", params)

        meta     = ContentMeta.from_dict(data.get("meta", {}))
        speakers = [Speaker.from_dict(s) for s in data.get("speakerInfo", [])]
        segments = [Segment.from_dict(s) for s in data.get("segments", [])]
        merged   = [Segment.from_dict(s) for s in data.get("mergedSegments", [])]

        return ContentDetail(
            meta=meta,
            speakers=speakers,
            segments=segments,
            merged_segments=merged,
            status=data.get("status", ""),
        )

    def get_transcript_text(
        self,
        content_id: str,
        use_merged: bool = True,
        speaker_separator: str = "\n",
    ) -> str:
        """
        상담 텍스트를 '화자명: 발화내용' 형태의 문자열로 반환

        Returns
        -------
        str  예시)
            참석자 1: 안녕하세요, 오늘 상담 목적은...
            참석자 2: 네, 저는 IT 분야에 관심이 있어서요...
        """
        detail      = self.get_content_detail(content_id, tabs=["segments", "speakerInfo"])
        speaker_map = {s.speaker_id: s.label for s in detail.speakers}

        segs = detail.merged_segments if use_merged else detail.segments
        if not segs:
            segs = detail.segments

        lines = []
        for seg in segs:
            speaker_name = speaker_map.get(seg.speaker_id, f"참석자 {seg.speaker_id}")
            lines.append(f"{speaker_name}: {seg.text}")

        return speaker_separator.join(lines)

    def get_conversation(
        self,
        content_id: str,
        use_merged: bool = True,
        criteria: dict | None = None,
        weights: dict | None = None,
    ) -> dict:
        """
        ★ 핵심 기능: segments API를 호출하고 AI 자율 판별 방식으로 분석 프롬프트까지 생성

        Returns
        -------
        dict:
            "content_id"      : str
            "title"           : str
            "start_time"      : str | None
            "text"            : str   중립 대화 텍스트
            "plain_text"      : str   레이블 없는 텍스트
            "lines"           : list  발화 단위 상세 리스트
            "speech_stats"    : dict  화자별 발화량·비중 통계
            "analysis_prompt" : str   LLM에 바로 전송할 수 있는 완성 프롬프트
            "speakers"        : list  [{"id": int, "name": str}, ...]
            "segment_count"   : int
        """
        detail       = self.get_content_detail(content_id, tabs=["segments", "speakerInfo"])
        conv         = build_conversation_text(detail, use_merged)
        speech_stats = compute_speech_stats(detail, use_merged)
        prompt       = build_analysis_prompt(conv["text"], speech_stats, criteria, weights)

        return {
            "content_id":      content_id,
            "title":           detail.meta.title,
            "start_time":      detail.meta.meeting_start_time,
            "text":            conv["text"],
            "plain_text":      conv["plain_text"],
            "lines":           conv["lines"],
            "speech_stats":    speech_stats,
            "analysis_prompt": prompt,
            "speakers":        [{"id": s.speaker_id, "name": s.label}
                                for s in detail.speakers],
            "segment_count":   len(conv["lines"]),
        }

    def fetch_all_transcripts(
        self,
        status_filter: str = "DONE",
        max_count: int = 50,
    ) -> list[dict]:
        """
        완료된 상담 목록을 순서대로 가져와 각각의 transcript를 함께 반환

        Returns
        -------
        list of dict, 각 항목:
            {
                "content_id", "title", "start_time", "end_time", "type",
                "text", "plain_text", "speech_stats", "analysis_prompt",
                "speakers", "segment_count",
                "error": str | None  (실패 시 에러 메시지)
            }
        """
        contents = self.get_content_list(status_filter=status_filter)
        results  = []

        for item in contents[:max_count]:
            content_id = item.get("contentId", "")
            title      = item.get("editedTitle") or item.get("title", "")

            try:
                detail       = self.get_content_detail(content_id, tabs=["segments", "speakerInfo"])
                conv         = build_conversation_text(detail)
                speech_stats = compute_speech_stats(detail)
                prompt       = build_analysis_prompt(conv["text"], speech_stats)

                results.append({
                    "content_id":      content_id,
                    "title":           title,
                    "start_time":      detail.meta.meeting_start_time,
                    "end_time":        detail.meta.meeting_end_time,
                    "type":            detail.meta.record_type,
                    "text":            conv["text"],
                    "plain_text":      conv["plain_text"],
                    "speech_stats":    speech_stats,
                    "analysis_prompt": prompt,
                    "speakers":        [{"id": s.speaker_id, "name": s.label}
                                        for s in detail.speakers],
                    "segment_count":   len(conv["lines"]),
                    "error":           None,
                })

            except TimbloAPIError as e:
                results.append({
                    "content_id":      content_id,
                    "title":           title,
                    "start_time":      item.get("meetingStartTime"),
                    "end_time":        item.get("meetingEndTime"),
                    "type":            item.get("type", ""),
                    "text":            "",
                    "plain_text":      "",
                    "speech_stats":    {},
                    "analysis_prompt": "",
                    "speakers":        [],
                    "segment_count":   0,
                    "error":           str(e),
                })

        return results


# ──────────────────────────────────────────────────────────
# 연결 테스트 유틸리티
# ──────────────────────────────────────────────────────────

def test_connection(api_base: str, api_key: str, email: str) -> dict:
    """
    API 연결 상태를 간단히 확인하고 결과를 반환

    Returns
    -------
    {"ok": bool, "count": int, "message": str}
    """
    try:
        client   = TimbloClient(api_base=api_base, api_key=api_key, email=email)
        contents = client.get_content_list(status_filter="DONE")
        return {
            "ok":      True,
            "count":   len(contents),
            "message": f"연결 성공 — 완료된 상담 {len(contents)}건 조회됨",
        }
    except TimbloAPIError as e:
        return {"ok": False, "count": 0, "message": str(e)}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "count": 0,
                "message": "서버에 연결할 수 없습니다. TIMBLO_API_BASE URL을 확인하세요."}
    except requests.exceptions.Timeout:
        return {"ok": False, "count": 0, "message": "요청 시간이 초과됐습니다."}
    except Exception as e:
        return {"ok": False, "count": 0, "message": f"예상치 못한 오류: {e}"}


# ──────────────────────────────────────────────────────────
# 직접 실행 시 간단한 동작 확인
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    API_BASE = "https://your-timblo-api-base-url.com"
    API_KEY  = "your_api_key_here"
    EMAIL    = "your@email.com"

    print("=" * 50)
    print("팀블로 API 연결 테스트")
    print("=" * 50)

    result = test_connection(API_BASE, API_KEY, EMAIL)
    status_icon = "✅" if result["ok"] else "❌"
    print(f"{status_icon} {result['message']}")

    if result["ok"] and result["count"] > 0:
        client   = TimbloClient(API_BASE, API_KEY, EMAIL)
        contents = client.get_content_list()
        first_id = contents[0]["contentId"]

        print(f"\n첫 번째 상담 분석 미리보기 (contentId: {first_id})")
        print("-" * 54)

        conv = client.get_conversation(first_id)

        print(f"제목         : {conv['title']}")
        print(f"시작 시각    : {conv['start_time']}")
        print(f"총 발화 수   : {conv['segment_count']}개")
        print(f"화자 목록    : {[s['name'] for s in conv['speakers']]}")
        print()

        print("[화자별 발화 통계]")
        for sid, s in conv["speech_stats"].items():
            print(f"  {s['name']}: {s['utterance_count']}회 발화 | "
                  f"발화 비중 {s['talk_ratio']}%")
        print()

        print("[중립 대화 텍스트 (처음 500자)]")
        print(conv["text"][:500] + ("..." if len(conv["text"]) > 500 else ""))
        print()

        print("[AI 분석 프롬프트 (처음 800자) — LLM에 그대로 전송 가능]")
        print(conv["analysis_prompt"][:800] + "...")
