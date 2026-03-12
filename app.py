import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import json
import os
import re
import time

_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
except ImportError:
    pass

from timblo_api import (
    TimbloClient, TimbloAPIError,
    build_conversation_text, compute_speech_stats,
)

# ══════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════
APP_PASSWORD = "timblo2024"
PAGE_SIZE    = 20

QUALITY_WEIGHTS = {
    "니즈파악": 20, "설명력": 15, "설득력": 20,
    "공감":     20, "반론대응": 15, "마무리": 10,
}
FLOW_STAGES   = ["고객상황파악", "문제공감", "직업설명", "수익구조설명", "사례제시", "행동유도"]
QUESTION_CATS = ["공부기간", "시험난이도", "수익가능성", "가격", "취업가능성", "기타"]

# 채널 탭 정의
CHANNEL_TABS = ["방문 상담", "비대면 상담", "전화 상담", "공통/기타"]
CHANNEL_KEYS = ["방문",     "비대면",      "전화",     "공통"]

# 기본 루브릭 (채널별 초기값 공용)
_DEFAULT_RUBRIC = [
    {"항목": "니즈파악",  "배점": 20, "기준": "고객의 상황·목표·제약을 정확히 파악했는가"},
    {"항목": "설명력",    "배점": 15, "기준": "훈련과정과 혜택을 명확하고 이해하기 쉽게 설명했는가"},
    {"항목": "설득력",    "배점": 20, "기준": "고객의 참여 동기를 효과적으로 이끌어냈는가"},
    {"항목": "공감",      "배점": 20, "기준": "고객의 감정과 상황에 충분히 공감했는가"},
    {"항목": "반론대응",  "배점": 15, "기준": "고객의 우려와 반론을 논리적으로 해소했는가"},
    {"항목": "마무리",    "배점": 10, "기준": "다음 단계를 명확히 안내하고 마무리했는가"},
]

# 사용자 커스텀 설정 기본값
DEFAULT_USER_CONFIG = {
    "quality_weights": dict(QUALITY_WEIGHTS),   # 하위 호환용 (레거시)
    "channel_rubrics": {k: [dict(r) for r in _DEFAULT_RUBRIC] for k in CHANNEL_KEYS},
    "flow_stages":     list(FLOW_STAGES),
    "system_prompt": (
        "당신은 직업훈련 상담 품질 평가 전문가입니다.\n"
        "상담사가 고객의 직업훈련 참여 결심을 돕는 과정을 면밀히 분석하세요.\n"
        "고객의 상황(나이, 경력, 목표)을 정확히 파악하고, 상담사가 맞춤형 솔루션을 제공했는지 평가하세요."
    ),
    "mandatory_items": [
        "훈련과정 환급 안내",
        "수강료 및 지원금 안내",
        "훈련 일정 및 수료 조건 안내",
    ],
}

# 설정을 로컬 파일에 저장/로드
_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "user_config.json")

def load_user_config() -> dict:
    try:
        if os.path.exists(_CONFIG_FILE):
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            cfg = dict(DEFAULT_USER_CONFIG)
            cfg.update(saved)
            # 레거시: channel_rubrics 없으면 quality_weights에서 생성
            if "channel_rubrics" not in saved:
                qw = saved.get("quality_weights", dict(QUALITY_WEIGHTS))
                fallback = [{"항목": k, "배점": v, "기준": ""} for k, v in qw.items()]
                cfg["channel_rubrics"] = {k: [dict(r) for r in fallback] for k in CHANNEL_KEYS}
            return cfg
    except Exception:
        pass
    return dict(DEFAULT_USER_CONFIG)

def save_user_config(cfg: dict):
    try:
        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ── 분석 결과 영구 저장 ──
_RESULTS_FILE = os.path.join(os.path.dirname(__file__), "analysis_results.json")

def load_analysis_results() -> dict:
    """앱 시작 시 로컬 파일에서 분석 결과를 로드합니다."""
    try:
        if os.path.exists(_RESULTS_FILE):
            with open(_RESULTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[LOAD] 분석 결과 로드 실패: {e}", flush=True)
    return {}

def save_analysis_results(analyzed: dict):
    """분석 완료 즉시 로컬 파일에 전체 결과를 저장합니다."""
    try:
        with open(_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(analyzed, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[SAVE] 분석 결과 저장 실패: {e}", flush=True)

def delete_analysis_results():
    """분석 결과 파일을 삭제합니다."""
    try:
        if os.path.exists(_RESULTS_FILE):
            os.remove(_RESULTS_FILE)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════
def get_env_config() -> dict:
    """설정값 우선순위: st.secrets (Cloud) → 환경변수 → .env 파일 → 코드 기본값"""
    def _s(key, fallback=""):
        # 1순위: Streamlit Cloud secrets
        try:
            v = st.secrets.get(key)
            if v: return str(v).strip()
        except Exception:
            pass
        # 2순위: 환경변수 / .env 파일 (dotenv로 이미 로드됨)
        v = os.getenv(key, "").strip()
        return v if v else fallback

    return {
        "api_base":   _s("TIMBLO_API_BASE",  "https://demo.timblo.io/api"),
        "api_key":    _s("TIMBLO_API_KEY",    "cm8o8cqet000014gd7ig5cxrw"),
        "email":      _s("TIMBLO_EMAIL",      "sorizava_counsel@timbel.net"),
        "gemini_key": _s("GEMINI_API_KEY"),
        "openai_key": _s("OPENAI_API_KEY"),
    }

def is_timblo_ready(cfg): return bool(cfg["api_base"] and cfg["api_key"] and cfg["email"])
def is_gemini_ready(cfg): return bool(cfg.get("gemini_key"))
def is_openai_ready(cfg): return bool(cfg.get("openai_key"))
def get_timblo_client(cfg):
    return TimbloClient(cfg["api_base"], cfg["api_key"], cfg["email"]) if is_timblo_ready(cfg) else None


# ══════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════
def parse_title(title: str) -> dict:
    """제목에서 상담방식/지사/직원명/상담자를 파싱합니다.
    - 맨 앞의 (신) 등 접두사가 있어도 첫 번째 [...] 를 상담방식으로 인식합니다.
    - 예: (신)[방문]강남_서채윤(남은도) → 방식:방문, 지사:강남, 직원:서채윤, 상담자:남은도
    """
    mode = branch = staff = client_name = ""
    try:
        raw = (title or "").strip()
        # 접두사 무시: 문자열 내 첫 [...] 를 어디서든 찾음 (^ 앵커 제거)
        m = re.search(r"\[([^\]]+)\]", raw)
        if m:
            mode = m.group(1).strip()
            rest = raw[m.end():]          # ] 이후 문자열
        else:
            rest = raw
        # ] 뒤 ~ _ 앞 → 지사
        if "_" in rest:
            branch, rest = rest.split("_", 1)
        branch = branch.strip()
        # 나머지: 직원명(상담자)
        m2 = re.match(r"([^(（]+)(?:[（(]([^)）]+)[）)])?", rest)
        if m2:
            staff       = m2.group(1).strip()
            client_name = (m2.group(2) or "").strip()
    except Exception:
        pass  # 파싱 실패 시 빈값 반환, 프로그램 중단 없음
    return {"mode": mode, "branch": branch, "staff": staff, "client": client_name}


def parse_date_str(raw):
    if not raw: return ""
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except:
        return raw[:10] if len(raw) >= 10 else raw


def fmt_date(raw):
    if not raw: return "-"
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return raw[:16] if len(raw) >= 16 else raw


def calc_total_score(quality_scores: dict, rubric: list = None) -> float:
    if not quality_scores: return 0.0
    if rubric:
        return round(sum((quality_scores.get(r["항목"], 3) / 5) * r["배점"] for r in rubric), 1)
    return round(sum((quality_scores.get(k, 3) / 5) * w for k, w in QUALITY_WEIGHTS.items()), 1)


def score_grade(score: float) -> str:
    if score >= 85: return "우수"
    if score >= 70: return "양호"
    if score >= 60: return "보통"
    return "개선필요"


def score_color(score: float) -> str:
    if score >= 85: return "#2e7d32"
    if score >= 70: return "#1565c0"
    if score >= 60: return "#f57c00"
    return "#c62828"


# ══════════════════════════════════════════════════════════
# 필수 안내 항목 하드 매칭 키워드 (대화 텍스트에서 발견 시 강제 PASS)
# 항목명에 아래 key 문자열이 포함되면 해당 keyword 목록으로 매핑
# ══════════════════════════════════════════════════════════
MANDATORY_KEYWORD_MAP: dict = {
    # 웍스파이(Worksfy) 실제 수요 인증
    "웍스파이": [
        "웍스파이", "워크스파이", "워크스", "웍스", "Worksfy", "worksfy",
        "일감 플랫폼", "일감플랫폼", "일거리 사이트", "일거리사이트",
        "플랫폼", "프리랜서 플랫폼", "협회 플랫폼",
    ],
    # 가격 인상 예정 고지
    "가격 인상": [
        "가격 인상", "가격인상", "금액 변동", "금액변동",
        "장비값 오른", "팀벨 인상", "팀벨인상", "인상 예정", "인상예정",
        "오를 예정", "오르기 전", "가격이 오", "비용 인상", "비용인상",
        "가격 올", "금액 올",
    ],
    # 구매 적기 어필
    "구매 적기": [
        "지금이 적기", "2026년", "수정 시간 10분", "수정시간 10분",
        "난이도 하향", "취득 적기", "적기", "지금이 최적기",
        "지금 시작", "이번 달", "한시적", "지금이 기회",
        "기회", "지금 아니면",
    ],
}


def hard_match_mandatory(conv_text: str, mandatory_items: list, ai_result: dict) -> dict:
    """대화 원문에서 키워드를 직접 스캔하여 AI 누락 보정.
    키워드가 발견되면 해당 항목을 강제로 done=True 처리.
    이미 True인 항목은 그대로 유지.
    """
    mc = dict(ai_result.get("mandatory_check") or {})
    text_lower = conv_text.lower()

    for item in mandatory_items:
        # 이미 done=True 이면 건드리지 않음
        existing = mc.get(item)
        if isinstance(existing, dict) and _coerce_done(existing.get("done", False)):
            continue
        if isinstance(existing, bool) and existing:
            continue

        # 이 항목에 매핑할 키워드 목록 결정
        kw_list: list = []
        for key_pattern, kws in MANDATORY_KEYWORD_MAP.items():
            if key_pattern in item:
                kw_list = kws
                break

        if not kw_list:
            continue

        # 키워드 스캔
        found_kw = next((kw for kw in kw_list if kw.lower() in text_lower), None)
        if found_kw:
            mc[item] = {
                "done": True,
                "evidence": f"[하드매칭] 대화 텍스트에서 키워드 '{found_kw}' 직접 발견",
                "미준수_사유": "",
            }

    ai_result["mandatory_check"] = mc
    return ai_result


# ══════════════════════════════════════════════════════════
# 필수 안내 체크 정제 헬퍼
# ══════════════════════════════════════════════════════════
def _coerce_done(raw) -> bool:
    """AI 응답의 done 값을 반드시 bool로 강제 변환합니다."""
    if isinstance(raw, bool): return raw
    if isinstance(raw, (int, float)): return bool(raw)
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "yes", "예", "o", "✅", "1", "완료", "함")
    return False

def parse_mandatory_check(raw: dict) -> dict:
    """AI가 반환한 mandatory_check를 정제합니다.
    - done 필드를 반드시 bool로 변환 (문자열/숫자 등 모두 처리)
    - evidence 필드 보장 (note/근거 등 구버전 키도 흡수)
    """
    if not raw or not isinstance(raw, dict):
        return {}
    result = {}
    for item, val in raw.items():
        if isinstance(val, dict):
            done_raw    = val.get("done", val.get("준수", val.get("check", False)))
            done_bool   = _coerce_done(done_raw)
            evidence    = str(val.get("evidence", val.get("note", val.get("근거", "")))).strip()
            miss_reason = str(val.get("미준수_사유", val.get("reason", ""))).strip()
        elif isinstance(val, (bool, int, float, str)):
            done_bool   = _coerce_done(val)
            evidence    = ""
            miss_reason = ""
        else:
            done_bool   = False
            evidence    = ""
            miss_reason = ""
        result[item] = {
            "done":     done_bool,
            "evidence": evidence    or "판단 근거 없음",
            "미준수_사유": miss_reason if not done_bool else "",
        }
    return result


# ══════════════════════════════════════════════════════════
# 채널 루브릭 라우터
# ══════════════════════════════════════════════════════════
def get_rubric_for_mode(mode: str, user_cfg: dict) -> list:
    """mode 문자열을 CHANNEL_KEYS에 매핑해 해당 채점표를 반환. 미매핑 시 '공통' 반환."""
    rubrics = user_cfg.get("channel_rubrics", {})
    if not rubrics:
        qw = user_cfg.get("quality_weights", dict(QUALITY_WEIGHTS))
        return [{"항목": k, "배점": v, "기준": ""} for k, v in qw.items()]
    mode_str = (mode or "").strip()
    for key in CHANNEL_KEYS[:-1]:   # 방문, 비대면, 전화 (공통 제외)
        if key in mode_str:
            return rubrics.get(key, rubrics.get("공통", list(_DEFAULT_RUBRIC)))
    return rubrics.get("공통", list(_DEFAULT_RUBRIC))


# ══════════════════════════════════════════════════════════
# 심층 분석 프롬프트
# ══════════════════════════════════════════════════════════
def build_deep_prompt(conversation_text: str, speech_stats: dict, title_meta: dict,
                      user_cfg: dict = None) -> str:
    if user_cfg is None:
        user_cfg = DEFAULT_USER_CONFIG

    rubric          = get_rubric_for_mode(title_meta.get("mode", ""), user_cfg)
    u_flow_stages   = user_cfg.get("flow_stages", list(FLOW_STAGES))
    system_prompt   = user_cfg.get("system_prompt", "").strip()
    mandatory_items = user_cfg.get("mandatory_items", [])

    # 메타정보
    meta_parts = []
    if title_meta.get("mode"):   meta_parts.append(f"상담방식: {title_meta['mode']}")
    if title_meta.get("branch"): meta_parts.append(f"지사: {title_meta['branch']}")
    if title_meta.get("staff"):  meta_parts.append(f"담당 직원: {title_meta['staff']}")
    if title_meta.get("client"): meta_parts.append(f"상담자: {title_meta['client']}")
    meta_block = "\n".join(meta_parts) if meta_parts else "(정보 없음)"

    # 발화 통계
    stats_lines = []
    for sid, s in speech_stats.items():
        stats_lines.append(
            f"  - {s['name']}: 발화 {s['utterance_count']}회 | "
            f"총 {s['total_chars']}자 | 발화 비중 {s['talk_ratio']}%"
        )
    stats_block = "\n".join(stats_lines) if stats_lines else "  (통계 없음)"

    # 품질 점수 항목 설명 (채널별 루브릭 — 상세 기준 포함)
    q_lines = [
        f'   - {r["항목"]}(배점{r["배점"]}): {r["기준"] if r.get("기준","").strip() else "해당 역량을 1~5점으로 평가"}'
        for r in rubric
    ]
    q_block  = "\n".join(q_lines)
    q_schema = ", ".join(f'"{r["항목"]}": 3' for r in rubric)

    # 흐름 단계 목록
    flow_list   = ", ".join(u_flow_stages)
    flow_schema = ", ".join(f'"{s}": false' for s in u_flow_stages)

    # 필수 안내 항목 + 하드매칭 키워드 힌트 블록
    if mandatory_items:
        mand_lines = [f"   - {item}" for item in mandatory_items]
        mand_block = "\n".join(mand_lines)
        mand_schema_pairs = ", ".join(
            f'"{item}": {{"done": false, "evidence": "판단근거 또는 관련 발화 인용", "미준수_사유": ""}}'
            for item in mandatory_items
        )

        # 항목별 동의어 힌트 자동 생성
        kw_hint_lines = []
        for item in mandatory_items:
            for key_pat, kw_list in MANDATORY_KEYWORD_MAP.items():
                if key_pat in item:
                    kw_hint_lines.append(
                        f'   · "{item}" → 아래 중 하나라도 발견되면 즉시 done:true\n'
                        f'     인식 키워드: {", ".join(kw_list[:8])}'
                    )
                    break
        kw_hint_block = "\n".join(kw_hint_lines) if kw_hint_lines else ""

        mand_instruction = f"""
11. 필수 안내 항목 점검 ★★★★★ 최우선 지시 — 전체 프롬프트에서 가장 중요 ★★★★★

점검 대상 항목:
{mand_block}

━━━ 항목별 인식 키워드 (하드매칭 목록) ━━━
{kw_hint_block}

━━━ 핵심 판단 원칙 (절대 위반 금지) ━━━

① 위 키워드 목록 중 하나라도 대화 텍스트에 등장하면 → 해당 항목 즉시 done: true
   (STT 오타/유사발음/구어체 포함: 웍스파이≈웍스≈워크스파이≈워크스≈Worksfy)

② 아래 상황도 모두 done: true:
   - 항목의 핵심 개념이 대화 중 한 번이라도 언급·암시
   - 상담원이 해당 주제로 설명·언급한 사실이 문맥으로 확인
   - 고객이 해당 주제로 질문하고 상담원이 답변

③ done: false는 오직 아래 경우만:
   - 전체 대화에서 해당 주제·키워드·개념이 단 한 번도 등장하지 않았을 때

④ 판단이 1%라도 불확실하면 → done: true (관대한 기준 적용)

━━━ JSON 출력 규칙 ━━━
· done 필드: 반드시 boolean (true 또는 false) 만 허용. "True", "O", "준수" 등 문자열 절대 금지.
· evidence 필드: 실제 대화에서 발견한 발화 내용 직접 인용 (true/false 모두 필수 기재).
· 미준수_사유 필드: done이 false일 때만 구체적 사유 기재. true이면 반드시 빈 문자열("").
"""
        mand_json = f',\n  "mandatory_check": {{{mand_schema_pairs}}}'
    else:
        mand_instruction = ""
        mand_json = ',\n  "mandatory_check": {}'

    # 시스템 프롬프트 (페르소나)
    persona_line = system_prompt if system_prompt else "당신은 직업훈련 상담 품질 평가 전문가입니다."

    # 채널별 채점표 최상단 요약 블록 (AI가 채점 기준을 놓치지 않도록 상단 배치)
    rubric_summary_lines = [
        f'   ▶ {r["항목"]} (배점 {r["배점"]}점): {(r.get("기준","") or "해당 역량 평가")[:60]}...'
        if len(r.get("기준","")) > 60 else
        f'   ▶ {r["항목"]} (배점 {r["배점"]}점): {r.get("기준","해당 역량 평가")}'
        for r in rubric
    ]
    rubric_summary = "\n".join(rubric_summary_lines)
    total_max = sum(r["배점"] for r in rubric)

    return f"""{persona_line}
아래 상담 대화를 분석하여 반드시 지정된 JSON 형식으로만 응답하세요.
JSON 외의 텍스트, 설명, 마크다운 코드블록을 절대 포함하지 마세요.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【채점표 최우선 기준 — 반드시 이 기준으로만 채점】
총 배점: {total_max}점 기준 / 각 항목 1~5점 (배점×점수/5 = 항목 득점)
{rubric_summary}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

★★★ 채점 핵심 원칙 (반드시 적용) ★★★
① 관대한 채점: 상담사가 해당 항목의 핵심 내용을 조금이라도 언급했다면 최소 3점 이상 부여.
   완벽하지 않아도 노력이 보이면 4점. 탁월하게 수행했다면 5점.
② 키워드 보너스: 웍스파이, 수정시간 10분, 가격 인상, 무이자, 타임머신, 약어기능 등
   핵심 키워드가 명시적으로 언급된 경우 해당 항목 최소 4점 보장.
③ STT 오타 허용: 음성인식 오류로 발생한 유사 발음/구어체를 정상 언급으로 인정.
④ 짧은 상담도 핵심만 있으면 고점 가능: 발화량이 적어도 핵심 내용을 전달했다면 높은 점수.
⑤ 점수 분포: 60점 이하 집중을 피하고, 현실적인 역량을 반영하여 70~90점대로 분산.

[상담 메타정보]
{meta_block}

[화자별 발화 통계]
{stats_block}

[전체 대화 내용]
{conversation_text}

[분석 지시 — 아래 항목을 모두 분석하여 하나의 JSON으로 반환]

1. 역할 판별: 발화 내용 성격(설명/질문/고민)을 기반으로 상담원/고객을 판별하세요.

2. 품질 점수 (각 항목 1~5점, 위 채점 원칙 必 적용):
{q_block}

3. 흐름 분석: 아래 단계 포함 여부 (true/false)
   {flow_list}

4. 강점 3가지 (각 명확한 한 문장 — 실제 대화 근거 기반)
5. 개선점 3가지 (각 명확한 한 문장, 경청 문제 포함)
6. 키워드: 고객 발화 TOP 20 단어, 상담사 발화 TOP 20 단어 (의미있는 명사/동사)
7. 질문 패턴: 고객이 언급한 각 주제별 횟수 (없으면 0)
   공부기간, 시험난이도, 수익가능성, 가격, 취업가능성, 기타
8. 감정 분석: 고객 발화 기준 긍정/중립/부정 비율 (합계 반드시 1.0)
9. 위험 신호: "고민해보겠습니다, 비싸네요, 알아보고 올게요, 시간이 없어요" 등 탐지 문구 목록
10. 코칭 멘트: 종합적인 개선 방향 피드백 (3~5문장){mand_instruction}

반드시 아래 JSON 구조로만 반환하세요:
{{
  "identified_counselor": "참석자 N",
  "identified_customer": "참석자 N",
  "role_reason": "역할 판별 근거 1~2문장",
  "quality_scores": {{{q_schema}}},
  "flow_stages": {{{flow_schema}}},
  "strengths": ["강점1", "강점2", "강점3"],
  "improvements": ["개선점1", "개선점2", "개선점3"],
  "customer_keywords": ["키워드1", "키워드2"],
  "counselor_keywords": ["키워드1", "키워드2"],
  "question_patterns": {{"공부기간": 0, "시험난이도": 0, "수익가능성": 0, "가격": 0, "취업가능성": 0, "기타": 0}},
  "sentiment": {{"positive": 0.3, "neutral": 0.5, "negative": 0.2}},
  "risk_signals": [],
  "coaching": "코칭 멘트 전문"{mand_json}
}}"""


# ══════════════════════════════════════════════════════════
# 트랜스크립트 캐시
# ══════════════════════════════════════════════════════════
def fetch_transcript(client, content_id: str, title_meta: dict) -> dict:
    if content_id in st.session_state.transcripts:
        return st.session_state.transcripts[content_id]
    detail = client.get_content_detail(content_id, tabs=["segments", "speakerInfo"])
    conv   = build_conversation_text(detail, use_merged=True)
    if not conv["text"].strip():
        raise ValueError(f"[EMPTY_DATA] {content_id} — 텍스트 없음")
    if not conv["lines"]:
        raise ValueError(f"[EMPTY_DATA] {content_id} — 발화 0건")
    speech_stats = compute_speech_stats(detail, use_merged=True)
    prompt = build_deep_prompt(conv["text"], speech_stats, title_meta,
                               user_cfg=st.session_state.get("user_config"))
    cached = {
        "text": conv["text"], "lines": conv["lines"],
        "speech_stats": speech_stats, "analysis_prompt": prompt,
        "segment_count": len(conv["lines"]),
    }
    st.session_state.transcripts[content_id] = cached
    return cached


# ══════════════════════════════════════════════════════════
# 에러 분류 + JSON 파싱
# ══════════════════════════════════════════════════════════
def classify_error(e: Exception) -> str:
    msg = str(e).lower()
    if "empty_data" in str(e) or "텍스트 없음" in str(e): return "EMPTY_DATA"
    if "rate" in msg or "quota" in msg or "429" in msg or "resource_exhausted" in msg: return "RATE_LIMIT"
    if "api_key" in msg or "invalid_argument" in msg or "401" in msg or "403" in msg: return "AUTH"
    if "404" in msg or "not found" in msg: return "MODEL_NOT_FOUND"
    if "json" in msg: return "JSON_PARSE"
    if "connection" in msg or "timeout" in msg: return "NETWORK"
    return "UNKNOWN"


def parse_ai_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if "```" in text:
        for part in text.split("```"):
            p = part.strip()
            if p.startswith("json"): p = p[4:].strip()
            if p.startswith("{"): text = p; break
    s = text.find("{"); e = text.rfind("}")
    if s != -1 and e != -1: text = text[s:e+1]
    return json.loads(text)


def build_record(content_id, title, start_time, result, speech_stats, segment_count, engine,
                 rubric: list = None) -> dict:
    qs = result.get("quality_scores", {})
    if rubric:
        for row in rubric:
            if row["항목"] not in qs:
                qs[row["항목"]] = 3
    else:
        for k in QUALITY_WEIGHTS:
            if k not in qs: qs[k] = 3
    total = calc_total_score(qs, rubric)
    return {
        "id": content_id, "date": parse_date_str(start_time), "title": title,
        "counselor":          result.get("identified_counselor", "참석자 1"),
        "customer":           result.get("identified_customer",  "참석자 2"),
        "role_reason":        result.get("role_reason", ""),
        "quality_scores":     qs,
        "rubric_used":        rubric or [],
        "total_score":        total,
        "flow_stages":        result.get("flow_stages", {s: False for s in FLOW_STAGES}),
        "strengths":          result.get("strengths", []),
        "improvements":       result.get("improvements", []),
        "customer_keywords":  result.get("customer_keywords", []),
        "counselor_keywords": result.get("counselor_keywords", []),
        "question_patterns":  result.get("question_patterns", {}),
        "sentiment":          result.get("sentiment", {"positive": 0.33, "neutral": 0.34, "negative": 0.33}),
        "risk_signals":       result.get("risk_signals", []),
        "coaching":           result.get("coaching", ""),
        "mandatory_check":    parse_mandatory_check(result.get("mandatory_check", {})),
        "speech_stats":       speech_stats,
        "segment_count":      segment_count,
        "engine":             engine,
        "needs_review":       total < 60,
        "excellent":          total >= 85,
    }


# ══════════════════════════════════════════════════════════
# AI 분석 함수
# ══════════════════════════════════════════════════════════
def run_gemini_analysis(client, content_id, title, start_time, title_meta, gemini_api_key) -> dict:
    import google.generativeai as genai
    cached = fetch_transcript(client, content_id, title_meta)
    genai.configure(api_key=gemini_api_key)
    try:
        model    = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config={"response_mime_type": "application/json"},
        )
        response = model.generate_content(cached["analysis_prompt"])
        raw_text = response.text.strip()
    except Exception as e:
        raise RuntimeError(f"[{classify_error(e)}] Gemini 호출 실패: {e}") from e
    try:
        result = parse_ai_json(raw_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"[JSON_PARSE] Gemini JSON 파싱 실패: {e}") from e
    _ucfg  = st.session_state.get("user_config") or DEFAULT_USER_CONFIG
    rubric = get_rubric_for_mode(title_meta.get("mode", ""), _ucfg)
    # 하드매칭: 대화 텍스트 스캔으로 AI 누락 키워드 강제 PASS
    result = hard_match_mandatory(cached["text"], _ucfg.get("mandatory_items", []), result)
    return build_record(content_id, title, start_time, result,
                        cached["speech_stats"], cached["segment_count"], "Gemini", rubric=rubric)


def run_openai_analysis(client, content_id, title, start_time, title_meta, openai_api_key) -> dict:
    from openai import OpenAI as _OpenAI
    cached = fetch_transcript(client, content_id, title_meta)
    oa = _OpenAI(api_key=openai_api_key)
    try:
        resp = oa.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": cached["analysis_prompt"]}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw_text = resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"[{classify_error(e)}] OpenAI 호출 실패: {e}") from e
    try:
        result = parse_ai_json(raw_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"[JSON_PARSE] OpenAI JSON 파싱 실패: {e}") from e
    _ucfg  = st.session_state.get("user_config") or DEFAULT_USER_CONFIG
    rubric = get_rubric_for_mode(title_meta.get("mode", ""), _ucfg)
    # 하드매칭: 대화 텍스트 스캔으로 AI 누락 키워드 강제 PASS
    result = hard_match_mandatory(cached["text"], _ucfg.get("mandatory_items", []), result)
    return build_record(content_id, title, start_time, result,
                        cached["speech_stats"], cached["segment_count"], "ChatGPT", rubric=rubric)


def run_hybrid_analysis(client, content_id, title, start_time, title_meta, cfg) -> dict:
    gemini_key = cfg.get("gemini_key")
    openai_key = cfg.get("openai_key")
    if gemini_key:
        try:
            return run_gemini_analysis(client, content_id, title, start_time, title_meta, gemini_key)
        except RuntimeError as e:
            if classify_error(e) == "RATE_LIMIT" and openai_key:
                safe = str(e).encode("ascii", errors="replace").decode("ascii")
                print(f"[FALLBACK] Gemini 한도 -> ChatGPT: {safe}", flush=True)
            elif not openai_key:
                raise
            else:
                raise
    if openai_key:
        return run_openai_analysis(client, content_id, title, start_time, title_meta, openai_key)
    raise RuntimeError("Gemini와 OpenAI 키가 모두 없습니다.")


# ══════════════════════════════════════════════════════════
# 차트 헬퍼
# ══════════════════════════════════════════════════════════
def make_radar_chart(cats, vals, color="royalblue", height=320,
                     overlay_vals=None, overlay_name="선택 상담자", overlay_color="#e74c3c"):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor="rgba(65,105,225,0.15)",
        line=dict(color=color, width=2), name="전체 평균",
        mode="lines+markers+text",
        text=[f"{v:.0f}" for v in vals] + [""],
        textposition="top center",
        textfont=dict(size=11, color=color),
    ))
    if overlay_vals:
        fig.add_trace(go.Scatterpolar(
            r=overlay_vals + [overlay_vals[0]], theta=cats + [cats[0]],
            fill="toself", fillcolor="rgba(231,76,60,0.15)",
            line=dict(color=overlay_color, width=2.5, dash="solid"), name=overlay_name,
            mode="lines+markers+text",
            text=[f"{v:.0f}" for v in overlay_vals] + [""],
            textposition="top center",
            textfont=dict(size=11, color=overlay_color),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=bool(overlay_vals),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=30, b=40, l=40, r=40), height=height,
    )
    return fig


def make_sentiment_pie(sentiment: dict, height=220):
    labels = ["긍정", "중립", "부정"]
    values = [
        round(sentiment.get("positive", 0.33) * 100, 1),
        round(sentiment.get("neutral",  0.34) * 100, 1),
        round(sentiment.get("negative", 0.33) * 100, 1),
    ]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker_colors=["#4CAF50", "#9E9E9E", "#F44336"],
        hole=0.4, textinfo="label+percent",
    ))
    fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=height)
    return fig


# ══════════════════════════════════════════════════════════
# 페이지 설정 + CSS (Glassmorphism 로그인 포함)
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="상담 품질 분석 대시보드", page_icon="📊",
    layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
.stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; padding: 8px 18px; }
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #f5f7ff 0%, #eef1ff 100%);
    border-radius: 12px; padding: 14px 18px; border: 1px solid #dde3ff;
}
div[data-testid="stNumberInput"] input { text-align: center; }

/* Glassmorphism 로그인 */
.glass-box {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 24px;
    padding: 48px 40px;
    box-shadow: 0 8px 32px rgba(31,38,135,0.37);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# 인증 — Glassmorphism 로그인 화면 (보호 구역: 절대 변경 금지)
# ══════════════════════════════════════════════════════════
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
    <style>
    .stApp > header { display: none; }
    [data-testid="stSidebar"] { display: none; }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        min-height: 100vh;
    }
    </style>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
        <div class="glass-box">
            <div style="font-size:3.2rem; margin-bottom:8px;">📊</div>
            <div style="color:white; font-size:1.6rem; font-weight:800; margin-bottom:6px;">
                상담 품질 분석
            </div>
            <div style="color:rgba(255,255,255,0.8); font-size:0.95rem; margin-bottom:32px;">
                전문가용 AI 대시보드
            </div>
        </div>
        """, unsafe_allow_html=True)
        pw = st.text_input("", type="password", placeholder="비밀번호를 입력하세요",
                           label_visibility="collapsed")
        if st.button("로그인", use_container_width=True, type="primary"):
            if pw == APP_PASSWORD:
                st.session_state.authenticated = True
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("비밀번호가 올바르지 않습니다.")
        st.caption("기본 비밀번호: `timblo2024`")
    st.stop()


# ══════════════════════════════════════════════════════════
# 세션 상태 초기화
# ══════════════════════════════════════════════════════════
_defaults = {
    "content_list": [], "content_list_all": [],
    "analyzed": {}, "transcripts": {},
    "list_loaded": False, "selected_id": None,
    "api_connected": None, "api_error_msg": "",
    "current_page": 0,
    "auto_queue": [], "auto_failed": [],
    "auto_limit": 20, "auto_delay": 3,
    "auto_current": "", "auto_done": 0,
    "user_config": None,        # 사용자 커스텀 설정 (None이면 파일에서 로드)
    "results_loaded": False,    # 파일에서 분석 결과 로드 여부
    "reanalyze_target": None,   # 목록에서 재분석 요청된 content_id
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# user_config 로드 (세션 최초 1회)
if st.session_state.user_config is None:
    st.session_state.user_config = load_user_config()

# 분석 결과 로드 (세션 최초 1회 — 파일에서 영구 데이터 복원)
if not st.session_state.results_loaded:
    _file_results = load_analysis_results()
    if _file_results:
        for _k, _v in _file_results.items():
            if _k not in st.session_state.analyzed:
                # 구버전 mandatory_check(note 필드 등)를 신버전(evidence) 포맷으로 자동 마이그레이션
                if isinstance(_v, dict) and "mandatory_check" in _v:
                    _v["mandatory_check"] = parse_mandatory_check(_v["mandatory_check"])
                st.session_state.analyzed[_k] = _v
    st.session_state.results_loaded = True


# ══════════════════════════════════════════════════════════
# 설정 + 클라이언트
# ══════════════════════════════════════════════════════════
cfg          = get_env_config()
client       = get_timblo_client(cfg)
analyzed     = st.session_state.analyzed
content_list = st.session_state.content_list


# ══════════════════════════════════════════════════════════
# 목록 로드 (최초 1회)
# ══════════════════════════════════════════════════════════
if not st.session_state.list_loaded and client:
    try:
        loaded        = client.get_content_list(status_filter="DONE")
        loaded_sorted = sorted(loaded, key=lambda x: x.get("meetingStartTime",""), reverse=True)
        loaded_2026   = [i for i in loaded_sorted if i.get("meetingStartTime","") >= "2026-01-01"]
        target        = loaded_2026 if loaded_2026 else loaded_sorted[:100]
        for item in target:
            p = parse_title(item.get("editedTitle") or item.get("title",""))
            item["_mode"] = p["mode"]; item["_branch"] = p["branch"]
            item["_staff"] = p["staff"]; item["_client"] = p["client"]
        st.session_state.content_list     = target
        st.session_state.content_list_all = loaded_sorted
        st.session_state.list_loaded      = True
        st.session_state.api_connected    = True
        content_list = target
    except Exception as e:
        st.session_state.api_connected = False
        st.session_state.api_error_msg = str(e)


# ══════════════════════════════════════════════════════════
# 자동 분석 처리 (렌더링 사이클당 1건)
# ══════════════════════════════════════════════════════════
if st.session_state.auto_queue and client and (is_gemini_ready(cfg) or is_openai_ready(cfg)):
    cid      = st.session_state.auto_queue[0]
    matching = [i for i in content_list if i.get("contentId") == cid]

    if matching and cid not in analyzed:
        item   = matching[0]
        title  = item.get("editedTitle") or item.get("title", cid)
        tmeta  = {"mode": item.get("_mode",""), "branch": item.get("_branch",""),
                  "staff": item.get("_staff",""), "client": item.get("_client","")}
        st.session_state.auto_current = title
        safe = title.encode("ascii", errors="replace").decode("ascii")
        done_n = st.session_state.auto_done
        print(f"[AUTO] [{done_n+1}/{st.session_state.auto_limit}] {safe}", flush=True)
        try:
            record = run_hybrid_analysis(
                client=client, content_id=cid,
                title=title, start_time=item.get("meetingStartTime"),
                title_meta=tmeta, cfg=cfg,
            )
            st.session_state.analyzed[cid] = record
            save_analysis_results(st.session_state.analyzed)   # 즉시 파일 저장
            st.session_state.auto_done    += 1
            print(f"[AUTO] OK [{record.get('engine','?')}] {record['total_score']}점", flush=True)
        except Exception as e:
            etype  = classify_error(e)
            safe_e = str(e).encode("ascii", errors="replace").decode("ascii")
            print(f"[AUTO] FAIL {etype}: {safe_e}", flush=True)
            st.session_state.auto_failed.append({
                "cid": cid, "title": title, "error_type": etype,
                "error_msg": str(e), "time": datetime.now().strftime("%H:%M:%S"),
            })
            if etype == "RATE_LIMIT":
                time.sleep(st.session_state.auto_delay + 10)

    st.session_state.auto_queue.pop(0)
    if st.session_state.auto_queue:
        time.sleep(st.session_state.auto_delay)
    st.rerun()


# ══════════════════════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 메뉴")
    st.markdown("---")

    if st.session_state.api_connected is True:
        st.success(f"팀블로 연결됨  \n총 **{len(content_list)}건**")
    elif st.session_state.api_connected is False:
        st.error("연결 실패")
        st.caption(st.session_state.api_error_msg[:80])
        if st.button("재연결", use_container_width=True):
            st.session_state.list_loaded = False; st.rerun()
    else:
        st.warning("연결 확인 중...")

    ai_list = (["Gemini"] if is_gemini_ready(cfg) else []) + (["ChatGPT"] if is_openai_ready(cfg) else [])
    if ai_list:
        st.success(f"AI: {' + '.join(ai_list)}")
    else:
        st.error("AI 키 미설정")

    st.markdown("---")
    st.markdown("**데이터 필터**")

    all_modes    = sorted(set(i.get("_mode","")   for i in content_list if i.get("_mode")))
    all_branches = sorted(set(i.get("_branch","") for i in content_list if i.get("_branch")))
    all_staff    = sorted(set(i.get("_staff","")  for i in content_list if i.get("_staff")))

    f_mode   = st.selectbox("상담방식", ["전체"] + all_modes,   key="sb_mode")
    f_branch = st.selectbox("지사",     ["전체"] + all_branches, key="sb_branch")
    f_staff  = st.selectbox("직원명",   ["전체"] + all_staff,   key="sb_staff")
    f_grade  = st.selectbox("품질등급",
                            ["전체","우수(85+)","양호(70+)","보통(60+)","개선필요(60-)"],
                            key="sb_grade")

    # 상담자 선택 — 분석 완료 레코드 기반 (레이더 오버레이 + 전환 예측용)
    _analyzed_staff = sorted(set(
        parse_title(r.get("title","")).get("staff","") or "미확인"
        for r in analyzed.values()
        if parse_title(r.get("title","")).get("staff","")
    ))
    f_counselor = st.selectbox("상담자 선택", ["전체 보기"] + _analyzed_staff, key="sb_counselor")

    st.markdown("**기간 필터**")
    # content_list + analyzed 양쪽에서 날짜 풀 수집 (기본값: 전체 기간)
    _cl_dates = [parse_date_str(i.get("meetingStartTime","")) for i in content_list]
    _an_dates = [r["date"] for r in analyzed.values() if r.get("date")]
    _date_pool = sorted(set(d for d in _cl_dates + _an_dates if d))
    if _date_pool:
        _fd_min = datetime.strptime(_date_pool[0],  "%Y-%m-%d").date()
        _fd_max = datetime.strptime(_date_pool[-1], "%Y-%m-%d").date()
        date_range = st.date_input(
            "기간", value=(_fd_min, _fd_max),
            label_visibility="collapsed",
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            if date_range[0] == _fd_min and date_range[1] == _fd_max:
                st.caption("전체 기간")
            else:
                st.caption(f"{date_range[0]} ~ {date_range[1]}")
    else:
        date_range = None
        st.caption("날짜 정보 없음")

    st.markdown("---")
    st.markdown("**자동 분석**")

    n_done = len(analyzed)
    n_todo = len([i for i in content_list if i.get("contentId") not in analyzed])
    n_fail = len(st.session_state.auto_failed)
    st.caption(f"완료 {n_done}건  |  미분석 {n_todo}건  |  오류 {n_fail}건")

    g_cnt = sum(1 for r in analyzed.values() if r.get("engine") == "Gemini")
    c_cnt = sum(1 for r in analyzed.values() if r.get("engine") == "ChatGPT")
    if g_cnt or c_cnt:
        st.caption(f"[G] Gemini {g_cnt}건  |  [C] ChatGPT {c_cnt}건")

    if st.session_state.auto_queue:
        total_q = st.session_state.auto_limit
        done_q  = st.session_state.auto_done
        remain  = len(st.session_state.auto_queue)
        st.progress(done_q / max(total_q, 1))
        st.markdown(f"**{done_q}/{total_q}건** 완료 | 남은: {remain}건")
        cur = st.session_state.auto_current
        if cur: st.caption(f"처리 중: {cur[:24]}...")
        if st.button("중단", use_container_width=True):
            st.session_state.auto_queue = []; st.session_state.auto_current = ""; st.rerun()
    else:
        a_limit = st.number_input("분석 건수", min_value=1, max_value=500,
                                  value=st.session_state.auto_limit, step=10, key="sb_limit")
        a_delay = st.number_input("건당 대기(초)", min_value=2, max_value=30,
                                  value=st.session_state.auto_delay, step=1, key="sb_delay")
        col_s, col_r = st.columns(2)
        with col_s:
            if st.button("시작", type="primary", use_container_width=True):
                st.session_state.auto_limit = a_limit
                st.session_state.auto_delay = a_delay
                st.session_state.auto_done  = 0
                unanalyzed = sorted(
                    [i for i in content_list if i.get("contentId") not in analyzed],
                    key=lambda x: x.get("meetingStartTime",""), reverse=True,
                )
                st.session_state.auto_queue = [i["contentId"] for i in unanalyzed[:a_limit]]
                st.rerun()
        with col_r:
            nf = len(st.session_state.auto_failed)
            if nf and st.button(f"재시도({nf})", use_container_width=True):
                ids = [f["cid"] for f in st.session_state.auto_failed]
                st.session_state.auto_failed = []
                st.session_state.auto_limit  = nf
                st.session_state.auto_done   = 0
                st.session_state.auto_queue  = ids
                st.rerun()

    if st.session_state.auto_failed:
        with st.expander(f"오류 상세 ({n_fail}건)"):
            for f in st.session_state.auto_failed[-8:]:
                st.markdown(f"**{f['time']}** `{f['error_type']}`")
                st.caption(f"{f['title'][:30]}")
            if st.button("기록 지우기", key="clr_fail"):
                st.session_state.auto_failed = []; st.rerun()

    st.markdown("---")
    if st.button("로그아웃", use_container_width=True):
        st.session_state.authenticated = False; st.rerun()


# ══════════════════════════════════════════════════════════
# 필터 적용 헬퍼
# ══════════════════════════════════════════════════════════
def apply_filters(records: list) -> list:
    out = records
    if f_mode   != "전체": out = [r for r in out if parse_title(r.get("title","")).get("mode","")   == f_mode]
    if f_branch != "전체": out = [r for r in out if parse_title(r.get("title","")).get("branch","") == f_branch]
    if f_staff  != "전체": out = [r for r in out if parse_title(r.get("title","")).get("staff","")  == f_staff]
    if f_grade == "우수(85+)":       out = [r for r in out if r.get("total_score", 0) >= 85]
    elif f_grade == "양호(70+)":     out = [r for r in out if 70 <= r.get("total_score", 0) < 85]
    elif f_grade == "보통(60+)":     out = [r for r in out if 60 <= r.get("total_score", 0) < 70]
    elif f_grade == "개선필요(60-)": out = [r for r in out if r.get("total_score", 0) < 60]
    if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        s_d, e_d = date_range
        out = [r for r in out if r.get("date") and
               s_d <= datetime.strptime(r["date"], "%Y-%m-%d").date() <= e_d]
    return out


# ══════════════════════════════════════════════════════════
# 탭 구성
# ══════════════════════════════════════════════════════════
st.title("📊 상담 품질 분석 대시보드")
tab_dash, tab_list, tab_settings = st.tabs(["📈  대시보드", "📋  상담 목록", "⚙️  설정"])


# ══════════════════════════════════════════════════════════
# TAB 1 — 대시보드
# ══════════════════════════════════════════════════════════
with tab_dash:
    records_all      = list(analyzed.values())
    records_filtered = apply_filters(records_all)

    if not records_filtered:
        # 미분석 상태: content_list 기반 현황
        if content_list:
            df_all       = pd.DataFrame(content_list)
            df_all["날짜"]   = df_all["meetingStartTime"].apply(parse_date_str)
            df_all["날짜dt"] = pd.to_datetime(df_all["날짜"], errors="coerce")
            today   = datetime.now().date()
            w_start = today - timedelta(days=6)
            m_start = today - timedelta(days=29)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("전체 상담",  f"{len(df_all):,}건")
            c2.metric("오늘",       f"{int((df_all['날짜dt'].dt.date == today).sum())}건")
            c3.metric("주간(7일)",  f"{int((df_all['날짜dt'].dt.date >= w_start).sum())}건")
            c4.metric("분석 완료",  f"{len(analyzed)}건")
            r30 = df_all[df_all["날짜dt"].dt.date >= m_start].copy()
            if not r30.empty:
                daily = r30.groupby("날짜").size().reset_index(name="건수")
                daily["날짜"] = pd.to_datetime(daily["날짜"])
                st.markdown("**최근 30일 일별 상담 건수**")
                fig_d = px.bar(daily, x="날짜", y="건수", color_discrete_sequence=["#4169E1"])
                fig_d.update_layout(margin=dict(t=10, b=10), height=230, xaxis_title="")
                st.plotly_chart(fig_d, use_container_width=True)
        st.info("좌측 **시작** 버튼으로 자동 분석을 시작하세요.")

    else:
        # ── KPI ──
        scores    = [r["total_score"] for r in records_filtered]
        avg_score = round(sum(scores) / len(scores), 1)
        excellent = sum(1 for s in scores if s >= 85)
        needs_rev = sum(1 for s in scores if s < 60)
        flow_ok   = sum(1 for r in records_filtered if sum(r.get("flow_stages",{}).values()) >= 4)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("분석 완료",      f"{len(records_filtered)}건")
        c2.metric("평균 점수",      f"{avg_score}점")
        c3.metric("우수 (85+)",     f"{excellent}건", f"{excellent/len(records_filtered)*100:.0f}%")
        c4.metric("개선필요 (60-)", f"{needs_rev}건",  f"{needs_rev/len(records_filtered)*100:.0f}%")
        c5.metric("흐름 완성 (4+)", f"{flow_ok}건")

        st.markdown("---")

        # ── 상담방식별 + 지사별 ──
        col_mode, col_branch = st.columns(2)

        mode_data: dict = {}
        for r in records_filtered:
            m = parse_title(r.get("title","")).get("mode","기타") or "기타"
            mode_data.setdefault(m, []).append(r["total_score"])
        if mode_data:
            m_df = pd.DataFrame([{"상담방식": m, "건수": len(v),
                                   "평균점수": round(sum(v)/len(v),1)}
                                  for m, v in mode_data.items()]).sort_values("평균점수", ascending=False)
            with col_mode:
                st.markdown("**상담방식별 평균 점수**")
                fig_m = px.bar(m_df, x="상담방식", y="평균점수",
                               color="평균점수", color_continuous_scale="RdYlGn",
                               range_y=[0,100], text="평균점수")
                fig_m.update_traces(texttemplate="%{text}점", textposition="outside")
                fig_m.update_layout(coloraxis_showscale=False, margin=dict(t=10,b=10), height=260)
                st.plotly_chart(fig_m, use_container_width=True)

        branch_data: dict = {}
        for r in records_filtered:
            b = parse_title(r.get("title","")).get("branch","미확인") or "미확인"
            branch_data.setdefault(b, []).append(r["total_score"])
        if branch_data:
            b_df = pd.DataFrame([{"지사": b, "건수": len(v),
                                   "평균점수": round(sum(v)/len(v),1)}
                                  for b, v in branch_data.items()]).sort_values("평균점수", ascending=True)
            with col_branch:
                st.markdown("**지사별 평균 점수 랭킹**")
                fig_b = px.bar(b_df, x="평균점수", y="지사", orientation="h",
                               color="평균점수", color_continuous_scale="RdYlGn",
                               range_x=[0,100], text="평균점수")
                fig_b.update_traces(texttemplate="%{text}점", textposition="outside")
                fig_b.update_layout(coloraxis_showscale=False, margin=dict(t=10,b=10), height=260)
                st.plotly_chart(fig_b, use_container_width=True)

        st.markdown("---")

        # ── 6대 지표 레이더 + 직원 랭킹 ──
        col_radar, col_srank = st.columns([2, 3])

        with col_radar:
            _radar_label = f"{f_counselor} vs 전체 평균" if f_counselor != "전체 보기" else "품질 지표 평균"
            st.markdown(f"**{_radar_label}**")
            # 동적 카테고리: 레코드별 rubric_used 또는 quality_scores 키 수집
            seen_cats: dict = {}  # 삽입 순서 보존
            for r in records_filtered:
                rubric_r = r.get("rubric_used") or []
                if rubric_r:
                    for row in rubric_r:
                        seen_cats.setdefault(row["항목"], None)
                else:
                    for k in r.get("quality_scores", {}):
                        seen_cats.setdefault(k, None)
            cats = list(seen_cats.keys()) or list(QUALITY_WEIGHTS.keys())
            qs_agg = {k: [] for k in cats}
            for r in records_filtered:
                qs = r.get("quality_scores", {})
                for k in cats:
                    if k in qs:
                        qs_agg[k].append(qs[k])
            qs_avg = {k: round(sum(v)/len(v)/5*100, 1) if v else 0 for k, v in qs_agg.items()}
            vals = [qs_avg.get(c, 0) for c in cats]
            # 선택 상담자 오버레이
            _overlay_vals = None
            if f_counselor != "전체 보기":
                _counsel_recs = [r for r in records_filtered
                                 if parse_title(r.get("title","")).get("staff","") == f_counselor]
                if _counsel_recs:
                    _c_agg = {k: [] for k in cats}
                    for r in _counsel_recs:
                        qs = r.get("quality_scores", {})
                        for k in cats:
                            if k in qs: _c_agg[k].append(qs[k])
                    _overlay_vals = [
                        round(sum(v)/len(v)/5*100, 1) if v else 0
                        for v in [_c_agg[k] for k in cats]
                    ]
            st.plotly_chart(
                make_radar_chart(cats, vals,
                                 overlay_vals=_overlay_vals,
                                 overlay_name=f_counselor),
                use_container_width=True,
            )

        with col_srank:
            st.markdown("**직원별 평균 점수 (상위 15명)**")
            staff_data: dict = {}
            for r in records_filtered:
                s = parse_title(r.get("title","")).get("staff","") or r.get("counselor","미확인")
                staff_data.setdefault(s, []).append(r["total_score"])
            if staff_data:
                s_df = pd.DataFrame([{"직원명": s, "건수": len(v),
                                       "평균점수": round(sum(v)/len(v),1)}
                                      for s, v in staff_data.items()]) \
                           .sort_values("평균점수", ascending=False).head(15)
                fig_s = px.bar(s_df, x="직원명", y="평균점수",
                               color="평균점수", color_continuous_scale="RdYlGn",
                               range_y=[0,100], text="평균점수")
                fig_s.update_traces(texttemplate="%{text}점", textposition="outside")
                fig_s.update_layout(coloraxis_showscale=False, margin=dict(t=10,b=10),
                                    height=360, xaxis_tickangle=-30)
                st.plotly_chart(fig_s, use_container_width=True)

        st.markdown("---")

        # ── 키워드 + 질문패턴 + 감정 ──
        col_kw, col_qp, col_sent = st.columns(3)

        with col_kw:
            st.markdown("**고객 키워드 TOP 15**")
            kw_cnt: dict = {}
            for r in records_filtered:
                for kw in r.get("customer_keywords", []):
                    kw_cnt[kw] = kw_cnt.get(kw, 0) + 1
            if kw_cnt:
                kw_df = pd.DataFrame(list(kw_cnt.items()), columns=["키워드","빈도"])
                kw_df = kw_df.sort_values("빈도", ascending=True).tail(15)
                fig_kw = px.bar(kw_df, x="빈도", y="키워드", orientation="h",
                                color_discrete_sequence=["#7986CB"])
                fig_kw.update_layout(margin=dict(t=10,b=10), height=340, xaxis_title="")
                st.plotly_chart(fig_kw, use_container_width=True)
            else:
                st.caption("AI 분석 후 표시됩니다")

        with col_qp:
            st.markdown("**질문 패턴 빈도**")
            qp_agg: dict = {c: 0 for c in QUESTION_CATS}
            for r in records_filtered:
                for cat, cnt in r.get("question_patterns", {}).items():
                    if cat in qp_agg: qp_agg[cat] += cnt
            qp_df = pd.DataFrame(list(qp_agg.items()), columns=["주제","빈도"]) \
                        .sort_values("빈도", ascending=True)
            fig_qp = px.bar(qp_df, x="빈도", y="주제", orientation="h",
                            color_discrete_sequence=["#26A69A"])
            fig_qp.update_layout(margin=dict(t=10,b=10), height=340, xaxis_title="")
            st.plotly_chart(fig_qp, use_container_width=True)

        with col_sent:
            st.markdown("**감정 비율 (고객 발화)**")
            n_r = len(records_filtered)
            sp = sum(r.get("sentiment",{}).get("positive",0.33) for r in records_filtered) / n_r
            sn = sum(r.get("sentiment",{}).get("neutral",0.34)  for r in records_filtered) / n_r
            sg = sum(r.get("sentiment",{}).get("negative",0.33) for r in records_filtered) / n_r
            st.plotly_chart(make_sentiment_pie({"positive":sp,"neutral":sn,"negative":sg}),
                            use_container_width=True)
            risk_all: list = []
            for r in records_filtered: risk_all.extend(r.get("risk_signals",[]))
            if risk_all:
                risk_cnt: dict = {}
                for rs in risk_all: risk_cnt[rs] = risk_cnt.get(rs, 0) + 1
                top5 = sorted(risk_cnt.items(), key=lambda x: x[1], reverse=True)[:5]
                st.markdown("**위험 신호 TOP 5**")
                for rs, cnt in top5: st.caption(f"⚠️ {rs} ({cnt}건)")

        st.markdown("---")

        # ── 기간별 추이 ──
        if len(records_filtered) >= 2:
            t_data = [{"날짜": r["date"], "점수": r["total_score"]}
                      for r in records_filtered if r.get("date")]
            if t_data:
                t_df  = pd.DataFrame(t_data)
                t_df["날짜"] = pd.to_datetime(t_df["날짜"])
                t_agg = t_df.groupby("날짜")["점수"].mean().round(1).reset_index()
                fig_t = px.line(t_agg, x="날짜", y="점수", markers=True, range_y=[0,100],
                                color_discrete_sequence=["royalblue"])
                fig_t.add_hline(y=85, line_dash="dot", line_color="green",
                                annotation_text="우수(85점)", annotation_position="bottom right")
                fig_t.add_hline(y=60, line_dash="dot", line_color="red",
                                annotation_text="개선필요(60점)", annotation_position="top right")
                fig_t.update_layout(margin=dict(t=20,b=20), hovermode="x unified")
                st.subheader("기간별 점수 추이")
                st.plotly_chart(fig_t, use_container_width=True)

        # ── 결과 테이블 ──
        st.markdown("---")
        st.subheader("분석 결과 테이블")
        _tbl_mand_items = st.session_state.user_config.get("mandatory_items", [])
        rows = []
        for r in records_filtered:
            p  = parse_title(r.get("title",""))
            mc = r.get("mandatory_check", {})

            # 필수 안내: 항목별 ✅/❌ 개별 나열 (비율/숫자 표기 금지)
            if _tbl_mand_items:
                mand_parts = []
                for mi in _tbl_mand_items:
                    raw_val = mc.get(mi) if mc else None
                    if raw_val is None:
                        done = False
                    elif isinstance(raw_val, dict):
                        done = _coerce_done(raw_val.get("done", False))
                    else:
                        done = _coerce_done(raw_val)
                    mand_parts.append(("✅ " if done else "❌ ") + mi)
                if not mc:  # 아직 mandatory_check 없는 레코드
                    mand_text = "  |  ".join("❓ " + mi for mi in _tbl_mand_items)
                else:
                    mand_text = "  |  ".join(mand_parts)
            else:
                mand_text = "-"

            rows.append({
                "날짜":     r["date"],
                "상담방식": p["mode"],
                "지사":     p["branch"],
                "직원명":   p["staff"],
                "상담자":   p["client"],
                "총점":     r["total_score"],
                "등급":     score_grade(r["total_score"]),
                "엔진":     r.get("engine",""),
                "흐름":     f"{sum(r.get('flow_stages',{}).values())}/{len(r.get('flow_stages',{})) or 6}",
                "필수 안내": mand_text,
            })

        # components.html → overflow-x:auto 보장 (st.dataframe clipping 해결)
        def _score_color(s):
            if s >= 85: return "#27ae60"
            if s >= 70: return "#2980b9"
            if s >= 60: return "#f39c12"
            return "#e74c3c"

        _tbl_body = ""
        for row in rows:
            sc  = row["총점"]
            col = _score_color(sc)
            # 필수 안내 셀: ✅/❌ 뱃지로 변환
            _mand_html = ""
            for part in row["필수 안내"].split("  |  "):
                part = part.strip()
                if part.startswith("✅"):
                    _mand_html += (f'<span style="background:#e8f8f0;border:1px solid #27ae60;'
                                   f'border-radius:10px;padding:2px 7px;margin:2px;'
                                   f'white-space:nowrap;font-size:12px;">{part}</span>')
                elif part.startswith("❌"):
                    _mand_html += (f'<span style="background:#fdf2f2;border:1px solid #e74c3c;'
                                   f'border-radius:10px;padding:2px 7px;margin:2px;'
                                   f'white-space:nowrap;font-size:12px;">{part}</span>')
                elif part.startswith("❓"):
                    _mand_html += (f'<span style="background:#f9f9f9;border:1px solid #bbb;'
                                   f'border-radius:10px;padding:2px 7px;margin:2px;'
                                   f'white-space:nowrap;font-size:12px;">{part}</span>')
                else:
                    _mand_html += f'<span style="font-size:12px">{part}</span>'

            _tbl_body += (
                f'<tr style="border-bottom:1px solid #eee;">'
                f'<td style="white-space:nowrap;padding:6px 10px;">{row["날짜"]}</td>'
                f'<td style="white-space:nowrap;padding:6px 10px;">{row["상담방식"]}</td>'
                f'<td style="white-space:nowrap;padding:6px 10px;">{row["지사"]}</td>'
                f'<td style="white-space:nowrap;padding:6px 10px;font-weight:600;">{row["직원명"]}</td>'
                f'<td style="white-space:nowrap;padding:6px 10px;">{row["상담자"]}</td>'
                f'<td style="white-space:nowrap;padding:6px 10px;color:{col};font-weight:700;">{sc:.1f}점</td>'
                f'<td style="white-space:nowrap;padding:6px 10px;color:{col};">{row["등급"]}</td>'
                f'<td style="white-space:nowrap;padding:6px 10px;">{row["엔진"]}</td>'
                f'<td style="white-space:nowrap;padding:6px 10px;">{row["흐름"]}</td>'
                f'<td style="padding:6px 10px;min-width:380px;">{_mand_html if row["필수 안내"] != "-" else "-"}</td>'
                f'</tr>'
            )

        _tbl_html = f"""
<div style="overflow-x:auto;overflow-y:auto;max-height:480px;
            border:1px solid #e0e0e0;border-radius:8px;font-family:sans-serif;">
  <table style="border-collapse:collapse;width:max-content;font-size:13px;">
    <thead>
      <tr style="background:#f0f2f6;position:sticky;top:0;z-index:2;">
        <th style="padding:8px 12px;border-bottom:2px solid #ccc;white-space:nowrap;">날짜</th>
        <th style="padding:8px 12px;border-bottom:2px solid #ccc;white-space:nowrap;">방식</th>
        <th style="padding:8px 12px;border-bottom:2px solid #ccc;white-space:nowrap;">지사</th>
        <th style="padding:8px 12px;border-bottom:2px solid #ccc;white-space:nowrap;">직원명</th>
        <th style="padding:8px 12px;border-bottom:2px solid #ccc;white-space:nowrap;">상담자</th>
        <th style="padding:8px 12px;border-bottom:2px solid #ccc;white-space:nowrap;">총점</th>
        <th style="padding:8px 12px;border-bottom:2px solid #ccc;white-space:nowrap;">등급</th>
        <th style="padding:8px 12px;border-bottom:2px solid #ccc;white-space:nowrap;">엔진</th>
        <th style="padding:8px 12px;border-bottom:2px solid #ccc;white-space:nowrap;">흐름</th>
        <th style="padding:8px 12px;border-bottom:2px solid #ccc;min-width:380px;">필수 안내 항목별 ✅/❌</th>
      </tr>
    </thead>
    <tbody>{_tbl_body}</tbody>
  </table>
</div>"""
        components.html(_tbl_html, height=min(520, 60 + 40 * len(rows)), scrolling=False)

        # ── 지사/직원별 상담방식 통계 피벗 표 ──
        st.markdown("---")
        st.subheader("지사/직원별 상담방식 통계")
        _pv_rows = []
        for r in records_filtered:
            p = parse_title(r.get("title",""))
            _pv_rows.append({
                "지사":     p["branch"] or "미확인",
                "직원명":   p["staff"]  or "미확인",
                "상담방식": p["mode"]   or "미분류",
            })
        if _pv_rows:
            _pv_df = pd.DataFrame(_pv_rows)
            _pivot = (
                pd.crosstab([_pv_df["지사"], _pv_df["직원명"]], _pv_df["상담방식"])
                .fillna(0).astype(int)
            )
            _pivot.insert(0, "합계", _pivot.sum(axis=1))
            _pivot = _pivot.sort_values("합계", ascending=False)
            st.dataframe(_pivot, use_container_width=True)
        else:
            st.info("분석 데이터가 없습니다.")

        # ── 필수 안내 준수율 차트 ──
        mand_items = st.session_state.user_config.get("mandatory_items", [])
        # 분석 완료 레코드 전체 대상 (mandatory_check 없는 구버전 레코드도 미준수로 집계)
        mand_records = records_filtered
        mand_analyzed = [r for r in mand_records if r.get("mandatory_check")]
        if mand_items and mand_analyzed:
            st.markdown("---")
            st.subheader("필수 안내 준수율")
            mand_stats = []
            for item in mand_items:
                done_count = 0
                for r in mand_analyzed:
                    raw_val = r.get("mandatory_check", {}).get(item)
                    if raw_val is None:
                        continue
                    # 구버전(dict with note) / 신버전(dict with evidence) / 직접 bool 모두 처리
                    if isinstance(raw_val, dict):
                        done_raw = raw_val.get("done", raw_val.get("준수", False))
                    else:
                        done_raw = raw_val
                    if _coerce_done(done_raw):
                        done_count += 1
                rate = round(done_count / len(mand_analyzed) * 100, 1)
                mand_stats.append({"항목": item, "준수율(%)": rate,
                                   "준수": done_count, "미준수": len(mand_analyzed) - done_count,
                                   "분석건수": len(mand_analyzed)})
            mand_df = pd.DataFrame(mand_stats).sort_values("준수율(%)", ascending=True)
            fig_mand = px.bar(mand_df, x="준수율(%)", y="항목", orientation="h",
                              color="준수율(%)", color_continuous_scale="RdYlGn",
                              range_x=[0, 100], text="준수율(%)",
                              hover_data={"준수": True, "미준수": True, "분석건수": True})
            fig_mand.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_mand.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10),
                                   height=50 + 50 * len(mand_items))
            st.plotly_chart(fig_mand, use_container_width=True)
            # 요약 수치 표
            st.dataframe(mand_df[["항목","준수율(%)","준수","미준수","분석건수"]],
                         use_container_width=True, hide_index=True)

        # ── 성공 패턴 & Golden Phrases (항상 표시) ──
        st.markdown("---")
        st.subheader("💡 AI 전략 인사이트")
        _excellent_recs = [r for r in records_filtered if r.get("total_score", 0) >= 85]

        if not _excellent_recs:
            st.info("우수 상담(85점 이상) 데이터가 아직 없습니다. 분석을 더 진행하면 성공 패턴이 자동으로 추출됩니다.")
        else:
            st.caption(f"우수 상담(85점 이상) {len(_excellent_recs)}건에서 추출한 핵심 패턴")

            # 성공 키워드: 상담사 키워드 빈도 집계
            _succ_kw: dict = {}
            for r in _excellent_recs:
                for kw in r.get("counselor_keywords", []):
                    _succ_kw[kw] = _succ_kw.get(kw, 0) + 1
            _top_kw = sorted(_succ_kw.items(), key=lambda x: x[1], reverse=True)[:12]

            # Golden Phrases: strengths에서 수집
            _phrases: list = []
            for r in _excellent_recs:
                for s in r.get("strengths", []):
                    if s and s not in _phrases:
                        _phrases.append(s)
                if len(_phrases) >= 9:
                    break

            col_kw_s, col_phrases = st.columns([1, 2])
            with col_kw_s:
                st.markdown("**🏅 핵심 성공 키워드**")
                if _top_kw:
                    _kw_tags = " ".join(
                        f'<span style="background:#e8f4fd;border:1px solid #3498db;'
                        f'border-radius:14px;padding:4px 10px;margin:3px;'
                        f'display:inline-block;font-size:13px;color:#1a5276;">'
                        f'{"🔥" if cnt >= 3 else "✦"} {kw} <b>({cnt})</b></span>'
                        for kw, cnt in _top_kw
                    )
                    st.markdown(_kw_tags, unsafe_allow_html=True)
                else:
                    st.info("키워드 데이터가 부족합니다.")

            with col_phrases:
                st.markdown("**💬 Golden Phrases — 설득 핵심 문장**")
                if _phrases:
                    for i, phrase in enumerate(_phrases[:6]):
                        _bg = "#fff9e6" if i % 2 == 0 else "#f0fff4"
                        st.markdown(
                            f'<div style="background:{_bg};border-left:4px solid #f39c12;'
                            f'border-radius:6px;padding:8px 14px;margin:5px 0;'
                            f'font-size:13.5px;line-height:1.5;">'
                            f'💬 {phrase}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("우수 상담의 강점 데이터가 아직 부족합니다.")

        # ── 상담방식별 Top 5 랭킹 ──
        st.markdown("---")
        st.subheader("🏆 상담방식별 Top 5 랭킹")
        _ch_cols = st.columns(3)
        _channel_list = [("방문", _ch_cols[0]), ("비대면", _ch_cols[1]), ("전화", _ch_cols[2])]
        for _ch_key, _ch_col in _channel_list:
            _ch_recs = [r for r in records_filtered
                        if _ch_key in (parse_title(r.get("title","")).get("mode","") or "")]
            with _ch_col:
                st.markdown(f"**{_ch_key} 상담 Top 5**")
                if _ch_recs:
                    _ch_staff: dict = {}
                    for r in _ch_recs:
                        s = parse_title(r.get("title","")).get("staff","") or "미확인"
                        _ch_staff.setdefault(s, []).append(r["total_score"])
                    _ch_df = pd.DataFrame([
                        {"직원명": s, "건수": len(v), "평균점수": round(sum(v)/len(v), 1)}
                        for s, v in _ch_staff.items()
                    ]).sort_values("평균점수", ascending=False).head(5).reset_index(drop=True)
                    _ch_df.index = _ch_df.index + 1  # 1-based rank
                    st.dataframe(
                        _ch_df,
                        use_container_width=True,
                        hide_index=False,
                        column_config={
                            "직원명":  st.column_config.TextColumn("직원명"),
                            "건수":    st.column_config.NumberColumn("건수", format="%d건"),
                            "평균점수": st.column_config.ProgressColumn(
                                "평균점수", min_value=0, max_value=100, format="%.1f점"),
                        },
                    )
                else:
                    st.caption("해당 방식 데이터 없음")

        # ── 전환 가능성 예측 ──
        st.markdown("---")
        st.subheader("🎯 전환 가능성 예측")

        def _calc_conversion(r) -> int:
            base   = r.get("total_score", 0)
            flow   = r.get("flow_stages", {})
            f_done = sum(1 for v in flow.values() if v)
            f_tot  = len(flow) or 6
            pos    = r.get("sentiment", {}).get("positive", 0.33)
            risk   = len(r.get("risk_signals", []))
            score  = base * 0.80 + (f_done / f_tot) * 12 + pos * 8 - risk * 2
            return int(min(100, max(0, round(score))))

        _staff_conv: dict = {}
        for r in records_filtered:
            s = parse_title(r.get("title","")).get("staff","") or "미확인"
            _staff_conv.setdefault(s, []).append(_calc_conversion(r))

        _conv_df = pd.DataFrame([
            {"직원명": s, "전환가능성": round(sum(v)/len(v), 1), "건수": len(v)}
            for s, v in _staff_conv.items()
        ]).sort_values("전환가능성", ascending=False)

        _avg_conv = round(_conv_df["전환가능성"].mean(), 1) if not _conv_df.empty else 0

        if f_counselor != "전체 보기":
            # 선택 상담자 metric
            _sel_rows = _conv_df[_conv_df["직원명"] == f_counselor]
            if not _sel_rows.empty:
                _sel_conv = _sel_rows.iloc[0]["전환가능성"]
                _delta = round(_sel_conv - _avg_conv, 1)
                col_m1, col_m2, col_m3 = st.columns([1, 1, 3])
                with col_m1:
                    st.metric(f"{f_counselor}", f"{_sel_conv}점",
                              delta=f"{_delta:+.1f} vs 평균")
                with col_m2:
                    st.metric("전체 평균", f"{_avg_conv}점")
            else:
                st.caption(f"{f_counselor}의 전환 가능성 데이터가 없습니다")
        else:
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("전체 평균 전환가능성", f"{_avg_conv}점")
            _top_conv = _conv_df.iloc[0] if not _conv_df.empty else None
            if _top_conv is not None:
                col_m2.metric(f"최고 ({_top_conv['직원명']})", f"{_top_conv['전환가능성']}점")

        # 막대 차트 + 평균선
        if not _conv_df.empty:
            _conv_plot = _conv_df.sort_values("전환가능성", ascending=True).tail(20)
            _highlight = f_counselor if f_counselor != "전체 보기" else None
            _colors = [
                "#e74c3c" if _highlight and row["직원명"] == _highlight else "#3498db"
                for _, row in _conv_plot.iterrows()
            ]
            fig_conv = go.Figure(go.Bar(
                x=_conv_plot["전환가능성"],
                y=_conv_plot["직원명"],
                orientation="h",
                marker_color=_colors,
                text=[f"{v}점" for v in _conv_plot["전환가능성"]],
                textposition="outside",
            ))
            fig_conv.add_vline(
                x=_avg_conv, line_dash="dash", line_color="orange", line_width=2,
                annotation_text=f"평균 {_avg_conv}점",
                annotation_position="top right",
            )
            fig_conv.update_layout(
                xaxis=dict(range=[0, 105], title="전환가능성 점수"),
                margin=dict(t=20, b=20),
                height=max(280, 32 * len(_conv_plot)),
            )
            st.plotly_chart(fig_conv, use_container_width=True)

        # ── 업로드 현황 요약 (분석 결과 테이블과 동일 데이터 기반) ──
        st.markdown("---")
        st.subheader("업로드 현황 요약")
        st.caption("※ 아래 '분석 결과 테이블'과 동일한 기준으로 집계됩니다")
        if records_filtered:
            _up_rows = []
            for r in records_filtered:
                p = parse_title(r.get("title", ""))
                _up_rows.append({
                    "상담방식": p.get("mode") or "미분류",
                    "지사":     p.get("branch") or "미확인",
                    "직원명":   p.get("staff") or "미확인",
                })
            _up_df = pd.DataFrame(_up_rows)

            col_sum1, col_sum2, col_sum3 = st.columns(3)
            with col_sum1:
                st.markdown("**상담방식별**")
                _mode_sum = (
                    _up_df.groupby("상담방식").size()
                    .reset_index(name="건수")
                    .sort_values("건수", ascending=False)
                )
                st.dataframe(_mode_sum, hide_index=True, use_container_width=True,
                             column_config={"건수": st.column_config.NumberColumn("건수", format="%d건")})
            with col_sum2:
                st.markdown("**지사별**")
                _branch_sum = (
                    _up_df.groupby("지사").size()
                    .reset_index(name="건수")
                    .sort_values("건수", ascending=False)
                )
                st.dataframe(_branch_sum, hide_index=True, use_container_width=True,
                             column_config={"건수": st.column_config.NumberColumn("건수", format="%d건")})
            with col_sum3:
                st.markdown("**직원별 (상위 10명)**")
                _staff_sum = (
                    _up_df.groupby("직원명").size()
                    .reset_index(name="건수")
                    .sort_values("건수", ascending=False)
                    .head(10)
                )
                st.dataframe(_staff_sum, hide_index=True, use_container_width=True,
                             column_config={"건수": st.column_config.NumberColumn("건수", format="%d건")})
        else:
            st.info("분석 데이터가 없습니다.")


# ══════════════════════════════════════════════════════════
# TAB 2 — 상담 목록
# ══════════════════════════════════════════════════════════
with tab_list:
    st.markdown("## 상담 목록")

    if not content_list:
        if st.button("목록 불러오기", type="primary"):
            st.session_state.list_loaded = False; st.rerun()
        st.info("목록이 없습니다.")
    else:
        col_sq, col_sf, col_sr = st.columns([4, 2, 1])
        with col_sq:
            search_q = st.text_input("검색", placeholder="제목·날짜 검색", label_visibility="collapsed")
        with col_sf:
            status_f = st.selectbox("상태", ["전체","미분석","분석완료","우수(85+)","개선필요(60-)"],
                                    label_visibility="collapsed")
        with col_sr:
            if st.button("새로고침"):
                st.session_state.list_loaded = False; st.session_state.current_page = 0; st.rerun()

        def item_score(cid):
            r = analyzed.get(cid); return r["total_score"] if r else None

        fl = content_list
        if search_q:
            q = search_q.lower()
            fl = [i for i in fl if q in (i.get("editedTitle") or i.get("title","")).lower()
                  or q in parse_date_str(i.get("meetingStartTime",""))]
        if f_mode   != "전체": fl = [i for i in fl if i.get("_mode","")   == f_mode]
        if f_branch != "전체": fl = [i for i in fl if i.get("_branch","") == f_branch]
        if f_staff  != "전체": fl = [i for i in fl if i.get("_staff","")  == f_staff]
        if status_f == "미분석":         fl = [i for i in fl if i.get("contentId") not in analyzed]
        elif status_f == "분석완료":     fl = [i for i in fl if i.get("contentId") in analyzed]
        elif status_f == "우수(85+)":    fl = [i for i in fl if (item_score(i.get("contentId","")) or 0) >= 85]
        elif status_f == "개선필요(60-)": fl = [i for i in fl if 0 < (item_score(i.get("contentId","")) or 999) < 60]

        total_items = len(fl)
        total_pages = max(1, (total_items + PAGE_SIZE - 1) // PAGE_SIZE)
        page = max(0, min(st.session_state.current_page, total_pages - 1))
        st.session_state.current_page = page
        st.caption(f"총 **{total_items:,}건** | 분석완료 **{len(analyzed)}건** | 페이지 {page+1}/{total_pages}")

        h0, h1, h2, h3, h4, h5 = st.columns([3, 2, 1.5, 1, 0.8, 0.7])
        h0.markdown("**제목**"); h1.markdown("**방식/지사/직원**")
        h2.markdown("**날짜**"); h3.markdown("**상태**"); h4.markdown("**열기**"); h5.markdown("**재분석**")
        st.markdown('<hr style="margin:4px 0;">', unsafe_allow_html=True)

        for item in fl[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]:
            cid       = item.get("contentId","")
            title     = item.get("editedTitle") or item.get("title","(제목 없음)")
            is_done   = cid in analyzed
            is_active = cid == st.session_state.selected_id

            c0, c1, c2, c3, c4, c5 = st.columns([3, 2, 1.5, 1, 0.8, 0.7])
            with c0:
                pref = "▶ " if is_active else ""
                st.markdown(f"{pref}**{title[:32]}**{'...' if len(title)>32 else ''}")
            with c1:
                parts = [x for x in [item.get("_mode",""), item.get("_branch",""), item.get("_staff","")] if x]
                st.caption(" / ".join(parts) if parts else "-")
            with c2:
                st.caption(fmt_date(item.get("meetingStartTime")))
            with c3:
                if is_done:
                    sc  = item_score(cid)
                    eng = analyzed[cid].get("engine","")
                    ebadge = "[G]" if eng == "Gemini" else "[C]" if eng == "ChatGPT" else ""
                    st.markdown(f"{score_grade(sc)} {ebadge}")
                else:
                    st.markdown("미분석")
            with c4:
                if is_active:
                    if st.button("닫기", key=f"cls_{cid}"):
                        st.session_state.selected_id = None; st.rerun()
                else:
                    if st.button("열기" if is_done else "분석", key=f"sel_{cid}"):
                        st.session_state.selected_id = cid; st.rerun()
            with c5:
                if is_done:
                    if st.button("🔄", key=f"rean_{cid}", help="최신 설정으로 재분석 (기존 결과 덮어쓰기)"):
                        # 트랜스크립트 캐시 제거 → 최신 설정 프롬프트로 재생성
                        st.session_state.transcripts.pop(cid, None)
                        st.session_state.selected_id    = cid
                        st.session_state.reanalyze_target = cid
                        st.rerun()

        # 페이지네이션
        st.markdown("")
        MAX_BTNS = 9; half_pg = MAX_BTNS // 2
        start_p = max(0, min(page - half_pg, total_pages - MAX_BTNS))
        end_p   = min(total_pages, start_p + MAX_BTNS)
        p_cols  = st.columns(2 + (end_p - start_p))
        if p_cols[0].button("◀", disabled=(page == 0)):
            st.session_state.current_page = page - 1; st.rerun()
        for idx, pg in enumerate(range(start_p, end_p)):
            if p_cols[idx+1].button(f"**{pg+1}**" if pg == page else str(pg+1), key=f"pg_{pg}"):
                st.session_state.current_page = pg; st.rerun()
        if p_cols[-1].button("▶", disabled=(page >= total_pages - 1)):
            st.session_state.current_page = page + 1; st.rerun()

        # ── 선택 상담 상세 패널 ──
        sel_id = st.session_state.selected_id
        if sel_id and any(i.get("contentId") == sel_id for i in content_list):
            sel_item  = next(i for i in content_list if i.get("contentId") == sel_id)
            sel_title = sel_item.get("editedTitle") or sel_item.get("title","")
            sel_meta  = {"mode": sel_item.get("_mode",""), "branch": sel_item.get("_branch",""),
                         "staff": sel_item.get("_staff",""), "client": sel_item.get("_client","")}

            st.markdown("---")
            st.markdown(f"### {sel_title}")
            mpx = [x for x in [sel_meta["mode"], sel_meta["branch"],
                                sel_meta["staff"], sel_meta["client"]] if x]
            if mpx: st.caption(" | ".join(mpx))
            st.caption(f"날짜: {fmt_date(sel_item.get('meetingStartTime'))}")

            pt_ai, pt_stt, pt_detail = st.tabs(["🤖 AI 분석", "📝 STT 전문", "📊 상세 리포트"])

            with pt_ai:
                existing = analyzed.get(sel_id)
                if existing:
                    sc    = existing["total_score"]
                    eng   = existing.get("engine","")
                    color = score_color(sc)
                    st.markdown(
                        f"**총점: <span style='color:{color}; font-size:1.6rem; font-weight:800;'>"
                        f"{sc}점</span>** &nbsp; `{score_grade(sc)}` &nbsp; "
                        f"`{'[G] Gemini' if eng=='Gemini' else '[C] ChatGPT' if eng=='ChatGPT' else eng}`",
                        unsafe_allow_html=True,
                    )

                btn_lbl = "🔄 재분석 (최신 설정 적용)" if existing else "🤖 AI 분석 시작"
                # 목록에서 재분석 요청이 들어온 경우 자동 트리거
                _auto_reanalyze = st.session_state.get("reanalyze_target") == sel_id
                if _auto_reanalyze:
                    st.session_state.reanalyze_target = None
                do_analyze = st.button(btn_lbl, key=f"ai_{sel_id}", type="primary") or _auto_reanalyze
                if do_analyze:
                    try:
                        # 재분석 시 트랜스크립트 캐시 강제 초기화 → 최신 설정으로 프롬프트 재생성
                        st.session_state.transcripts.pop(sel_id, None)
                        with st.spinner("AI 심층 분석 중... (Gemini → ChatGPT 자동 전환)"):
                            record = run_hybrid_analysis(
                                client=client, content_id=sel_id,
                                title=sel_title, start_time=sel_item.get("meetingStartTime"),
                                title_meta=sel_meta, cfg=cfg,
                            )
                        st.session_state.analyzed[sel_id] = record
                        save_analysis_results(st.session_state.analyzed)  # 즉시 파일 저장
                        st.success(f"분석 완료! 총점: **{record['total_score']}점** [{record.get('engine','')}]")
                        st.rerun()
                    except Exception as e:
                        st.error(f"분석 실패: {e}")

                if existing:
                    st.markdown("---")
                    col_ra, col_flow = st.columns([1, 1])
                    with col_ra:
                        qs   = existing.get("quality_scores", {})
                        rubric_e = existing.get("rubric_used") or \
                                   [{"항목": k, "배점": v, "기준": ""} for k, v in QUALITY_WEIGHTS.items()]
                        cats = [r["항목"] for r in rubric_e] or list(qs.keys()) or list(QUALITY_WEIGHTS.keys())
                        vals = [round(qs.get(c, 3)/5*100, 1) for c in cats]
                        st.markdown("**품질 지표**")
                        st.plotly_chart(make_radar_chart(cats, vals, height=280), use_container_width=True)
                    with col_flow:
                        st.markdown("**흐름 6단계**")
                        flow = existing.get("flow_stages", {})
                        for stage in FLOW_STAGES:
                            icon = "✅" if flow.get(stage, False) else "❌"
                            st.markdown(f"{icon} {stage}")

                    col_st, col_im = st.columns(2)
                    with col_st:
                        st.success("**강점**")
                        for s in existing.get("strengths", []): st.markdown(f"• {s}")
                    with col_im:
                        st.warning("**개선점**")
                        for imp in existing.get("improvements", []): st.markdown(f"• {imp}")

                    col_sent2, col_risk = st.columns(2)
                    with col_sent2:
                        st.markdown("**감정 분석**")
                        st.plotly_chart(make_sentiment_pie(existing.get("sentiment",{})),
                                        use_container_width=True)
                    with col_risk:
                        st.markdown("**위험 신호**")
                        risks = existing.get("risk_signals", [])
                        if risks:
                            for rs in risks: st.markdown(f"⚠️ {rs}")
                        else:
                            st.caption("탐지된 위험 신호 없음")

                    if existing.get("coaching"):
                        st.info(f"**코칭 리포트**\n\n{existing['coaching']}")

                    qp = existing.get("question_patterns", {})
                    if any(v > 0 for v in qp.values()):
                        st.markdown("**질문 패턴**")
                        qpd = pd.DataFrame(list(qp.items()), columns=["주제","빈도"])
                        fig_qp2 = px.bar(qpd, x="주제", y="빈도", color_discrete_sequence=["#26A69A"])
                        fig_qp2.update_layout(margin=dict(t=10,b=10), height=200)
                        st.plotly_chart(fig_qp2, use_container_width=True)

            with pt_stt:
                cached_t = st.session_state.transcripts.get(sel_id)
                if cached_t is None:
                    with st.spinner("STT 텍스트 로딩 중..."):
                        try:
                            cached_t = fetch_transcript(client, sel_id, sel_meta)
                            st.rerun()
                        except Exception as e:
                            st.error(f"STT 로드 실패: {e}")
                if cached_t:
                    st.caption(f"총 {cached_t['segment_count']}개 발화")
                    if cached_t.get("speech_stats"):
                        ss_rows = [{"화자": s.get("name",""), "발화수": s.get("utterance_count",0),
                                    "총글자": s.get("total_chars",0), "발화비중(%)": s.get("talk_ratio",0)}
                                   for s in cached_t["speech_stats"].values()]
                        st.dataframe(pd.DataFrame(ss_rows), use_container_width=True, hide_index=True)
                    with st.expander("전체 STT 텍스트"):
                        st.text(cached_t["text"])

            with pt_detail:
                existing2 = analyzed.get(sel_id)
                if not existing2:
                    st.info("AI 분석 후 상세 리포트가 표시됩니다.")
                else:
                    col_kw1, col_kw2 = st.columns(2)
                    with col_kw1:
                        st.markdown("**고객 키워드 TOP 20**")
                        ck = existing2.get("customer_keywords", [])[:20]
                        if ck:
                            k1 = pd.DataFrame({"키워드": ck, "순위": range(1, len(ck)+1)})
                            fig_c1 = px.bar(k1, x="순위", y="키워드", orientation="h",
                                            color_discrete_sequence=["#7986CB"])
                            fig_c1.update_layout(yaxis=dict(autorange="reversed"),
                                                 margin=dict(t=10,b=10), height=320)
                            st.plotly_chart(fig_c1, use_container_width=True)
                    with col_kw2:
                        st.markdown("**상담사 키워드 TOP 20**")
                        ck2 = existing2.get("counselor_keywords", [])[:20]
                        if ck2:
                            k2 = pd.DataFrame({"키워드": ck2, "순위": range(1, len(ck2)+1)})
                            fig_c2 = px.bar(k2, x="순위", y="키워드", orientation="h",
                                            color_discrete_sequence=["#26A69A"])
                            fig_c2.update_layout(yaxis=dict(autorange="reversed"),
                                                 margin=dict(t=10,b=10), height=320)
                            st.plotly_chart(fig_c2, use_container_width=True)

                    qs3 = existing2.get("quality_scores", {})
                    rubric3 = existing2.get("rubric_used") or \
                              [{"항목": k, "배점": v, "기준": ""} for k, v in QUALITY_WEIGHTS.items()]
                    qs_rows = [{"항목": r["항목"],
                                "점수(1~5)": qs3.get(r["항목"], 0),
                                "배점": r["배점"],
                                "획득점수": round(qs3.get(r["항목"], 0) / 5 * r["배점"], 1),
                                "평가 기준": r.get("기준", "")}
                               for r in rubric3]
                    st.markdown("**항목별 점수**")
                    st.dataframe(pd.DataFrame(qs_rows), use_container_width=True, hide_index=True)

                    if existing2.get("role_reason"):
                        with st.expander("역할 판별 근거"):
                            st.markdown(existing2["role_reason"])

                    # ── 필수 안내 항목 점검 결과 ──
                    mc2 = existing2.get("mandatory_check", {})
                    if mc2:
                        st.markdown("---")
                        st.markdown("**필수 안내 항목 점검**")
                        mc_rows = []
                        for mc_item, mc_val in mc2.items():
                            if isinstance(mc_val, dict):
                                done_raw    = mc_val.get("done", mc_val.get("준수", False))
                                evidence    = str(mc_val.get("evidence", mc_val.get("note", "근거 없음"))).strip()
                                miss_reason = str(mc_val.get("미준수_사유", mc_val.get("reason", ""))).strip()
                            else:
                                done_raw    = mc_val
                                evidence    = ""
                                miss_reason = ""
                            done_bool = _coerce_done(done_raw)
                            mc_rows.append({
                                "항목":      mc_item,
                                "준수여부":  "✅ 완료" if done_bool else "❌ 미준수",
                                "판단 근거": evidence    or "근거 없음",
                                "미준수 사유": miss_reason if not done_bool else "-",
                            })
                        mc_df = pd.DataFrame(mc_rows)
                        st.dataframe(mc_df, use_container_width=True, hide_index=True,
                                     column_config={
                                         "항목":      st.column_config.TextColumn("항목",      width="small"),
                                         "준수여부":  st.column_config.TextColumn("준수여부",  width="small"),
                                         "판단 근거": st.column_config.TextColumn("판단 근거 (AI)",  width="medium"),
                                         "미준수 사유": st.column_config.TextColumn("미준수 사유 (AI)", width="medium"),
                                     })


# ══════════════════════════════════════════════════════════
# TAB 3 — 설정
# ══════════════════════════════════════════════════════════
with tab_settings:
    st.markdown("## 설정")
    st.markdown("---")

    st.markdown("### 분석 엔진 상태")
    col_ga, col_oa = st.columns(2)
    with col_ga:
        st.metric("Gemini 2.0 Flash", "연결됨" if is_gemini_ready(cfg) else "미설정")
        st.caption("JSON 모드 (response_mime_type: application/json)")
    with col_oa:
        st.metric("ChatGPT gpt-4o-mini", "연결됨" if is_openai_ready(cfg) else "미설정")
        st.caption("JSON 모드 (response_format: json_object)")

    st.markdown("---")

    u_cfg = st.session_state.user_config  # shorthand

    # ── 1. 채널별 품질 평가 채점표 ──
    st.markdown("### 1️⃣ 채널별 품질 평가 채점표")
    st.caption("상담방식 태그([방문]/[비대면]/[전화])에 따라 AI가 해당 탭의 채점표를 자동 적용합니다. 배점 합계는 반드시 100점.")
    channel_rubrics_src = u_cfg.get("channel_rubrics", {k: list(_DEFAULT_RUBRIC) for k in CHANNEL_KEYS})
    ch_tab_uis = st.tabs(CHANNEL_TABS)
    edited_rubrics: dict = {}
    all_weights_valid = True
    for ch_tab_ui, ch_key in zip(ch_tab_uis, CHANNEL_KEYS):
        with ch_tab_ui:
            src_rows = channel_rubrics_src.get(ch_key, list(_DEFAULT_RUBRIC))
            # 컬럼 보정: 모든 row에 3개 키 보장
            src_rows = [{"항목": r.get("항목",""), "배점": r.get("배점", 0), "기준": r.get("기준","")}
                        for r in src_rows]
            ch_df_edited = st.data_editor(
                pd.DataFrame(src_rows) if src_rows else pd.DataFrame({"항목": [""], "배점": [0], "기준": [""]}),
                column_config={
                    "항목": st.column_config.TextColumn("평가 항목명", width="small"),
                    "배점": st.column_config.NumberColumn("배점", min_value=0, max_value=50, step=1, width="small"),
                    "기준": st.column_config.TextColumn("상세 평가 기준 (AI 프롬프트에 주입됨)", width="large"),
                },
                num_rows="dynamic", use_container_width=True, hide_index=True,
                key=f"rubric_editor_{ch_key}",
            )
            ch_total = int(ch_df_edited["배점"].sum()) if not ch_df_edited.empty else 0
            if ch_total == 100:
                st.success(f"총합: {ch_total}점 ✅  — 저장 시 분석에 적용됩니다")
            else:
                st.warning(f"⚠️ 총합: {ch_total}점 — 100점이 되어야 분석이 가능합니다")
                all_weights_valid = False
            edited_rubrics[ch_key] = [
                {"항목": str(row["항목"]).strip(), "배점": int(row["배점"]), "기준": str(row.get("기준","")).strip()}
                for _, row in ch_df_edited.iterrows()
                if str(row.get("항목","")).strip()
            ]

    st.markdown("---")

    # ── 2. 흐름 분석 단계 ──
    st.markdown("### 2️⃣ 흐름 분석 단계")
    st.caption("단계 목록을 편집하세요. 행 추가/삭제 가능합니다.")
    fs_rows = [{"단계": s} for s in u_cfg["flow_stages"]]
    fs_edited = st.data_editor(
        pd.DataFrame(fs_rows) if fs_rows else pd.DataFrame({"단계": [""]}),
        column_config={"단계": st.column_config.TextColumn("단계명")},
        num_rows="dynamic", use_container_width=True, hide_index=True, key="fs_editor",
    )

    st.markdown("---")

    # ── 3. AI 시스템 프롬프트 (페르소나) ──
    st.markdown("### 3️⃣ AI 분석 페르소나 (시스템 프롬프트)")
    st.caption("AI가 분석을 시작할 때 맨 앞에 붙는 역할 지시문입니다.")
    new_sys_prompt = st.text_area(
        label="시스템 프롬프트",
        value=u_cfg.get("system_prompt", ""),
        height=120, key="sys_prompt_area",
        placeholder="예: 당신은 직업훈련 상담 품질 평가 전문가입니다...",
    )

    st.markdown("---")

    # ── 4. 필수 안내 사항 체크리스트 ──
    st.markdown("### 4️⃣ 필수 안내 사항 (AI 체크리스트)")
    st.caption("AI가 상담 텍스트에서 해당 안내를 했는지 O/X와 근거를 JSON에 포함합니다.")
    mi_rows = [{"필수항목": m} for m in u_cfg.get("mandatory_items", [])]
    mi_edited = st.data_editor(
        pd.DataFrame(mi_rows) if mi_rows else pd.DataFrame({"필수항목": [""]}),
        column_config={"필수항목": st.column_config.TextColumn("필수 안내 항목")},
        num_rows="dynamic", use_container_width=True, hide_index=True, key="mi_editor",
    )

    st.markdown("---")

    # ── 저장 버튼 ──
    col_save, col_reset = st.columns([2, 1])
    with col_save:
        if st.button("💾 설정 저장", type="primary", use_container_width=True):
            if not all_weights_valid:
                st.error("채널별 배점 총합이 모두 100점이어야 저장할 수 있습니다. 각 탭을 확인하세요.")
            else:
                new_stages  = [row["단계"] for _, row in fs_edited.iterrows()
                               if str(row.get("단계","")).strip()]
                new_items   = [row["필수항목"] for _, row in mi_edited.iterrows()
                               if str(row.get("필수항목","")).strip()]
                new_cfg = {
                    "channel_rubrics": edited_rubrics,
                    "flow_stages":     new_stages,
                    "system_prompt":   new_sys_prompt.strip(),
                    "mandatory_items": new_items,
                }
                save_user_config(new_cfg)
                st.session_state.user_config = new_cfg
                # 트랜스크립트 프롬프트 캐시만 초기화 (분석 결과는 보존)
                st.session_state.transcripts = {}
                st.success(
                    "✅ 설정이 저장됐습니다.  \n"
                    "기존 분석 결과는 그대로 유지됩니다. "
                    "새 설정을 적용하려면 목록에서 🔄 재분석 버튼을 누르세요."
                )
                st.rerun()

    with col_reset:
        if st.button("↩️ 기본값 복원", use_container_width=True):
            save_user_config(DEFAULT_USER_CONFIG)
            st.session_state.user_config = dict(DEFAULT_USER_CONFIG)
            st.session_state.transcripts = {}  # 프롬프트 캐시만 초기화
            st.success("기본값으로 복원됐습니다. 기존 분석 결과는 보존됩니다."); st.rerun()

    st.markdown("---")
    st.markdown("### 🗄️ 데이터 관리")
    _n_analyzed = len(st.session_state.analyzed)
    _file_exists = os.path.exists(_RESULTS_FILE)
    _file_size_kb = round(os.path.getsize(_RESULTS_FILE) / 1024, 1) if _file_exists else 0
    col_dm1, col_dm2, col_dm3 = st.columns(3)
    col_dm1.metric("저장된 분석 건수", f"{_n_analyzed}건")
    col_dm2.metric("파일 저장 상태", "저장됨 ✅" if _file_exists else "없음")
    col_dm3.metric("파일 크기", f"{_file_size_kb} KB" if _file_exists else "-")
    st.caption(f"📁 저장 경로: `{_RESULTS_FILE}`")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("⚠️ 분석 결과 전체 초기화", use_container_width=True):
            st.session_state.analyzed    = {}
            st.session_state.transcripts = {}
            st.session_state.results_loaded = True
            delete_analysis_results()
            st.success("모든 분석 결과가 초기화됐습니다."); st.rerun()
    with col_c2:
        if st.button("목록 새로고침", use_container_width=True):
            st.session_state.list_loaded = False; st.rerun()
