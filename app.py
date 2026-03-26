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

# ── 구글 시트 연동 (선택적 — streamlit-gsheets-connection 미설치 시 해당 기능 비활성화) ──
try:
    from streamlit_gsheets import GSheetsConnection as _GSheetsConnection
    _GSHEETS_OK = True
except ImportError:
    _GSHEETS_OK = False

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

# ── 채널별 30개 항목 루브릭 (서버 배포 시에도 기본값으로 동작) ─────────────────
_DEFAULT_RUBRIC_방문 = [
    {"항목": "첫 인사 및 상담 시작 안내", "배점": 2, "기준": "상담 시작 시 자연스러운 인사와 함께 \"오늘 상담은 속기사 직업, 취업 분야, 준비 방법 등 3가지 핵심 내용으로 진행됩니다\"와 같이 상담 목적과 진행 방식을 명확히 안내했는지 평가한다."},
    {"항목": "긴장 완화 및 라포 형성", "배점": 2, "기준": "대면 상담의 긴장감을 풀기 위해 음료를 권하거나, 찾아오시는 길은 어땠는지 묻는 등 고객이 편안하게 상담에 참여할 수 있도록 부드러운 분위기를 조성했는지 평가한다."},
    {"항목": "상담 진행 구조 안내", "배점": 2, "기준": "PPT 등 시각 자료를 활용하여 상담 초반에 \"스펙 걱정 없는 전문직\", \"실제 돈 버는 프로세스\", \"지속 가능한 업무\" 등 오늘 설명할 전체적인 흐름을 시각적으로 안내했는지 확인한다."},
    {"항목": "고객 참여 유도", "배점": 2, "기준": "일방적인 설명에 그치지 않고, \"설명 들으시다가 궁금한 점이 생기면 언제든 편하게 질문해 주세요\"라고 고지하여 고객의 능동적인 참여를 유도했는지 평가한다."},
    {"항목": "관심 계기 파악", "배점": 3, "기준": "\"속기사라는 직업을 어떻게 알게 되셨나요? (유튜브, 블로그, 지인 등)\"라고 질문하여, 고객이 방문 상담을 신청하게 된 구체적인 계기와 유입 경로를 파악했는지 평가한다."},
    {"항목": "현재 상황 파악", "배점": 3, "기준": "고객의 현재 소속(직장인, 주부, 취준생 등)과 나이, 공부에 투자할 수 있는 하루 여유 시간 등을 구체적인 질문을 통해 정확히 확인했는지 평가한다."},
    {"항목": "목표 확인", "배점": 3, "기준": "고객이 속기사를 통해 이루고자 하는 궁극적인 목표가 전업 프리랜서인지, 투잡(부업)인지, 안정적인 공무원 취업인지 명확히 질문하여 타겟팅했는지 평가한다."},
    {"항목": "정보 수준 확인", "배점": 2, "기준": "\"혹시 속기 전용 키보드를 직접 보신 적 있으신가요?\", \"어느 정도 알아보고 오셨나요?\" 등의 질문으로 고객의 사전 지식수준을 점검했는지 평가한다."},
    {"항목": "맞춤형 상담 전개", "배점": 1, "기준": "파악된 고객의 상황(나이, 직업, 목표 등)에 맞추어, 공무원 루트 또는 프리랜서 루트 중 고객에게 더 적합한 방향으로 상담 비중을 조정하여 전개했는지 평가한다."},
    {"항목": "속기사 직업 정의 설명", "배점": 3, "기준": "단순 타자원이 아니라, '말소리부터 비언어적 표현, 감정까지 빠르고 정확하게 기록하는 기록 전문가'임을 명확히 정의하고 설명했는지 평가한다."},
    {"항목": "취업처 구조 설명", "배점": 3, "기준": "필기시험이 없는 법원/국회/의회 공무원, 방송국/기업체 정규직, 그리고 재택근무가 가능한 프리랜서(VOD, 라이브콘텐츠 등)로 취업 분야를 세분화하여 설명했는지 평가한다."},
    {"항목": "시장 전망 설명", "배점": 3, "기준": "장애인차별금지법(배리어프리 의무화), 공공기록물관리법, OTT 콘텐츠의 급증 등을 근거로 들어 속기사의 수요가 폭발적으로 증가하고 있는 긍정적 시장 전망을 설명했는지 평가한다."},
    {"항목": "사례 및 근거 활용", "배점": 2, "기준": "매뉴얼에 수록된 실제 합격자나 프리랜서 활동 사례(예: 30대 육아맘의 재택 성공기, 간호조무사 출신의 법원 취업 등) 중 고객 상황과 유사한 구체적 사례를 제시했는지 평가한다."},
    {"항목": "고객 맞춤 직무 연결", "배점": 1, "기준": "고객의 목표와 여건에 맞춰 \"고객님처럼 퇴근 후 2시간을 활용하시려면 VOD 자막 작업이 제격입니다\"와 같이 특정 직무를 1:1로 맞춤 연결해 주었는지 평가한다."},
    {"항목": "속기 키보드 구조 설명", "배점": 2, "기준": "자음, 모음, 받침을 한 번에 눌러 한 글자를 1~2초 만에 완성하는 '세벌식 동시 입력 방식'의 특징과 일반 키보드와의 결정적인 차이점을 명확히 설명했는지 평가한다."},
    {"항목": "키보드 기능 설명", "배점": 2, "기준": "소리자바 속기 키보드의 핵심인 자주 쓰는 단어를 저장하는 '약어 기능'과 놓친 음성을 되돌려 듣는 특허 기술 '타임머신 기능'을 구체적으로 설명했는지 평가한다."},
    {"항목": "키보드 체험 또는 시연", "배점": 2, "기준": "[방문 상담 핵심] 상담사가 직접 타임머신과 약어 기능을 시연하여 빠른 입력 속도를 눈으로 확인시켜 주거나, 고객이 직접 키보드를 눌러보고 체험할 수 있도록 유도했는지 평가한다."},
    {"항목": "플랫폼 및 서비스 연결 설명", "배점": 2, "기준": "'소리바로', '웹포스' 등 AI 협업 프로그램의 화면을 보여주고, 공식 일자리 플랫폼 '웍스파이'를 띄워 실시간 일거리가 업로드되는 화면을 눈으로 직접 확인시켜 주었는지 평가한다."},
    {"항목": "학습 과정 설명", "배점": 2, "기준": "소리자바 아카데미를 통한 무료 입문 강의부터 실시간 화상 강의, 실무 연수까지 이어지는 100% 온라인 학습 커리큘럼을 체계적으로 안내했는지 평가한다."},
    {"항목": "학습 기간 설명", "배점": 2, "기준": "프리랜서(VOD) 실무 투입까지는 하루 2~3시간 투자 시 약 3~4개월, 자격증 3급 취득까지는 평균 6개월이 소요된다는 점을 명확한 숫자로 설명했는지 평가한다."},
    {"항목": "자격증 구조 설명", "배점": 2, "기준": "한글속기 자격증(1~3급)이 필기 없이 실기 90% 절대평가임을 알리고, 2026년까지 한시적으로 10분의 수정 시간이 주어져 난이도가 대폭 하향(취득 적기)되었음을 고지했는지 평가한다."},
    {"항목": "실무 연결 설명", "배점": 2, "기준": "자격증 취득 전이라도 협회의 실무 연수를 거쳐 '웍스파이'를 통해 VOD 자막 등 프리랜서 작업에 즉시 투입될 수 있는 구조를 설명했는지 평가한다."},
    {"항목": "수익 구조 설명", "배점": 3, "기준": "고객의 투입 가능 시간에 맞춰 \"하루 2시간 투자 시 월 100~150만 원\", \"전업 시 월 300~600만 원\" 등 직군별/연차별 급여 테이블을 시각적으로 보여주며 수익성을 증명했는지 평가한다."},
    {"항목": "커리어 경로 설명", "배점": 2, "기준": "자격증 하나로 프리랜서로 시작해 향후 공공기관(속기공무원), 속기사무소 창업 등 세 가지 방향으로 커리어를 장기적으로 확장할 수 있음을 어필했는지 평가한다."},
    {"항목": "현실 균형 설명", "배점": 2, "기준": "무조건 쉽다는 장점만 부각하지 않고, \"초반 키보드 배열 적응 기간이 필요하다\", \"장비 구입이라는 초기 투자가 필수적이다\" 등 현실적인 조건과 노력의 필요성을 균형 있게 설명했는지 평가한다."},
    {"항목": "고객 질문 대응", "배점": 8, "기준": "고객이 제기한 질문에 객관적 자료를 근거로 논리적으로 답변했는지 평가한다. (예: \"AI가 대체하지 않나요?\" 질문에 \"오히려 AI가 못하는 뉘앙스와 감정을 보완하는 AI 데이터 전문 속기사가 각광받는다\"라고 대응)"},
    {"항목": "고객 불안 선제 언급", "배점": 10, "기준": "고객이 먼저 묻지 않더라도 \"대부분 기계값 300만 원이 부담된다고 하십니다\", \"포기자가 많다던데 내가 할 수 있을까 걱정하시죠?\" 등 주요 불안 요소를 선제적으로 끄집어내어 공감을 형성했는지 평가한다."},
    {"항목": "고객 상황 맞춤 불안 해소", "배점": 9, "기준": "선제 언급된 불안 요소에 대해 맞춤 솔루션을 제시했는지 평가한다. (비용 불안 시: 최대 24개월 무이자 할부로 월 10만 원대 안내 / 중도 포기 불안 시: 실무 병행으로 원금 회수가 빠름을 안내 / 중고 구매: 정회원 혜택 불가 경고)"},
    {"항목": "다음 행동 제시", "배점": 8, "기준": "상담 후 단순히 \"생각해 보세요\"가 아니라, 가예약, 색상 지정(재고 부족 어필), 당일 혜택 적용을 위한 결제 진행 등 명확하고 구체적인 다음 행동 지침을 제시했는지 평가한다."},
    {"항목": "결심 유도 및 상담 마무리", "배점": 10, "기준": "[방문 상담 핵심] 곧 있을 팀벨 장비 가격 인상 예정 사실을 고지하고, 약 800만 원 상당의 합격 풀케어 서비스 가치를 어필하며, 현금 결제 시 10만 원 추가 할인 및 당일 현장 방문 사은품(키스킨 등) 증정 혜택을 강조하여 당일 결제를 강력하고 성공적으로 유도했는지 평가한다."},
]

_DEFAULT_RUBRIC_비대면 = [
    {"항목": "첫 인사 및 상담 시작 안내", "배점": 2, "기준": "통화 가능 여부를 묻고 \"전화상담은 5~10분 정도 소요될 예정입니다\"라며 짧은 소요 시간과 상담 목적을 명확히 안내했는지 평가한다."},
    {"항목": "긴장 완화 및 라포 형성", "배점": 2, "기준": "유선상 고객이 편하게 들을 수 있도록 밝고 부드러운 음성으로 대화를 시작했는지 평가한다."},
    {"항목": "상담 진행 구조 안내", "배점": 2, "기준": "\"준비과정, 교육과정, 취업분야 순으로 핵심만 요약해서 설명해 드릴게요\"라고 짧은 상담의 흐름을 짚어주었는지 평가한다."},
    {"항목": "고객 참여 유도", "배점": 2, "기준": "\"추가로 궁금한 점이 있으시면 언제든 말씀해 주세요\"라고 안내하여 유선상 일방적인 전달이 되지 않도록 참여를 유도했는지 평가한다."},
    {"항목": "관심 계기 파악", "배점": 3, "기준": "속기사를 어떻게 알고 문의를 남겼는지 빠르게 질문하여 유입 경로를 파악했는지 평가한다."},
    {"항목": "현재 상황 파악", "배점": 3, "기준": "직장인, 주부, 취준생 등 직업을 파악하고 속기사를 알아본 기간을 빠르게 확인했는지 평가한다."},
    {"항목": "목표 확인", "배점": 3, "기준": "프리랜서(투잡/부업)인지, 공무원 등 출퇴근을 원하는지 질문하여 핵심 타겟을 설정했는지 평가한다."},
    {"항목": "정보 수준 확인", "배점": 2, "기준": "키보드 비용(342만 원) 등에 대해 대략적으로 알고 있는지 질문하여 사전 정보 수준을 파악했는지 평가한다."},
    {"항목": "맞춤형 상담 전개", "배점": 1, "기준": "파악된 상황에 맞춰 5~10분 내에 고객이 가장 궁금해할 내용(예: 투잡 희망 시 프리랜서 비중 확대)으로 시간을 효율적으로 배분했는지 평가한다."},
    {"항목": "속기사 직업 정의 설명", "배점": 3, "기준": "단순 타자가 아닌 '말소리부터 비언어적 표현까지 기록하는 전문가'임을 핵심만 간략히 정의했는지 평가한다."},
    {"항목": "취업처 구조 설명", "배점": 3, "기준": "프리랜서(VOD 등)와 공무원(법원 등)의 두 가지 큰 줄기로 나누어 간결하게 취업처를 안내했는지 평가한다."},
    {"항목": "시장 전망 설명", "배점": 3, "기준": "OTT 영상 자막 수요 등 속기사 수요가 폭발적으로 늘고 있어 진입하기 좋은 시기임을 어필했는지 평가한다."},
    {"항목": "사례 및 근거 활용", "배점": 2, "기준": "육아 병행이나 직장인 투잡 등 고객 상황에 맞는 짧은 성공 사례를 언급하여 공감을 주었는지 평가한다."},
    {"항목": "고객 맞춤 직무 연결", "배점": 1, "기준": "\"직장 병행이시면 VOD 자막 작업이 유연해서 좋습니다\"처럼 빠른 직무 매칭을 제공했는지 평가한다."},
    {"항목": "속기 키보드 구조 설명", "배점": 2, "기준": "여러 글자를 동시에 누르는 방식으로 일반 키보드와 구조가 달라 전용 키보드가 필수임을 설명했는지 평가한다."},
    {"항목": "키보드 기능 설명", "배점": 2, "기준": "자주 쓰는 단어를 묶는 약어 기능과 돌려 듣는 타임머신 등 필수 기능을 간략히 안내했는지 평가한다."},
    {"항목": "키보드 체험 또는 시연", "배점": 2, "기준": "[비대면 핵심] \"유선이라 직접 보여드릴 수 없으니 방문하셔서 직접 키보드를 체험해 보시길 권해드립니다\"라고 방문 체험의 필요성을 어필했는지 평가한다."},
    {"항목": "플랫폼 및 서비스 연결 설명", "배점": 2, "기준": "프리랜서 일거리를 제공받을 수 있는 협회 공식 플랫폼 '웍스파이'를 간략히 언급했는지 평가한다."},
    {"항목": "학습 과정 설명", "배점": 2, "기준": "소리자바 아카데미를 통한 전 과정 100% 온라인 학습 및 실시간 화상 강의가 가능함을 설명했는지 평가한다."},
    {"항목": "학습 기간 설명", "배점": 2, "기준": "3급 자격증 기준 평균 6개월~1년, 프리랜서 실무 투입은 약 3~4개월 연습 시 가능함을 명확한 수치로 전달했는지 평가한다."},
    {"항목": "자격증 구조 설명", "배점": 2, "기준": "내년(2026년)까지 한시적으로 수정 시간 10분이 주어져 자격증 취득 적기임을 고지했는지 평가한다."},
    {"항목": "실무 연결 설명", "배점": 2, "기준": "자격증 취득 전이라도 실력을 갖추면 프리랜서 연수를 거쳐 실무 진입이 가능함을 설명했는지 평가한다."},
    {"항목": "수익 구조 설명", "배점": 3, "기준": "분야나 실력에 따라 다르나, 부업 시 월 100~200만 원 수익이 가능함을 구체적 수치로 설명했는지 평가한다."},
    {"항목": "커리어 경로 설명", "배점": 2, "기준": "프리랜서 활동 후 공공기관으로 커리어를 확장할 수 있음을 간략히 설명했는지 평가한다."},
    {"항목": "현실 균형 설명", "배점": 2, "기준": "난이도는 높지 않으나 꾸준한 연습(하루 2시간 등)이 뒷받침되어야 함을 현실적으로 짚어주었는지 평가한다."},
    {"항목": "고객 질문 대응", "배점": 8, "기준": "AI 대체 우려 등에 대해 \"오히려 AI가 못하는 영역을 사람이 보완하여 수요가 증가한다\"라고 논리적으로 대응했는지 평가한다."},
    {"항목": "고객 불안 선제 언급", "배점": 10, "기준": "고객이 말하기 전에 \"검색해 보시면 일반 쇼핑몰에 10만 원대 키보드가 나오는데, 이건 공인된 장비가 아니라 시험/실무용으로 쓸 수 없다\"며 중고/저가 구매 불안이나 유혹을 선제적으로 짚었는지 평가한다."},
    {"항목": "고객 상황 맞춤 불안 해소", "배점": 10, "기준": "비용 부담 시 \"무이자 최대 24개월로 월 10만 원대\" 솔루션을 제시하여 단순 지출이 아닌 내 기술을 위한 투자로 인식시켰는지 평가한다."},
    {"항목": "다음 행동 제시", "배점": 8, "기준": "[비대면 핵심] 통화 종료 후 속기 수요를 눈으로 볼 수 있는 '웍스파이' 등 관련 자료를 문자로 발송하겠다고 명확히 안내했는지 평가한다."},
    {"항목": "결심 유도 및 상담 마무리", "배점": 9, "기준": "[비대면 핵심] 팀벨 장비 가격 인상 예정 사실을 고지하고, \"1차 전화 상담 후 방문 시 키보드 구매 추가 5만 원 할인이 적용됩니다\"라며 대면 방문 예약 및 빠른 결정을 성공적으로 유도했는지 평가한다."},
]

_DEFAULT_RUBRIC_전화 = [
    {"항목": "첫 인사 및 상담 시작 안내", "배점": 2, "기준": "\"전화상담으로 신청 주셨으며 시간은 15~20분 정도 소요됩니다\"라며 심층 상담의 소요 시간을 사전 안내했는지 평가한다."},
    {"항목": "긴장 완화 및 라포 형성", "배점": 2, "기준": "\"예약 주신 시간에 맞춰 전화드렸습니다\"라며 예의 바르고 친절하게 라포를 형성했는지 평가한다."},
    {"항목": "상담 진행 구조 안내", "배점": 2, "기준": "속기사의 정의, 취업 분야, 준비 방법 순으로 상세하게 설명할 것임을 안내했는지 평가한다."},
    {"항목": "고객 참여 유도", "배점": 2, "기준": "심층 대화 중 \"설명을 들으시다가 궁금한 점이 생기면 언제든 말씀해주세요\"라고 적극적인 질의응답을 열어두었는지 평가한다."},
    {"항목": "관심 계기 파악", "배점": 3, "기준": "인스타그램, 유튜브, 지인 등 어떠한 구체적 경로를 통해 속기를 접하게 되었는지 심도 있게 파악했는지 평가한다."},
    {"항목": "현재 상황 파악", "배점": 3, "기준": "공부에 투자할 수 있는 시간 여력을 확인하기 위해 현재 소속(직장, 취준생, 주부 등)을 정확히 파악했는지 평가한다."},
    {"항목": "목표 확인", "배점": 3, "기준": "공무원, 프리랜서(투잡/재택/데이터전문), 창업 등 고객이 진정으로 희망하는 궁극적 목표를 파악했는지 평가한다."},
    {"항목": "정보 수준 확인", "배점": 2, "기준": "\"속기사에 대해 어느 정도 궁금한 점을 가지고 계신가요?\"라며 비용, 취득 기간, 비전 등 고객의 정보 수준과 가장 큰 궁금증을 확인했는지 평가한다."},
    {"항목": "맞춤형 상담 전개", "배점": 1, "기준": "파악한 고객의 목표(예: 프리랜서)에 맞춰 VOD, 라콘, AI데이터 등 해당 분야 위주로 딥다이브하여 맞춤 전개했는지 평가한다."},
    {"항목": "속기사 직업 정의 설명", "배점": 3, "기준": "말소리뿐만 아니라 비언어적 표현까지 빠르고 정확하게 기록하는 기록 전문가로 전문성을 강조하여 정의했는지 평가한다."},
    {"항목": "취업처 구조 설명", "배점": 3, "기준": "국회, 법원, 검찰 등 공무원 분야와 VOD, 방송 자막 등 프리랜서 분야를 매우 구체적인 직렬을 들어 세분화하여 설명했는지 평가한다."},
    {"항목": "시장 전망 설명", "배점": 3, "기준": "장애인차별금지법, 공공기록물관리법, 그리고 OTT 콘텐츠의 확대로 속기사 인력 수요가 급증하고 있음을 객관적 근거를 들어 설명했는지 평가한다."},
    {"항목": "사례 및 근거 활용", "배점": 2, "기준": "수많은 합격생 리스트 중 고객의 성별, 연령대, 직업이 유사한 1~2명의 실제 데이터(예: 직장 병행하여 프리랜서 진입 사례 등)를 구체적으로 활용했는지 평가한다."},
    {"항목": "고객 맞춤 직무 연결", "배점": 1, "기준": "\"고객님 상황이라면 하루 2시간 투자로 VOD가 제격입니다\"처럼 구체적인 직무를 타겟팅해 주었는지 평가한다."},
    {"항목": "속기 키보드 구조 설명", "배점": 2, "기준": "일반 키보드로 할 수 없는 세벌식 동시 입력의 원리를 설명하여 장비 구매의 당위성을 논리적으로 납득시켰는지 평가한다."},
    {"항목": "키보드 기능 설명", "배점": 2, "기준": "타임머신, 약어 등 속기 전용 키보드만이 가진 특수 기능을 심층적으로 설명했는지 평가한다."},
    {"항목": "키보드 체험 또는 시연", "배점": 2, "기준": "전화상담이므로 \"방문하시면 속기 키보드 체험은 물론 자세한 정보를 눈으로 보며 안내받으실 수 있습니다\"라며 대면 체험을 적극 권유했는지 평가한다."},
    {"항목": "플랫폼 및 서비스 연결 설명", "배점": 2, "기준": "'웍스파이(Worksfy)' 플랫폼을 통해 자격증 취득 전후로 일거리가 배정되는 구조를 명확히 설명했는지 평가한다."},
    {"항목": "학습 과정 설명", "배점": 2, "기준": "소리자바 아카데미의 무료 입문 강의 제공과 실시간 화상 강의를 통한 100% 재택 학습 커리큘럼을 체계적으로 설명했는지 평가한다."},
    {"항목": "학습 기간 설명", "배점": 2, "기준": "VOD 약 1~2달, 라콘 약 4~5개월 등 각 분야별 진입에 필요한 구체적 준비 기간을 수치화하여 설명했는지 평가한다."},
    {"항목": "자격증 구조 설명", "배점": 2, "기준": "필기 없이 실기 90%로 평가되며, 2026년까지 수정 시간 10분이 추가되어 난이도가 크게 낮아졌음을 강조했는지 평가한다."},
    {"항목": "실무 연결 설명", "배점": 2, "기준": "자격증 유무와 관계없이 연수를 통해 중급 수준(약 3개월)부터 실무(웍스파이)에 투입되어 수익 창출이 가능하다는 점을 설명했는지 평가한다."},
    {"항목": "수익 구조 설명", "배점": 3, "기준": "실력에 따른 단가 차등을 설명하며, 부업 시 100~200만 원, 전업 고수익 시 400~600만 원의 현실적 기대 수익을 제시했는지 평가한다."},
    {"항목": "커리어 경로 설명", "배점": 2, "기준": "프리랜서로 시작해 공공기관, 나아가 창업까지 연결될 수 있는 속기사의 생애 주기를 짚어주었는지 평가한다."},
    {"항목": "현실 균형 설명", "배점": 2, "기준": "수익을 내기 위해서는 학습 초반 단기적인 집중 투자(실력 완성)와 노력이 수반되어야 한다는 점을 균형 있게 짚었는지 평가한다."},
    {"항목": "고객 질문 대응", "배점": 10, "기준": "AI 대체나 수익의 불안정성에 대한 질문을 회피하지 않고 \"AI와 협업하는 AI속기사가 신설되는 등 오히려 수요가 는다\"라며 정면으로 방어했는지 평가한다."},
    {"항목": "고객 불안 선제 언급", "배점": 10, "기준": "\"비용 때문에 시작 못하고 흐지부지 넘어갔다가 2~3년 뒤 후회하는 분들이 많습니다\" 등 망설임을 유발하는 주요 고민(비용, 포기)을 선제적으로 끄집어냈는지 평가한다."},
    {"항목": "고객 상황 맞춤 불안 해소", "배점": 8, "기준": "중도 포기 우려 시 \"실무를 병행하며 돈을 벌면서 하므로 동기부여가 됩니다\", 비용 불안 시 \"12~24개월 무이자 시 월 10만 원대\"라며 논리적인 맞춤 솔루션을 주었는지 평가한다."},
    {"항목": "다음 행동 제시", "배점": 8, "기준": "[전화 핵심] 상담 종료 시 \"웍스파이 등 실제 프로젝트 수요를 확인할 수 있는 자료를 문자로 보내드리겠습니다\"라며 명확한 후속 안내를 했는지 평가한다."},
    {"항목": "결심 유도 및 상담 마무리", "배점": 9, "기준": "[전화 핵심] 팀벨 장비 가격 인상 예정 사실을 고지하고 \"지금이 혜택을 다 챙기실 수 있는 가장 좋은 타이밍\"이라며 망설임(\"100% 확신을 갖고 시작하는 사람은 없다\")을 극복시키고 결정/가예약을 강력히 촉구했는지 평가한다."},
]

_DEFAULT_RUBRIC_공통 = [
    {"항목": "첫 인사 및 상담 시작 안내", "배점": 2, "기준": "고객 상황에 맞는 정중한 첫인상과 함께 오늘 상담에서 다룰 주요 안건(목적)을 분명하게 제시하며 주도권을 잡았는지 평가한다."},
    {"항목": "긴장 완화 및 라포 형성", "배점": 2, "기준": "고객이 낯선 직업에 대해 문의하며 갖는 부담감을 줄여주기 위해 공감대 형성 멘트를 적절히 사용했는지 평가한다."},
    {"항목": "상담 진행 구조 안내", "배점": 2, "기준": "설명이 중구난방이 되지 않도록, 상담 초반에 어떤 순서(비전-교육-비용 등)로 이야기를 풀어나갈지 구조를 안내했는지 평가한다."},
    {"항목": "고객 참여 유도", "배점": 2, "기준": "혼자서 설명만 늘어놓지 않고, 중간중간 \"여기까지 이해되셨나요?\", \"궁금한 점 있으세요?\"라며 반응을 살피고 질문을 유도했는지 평가한다."},
    {"항목": "관심 계기 파악", "배점": 3, "기준": "고객이 먼저 속기사에 관심을 갖게 된 개인적인 동기(계기)를 명확히 질문하여 대화의 맥락을 형성했는지 평가한다."},
    {"항목": "현재 상황 파악", "배점": 3, "기준": "직장, 가사, 공부 여력 등 고객이 현재 처한 물리적/시간적 상황을 구체적으로 질문하여 파악했는지 평가한다."},
    {"항목": "목표 확인", "배점": 3, "기준": "단기적 수익(부업)인지, 장기적 취업(공무원)인지 고객이 기대하는 궁극적 목표를 확인했는지 평가한다."},
    {"항목": "정보 수준 확인", "배점": 2, "기준": "키보드 가격대, 자격증 유무 등 속기에 대해 이미 알고 있는 지식이 어느 정도인지 점검하여 설명의 깊이를 조절했는지 평가한다."},
    {"항목": "맞춤형 상담 전개", "배점": 1, "기준": "고객의 상황과 목표에 맞추어 불필요한 설명은 줄이고, 고객이 가장 원하는 분야에 시간을 집중적으로 할애했는지 평가한다."},
    {"항목": "속기사 직업 정의 설명", "배점": 3, "기준": "속기사를 단순 타이핑 직업이 아닌 '말의 뉘앙스와 맥락까지 정확하게 기록하는 기록 전문가'로 품격 있게 정의했는지 평가한다."},
    {"항목": "취업처 구조 설명", "배점": 3, "기준": "공공기관(법원, 국회 등), 기업체, 프리랜서(VOD, 라콘 등)로 속기사가 뻗어나갈 수 있는 모든 취업 구조를 안내했는지 평가한다."},
    {"항목": "시장 전망 설명", "배점": 3, "기준": "장애인차별금지법, 공공기록물관리법, OTT 콘텐츠 증가 등의 객관적 법령과 시장 상황을 근거로 폭발적인 수요 전망을 설명했는지 평가한다."},
    {"항목": "사례 및 근거 활용", "배점": 2, "기준": "매뉴얼의 '맞춤 사례 모음'에 기반하여, 고객과 동일한 나이대나 직업군이 속기로 성공한 실제 인물 케이스를 인용하여 신뢰감을 주었는지 평가한다."},
    {"항목": "고객 맞춤 직무 연결", "배점": 1, "기준": "파악된 고객 니즈에 가장 잘 맞는 단 1개의 직무(예: 라이브콘텐츠, 법원 등)를 콕 집어 고객만의 최적 경로로 추천했는지 평가한다."},
    {"항목": "속기 키보드 구조 설명", "배점": 2, "기준": "자음/모음/받침을 동시에 누르는 세벌식 구조를 설명하며 전용 장비가 왜 반드시 필요한지 기술적 차이를 명확히 했는지 평가한다."},
    {"항목": "키보드 기능 설명", "배점": 2, "기준": "긴 단어를 단축하는 약어 기능과 놓친 음성을 잡는 타임머신 기능을 설명하여 업무 효율성을 증명했는지 평가한다."},
    {"항목": "키보드 체험 또는 시연", "배점": 2, "기준": "대면/비대면 환경에 맞추어 시연을 진행하거나, 방문을 유도하여 키보드의 우수성을 직접 체감할 수 있는 방안을 제시했는지 평가한다."},
    {"항목": "플랫폼 및 서비스 연결 설명", "배점": 2, "기준": "자격증 취득 전후로 일을 배정받는 공식 프리랜서 플랫폼 '웍스파이(Worksfy)' 및 AI 협업 툴(소리바로 등)의 존재를 설명했는지 평가한다."},
    {"항목": "학습 과정 설명", "배점": 2, "기준": "소리자바 아카데미를 통한 100% 온라인 교육, 실시간 화상 강의, 연수까지 체계적인 원스톱 학습 커리큘럼을 안내했는지 평가한다."},
    {"항목": "학습 기간 설명", "배점": 2, "기준": "\"3급 기준 평균 6개월\", \"프리랜서 실무 약 3개월\" 등 불명확하지 않은 구체적인 숫자로 예상 소요 기간을 제시했는지 평가한다."},
    {"항목": "자격증 구조 설명", "배점": 2, "기준": "한글속기가 실기 90% 절대평가임을 설명하고, 2026년까지 한시적 10분 수정 시간 부여로 난이도가 대폭 완화된 점을 필수 고지했는지 평가한다."},
    {"항목": "실무 연결 설명", "배점": 2, "기준": "자격증을 따기 전이어도 연습만 꾸준히 하면 연수를 거쳐 실무 작업(VOD 자막 등)에 바로 투입될 수 있음을 설명했는지 평가한다."},
    {"항목": "수익 구조 설명", "배점": 3, "기준": "투입하는 시간(하루 2시간 vs 풀타임)에 따라 100만 원~600만 원까지 기대할 수 있는 수익의 폭과 단가 차등 구조를 설명했는지 평가한다."},
    {"항목": "커리어 경로 설명", "배점": 2, "기준": "지금 시작해서 프리랜서로 경험을 쌓고 향후 안정적인 공무원이나 속기사무소 창업으로 나아갈 수 있는 평생 커리어 비전을 제시했는지 평가한다."},
    {"항목": "현실 균형 설명", "배점": 2, "기준": "무조건 다 돈을 잘 번다고 포장하지 않고, 단기간(3~6개월) 집중적인 훈련과 꾸준함이 있어야 수익이 발생한다는 현실을 균형감 있게 짚었는지 평가한다."},
    {"항목": "고객 질문 대응", "배점": 9, "기준": "AI 기술 대체 우려, 비용, 나이 제한 등에 대한 반론에 대해 '대응 멘트 모음'의 논리(AI 협업, 원금 단기 회수 등)를 완벽히 적용해 방어했는지 평가한다."},
    {"항목": "고객 불안 선제 언급", "배점": 10, "기준": "고객이 차마 묻지 못한 중고 키보드 구매 시 정회원 혜택 박탈 위험을 선제적으로 경고하고 비용 부담 등 공통 불안 요소를 먼저 꺼내 공감했는지 평가한다."},
    {"항목": "고객 상황 맞춤 불안 해소", "배점": 10, "기준": "고객이 처한 자금/시간 부족 등 불안 요소에 대해 12~24개월 무이자 할부 등 실질적인 해결 솔루션을 제시하여 불안감을 제거했는지 평가한다."},
    {"항목": "다음 행동 제시", "배점": 8, "기준": "당일 결제, 색상 예약, 웍스파이 자료 문자 발송 확인, 혹은 방문 상담 예약 등 고객이 통화/상담 직후 취해야 할 액션을 아주 구체적으로 지시했는지 평가한다."},
    {"항목": "결심 유도 및 상담 마무리", "배점": 8, "기준": "[공통 핵심] 곧 있을 팀벨 장비 가격 인상 예정 사실을 강력히 고지하고, 오늘이 키스킨 증정/무이자 등 혜택을 챙길 수 있는 최적기임을 어필하며 당일 가예약/결제를 확정형 멘트로 이끌어냈는지 평가한다."},
]

# 하위 호환용 — 레거시 코드에서 _DEFAULT_RUBRIC 을 직접 참조하는 경우를 위해 유지
_DEFAULT_RUBRIC = _DEFAULT_RUBRIC_공통

# 사용자 커스텀 설정 기본값
DEFAULT_USER_CONFIG = {
    "quality_weights": dict(QUALITY_WEIGHTS),   # 하위 호환용 (레거시)
    "channel_rubrics": {
        "방문":  [dict(r) for r in _DEFAULT_RUBRIC_방문],
        "비대면": [dict(r) for r in _DEFAULT_RUBRIC_비대면],
        "전화":  [dict(r) for r in _DEFAULT_RUBRIC_전화],
        "공통":  [dict(r) for r in _DEFAULT_RUBRIC_공통],
    },
    "flow_stages": ["고객상황파악", "문제공감", "직업설명", "수익구조설명", "사례제시", "행동유도"],
    "system_prompt": (
        "당신은 직업훈련 상담 품질 평가 전문가입니다.\n"
        "상담사가 고객의 직업훈련 참여 결심을 돕는 과정을 면밀히 분석하세요.\n"
        "고객의 상황(나이, 경력, 목표)을 정확히 파악하고, 상담사가 맞춤형 솔루션을 제공했는지 평가하세요."
    ),
    "mandatory_items": [
        "웍스파이(Worksfy) 실제 수요 인증",
        "가격 인상 예정 고지",
        "구매 적기 어필",
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
# CRM 구글 시트 연동 유틸리티
# ══════════════════════════════════════════════════════════

# 지사 탭 목록 (구글 시트에서 읽을 워크시트 이름)
_CRM_BRANCHES = ["강남", "가산", "대전", "대구", "광주", "부산"]

# 구글 시트 컬럼 위치 (0-based 인덱스; 헤더 행 제외)
# A=0, B=1, ... H=7, ... X=23, Y=24, ... AI=34, AJ=35
_CRM_COL_CUSTOMER = 5    # F열: 가입자명(고객명)
_CRM_COL_DATE     = 7    # H열: 상담 날짜
_CRM_COL_STAFF    = 23   # X열: 상담 직원
_CRM_COL_MODE     = 24   # Y열: 상담 방식
_CRM_COL_PURCHASE = 34   # AI열: 구입 여부 (결제완료 / 예약 / 미결제 등)
_CRM_COL_PURCH_DT = 35   # AJ열: 구입 날짜

# 결제 성공 판단 키워드 — 시트 AI열에 정확히 "구입"이 기입된 경우만 전환 성공
_PURCHASE_KEYWORD = "구입"

# ── 월별 통계 시트 열 인덱스 (YYYY-MM 형식 단일 시트, 17~32행 기준)
# A=0:지사명  B=1:직원명  Q=16:방문상담  W=22:비대면상담  AC=28:전화상담
_CRM_SUM_COL_BRANCH = 0   # A열: 지사명
_CRM_SUM_COL_STAFF  = 1   # B열: 직원명
_CRM_SUM_COL_VISIT  = 16  # Q열: 방문 상담 건수 (합산)
_CRM_SUM_COL_ONLINE = 22  # W열: 비대면 상담 건수 (합산)
_CRM_SUM_COL_PHONE  = 28  # AC열: 전화 상담 건수 (합산)
# 데이터 영역: 엑셀 16행(제목줄) ~ 32행(이세희)
# header=None으로 전체 읽기 → iloc[14:32] = 엑셀 15행~32행 → 제목줄 포함
# 이후 B열 필터로 '직원명'·'합계'·빈칸 제거 → 17행(서채윤)~32행(이세희) 실데이터만 남음
_CRM_SUM_ROW_START  = 14  # iloc 시작: 엑셀 15행 (제목줄 포함)
_CRM_SUM_ROW_END    = 32  # iloc 끝(exclusive): 엑셀 32행(이세희) 포함

def _norm_key(s) -> str:
    """이름 비교용 정규화: 일반공백·비분리공백(\\u00a0)·전각공백(\\u3000)·zero-width(\\u200b) 모두 제거."""
    return re.sub(r"[\s\u00a0\u3000\u200b]+", "", str(s or "")).strip()

def _pure_korean(s) -> str:
    """이름에서 한글(가-힣)만 추출. 괄호·특수문자·숫자·영문·공백 모두 제거.
    (신)·[방문] 등 접두사가 남아 있어도 순수 한글 이름 비교가 가능해진다.
    주의: 최소 2자 미만이면 빈 문자열 반환 (오탐 방지).
    """
    k = re.sub(r"[^가-힣]", "", str(s or ""))
    return k if len(k) >= 2 else ""

# ── CRM 구글 시트 기본 URL (공개 공유 링크 — secrets.toml 없이도 동작) ──
# 시트가 "링크 있는 모든 사용자 - 뷰어" 이상 공개 상태여야 합니다.
_CRM_SHEET_ID  = "1N-8DO73uT9tzTQgUYL9EOCdirVW6CrNA6NRJGOR8Wus"
_CRM_SHEET_URL = f"https://docs.google.com/spreadsheets/d/{_CRM_SHEET_ID}"

# ── CRM 캐시 TTL 설정 ──────────────────────────────────────────────────────
# 6시간 = 21600초  /  3시간(선희님 요청 시) = 10800초
_CRM_CACHE_TTL = 21600


@st.cache_data(ttl=_CRM_CACHE_TTL)
def load_crm_data(sheet_id: str = _CRM_SHEET_ID, cutoff_date: str = "") -> pd.DataFrame:
    """
    구글 시트 지사별 탭에서 CRM 데이터를 모두 불러와 하나의 DataFrame으로 병합합니다.
    @st.cache_data(ttl=_CRM_CACHE_TTL) — 6시간 캐시 (변경: _CRM_CACHE_TTL 상수).

    cutoff_date: "YYYY-MM-DD" 형식. 호출 시 오늘 날짜를 전달하면 날짜가 바뀔 때마다
    캐시 키가 달라져 자동으로 새 캐시 엔트리가 생성됩니다 (미래 날짜 완전 차단).

    ── 인증 방식 (자동 선택) ──────────────────────────────────────────
    방법 A (우선): .streamlit/secrets.toml 에 서비스 계정 설정 시 → st.connection 사용
    방법 B (기본): 설정 없음 → 공개 시트 CSV 내보내기 URL로 직접 읽기
                  (시트가 "링크 있는 모든 사용자" 공개 설정이어야 함)

    ── secrets.toml 서비스 계정 설정 예시 ────────────────────────────
    [connections.gsheets]
    spreadsheet = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit"
    type = "gsheets"

    [connections.gsheets.credentials]
    type = "service_account"
    project_id = "your-gcp-project-id"
    private_key_id = "abc123..."
    private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
    client_email = "your-sa@your-project.iam.gserviceaccount.com"
    client_id = "123456789"
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."
    universe_domain = "googleapis.com"
    ※ private_key 내 실제 줄바꿈은 두 글자 \\n 으로 입력하세요.
    ── Streamlit Cloud 배포 시: 앱 Settings > Secrets 에 동일 내용 ──
    """
    frames: list[pd.DataFrame] = []

    # secrets.toml 에 connections.gsheets 설정 여부 확인
    _has_secrets = False
    if _GSHEETS_OK:
        try:
            _has_secrets = bool(st.secrets.get("connections", {}).get("gsheets"))
        except Exception:
            pass

    for branch in _CRM_BRANCHES:
        try:
            if _has_secrets:
                # ── 방법 A: 서비스 계정 인증 (비공개 시트 가능) ──
                conn = st.connection("gsheets", type=_GSheetsConnection)
                df_raw = conn.read(
                    spreadsheet=f"https://docs.google.com/spreadsheets/d/{sheet_id}",
                    worksheet=branch,
                    ttl=_CRM_CACHE_TTL,
                )
            else:
                # ── 방법 B: 공개 시트 CSV 내보내기 (credentials 불필요) ──
                import urllib.parse
                csv_url = (
                    f"https://docs.google.com/spreadsheets/d/{sheet_id}"
                    f"/gviz/tq?tqx=out:csv&sheet={urllib.parse.quote(branch)}"
                )
                df_raw = pd.read_csv(csv_url, header=0)

            if df_raw is None or df_raw.empty:
                continue

            n_cols = df_raw.shape[1]
            col_map = {
                "가입자명":     _CRM_COL_CUSTOMER,
                "상담날짜":     _CRM_COL_DATE,
                "상담직원":     _CRM_COL_STAFF,
                "상담방식_crm": _CRM_COL_MODE,
                "구입여부":     _CRM_COL_PURCHASE,
                "구입날짜":     _CRM_COL_PURCH_DT,
            }
            # 시트 열 수가 부족한 경우 존재하는 열만 추출
            valid = {k: v for k, v in col_map.items() if v < n_cols}
            if "가입자명" not in valid:
                continue

            sub = df_raw.iloc[:, list(valid.values())].copy()
            sub.columns = list(valid.keys())
            sub["지사_crm"] = branch

            # ▶ 날짜 다형식 파싱: "2024.03.15" / "2024/03/15" / "2024-03-15" 모두 인식
            if "상담날짜" in sub.columns:
                _date_str = (
                    sub["상담날짜"].astype(str)
                    .str.strip()
                    .str.replace(r"[./]", "-", regex=True)   # 마침표·슬래시 → 하이픈
                )
                sub["상담날짜_dt"] = pd.to_datetime(_date_str, errors="coerce")
            else:
                sub["상담날짜_dt"] = pd.NaT
            sub["상담날짜_dt"] = sub["상담날짜_dt"].astype("datetime64[ns]")

            # ▶▶ 루프 내 즉시 가지치기: 미래 행을 브랜치 단위로 영구 삭제
            #    cutoff_date 파라미터(호출 시 오늘 날짜)를 사용 → 캐시 고정 문제 없음
            _cutoff = pd.Timestamp(cutoff_date) if cutoff_date else pd.Timestamp(datetime.now().date())
            sub = sub.loc[
                sub["상담날짜_dt"].notna() & (sub["상담날짜_dt"] <= _cutoff)
            ].copy()
            if sub.empty:
                continue

            frames.append(sub)
        except Exception:
            pass    # 개별 시트 실패 시 나머지 계속 진행

    if not frames:
        return pd.DataFrame()

    crm = pd.concat(frames, ignore_index=True)

    # 가입자명 정제 및 빈 행 제거
    crm["가입자명"] = crm["가입자명"].astype(str).str.strip()
    crm = crm[~crm["가입자명"].isin(["", "nan", "None"])]

    # 매핑 키: 이름 내 모든 공백 제거 후 비교 (모듈 레벨 _norm_key 사용)
    crm["가입자명_key"] = crm["가입자명"].apply(_norm_key)
    if "상담직원" in crm.columns:
        crm["상담직원_key"] = crm["상담직원"].astype(str).apply(_norm_key)
    else:
        crm["상담직원_key"] = ""

    # 날짜 정규화 — 상담날짜_dt는 각 sub에서 이미 생성됨, str 열만 추가
    if "상담날짜_dt" not in crm.columns:
        crm["상담날짜_dt"] = pd.to_datetime(
            crm["상담날짜"].astype(str).str.strip().str.replace(r"[./]", "-", regex=True)
            if "상담날짜" in crm.columns else pd.Series(dtype="object"),
            errors="coerce",
        ).astype("datetime64[ns]")

    # timezone 제거 (Google Sheets가 tz-aware 반환 시 비교 오류 방지)
    if hasattr(crm["상담날짜_dt"].dtype, "tz") and crm["상담날짜_dt"].dt.tz is not None:
        crm["상담날짜_dt"] = crm["상담날짜_dt"].dt.tz_localize(None)

    crm["상담날짜_str"] = crm["상담날짜_dt"].dt.strftime("%Y-%m-%d").fillna("")

    # ▶ 2차 방어: concat 후 재확인 (루프에서 이미 처리됐지만 확실히)
    _today_cutoff = pd.Timestamp(cutoff_date) if cutoff_date else pd.Timestamp(datetime.now().date())
    crm = crm[crm["상담날짜_dt"].notna() & (crm["상담날짜_dt"] <= _today_cutoff)].copy()

    return crm


@st.cache_data(ttl=_CRM_CACHE_TTL)
def load_crm_purchase_status(sheet_id: str = _CRM_SHEET_ID) -> dict:
    """
    가벼운 결제 상태 전용 로드 — AI열(인덱스 34)과 매칭 키(F열·X열)만 읽어 반환.
    {(가입자명_key, 상담직원_key): "구입" | ""} 형태의 dict를 반환합니다.
    load_crm_data()와 동일한 TTL(_CRM_CACHE_TTL)로 캐시되며,
    대시보드 렌더링 시마다 캐시 만료 여부를 자동으로 확인합니다.
    """
    _has_secrets = False
    if _GSHEETS_OK:
        try:
            _has_secrets = bool(st.secrets.get("connections", {}).get("gsheets"))
        except Exception:
            pass

    status_map: dict = {}
    for branch in _CRM_BRANCHES:
        try:
            if _has_secrets:
                conn   = st.connection("gsheets", type=_GSheetsConnection)
                df_raw = conn.read(
                    spreadsheet=f"https://docs.google.com/spreadsheets/d/{sheet_id}",
                    worksheet=branch,
                    ttl=_CRM_CACHE_TTL,
                )
            else:
                import urllib.parse
                csv_url = (
                    f"https://docs.google.com/spreadsheets/d/{sheet_id}"
                    f"/gviz/tq?tqx=out:csv&sheet={urllib.parse.quote(branch)}"
                )
                df_raw = pd.read_csv(csv_url, header=0)

            if df_raw is None or df_raw.empty:
                continue
            n = df_raw.shape[1]
            for _, row in df_raw.iterrows():
                cust_key  = _norm_key(row.iloc[_CRM_COL_CUSTOMER]) if _CRM_COL_CUSTOMER < n else ""
                staff_key = _norm_key(row.iloc[_CRM_COL_STAFF])    if _CRM_COL_STAFF    < n else ""
                purch_val = str(row.iloc[_CRM_COL_PURCHASE]).strip() if _CRM_COL_PURCHASE < n else ""
                if cust_key:
                    status_map[(cust_key, staff_key)] = purch_val
        except Exception:
            pass
    return status_map


@st.cache_data(ttl=_CRM_CACHE_TTL)
def load_crm_summary(sheet_name: str = "") -> pd.DataFrame:
    """
    YYYY-MM 월별 시트 17~32행에서 직원별 합산 상담 건수를 반환합니다 (요약 통계 전용).

    반환: summary_df
       열: [지사, 직원명, 직원명_key, 방문_CRM, 비대면_CRM, 전화_CRM]

    행 범위:
      header=None 기준: 엑셀 1행=iloc[0] → iloc[14:32] = 엑셀 15~32행
      이후 B열 필터로 헤더·합계·빈칸 제거 → 엑셀 17행(서채윤)~32행(이세희) 실데이터만 남음

    매출 전환 분석용 상세 결제 데이터는 load_crm_detailed_matching_data() 를 사용하세요.
    """
    import urllib.parse

    _has_secrets = False
    if _GSHEETS_OK:
        try:
            _has_secrets = bool(st.secrets.get("connections", {}).get("gsheets"))
        except Exception:
            pass

    _ws = sheet_name if sheet_name else datetime.now().strftime("%Y-%m")

    # ═══════════════════════════════════════════════════════════
    # PART A — YYYY-MM 월별 시트 → summary_df (17~32행)
    # ═══════════════════════════════════════════════════════════
    summary_df = pd.DataFrame()
    try:
        if _has_secrets:
            conn   = st.connection("gsheets", type=_GSheetsConnection)
            df_all = conn.read(
                spreadsheet=f"https://docs.google.com/spreadsheets/d/{_CRM_SHEET_ID}",
                worksheet=_ws,
                ttl=_CRM_CACHE_TTL,
            )
            if df_all is not None and not df_all.empty:
                df_raw = df_all.reset_index(drop=True)
            else:
                df_raw = pd.DataFrame()
        else:
            csv_url = (
                f"https://docs.google.com/spreadsheets/d/{_CRM_SHEET_ID}"
                f"/gviz/tq?tqx=out:csv&sheet={urllib.parse.quote(_ws)}"
            )
            df_raw = pd.read_csv(csv_url, header=None, dtype=str)

        if df_raw is not None and not df_raw.empty:
            # STEP 1: iloc[14:32] = 엑셀 15~32행
            df_raw = df_raw.iloc[_CRM_SUM_ROW_START:_CRM_SUM_ROW_END].reset_index(drop=True)

            n_cols = df_raw.shape[1]
            if not df_raw.empty and n_cols > _CRM_SUM_COL_STAFF:
                # STEP 2: B열 필터
                _b_col = df_raw.iloc[:, _CRM_SUM_COL_STAFF].astype(str).str.strip()
                _skip_b = {"", "nan", "None", "직원명", "합계", "합 계"}
                df_raw = df_raw[~_b_col.isin(_skip_b)].reset_index(drop=True)

                if not df_raw.empty:
                    col_map = {
                        "지사":       _CRM_SUM_COL_BRANCH,
                        "직원명":     _CRM_SUM_COL_STAFF,
                        "방문_CRM":   _CRM_SUM_COL_VISIT,
                        "비대면_CRM": _CRM_SUM_COL_ONLINE,
                        "전화_CRM":   _CRM_SUM_COL_PHONE,
                    }
                    valid = {k: v for k, v in col_map.items() if v < n_cols}
                    if "직원명" in valid:
                        sub = df_raw.iloc[:, list(valid.values())].copy()
                        sub.columns = list(valid.keys())

                        if "지사" not in sub.columns:
                            sub["지사"] = ""

                        sub["지사"] = sub["지사"].astype(str).str.strip()
                        sub["지사"] = sub["지사"].replace({"nan": "", "None": ""})
                        sub["지사"] = sub["지사"].replace("", np.nan).ffill().fillna("")

                        sub["직원명"] = sub["직원명"].astype(str).str.strip()
                        _skip_names = {"", "nan", "None", "직원명", "상담직원", "합계", "합 계", "이름", "성명"}
                        sub = sub[~sub["직원명"].isin(_skip_names)].copy()
                        sub = sub[sub["직원명"].str.len() >= 2].copy()

                        for col in ["방문_CRM", "비대면_CRM", "전화_CRM"]:
                            if col in sub.columns:
                                sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0.0)
                            else:
                                sub[col] = 0.0

                        sub = sub[(sub["방문_CRM"] + sub["비대면_CRM"] + sub["전화_CRM"]) > 0].copy()
                        sub["직원명_key"] = sub["직원명"].apply(_norm_key)

                        sub = sub.reset_index(drop=True)
                        sub["_row_order"] = sub.index
                        sub = (
                            sub.groupby(["지사", "직원명_key"], sort=False, as_index=False)
                            .agg(_row_order=("_row_order", "min"),
                                 직원명=("직원명", "first"),
                                 방문_CRM=("방문_CRM", "sum"),
                                 비대면_CRM=("비대면_CRM", "sum"),
                                 전화_CRM=("전화_CRM", "sum"))
                            .sort_values("_row_order")
                            .drop(columns=["_row_order"])
                            .reset_index(drop=True)
                        )
                        summary_df = sub
    except Exception:
        pass

    return summary_df


@st.cache_data(ttl=_CRM_CACHE_TTL)
def load_crm_detailed_matching_data(sheet_id: str = _CRM_SHEET_ID) -> pd.DataFrame:
    """
    매출 전환 분석 전용 — 지사별 탭(강남~부산)에서 4개 열만 추출합니다.
    통합 시트(Q/W/AC열)와 완전히 분리된 별도 로직입니다.

      F열(5):   가입자명  → 팀블로 상담자명과 매칭
      X열(23):  상담직원  → 팀블로 상담직원명과 매칭
      Y열(24):  상담방식  → 분석 보조 정보
      AI열(34): 구입여부  → '구입' 여부 판별 (_PURCHASE_KEYWORD)

    날짜 필터 없이 전체 행을 읽어 merge_timblo_crm()에 바로 넘길 수 있는 DataFrame으로 반환합니다.
    반환 열: 가입자명, 상담직원, 상담방식_crm, 구입여부, 지사_crm,
             가입자명_key, 상담직원_key, 상담날짜_str(빈값), 구입날짜(빈값)
    """
    import urllib.parse

    _has_secrets = False
    if _GSHEETS_OK:
        try:
            _has_secrets = bool(st.secrets.get("connections", {}).get("gsheets"))
        except Exception:
            pass

    frames: list[pd.DataFrame] = []
    for branch in _CRM_BRANCHES:
        try:
            if _has_secrets:
                conn   = st.connection("gsheets", type=_GSheetsConnection)
                df_raw = conn.read(
                    spreadsheet=f"https://docs.google.com/spreadsheets/d/{sheet_id}",
                    worksheet=branch,
                    ttl=_CRM_CACHE_TTL,
                )
            else:
                csv_url = (
                    f"https://docs.google.com/spreadsheets/d/{sheet_id}"
                    f"/gviz/tq?tqx=out:csv&sheet={urllib.parse.quote(branch)}"
                )
                df_raw = pd.read_csv(csv_url, header=0, dtype=str)

            if df_raw is None or df_raw.empty:
                continue

            n = df_raw.shape[1]
            # ── 지사별 시트 전용 열 인덱스 (통합 시트 Q/W/AC 와 완전히 다름) ──
            col_map = {
                "가입자명":     _CRM_COL_CUSTOMER,  # F열(5)
                "상담직원":     _CRM_COL_STAFF,      # X열(23)
                "상담방식_crm": _CRM_COL_MODE,       # Y열(24)
                "구입여부":     _CRM_COL_PURCHASE,   # AI열(34)
            }
            valid = {k: v for k, v in col_map.items() if v < n}
            if "가입자명" not in valid:
                continue

            sub = df_raw.iloc[:, list(valid.values())].copy()
            sub.columns = list(valid.keys())
            sub["지사_crm"] = branch

            sub["가입자명"] = sub["가입자명"].astype(str).str.strip()
            sub = sub[~sub["가입자명"].isin(["", "nan", "None"])].copy()
            if sub.empty:
                continue

            frames.append(sub)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    detail = pd.concat(frames, ignore_index=True)
    detail["가입자명_key"]  = detail["가입자명"].apply(_norm_key)
    detail["상담직원_key"]  = detail["상담직원"].astype(str).apply(_norm_key) if "상담직원" in detail.columns else ""
    if "구입여부" not in detail.columns:
        detail["구입여부"] = ""
    # merge_timblo_crm이 참조하는 열 — 없으면 빈값으로 채움
    detail["상담날짜_str"] = ""
    detail["구입날짜"]     = ""
    return detail


def merge_timblo_crm(records: list, crm_df: pd.DataFrame) -> pd.DataFrame:
    """
    팀블로 분석 레코드(list of dict)와 CRM 구글 시트 데이터를 Left Join으로 병합합니다.

    매칭 기준:
        상담자명(팀블로 제목) ↔ 가입자명(CRM)  &  직원명(팀블로) ↔ 상담직원(CRM)
        이름 내 공백 전부 제거 후 비교 (예: "남 은도" == "남은도")

    날짜 덮어쓰기 규칙:
        매칭 성공 → 대시보드 표기 날짜를 CRM 시트의 '상담날짜'로 교체
        매칭 실패 → 팀블로 원본 날짜 유지
    """
    if crm_df.empty or not records:
        return pd.DataFrame(records)

    rows = []
    for r in records:
        p          = parse_title(r.get("title", ""))
        client_key = _norm_key(p.get("client", ""))
        staff_key  = _norm_key(p.get("staff",  ""))
        row        = dict(r)

        # ── 매칭 4단계: 정확도 내림차순 ────────────────────────────────
        branch_val = (p.get("branch", "") or "").strip()
        client_kr  = _pure_korean(client_key)  # 순수 한글만 (fallback용)

        matched = pd.DataFrame()
        # Level-1: 지사 + 직원 + 고객명 3중 정확 매칭
        if branch_val and client_key and staff_key and "지사_crm" in crm_df.columns:
            matched = crm_df[
                (crm_df["지사_crm"]       == branch_val) &
                (crm_df["상담직원_key"]   == staff_key)  &
                (crm_df["가입자명_key"]   == client_key)
            ]
        # Level-2: 직원 + 고객명 정확 매칭
        if matched.empty and client_key and staff_key:
            matched = crm_df[
                (crm_df["가입자명_key"] == client_key) &
                (crm_df["상담직원_key"] == staff_key)
            ]
        # Level-3: 고객명 정확 매칭
        if matched.empty and client_key:
            matched = crm_df[crm_df["가입자명_key"] == client_key]
        # Level-4: 순수 한글 포함 매칭 (공백·특수문자 차이 흡수)
        if matched.empty and client_kr:
            matched = crm_df[
                crm_df["가입자명_key"].apply(
                    lambda v: (lambda k: bool(k and (client_kr in k or k in client_kr)))
                    (_pure_korean(v))
                )
            ]

        if not matched.empty:
            m = matched.iloc[0]
            # 날짜 덮어쓰기: CRM 상담날짜 우선
            crm_date = m.get("상담날짜_str", "")
            if crm_date:
                row["date"] = crm_date
            row["crm_구입여부"] = str(m.get("구입여부", "")).strip()
            row["crm_구입날짜"] = str(m.get("구입날짜", "")).strip()
            row["crm_지사"]     = str(m.get("지사_crm", "")).strip()
            row["crm_matched"]  = True
        else:
            row["crm_구입여부"] = ""
            row["crm_구입날짜"] = ""
            row["crm_지사"]     = ""
            row["crm_matched"]  = False

        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════
def get_env_config() -> dict:
    """설정값 우선순위: st.secrets (Cloud) → 환경변수 → .env 파일 → 코드 기본값"""
    def _s(key, fallback=""):
        # 1순위: Streamlit Cloud secrets
        try:
            v = st.secrets.get(key)
            if v:
                # 따옴표·개행·공백 전부 제거 (secrets.toml 입력 실수 방어)
                v = str(v).strip().strip('"').strip("'").strip()
                if v:
                    return v
        except Exception:
            pass
        # 2순위: 환경변수 / .env 파일 (dotenv로 이미 로드됨)
        v = os.getenv(key, "").strip().strip('"').strip("'").strip()
        return v if v else fallback

    cfg = {
        "api_base":   _s("TIMBLO_API_BASE",  "https://demo.timblo.io/api"),
        "api_key":    _s("TIMBLO_API_KEY",    "cm8o8cqet000014gd7ig5cxrw"),
        "email":      _s("TIMBLO_EMAIL",      "sorizava_counsel@timbel.net"),
        "gemini_key": _s("GEMINI_API_KEY"),
        "openai_key": _s("OPENAI_API_KEY"),
    }
    # 키 로드 결과를 터미널에 출력 (앞 8자만 — 디버깅용)
    _gk = cfg["gemini_key"]
    _ok = cfg["openai_key"]
    print(
        f"[ENV] gemini_key={'설정됨('+_gk[:8]+')' if _gk else '없음'}  "
        f"openai_key={'설정됨('+_ok[:8]+')' if _ok else '없음'}",
        flush=True,
    )
    return cfg

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
    파싱 규칙:
      1. 첫 번째 [...]  → 상담방식
      2. ] 이후 ~ _    → 지사
      3. _ 이후 ~ (    → 직원명  (언더바 다음, 첫 괄호 이전)
      4. 첫 번째 (...) → 상담자명 (괄호 안)
    """
    mode = branch = staff = client_name = ""
    try:
        raw = (title or "").strip()
        # 1. 상담방식: 첫 번째 [...] 탐색 (접두사 무시)
        m = re.search(r"\[([^\]]+)\]", raw)
        if m:
            mode = m.group(1).strip()
            rest = raw[m.end():]          # ] 이후 문자열
        else:
            rest = raw

        # 2. 지사: _ 이전 부분
        if "_" in rest:
            branch, rest = rest.split("_", 1)
        branch = branch.strip()

        # 3. 직원명: _ 이후 ~ 첫 번째 ( 이전
        # 4. 상담자명: 첫 번째 (...) 안
        m2 = re.match(r"([^(（（]+?)\s*[（(]([^)）)]+)[）)]", rest)
        if m2:
            staff       = m2.group(1).strip()
            client_name = m2.group(2).strip()
        else:
            # 괄호 없는 경우: 남은 전체가 직원명
            staff = rest.strip()
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
    """소수점 0.1 단위 정밀 채점 — AI가 반환한 배점 내 소수점 점수를 그대로 환산."""
    if not quality_scores: return 0.0
    def _to_float(v, default=3.0):
        try: return float(v)
        except (TypeError, ValueError): return default
    if rubric:
        total = sum(
            (_to_float(quality_scores.get(r["항목"], 3.0)) / 5.0) * r["배점"]
            for r in rubric
        )
        return round(total, 1)
    return round(
        sum((_to_float(quality_scores.get(k, 3.0)) / 5.0) * w for k, w in QUALITY_WEIGHTS.items()),
        1
    )


def score_grade(score: float) -> str:
    if score >= 95: return "S"
    if score >= 85: return "A"
    if score >= 70: return "B"
    if score >= 55: return "C"
    return "D"


def score_grade_label(score: float) -> str:
    """등급 + 한글 설명"""
    g = score_grade(score)
    labels = {"S": "S (최상)", "A": "A (우수)", "B": "B (양호)", "C": "C (보통)", "D": "D (개선필요)"}
    return labels.get(g, g)


def score_color(score: float) -> str:
    if score >= 95: return "#6a0dad"
    if score >= 85: return "#2e7d32"
    if score >= 70: return "#1565c0"
    if score >= 55: return "#f57c00"
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
        "워크", "수요", "일거리", "실제 수요", "실제수요",
    ],
    # 가격 인상 예정 고지
    "가격 인상": [
        "가격 인상", "가격인상", "금액 변동", "금액변동",
        "장비값 오른", "팀벨 인상", "팀벨인상", "인상 예정", "인상예정",
        "오를 예정", "오르기 전", "가격이 오", "비용 인상", "비용인상",
        "가격 올", "금액 올",
        "인상", "오를", "변동", "금액", "오르기", "팀벨", "가격이",
    ],
    # 구매 적기 어필
    "구매 적기": [
        "지금이 적기", "2026년", "수정 시간 10분", "수정시간 10분",
        "난이도 하향", "취득 적기", "적기", "지금이 최적기",
        "지금 시작", "이번 달", "한시적", "지금이 기회",
        "기회", "지금 아니면",
        "10분", "최적", "수정 시간", "지금",
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
def build_analysis_prompts(conversation_text: str, speech_stats: dict, title_meta: dict,
                           user_cfg: dict = None) -> tuple:
    """채점 시스템 프롬프트(system_text)와 대화 분석 프롬프트(user_text)를 반환."""
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
    q_schema = ", ".join(f'"{r["항목"]}": 3.0' for r in rubric)  # float 힌트

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

    # 페르소나
    persona_line = system_prompt if system_prompt else "당신은 직업훈련 상담 품질 평가 전문가입니다."

    # 채널별 채점표 (system role에 배치)
    rubric_lines = [
        f'  - {r["항목"]} [배점 {r["배점"]}점]: {r.get("기준","") or "해당 역량 평가"}'
        for r in rubric
    ]
    rubric_block = "\n".join(rubric_lines)
    total_max = sum(r["배점"] for r in rubric)

    # ── SYSTEM TEXT (채점 규칙 + 루브릭 고정 — 토큰 절약) ──
    system_text = f"""{persona_line}
반드시 JSON 형식으로만 응답하세요. JSON 외 텍스트·마크다운 절대 금지.

## 채점표 (총 배점 {total_max}점)
{rubric_block}

## 채점 원칙 ★★ 0.1점 단위 정밀 채점 필수 ★★
- 각 항목 점수는 0.0 ~ 5.0 범위의 소수점 한 자리 실수(float)로 반환할 것.
  예: 1.0 / 1.5 / 2.3 / 3.7 / 4.5 / 5.0
- 최종 점수 = (항목 점수 / 5.0) × 배점  →  배점 내에서 0.1점 단위로 세밀하게 분배
- 점수 기준 (기준 내에서 소수점 세분화 필수)
  · 5.0점: 세부 기준 완전 충족 + 구체적 언급·실행 확인
  · 4.0~4.9점: 대부분 충족, 세부 요소 일부 누락
  · 3.0~3.9점: 핵심만 언급, 구체성·깊이 부족
  · 2.0~2.9점: 언급은 있으나 피상적·짧음 (예: 2.0점 항목에서 간략 언급 → 0.8점 환산)
  · 1.0~1.9점: 매우 형식적, 핵심 메시지 전달 실패
  · 0.0~0.9점: 해당 항목 미수행 또는 완전 누락

## 발화 통계 기반 채점 원칙
- 화자별 발화 통계가 제공된다. talk_ratio와 total_chars를 채점에 반드시 반영할 것.
- 상담사 발화 비중(talk_ratio)이 높고 핵심 키워드가 포함된 항목 → 개선점 제외, 고득점(4.0~5.0) 부여
- 상담사 발화량이 적은 항목은 설명 충실도를 근거로 감점 가능

## 감점·미준수 판단 시 실제 발화 인용 의무
- 점수를 3.5 미만으로 부여하거나 개선점을 도출할 때는 반드시:
  · 어느 발화에서 부족함이 확인됐는지 대화 원문의 한 문장 이상을 직접 인용
  · improvements 항목 형식: "개선 필요 내용 — 근거: '실제 발화 인용'"
- 근거 없는 추상적 개선점(예: "경청이 부족합니다") 출력 금지

## 기타 원칙
- 점수 편중 금지: 특정 구간(3점대)에 집중하지 말고 0.0~5.0 전 구간 활용
- STT 오타 허용: 유사 발음·구어체를 정상 언급으로 인정
- 키워드 가점: 웍스파이/워크스, 가격 인상/팀벨 인상, 2026년/수정 시간 10분 등
  핵심 키워드 명시 언급 → 해당 항목 최소 4.0점 보장
- 짧아도 핵심 전달 시 고점 가능
{mand_instruction}"""

    # ── USER TEXT (대화 내용 + 분석 지시) ──
    user_text = f"""[상담 메타정보]
{meta_block}

[화자별 발화 통계 — 채점 시 반드시 참조]
{stats_block}
※ talk_ratio가 높은 화자가 해당 항목 키워드를 충분히 언급했다면 개선점 제외 + 고득점 부여

[전체 대화 내용]
{conversation_text}

[분석 지시 — 아래 항목 전부 분석 → 단일 JSON 반환]
1. 역할 판별: 발화 성격(설명/질문/고민)으로 상담원/고객 구분
2. 품질 점수 ★ 반드시 0.0~5.0 소수점 한 자리 float으로 반환 ★ (system 채점 원칙 必 적용):
{q_block}
   ※ 감점(3.5 미만) 시 해당 발화 인용 필수
3. 흐름 단계 포함 여부 (true/false): {flow_list}
4. 강점 3가지 (실제 대화 근거 기반 한 문장씩, 발화 직접 인용 포함)
5. 개선점 3가지 — 형식: "개선 내용 — 근거: '실제 발화 인용'" (추상적 나열 금지)
6. 고객 발화 TOP 20 키워드, 상담사 발화 TOP 20 키워드 (의미있는 명사/동사)
7. 질문 패턴 횟수: 공부기간, 시험난이도, 수익가능성, 가격, 취업가능성, 기타
8. 감정 분석: 긍정/중립/부정 비율 (합계 1.0)
9. 위험 신호 문구 목록 (고민, 비쌈, 시간없음 등)
10. 코칭 멘트 3~5문장
11. 리드 품질 점수 (1~100점 정수)
    고객(상담자)의 발화 내용만 분석해 구매 가능성을 아래 기준으로 점수화하세요.
    · 질문의 구체성 (30점): 막연한 궁금증이 아닌 구체적 준비 방법·일정·비용 질문
    · 구매 의지 표명 (30점): "해보고 싶다", "시작하려고", "신청하면" 등 적극적 의지 표현
    · 예산/시간 확보 (20점): 학습 시간·비용 마련 가능 여부 직접 언급
    · 사전 이해도 (20점): 속기사 직업 사전 조사·구체적 지식 보유 여부
    기준 없이 무조건 50점으로 설정하지 말 것. 대화 근거에 따라 1~100 전 구간 활용.
12. 전환 클로징 분석
    상담이 성공적으로 결제로 이어졌다고 판단할 때 사용됐을 법한 상담사 핵심 클로징 멘트 1~3문장을 추출하세요.
    고객이 결심을 굳히기 직전에 보인 질문·발화 패턴도 함께 추출하세요.

반드시 아래 JSON 구조로만 반환:
{{
  "identified_counselor": "참석자 N",
  "identified_customer": "참석자 N",
  "role_reason": "근거 1~2문장",
  "quality_scores": {{{q_schema}}},
  "flow_stages": {{{flow_schema}}},
  "strengths": ["강점1","강점2","강점3"],
  "improvements": ["개선점1","개선점2","개선점3"],
  "customer_keywords": ["키워드1","키워드2"],
  "counselor_keywords": ["키워드1","키워드2"],
  "question_patterns": {{"공부기간":0,"시험난이도":0,"수익가능성":0,"가격":0,"취업가능성":0,"기타":0}},
  "sentiment": {{"positive":0.3,"neutral":0.5,"negative":0.2}},
  "risk_signals": [],
  "coaching": "코칭 멘트",
  "lead_score": 50,
  "closing_phrases": ["클로징 멘트1","클로징 멘트2"],
  "decision_signals": ["결심 직전 고객 발화 패턴1"]{mand_json}
}}"""

    return system_text, user_text


def build_deep_prompt(conversation_text: str, speech_stats: dict, title_meta: dict,
                      user_cfg: dict = None) -> str:
    """하위 호환용 래퍼 — system+user를 합쳐 단일 문자열로 반환."""
    sys_t, usr_t = build_analysis_prompts(conversation_text, speech_stats, title_meta, user_cfg)
    return sys_t + "\n\n" + usr_t


# ══════════════════════════════════════════════════════════
# 텍스트 전처리
# ══════════════════════════════════════════════════════════
_MIN_CONV_CHARS = 100   # 이 미만이면 AI 호출 생략


def preprocess_conv_text(raw: str) -> str:
    """STT 노이즈 제거 + 핵심 발화만 남겨 토큰 절약."""
    lines = raw.split("\n")
    cleaned = []
    for line in lines:
        s = line.strip()
        # 3글자 미만 단독 발화(네/아/음 등) 제거
        if len(s) < 3:
            continue
        # 연속 동일 문구 제거
        if cleaned and s == cleaned[-1]:
            continue
        cleaned.append(s)
    text = "\n".join(cleaned)
    # 과도한 개행/공백 정리
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


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
    clean_text = preprocess_conv_text(conv["text"])
    if len(clean_text) < _MIN_CONV_CHARS:
        raise ValueError(f"[EMPTY_DATA] {content_id} — 텍스트 너무 짧음 ({len(clean_text)}자)")
    speech_stats = compute_speech_stats(detail, use_merged=True)
    sys_txt, usr_txt = build_analysis_prompts(clean_text, speech_stats, title_meta,
                                              user_cfg=st.session_state.get("user_config"))
    cached = {
        "text": conv["text"],          # 하드매칭용 원본
        "clean_text": clean_text,       # 전처리된 텍스트 (AI 전달용)
        "lines": conv["lines"],
        "speech_stats": speech_stats,
        "system_prompt_text": sys_txt,  # system role
        "user_prompt_text": usr_txt,    # user role
        "analysis_prompt": sys_txt + "\n\n" + usr_txt,  # 구형 호환
        "segment_count": len(conv["lines"]),
    }
    st.session_state.transcripts[content_id] = cached
    return cached


# ══════════════════════════════════════════════════════════
# 에러 분류 + JSON 파싱
# ══════════════════════════════════════════════════════════
def classify_error(e: Exception) -> str:
    msg       = str(e).lower()
    etype_cls = type(e).__name__.lower()   # 예: "authenticationerror", "permissiondeniederror"
    if "empty_data" in str(e) or "텍스트 없음" in str(e): return "EMPTY_DATA"
    if "rate" in msg or "quota" in msg or "429" in msg or "resource_exhausted" in msg: return "RATE_LIMIT"
    # AUTH: HTTP 401/403, OpenAI AuthenticationError, Gemini API_KEY_INVALID / PERMISSION_DENIED
    # ※ "invalid_argument" 제거 — JSON 파라미터 오류 등 비인증 오류가 AUTH로 오분류되던 문제 수정
    # ※ "[auth]" in msg — 이미 래핑된 RuntimeError에서도 AUTH 타입 보존
    _auth_signals = (
        "401" in msg or "403" in msg
        or "[auth]" in msg                     # 이미 래핑된 AUTH RuntimeError 재인식
        or "incorrect api key" in msg          # OpenAI: Incorrect API key provided
        or "no api key" in msg                 # OpenAI: No API key provided
        or "invalid_api_key" in msg            # OpenAI error code
        or "api_key_invalid" in msg            # Gemini
        or "permission_denied" in msg          # Gemini gRPC
        or "unauthenticated" in msg            # gRPC status
        or "authentication" in etype_cls       # openai.AuthenticationError
        or ("permission" in etype_cls and "denied" in msg)
    )
    if _auth_signals: return "AUTH"
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
        "lead_score":         int(result.get("lead_score", 50)),
        "closing_phrases":    result.get("closing_phrases", []),
        "decision_signals":   result.get("decision_signals", []),
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
    _clean_gkey = (gemini_api_key or "").strip().strip('"').strip("'").strip()
    if not _clean_gkey:
        raise RuntimeError("[AUTH] Gemini API 키가 비어 있습니다. .env 또는 Secrets에 GEMINI_API_KEY를 입력하세요.")
    cached = fetch_transcript(client, content_id, title_meta)
    genai.configure(api_key=_clean_gkey)
    try:
        model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config={"response_mime_type": "application/json"},
            system_instruction=cached["system_prompt_text"],   # ← system role 분리
        )
        response = model.generate_content(cached["user_prompt_text"])
        raw_text = response.text.strip()
    except Exception as e:
        raise RuntimeError(f"[{classify_error(e)}] Gemini 호출 실패: {e}") from e
    try:
        result = parse_ai_json(raw_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"[JSON_PARSE] Gemini JSON 파싱 실패: {e}") from e
    _ucfg  = st.session_state.get("user_config") or DEFAULT_USER_CONFIG
    rubric = get_rubric_for_mode(title_meta.get("mode", ""), _ucfg)
    result = hard_match_mandatory(cached["text"], _ucfg.get("mandatory_items", []), result)
    return build_record(content_id, title, start_time, result,
                        cached["speech_stats"], cached["segment_count"], "Gemini", rubric=rubric)


def run_openai_analysis(client, content_id, title, start_time, title_meta,
                        openai_api_key, model_id: str = "gpt-4o-mini") -> dict:
    from openai import OpenAI as _OpenAI
    _clean_okey = (openai_api_key or "").strip().strip('"').strip("'").strip()
    if not _clean_okey:
        raise RuntimeError("[AUTH] OpenAI API 키가 비어 있습니다. .env 또는 Secrets에 OPENAI_API_KEY를 입력하세요.")
    # sk-proj-… 형식도 유효 — startswith("sk-") 체크 제거
    cached = fetch_transcript(client, content_id, title_meta)
    oa = _OpenAI(api_key=_clean_okey)
    try:
        resp = oa.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": cached["system_prompt_text"]},  # ← system role
                {"role": "user",   "content": cached["user_prompt_text"]},
            ],
            temperature=0.15,
            response_format={"type": "json_object"},
        )
        raw_text = resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"[{classify_error(e)}] OpenAI({model_id}) 호출 실패: {e}") from e
    try:
        result = parse_ai_json(raw_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"[JSON_PARSE] OpenAI JSON 파싱 실패: {e}") from e
    _ucfg  = st.session_state.get("user_config") or DEFAULT_USER_CONFIG
    rubric = get_rubric_for_mode(title_meta.get("mode", ""), _ucfg)
    result = hard_match_mandatory(cached["text"], _ucfg.get("mandatory_items", []), result)
    engine_label = "ChatGPT-4o" if "gpt-4o" in model_id and "mini" not in model_id else "ChatGPT"
    return build_record(content_id, title, start_time, result,
                        cached["speech_stats"], cached["segment_count"], engine_label, rubric=rubric)


def run_hybrid_analysis(client, content_id, title, start_time, title_meta, cfg) -> dict:
    gemini_key = cfg.get("gemini_key")
    openai_key = cfg.get("openai_key")

    # ── 키 없음 → 즉시 AUTH 오류 (재시도 없음) ──
    if not gemini_key and not openai_key:
        raise RuntimeError(
            "[AUTH] API 키가 설정되지 않았습니다. "
            ".env 파일 또는 Streamlit Secrets에 GEMINI_API_KEY / OPENAI_API_KEY를 입력하세요."
        )

    sel_model = st.session_state.get("ai_model", "gpt-4o-mini")

    if gemini_key and sel_model == "gemini":
        try:
            return run_gemini_analysis(client, content_id, title, start_time, title_meta, gemini_key)
        except RuntimeError as e:
            etype = classify_error(e)
            if etype == "AUTH":
                # 인증 오류 → 즉시 전파, OpenAI 폴백 없음
                raise RuntimeError(
                    f"[AUTH] Gemini 인증 실패 — API 키를 확인하세요. 원본: {e}"
                ) from e
            if etype == "RATE_LIMIT" and openai_key:
                safe = str(e).encode("ascii", errors="replace").decode("ascii")
                print(f"[FALLBACK] Gemini 한도 -> ChatGPT: {safe}", flush=True)
                # fall-through to openai block below
            else:
                raise

    if openai_key:
        # RuntimeError는 run_openai_analysis 내에서 이미 [TYPE] 접두사로 래핑됨 → 그대로 전파
        return run_openai_analysis(
            client, content_id, title, start_time, title_meta,
            openai_key,
            model_id=sel_model if sel_model != "gemini" else "gpt-4o-mini",
        )

    if gemini_key:
        return run_gemini_analysis(client, content_id, title, start_time, title_meta, gemini_key)

    raise RuntimeError(
        "[AUTH] 사용 가능한 AI 키가 없습니다. "
        "사이드바에서 API 키 설정을 확인하세요."
    )


# ══════════════════════════════════════════════════════════
# 차트 헬퍼
# ══════════════════════════════════════════════════════════
def make_radar_chart(cats, vals, color="royalblue", height=320,
                     overlay_vals=None, overlay_name="선택 상담자", overlay_color="#e74c3c"):
    # ── 입력값 정제: None·NaN → 50.0, 범위 클램프 ──────────────────────────
    def _clean(v_list):
        out = []
        for v in v_list:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = 50.0
            out.append(max(0.0, min(100.0, fv)))
        return out

    r_vals = _clean(vals)
    # 폐합: 마지막 점을 첫 번째와 명시적으로 연결
    r_closed     = r_vals + [r_vals[0]]
    theta_closed = list(cats) + [cats[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r_closed, theta=theta_closed,
        fill="toself", fillcolor="rgba(65,105,225,0.15)",
        line=dict(color=color, width=2), name="전체 평균",
        mode="lines+markers+text",
        text=[f"{v:.0f}" for v in r_vals] + [""],
        textposition="top center",
        textfont=dict(size=9, color=color),
    ))
    if overlay_vals:
        o_vals   = _clean(overlay_vals)
        o_closed = o_vals + [o_vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=o_closed, theta=theta_closed,
            fill="toself", fillcolor="rgba(231,76,60,0.15)",
            line=dict(color=overlay_color, width=2.5, dash="solid"), name=overlay_name,
            mode="lines+markers+text",
            text=[f"{v:.0f}" for v in o_vals] + [""],
            textposition="top center",
            textfont=dict(size=9, color=overlay_color),
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor="rgba(180,180,180,0.35)",
                linecolor="rgba(180,180,180,0.35)",
                tickfont=dict(size=8, color="#888888"),
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color="#444444"),
                linecolor="rgba(180,180,180,0.4)",
            ),
        ),
        showlegend=bool(overlay_vals),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
        margin=dict(t=40, b=40, l=60, r=60),
        height=height,
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
    "ai_model": "gpt-4o-mini",  # 사용 AI 모델 (gpt-4o-mini / gpt-4o / gemini)
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# user_config 로드 (세션 최초 1회)
if st.session_state.user_config is None:
    st.session_state.user_config = load_user_config()

# 분석 결과 로드
# - 최초 1회 로드 (results_loaded == False)
# - analyzed가 비어있는데 파일은 있으면 항상 재시도 (results_loaded 여부 무관)
# - 강제 새로고침 버튼(force_reload_results) 클릭 시 재로드
_should_load = (
    not st.session_state.results_loaded
    or st.session_state.get("force_reload_results")
    or (not st.session_state.analyzed and not st.session_state.get("_load_tried_empty"))
)
if _should_load:
    _file_results = load_analysis_results()
    if _file_results:
        for _k, _v in _file_results.items():
            if _k not in st.session_state.analyzed:
                # 구버전 mandatory_check(note 필드 등)를 신버전(evidence) 포맷으로 자동 마이그레이션
                if isinstance(_v, dict) and "mandatory_check" in _v:
                    _v["mandatory_check"] = parse_mandatory_check(_v["mandatory_check"])
                st.session_state.analyzed[_k] = _v
    else:
        # 파일이 없거나 비어있음 — 루프 방지용 플래그
        st.session_state["_load_tried_empty"] = True
    st.session_state.results_loaded = True
    st.session_state["force_reload_results"] = False


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
            if etype == "AUTH":
                # 인증 오류 → 해당 항목만 실패 기록, 큐는 계속 진행 (큐 전체 중단 금지)
                # 사이드바에 마지막 AUTH 오류 메시지만 보관
                st.session_state["_auth_error_msg"] = str(e)
                print(f"[AUTO] AUTH 오류 — 해당 항목 스킵, 다음 항목 계속", flush=True)
            elif etype == "RATE_LIMIT":
                time.sleep(st.session_state.auto_delay + 10)

    # pop 시 IndexError 방지 (이미 다른 경로로 큐가 비워진 경우 대비)
    if st.session_state.auto_queue:
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
        st.caption("GEMINI_API_KEY 또는 OPENAI_API_KEY를 .env 파일에 입력하세요.")

    # AUTH 오류 발생 시 사이드바 경고 표시
    _auth_err = st.session_state.get("_auth_error_msg", "")
    if _auth_err:
        st.error(f"🔐 인증 오류 — 자동 분석 중단됨")
        st.caption(_auth_err[:120])
        if st.button("오류 닫기", key="btn_clear_auth_err", use_container_width=True):
            st.session_state["_auth_error_msg"] = ""
            st.rerun()

    st.markdown("---")
    st.markdown("**데이터 필터**")

    all_modes    = sorted(set(i.get("_mode","")   for i in content_list if i.get("_mode")))
    all_branches = sorted(set(i.get("_branch","") for i in content_list if i.get("_branch")))
    all_staff    = sorted(set(i.get("_staff","")  for i in content_list if i.get("_staff")))

    f_mode   = st.selectbox("상담방식", ["전체"] + all_modes,   key="sb_mode")
    f_branch = st.selectbox("지사",     ["전체"] + all_branches, key="sb_branch")
    f_staff  = st.selectbox("직원명",   ["전체"] + all_staff,   key="sb_staff")
    f_grade  = st.selectbox("품질등급",
                            ["전체","S(95+)","A(85+)","B(70+)","C(55+)","D(55-)"],
                            key="sb_grade")

    # 상담자 선택 — 분석 완료 레코드 기반 (레이더 오버레이 + 전환 예측용)
    _analyzed_staff = sorted(set(
        parse_title(r.get("title","")).get("staff","") or "미확인"
        for r in analyzed.values()
        if parse_title(r.get("title","")).get("staff","")
    ))
    f_counselor = st.selectbox("상담자 선택", ["전체 보기"] + _analyzed_staff, key="sb_counselor")

    st.markdown("**기간 필터**")
    # content_list + analyzed 양쪽에서 날짜 풀 수집
    _cl_dates  = [parse_date_str(i.get("meetingStartTime","")) for i in content_list]
    _an_dates  = [r["date"] for r in analyzed.values() if r.get("date")]
    _date_pool = sorted(set(d for d in _cl_dates + _an_dates if d))

    # 기본값: 당월 1일 ~ 오늘 (데이터 범위 내로 클램프)
    _today_date   = datetime.now().date()
    _month_1st    = _today_date.replace(day=1)

    if _date_pool:
        _fd_min = datetime.strptime(_date_pool[0],  "%Y-%m-%d").date()
        _fd_max = datetime.strptime(_date_pool[-1], "%Y-%m-%d").date()
        _def_start = max(_fd_min, _month_1st)
        _def_end   = min(_fd_max, _today_date)
        # 데이터가 모두 이번 달 이전인 경우 전체 기간으로 폴백
        if _def_start > _def_end:
            _def_start, _def_end = _fd_min, _fd_max
        # key="sb_daterange" → 첫 로드 시 기본값 적용, 이후 사용자 변경 값 유지
        date_range = st.date_input(
            "기간", value=(_def_start, _def_end),
            min_value=_fd_min, max_value=_fd_max,
            label_visibility="collapsed",
            key="sb_daterange",
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            if date_range[0] == _fd_min and date_range[1] == _fd_max:
                st.caption("전체 기간")
            elif date_range[0] == _month_1st and date_range[1] == _today_date:
                st.caption(f"이번 달 ({date_range[0]} ~ {date_range[1]})")
            else:
                st.caption(f"{date_range[0]} ~ {date_range[1]}")
    else:
        date_range = None
        st.caption("날짜 정보 없음")

    st.markdown("---")
    if st.button("🔄 CRM 데이터 갱신", use_container_width=True,
                 help="구글 시트를 즉시 다시 읽어옵니다 (CRM 캐시만 삭제)"):
        load_crm_data.clear()
        load_crm_purchase_status.clear()
        st.rerun()

    if st.button("⚡ 시스템 전체 초기화", use_container_width=True, type="primary",
                 help="모든 캐시·세션 변수를 비우고 통합 시트(요약)와 지사별 시트(결제상세)를 모두 새로 읽어옵니다"):
        st.cache_data.clear()
        st.cache_resource.clear()
        for _ss_key in list(st.session_state.keys()):
            if any(k in _ss_key.lower() for k in ("crm", "구입", "purchase", "matched", "nurak")):
                st.session_state[_ss_key] = None
        st.rerun()

    st.markdown("---")
    st.markdown("**자동 분석**")

    n_done = len(analyzed)
    n_todo = len([i for i in content_list if i.get("contentId") not in analyzed])
    n_fail = len(st.session_state.auto_failed)
    st.caption(f"완료 {n_done}건  |  미분석 {n_todo}건  |  오류 {n_fail}건")

    # 저장된 분석 결과가 보이지 않을 때 강제 새로고침
    if n_done == 0 and os.path.exists(_RESULTS_FILE):
        if st.button("📂 저장 결과 불러오기", use_container_width=True, type="primary"):
            st.session_state["force_reload_results"] = True
            st.session_state.results_loaded = False
            st.rerun()
        st.caption(f"💡 `analysis_results.json` 파일이 감지됨")

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
    st.markdown("**AI 모델 설정**")
    _model_opts = {
        "gpt-4o-mini": "⚡ GPT-4o mini (빠름·절약)",
        "gpt-4o":      "🚀 GPT-4o (고성능)",
        "gemini":      "✨ Gemini 2.0 Flash",
    }
    _cur_model = st.session_state.get("ai_model", "gpt-4o-mini")
    _sel = st.selectbox(
        "분석 모델",
        list(_model_opts.keys()),
        format_func=lambda k: _model_opts[k],
        index=list(_model_opts.keys()).index(_cur_model) if _cur_model in _model_opts else 0,
        key="sb_ai_model",
        label_visibility="collapsed",
    )
    if _sel != _cur_model:
        st.session_state.ai_model = _sel
        st.rerun()
    if _sel == "gpt-4o":
        st.caption("⚠️ 고성능 모드 — 토큰 비용 약 10배")

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
    if f_grade == "S(95+)":   out = [r for r in out if r.get("total_score", 0) >= 95]
    elif f_grade == "A(85+)": out = [r for r in out if 85 <= r.get("total_score", 0) < 95]
    elif f_grade == "B(70+)": out = [r for r in out if 70 <= r.get("total_score", 0) < 85]
    elif f_grade == "C(55+)": out = [r for r in out if 55 <= r.get("total_score", 0) < 70]
    elif f_grade == "D(55-)": out = [r for r in out if r.get("total_score", 0) < 55]
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

    # ── CRM 데이터 최상단 로드 ──────────────────────────────────────────────
    # load_crm_data()는 날짜 필터 없이 원시 데이터를 캐시 반환.
    # '오늘' 기준 필터는 반드시 여기(매 렌더링)에서 실시간으로 적용.
    _crm_df       = pd.DataFrame()   # 전체(오늘 이하) — merge_timblo_crm 등에 사용
    _crm_month_df = pd.DataFrame()   # 당월 1일 ~ 오늘 — 직원표·누락KPI 전용
    _crm_load_error = ""
    _crm_raw_n = 0
    _crm_valid_n = 0

    try:
        # cutoff_date를 오늘 날짜 문자열로 전달 → 날짜가 바뀌면 캐시 키 변경 → 자동 재로딩
        _today_str = str(datetime.now().date())
        _raw_crm   = load_crm_data(cutoff_date=_today_str)
        _crm_raw_n = len(_raw_crm)

        if not _raw_crm.empty:
            _now_today      = pd.Timestamp(datetime.now().date())
            _this_month_1st = _now_today.replace(day=1)

            # ─ 날짜 열 보장 + timezone 정규화 (캐시 구버전·tz-aware 모두 방어)
            _raw_crm = _raw_crm.copy()
            if "상담날짜_dt" not in _raw_crm.columns:
                _raw_crm["상담날짜_dt"] = pd.to_datetime(
                    _raw_crm["상담날짜"].astype(str).str.strip()
                    .str.replace(r"[./]", "-", regex=True)
                    if "상담날짜" in _raw_crm.columns else pd.Series(dtype="object"),
                    errors="coerce",
                ).astype("datetime64[ns]")
            # timezone-aware → naive 강제 변환
            if _raw_crm["상담날짜_dt"].dt.tz is not None:
                _raw_crm["상담날짜_dt"] = _raw_crm["상담날짜_dt"].dt.tz_localize(None)

            # ① 전체(오늘 이하): 미래 예약 2차 완전 차단
            _crm_df = _raw_crm[
                _raw_crm["상담날짜_dt"].notna() &
                (_raw_crm["상담날짜_dt"] <= _now_today)
            ].copy()

            # ② 당월(1일 ~ 오늘): 직원표·누락KPI 전용
            _crm_month_df = _crm_df[
                (_crm_df["상담날짜_dt"] >= _this_month_1st) &
                (_crm_df["상담날짜_dt"] <= _now_today)
            ].copy() if not _crm_df.empty else pd.DataFrame(columns=_crm_df.columns)

        _crm_valid_n = len(_crm_df)

        # ── 사이드바 디버그 캡션 ──
        st.sidebar.caption(
            f"📋 CRM: 전체 {_crm_raw_n}건 → 유효 {_crm_valid_n}건 "
            f"(미래 예약 {_crm_raw_n - _crm_valid_n}건 제외) | "
            f"당월 {len(_crm_month_df)}건"
        )
    except Exception as _e:
        _crm_load_error = str(_e)

    # ── 미분석 상태 + CRM 없음: 업로드 현황만 표시 ──
    if not records_filtered and _crm_df.empty:
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
        # ── CRM 요약 통계 로드 (직원별 현황 표 전용) ────────────────────────
        _sheet_name = datetime.now().strftime("%Y-%m")
        _crm_sum_df = load_crm_summary(sheet_name=_sheet_name)

        # ── 지사별 상세 결제 데이터 로드 (매출 전환 분석 전용) ───────────────
        _detail_crm_df = load_crm_detailed_matching_data()

        # ── 상세 결제 데이터로 레코드별 구매 여부 판별 ──────────────────────
        # {record_id: True(구입) / False(미구입)}
        _purch_by_id: dict = {}
        if not _detail_crm_df.empty:
            _has_branch_col = "지사_crm" in _detail_crm_df.columns
            for _r in records_filtered:
                _p_     = parse_title(_r.get("title", ""))
                _ck_    = _norm_key(_p_.get("client", ""))
                _sk_    = _norm_key(_p_.get("staff",  ""))
                _bk_    = (_p_.get("branch", "") or "").strip()
                _ck_kr_ = _pure_korean(_ck_)
                if not _ck_ and not _ck_kr_:
                    continue

                # Level-1: 지사 + 직원 + 고객명 3중 정확 매칭
                _mask_ = (
                    (_detail_crm_df["지사_crm"]     == _bk_) &
                    (_detail_crm_df["상담직원_key"] == _sk_) &
                    (_detail_crm_df["가입자명_key"] == _ck_)
                ) if (_has_branch_col and _bk_ and _sk_ and _ck_) else pd.Series(False, index=_detail_crm_df.index)

                # Level-2: 직원 + 고객명 정확 매칭
                if not _mask_.any() and _ck_ and _sk_:
                    _mask_ = (
                        (_detail_crm_df["상담직원_key"] == _sk_) &
                        (_detail_crm_df["가입자명_key"] == _ck_)
                    )

                # Level-3: 고객명 정확 매칭
                if not _mask_.any() and _ck_:
                    _mask_ = (_detail_crm_df["가입자명_key"] == _ck_)

                # Level-4: 순수 한글 포함 매칭 (특수문자·공백 차이 흡수)
                if not _mask_.any() and _ck_kr_:
                    _mask_ = _detail_crm_df["가입자명_key"].apply(
                        lambda v: (lambda k: bool(k and (
                            _ck_kr_ in k or k in _ck_kr_
                        )))(_pure_korean(v))
                    )

                if _mask_.any():
                    _pval_ = str(_detail_crm_df.loc[_mask_, "구입여부"].iloc[0]).strip()
                    _purch_by_id[_r.get("id")] = (_pval_ == _PURCHASE_KEYWORD)

            # ── 매칭 0건 시 디버그 출력 ──────────────────────────────────────
            if len(_purch_by_id) == 0 and records_filtered:
                _dbg_timblo = [
                    _norm_key(parse_title(_r.get("title", "")).get("client", ""))
                    for _r in records_filtered[:5]
                ]
                _dbg_sheet = _detail_crm_df["가입자명_key"].dropna().head(5).tolist()
                print(f"[CRM매칭 디버그] 팀블로 상담자명 예시(5): {_dbg_timblo}", flush=True)
                print(f"[CRM매칭 디버그] 시트 가입자명 예시(5):  {_dbg_sheet}", flush=True)
                print(f"[CRM매칭 디버그] 시트 전체 행 수: {len(_detail_crm_df)}", flush=True)

        # ── CRM 매칭: 상세 결제 데이터 우선, 없으면 전체 CRM 데이터 사용 ──
        _match_source = _detail_crm_df if not _detail_crm_df.empty else _crm_df
        _merged_df   = merge_timblo_crm(records_filtered, _match_source) if records_filtered and not _match_source.empty else pd.DataFrame()
        _matched_df  = _merged_df[_merged_df.get("crm_matched", pd.Series(False, index=_merged_df.index)) == True].copy() if not _merged_df.empty and "crm_matched" in _merged_df.columns else pd.DataFrame()

        # ── 결제 상태 경량 동기화 (_CRM_CACHE_TTL 주기마다 AI열만 재조회) ──
        def _is_purchased(val: str) -> bool:
            # 시트 AI열 값이 정확히 "구입"일 때만 전환 성공으로 판단
            return str(val).strip() == _PURCHASE_KEYWORD

        if not _matched_df.empty:
            try:
                _purch_map = load_crm_purchase_status()
                def _get_fresh_purch(row):
                    ck = _norm_key(row.get("client", ""))
                    sk = _norm_key(row.get("staff",  ""))
                    return _purch_map.get((ck, sk), row.get("crm_구입여부", ""))
                _matched_df["crm_구입여부"] = _matched_df.apply(_get_fresh_purch, axis=1)
            except Exception:
                pass  # 동기화 실패 시 기존 매칭 값 유지

        # 실제 결제 전환율 계산
        _crm_conv_rate = 0.0
        _crm_match_n   = 0
        if not _matched_df.empty and "crm_구입여부" in _matched_df.columns:
            _matched_df["결제성공"] = _matched_df["crm_구입여부"].apply(_is_purchased)
            _crm_match_n   = len(_matched_df)
            _crm_conv_rate = round(_matched_df["결제성공"].sum() / max(_crm_match_n, 1) * 100, 1)

        # 업로드 누락 인원 계산 (당월 기준 — _crm_month_df 직접 사용)
        _nurak_alert_n = 0
        if not _crm_month_df.empty and "상담직원_key" in _crm_month_df.columns:
            _kpi_today     = datetime.now().date()
            _kpi_month_str = _kpi_today.replace(day=1).strftime("%Y-%m-%d")
            _kpi_today_str = _kpi_today.strftime("%Y-%m-%d")
            # _crm_month_df: 이미 당월 1일 ~ 오늘으로 필터링 완료
            _crm_kpi = _crm_month_df
            # 팀블로: 당월 업로드만
            _timblo_src2 = st.session_state.get("content_list_all") or content_list or []
            _t2_rows = []
            for _item2 in _timblo_src2:
                _dt2 = parse_date_str(_item2.get("meetingStartTime", ""))
                if not _dt2 or _dt2 < _kpi_month_str or _dt2 > _kpi_today_str:
                    continue
                _p2  = parse_title(_item2.get("editedTitle") or _item2.get("title", ""))
                _br2 = _p2.get("branch", "") or "미확인"
                _s2  = _norm_key(_p2.get("staff", ""))
                if _s2: _t2_rows.append({"지사": _br2, "직원명_key": _s2})
            _t2_cnt = (pd.DataFrame(_t2_rows).groupby(["지사","직원명_key"]).size().reset_index(name="업로드건수")
                       if _t2_rows else pd.DataFrame(columns=["지사","직원명_key","업로드건수"]))
            _crm_grp2 = (_crm_kpi.groupby(["지사_crm","상담직원_key"]).size()
                         .reset_index(name="crm건수").rename(columns={"지사_crm":"지사","상담직원_key":"직원명_key"}))
            _nr2 = _crm_grp2.merge(_t2_cnt, on=["지사","직원명_key"], how="left")
            _nr2["업로드건수"] = _nr2["업로드건수"].fillna(0).astype(int)
            _nr2["누락건수"]   = (_nr2["crm건수"] - _nr2["업로드건수"]).clip(lower=0)
            # 직원별 총 누락 합계 기준 고유 인원수
            _nr2_person    = _nr2.groupby(["지사","직원명_key"])["누락건수"].sum().reset_index()
            _nurak_alert_n = int((_nr2_person["누락건수"] >= 5).sum())

        # ════════════════════════════════════════════
        # ① KPI 카드 4개 (슬림 구성)
        # ════════════════════════════════════════════
        scores        = [r["total_score"] for r in records_filtered] if records_filtered else []
        avg_score     = round(sum(scores) / len(scores), 1) if scores else 0.0
        _today_str    = datetime.now().strftime("%Y-%m-%d")
        _today_upload = sum(1 for r in analyzed.values() if r.get("date") == _today_str)

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("평균 AI 점수", f"{avg_score}점",
                    delta=score_grade(avg_score) if scores else None)
        if _crm_match_n:
            kpi2.metric("실제 결제 전환율", f"{_crm_conv_rate}%",
                        f"CRM 매칭 {_crm_match_n}건 기준")
        else:
            kpi2.metric("실제 결제 전환율", "CRM 데이터 없음")
        if _nurak_alert_n:
            kpi3.metric("5건 이상 누락 인원", f"{_nurak_alert_n}명",
                        delta="즉시 확인 필요", delta_color="inverse")
        else:
            kpi3.metric("5건 이상 누락 인원", "없음" if not _crm_df.empty else "CRM 미연결")
        kpi4.metric("오늘 업로드", f"{_today_upload}건")

        st.markdown("---")

        # ════════════════════════════════════════════
        # ② 직원별 상담방식별 업로드 현황 표 (당월 피벗)
        # ════════════════════════════════════════════
        _emp_today     = datetime.now().date()
        _emp_month_1st = _emp_today.replace(day=1)
        _emp_today_str = _emp_today.strftime("%Y-%m-%d")
        _emp_month_str = _emp_month_1st.strftime("%Y-%m-%d")

        st.subheader("📋 직원별 상담 업로드 현황")
        _sheet_label = _sheet_name   # 이미 else 블록 상단에서 정의됨
        st.caption(
            f"기준 시트: {_sheet_label} | CRM 합산 열(Q·W·AC) 직접 참조 | "
            "직원명(X열) 기준 전체 표시 | 총 누락 5건↑ 빨간 하이라이트"
        )

        # _crm_sum_df 는 else 블록 상단 load_crm_summary() 호출에서 이미 로드됨
        if _crm_sum_df.empty:
            st.info("CRM 시트 연동 후 직원별 현황이 표시됩니다.")
        else:
            # ── ② 팀블로 당월 업로드 건수 (방식별)
            _timblo_emp_src = st.session_state.get("content_list_all") or content_list or []
            _emp_rows = []
            for _it in _timblo_emp_src:
                _dt_str = parse_date_str(_it.get("meetingStartTime", ""))
                if not _dt_str or _dt_str < _emp_month_str or _dt_str > _emp_today_str:
                    continue
                _pe   = parse_title(_it.get("editedTitle") or _it.get("title", ""))
                _br_e = _pe.get("branch", "") or "미확인"
                _st_e = _norm_key(_pe.get("staff", ""))
                _md_e = _pe.get("mode", "") or "기타"
                if _st_e:
                    _emp_rows.append({"지사": _br_e, "직원명_key": _st_e, "상담방식": _md_e})

            if _emp_rows:
                _emp_timblo_long = (
                    pd.DataFrame(_emp_rows)
                    .groupby(["지사", "직원명_key", "상담방식"]).size()
                    .reset_index(name="업로드건수")
                )
                _up_wide = _emp_timblo_long.pivot_table(
                    index=["지사", "직원명_key"],
                    columns="상담방식", values="업로드건수",
                    aggfunc="sum", fill_value=0,
                ).reset_index()
                _up_wide.columns.name = None
                _up_modes = [c for c in _up_wide.columns if c not in ["지사", "직원명_key"]]
                _up_wide  = _up_wide.rename(columns={m: f"{m}_업로드" for m in _up_modes})
            else:
                _up_wide = pd.DataFrame(columns=["지사", "직원명_key"])

            # ── ③ CRM 합산 LEFT JOIN 팀블로 (CRM 직원은 업로드 0이어도 표시)
            _emp_wide = _crm_sum_df.merge(_up_wide, on=["지사", "직원명_key"], how="left").fillna(0)

            # 3가지 방식 업로드 컬럼 보장
            for _m in ["방문", "비대면", "전화"]:
                if f"{_m}_업로드" not in _emp_wide.columns:
                    _emp_wide[f"{_m}_업로드"] = 0
                _emp_wide[f"{_m}_업로드"] = _emp_wide[f"{_m}_업로드"].astype(float)

            # CRM 건수도 float 통일
            for _m in ["방문_CRM", "비대면_CRM", "전화_CRM"]:
                _emp_wide[_m] = pd.to_numeric(_emp_wide[_m], errors="coerce").fillna(0)

            # ── ④ 총합 및 지표
            _emp_wide["총_CRM"]    = _emp_wide["방문_CRM"]    + _emp_wide["비대면_CRM"]    + _emp_wide["전화_CRM"]
            _emp_wide["총_업로드"] = _emp_wide["방문_업로드"] + _emp_wide["비대면_업로드"] + _emp_wide["전화_업로드"]
            _emp_wide["총_누락건수"] = (_emp_wide["총_CRM"] - _emp_wide["총_업로드"]).clip(lower=0)
            _emp_wide["업로드율(%)"] = (
                _emp_wide["총_업로드"] / _emp_wide["총_CRM"].replace(0, np.nan) * 100
            ).round(1).fillna(0)

            # ── ⑤ 시트 원본 행 순서 보존 (load_crm_summary가 이미 정렬 완료)

            # ── ⑥ 지사명 병합 표현: 첫 행만 지사명, 나머지는 빈 문자열
            _emp_wide = _emp_wide.reset_index(drop=True)
            _emp_wide["지사_표시"] = _emp_wide["지사"].where(
                ~_emp_wide["지사"].duplicated(), ""
            )

            # ── ⑦ 합계 행 추가
            _sum_row = {
                "지사_표시": "합계",
                "직원명":    "",
                "방문_CRM":   _emp_wide["방문_CRM"].sum(),
                "방문_업로드": _emp_wide["방문_업로드"].sum(),
                "비대면_CRM":  _emp_wide["비대면_CRM"].sum(),
                "비대면_업로드": _emp_wide["비대면_업로드"].sum(),
                "전화_CRM":   _emp_wide["전화_CRM"].sum(),
                "전화_업로드": _emp_wide["전화_업로드"].sum(),
                "총_누락건수": _emp_wide["총_누락건수"].sum(),
                "업로드율(%)": round(
                    _emp_wide["총_업로드"].sum()
                    / max(_emp_wide["총_CRM"].sum(), 1) * 100, 1
                ),
            }
            _emp_final = pd.concat(
                [_emp_wide, pd.DataFrame([_sum_row])], ignore_index=True
            )

            # ── ⑧ 경고
            _emp_alert_n = int((_emp_wide["총_누락건수"] >= 5).sum())
            if _emp_alert_n:
                st.warning(f"⚠️ 총 누락 5건 이상 직원: **{_emp_alert_n}명** — 즉시 업로드 독촉 필요")

            def _style_emp_final(row):
                if row.get("지사_표시") == "합계":
                    return ["font-weight: bold; background-color: #f0f0f0"] * len(row)
                return (
                    ["background-color: #fce8e8"] * len(row)
                    if row.get("총_누락건수", 0) >= 5 else [""] * len(row)
                )

            _disp_cols = [
                "지사_표시", "직원명",
                "방문_CRM", "방문_업로드",
                "비대면_CRM", "비대면_업로드",
                "전화_CRM", "전화_업로드",
                "총_누락건수", "업로드율(%)",
            ]
            _show = _emp_final[[c for c in _disp_cols if c in _emp_final.columns]]
            st.dataframe(
                _show.style.apply(_style_emp_final, axis=1),
                use_container_width=True, hide_index=True,
                column_config={
                    "지사_표시":     st.column_config.TextColumn("지사"),
                    "직원명":        st.column_config.TextColumn("직원명"),
                    "방문_CRM":      st.column_config.NumberColumn("방문 CRM",      format="%.1f건"),
                    "방문_업로드":   st.column_config.NumberColumn("방문 업로드",   format="%.1f건"),
                    "비대면_CRM":    st.column_config.NumberColumn("비대면 CRM",    format="%.1f건"),
                    "비대면_업로드": st.column_config.NumberColumn("비대면 업로드", format="%.1f건"),
                    "전화_CRM":      st.column_config.NumberColumn("전화 CRM",      format="%.1f건"),
                    "전화_업로드":   st.column_config.NumberColumn("전화 업로드",   format="%.1f건"),
                    "총_누락건수":   st.column_config.NumberColumn("총 누락 건수",  format="%.1f건"),
                    "업로드율(%)":   st.column_config.ProgressColumn(
                        "업로드율", min_value=0, max_value=100, format="%.1f%%"
                    ),
                },
            )

        st.markdown("---")

        if records_filtered:
            # ════════════════════════════════════════════
            # ② 중단 1열: 품질 지표 레이더  |  AI점수×결제전환율
            # ════════════════════════════════════════════
            col_radar, col_conv = st.columns(2)

            with col_radar:
                _radar_label = f"{f_counselor} vs 전체 평균" if f_counselor != "전체 보기" else "품질 지표 레이더"
                st.markdown(f"**{_radar_label}**")

                # ── 5개 그룹 정의 (항목 순서 기준 슬라이싱) ──────────────────
                _RADAR_GROUPS = [
                    ("라포·도입",     slice(0,  4)),   # 항목 1~4
                    ("니즈 파악",     slice(4,  9)),   # 항목 5~9
                    ("직업·시장",     slice(9,  14)),  # 항목 10~14
                    ("서비스·시스템", slice(14, 22)),  # 항목 15~22
                    ("수익·클로징",   slice(22, 30)),  # 항목 23~30
                ]

                def _norm_cat(s: str) -> str:
                    """항목 키 정규화: 공백·탭 제거 후 소문자 변환."""
                    return re.sub(r"[\s\u00a0\u3000]+", "", str(s)).strip()

                def _build_qs_avg(recs: list, cat_list: list, fallback: float = 50.0) -> dict:
                    """레코드 목록에서 항목별 0~100 스케일 평균 반환.
                    - quality_scores 키를 정규화하여 공백 오탐 방지
                    - 데이터 없는 항목은 fallback 으로 보정
                    """
                    agg: dict = {k: [] for k in cat_list}
                    for r in recs:
                        qs_raw = r.get("quality_scores", {})
                        # 키 정규화 매핑: 정규화된 키 → 원본 값
                        qs_norm = {_norm_cat(k): v for k, v in qs_raw.items()}
                        for k in cat_list:
                            v = qs_norm.get(_norm_cat(k))
                            if v is not None:
                                try:
                                    agg[k].append(float(v))
                                except (TypeError, ValueError):
                                    pass
                    result = {}
                    for k, scores in agg.items():
                        if scores:
                            result[k] = round(sum(scores) / len(scores) / 5 * 100, 1)
                        else:
                            result[k] = fallback
                    return result

                def _group_avg(qs_dict: dict, ordered_cats: list, fallback: float = 50.0) -> list:
                    """ordered_cats 기준 슬라이싱 → 그룹별 평균(0~100).
                    그룹에 매핑된 항목이 하나도 없으면 fallback(전체 평균) 사용 → 도형 왜곡 방지."""
                    avgs = []
                    for _, sl in _RADAR_GROUPS:
                        grp_keys = ordered_cats[sl]
                        scores = [qs_dict[k] for k in grp_keys if k in qs_dict and qs_dict[k] != fallback]
                        avgs.append(round(sum(scores) / len(scores), 1) if scores else fallback)
                    return avgs

                # ── 전체 항목 목록 수집 (키 strip 처리) ─────────────────────
                seen_cats: dict = {}
                for r in records_filtered:
                    rubric_r = r.get("rubric_used") or []
                    if rubric_r:
                        for row in rubric_r:
                            seen_cats.setdefault(row["항목"].strip(), None)
                    else:
                        for k in r.get("quality_scores", {}):
                            seen_cats.setdefault(k.strip(), None)
                all_cats = list(seen_cats.keys()) or list(QUALITY_WEIGHTS.keys())

                # 전체 평균 (빈 그룹 fallback용 전체 평균값 계산)
                qs_avg      = _build_qs_avg(records_filtered, all_cats, fallback=50.0)
                _global_avg = round(sum(qs_avg.values()) / max(len(qs_avg), 1), 1)

                group_cats = [g[0] for g in _RADAR_GROUPS]
                vals = _group_avg(qs_avg, all_cats, fallback=_global_avg)

                # ── 선택 상담자 오버레이 ──────────────────────────────────────
                _overlay_vals = None
                if f_counselor != "전체 보기":
                    _counsel_recs = [r for r in records_filtered
                                     if parse_title(r.get("title","")).get("staff","") == f_counselor]
                    if _counsel_recs:
                        _c_avg        = _build_qs_avg(_counsel_recs, all_cats, fallback=_global_avg)
                        _overlay_vals = _group_avg(_c_avg, all_cats, fallback=_global_avg)

                st.plotly_chart(
                    make_radar_chart(
                        group_cats, vals,
                        overlay_vals=_overlay_vals, overlay_name=f_counselor,
                        height=340,
                    ),
                    use_container_width=True,
                )

            with col_conv:
                st.markdown("**AI 점수별 실제 결제 전환율**")
                if not _matched_df.empty and "crm_구입여부" in _matched_df.columns:
                    def _score_band(score: float) -> str:
                        if score >= 90: return "90점 이상"
                        if score >= 80: return "80점대"
                        if score >= 70: return "70점대"
                        return "60점 이하"
                    _band_order = ["60점 이하", "70점대", "80점대", "90점 이상"]
                    _mb = _matched_df.copy()
                    _mb["점수구간"] = _mb["total_score"].apply(_score_band)
                    _band_stats = (
                        _mb.groupby("점수구간")
                        .agg(전체건수=("결제성공", "count"), 결제건수=("결제성공", "sum"))
                        .reindex(_band_order).fillna(0).reset_index()
                    )
                    _band_stats["결제전환율(%)"] = (
                        _band_stats["결제건수"]
                        / _band_stats["전체건수"].replace(0, np.nan) * 100
                    ).round(1).fillna(0)
                    fig_corr = go.Figure()
                    fig_corr.add_trace(go.Bar(
                        x=_band_stats["점수구간"], y=_band_stats["결제전환율(%)"],
                        marker_color=["#e74c3c","#f39c12","#2980b9","#27ae60"],
                        text=[f"{v}%" for v in _band_stats["결제전환율(%)"]],
                        textposition="outside", name="결제 전환율",
                    ))
                    fig_corr.add_trace(go.Scatter(
                        x=_band_stats["점수구간"], y=_band_stats["결제전환율(%)"],
                        mode="lines+markers",
                        line=dict(color="#8e44ad", width=2.5, dash="dot"),
                        marker=dict(size=9, color="#8e44ad"), name="추세선",
                    ))
                    fig_corr.update_layout(
                        xaxis=dict(title="AI 점수 구간", categoryorder="array",
                                   categoryarray=_band_order),
                        yaxis=dict(title="결제 전환율 (%)", range=[0, 110]),
                        legend=dict(orientation="h", y=-0.22),
                        margin=dict(t=20, b=50), height=340,
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    st.caption(f"CRM 매칭 {_crm_match_n}건 / 결제 {int(_matched_df['결제성공'].sum())}건 기준")
                else:
                    st.info("CRM 매칭 데이터가 없습니다.\n팀블로 제목의 상담자명·직원명이 CRM 시트와 일치해야 합니다.")

            st.markdown("---")

            # ════════════════════════════════════════════
            # ③ 중단 2열: 고객 키워드 TOP15  |  위험 신호 TOP5
            # ════════════════════════════════════════════
            col_kw, col_risk = st.columns(2)

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
                    fig_kw.update_layout(margin=dict(t=10, b=10), height=360, xaxis_title="")
                    st.plotly_chart(fig_kw, use_container_width=True)
                else:
                    st.caption("AI 분석 후 표시됩니다.")

            with col_risk:
                st.markdown("**위험 신호 (거절 사유) TOP 5**")
                risk_all: list = []
                for r in records_filtered:
                    risk_all.extend(r.get("risk_signals", []))
                if risk_all:
                    risk_cnt: dict = {}
                    for rs in risk_all:
                        risk_cnt[rs] = risk_cnt.get(rs, 0) + 1
                    top5 = sorted(risk_cnt.items(), key=lambda x: x[1], reverse=True)[:5]
                    for i, (rs, cnt) in enumerate(top5):
                        _rank_color = ["#c0392b","#e74c3c","#e67e22","#f39c12","#f1c40f"][i]
                        st.markdown(
                            f'<div style="background:#fff8f8;border-left:4px solid {_rank_color};'
                            f'border-radius:6px;padding:10px 14px;margin:6px 0;font-size:14px;">'
                            f'<b>#{i+1}</b> ⚠️ {rs} <span style="color:#888;font-size:12px;">({cnt}건)</span></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("탐지된 위험 신호 없음")

            st.markdown("---")

            # ════════════════════════════════════════════
            # ④ 매출 전환 분석 (리드 점수 + 전환 성공 패턴)
            # ════════════════════════════════════════════
            st.subheader("💰 매출 전환 분석")

            # ── _purch_by_id + _matched_df 통합 → 단일 전환 데이터 소스 ────────
            # lead_score 유무에 관계없이 전체 records_filtered 대상
            # 점수는 total_score(AI 채점) 우선, lead_score 폴백
            _conv_data:    list = []  # [(record, score)]  전환 성공
            _no_conv_data: list = []  # [(record, score)]  미전환(확인됨)

            _mid_set = set()  # _matched_df id 집합 (빠른 조회)
            if not _matched_df.empty and "id" in _matched_df.columns:
                _mid_set = set(_matched_df["id"].tolist())

            for _lr in records_filtered:
                _lr_id  = _lr.get("id")
                _lr_sc  = float(_lr.get("total_score") or _lr.get("lead_score") or 0)
                _bought = _purch_by_id.get(_lr_id)

                # _matched_df 로 보완
                if _bought is None and _lr_id in _mid_set:
                    _sl = _matched_df[_matched_df["id"] == _lr_id]
                    if not _sl.empty and "crm_구입여부" in _sl.columns:
                        _bought = _is_purchased(_sl.iloc[0]["crm_구입여부"])

                if _bought is True:
                    _conv_data.append((_lr, _lr_sc))
                elif _bought is False:
                    _no_conv_data.append((_lr, _lr_sc))

            # ── KPI 계산 ─────────────────────────────────────────────────────
            _conv_scores_all = [sc for _, sc in _conv_data if sc > 0]
            _all_scores_all  = [sc for _, sc in (_conv_data + _no_conv_data) if sc > 0]
            _conv_avg_lead   = round(sum(_conv_scores_all) / len(_conv_scores_all), 1) if _conv_scores_all else None
            _all_avg_lead    = round(sum(_all_scores_all)  / len(_all_scores_all),  1) if _all_scores_all  else None
            _conv_count      = len(_conv_data)

            _lcol1, _lcol2, _lcol3 = st.columns(3)
            _lcol1.metric(
                "전환 성공 리드 평균 점수",
                f"{_conv_avg_lead}점" if _conv_avg_lead is not None else "데이터 부족",
                help="결제 완료 상담의 AI 채점 평균 (total_score 기준)",
            )
            _lcol2.metric(
                "전체 리드 평균 점수",
                f"{_all_avg_lead}점" if _all_avg_lead is not None else "분석 전",
            )
            _lcol3.metric(
                "전환 성공 건수",
                f"{_conv_count}건" if (_conv_data or _no_conv_data) else "CRM 매칭 필요",
            )

            st.markdown("")

            # ── 리드 점수별 전환율 차트 (X: 점수구간, Y: 전환율%)
            _col_lead_chart, _col_success = st.columns(2)

            with _col_lead_chart:
                st.markdown("**📈 리드 점수 구간별 실제 결제 전환율**")
                # _conv_data + _no_conv_data: total_score 기준, lead_score 폴백
                _all_scored = [
                    (sc, True)  for _, sc in _conv_data    if sc > 0
                ] + [
                    (sc, False) for _, sc in _no_conv_data if sc > 0
                ]
                if _all_scored:
                    def _lead_band(score: float) -> str:
                        b = (int(score) // 10) * 10
                        return f"{b}~{b+9}점"
                    _band_order = [f"{b}~{b+9}점" for b in range(0, 100, 10)]
                    _lband_stats: dict = {b: {"total": 0, "conv": 0} for b in _band_order}
                    for sc, conv in _all_scored:
                        b = _lead_band(sc)
                        if b in _lband_stats:
                            _lband_stats[b]["total"] += 1
                            if conv:
                                _lband_stats[b]["conv"] += 1
                    _lband_df = pd.DataFrame([
                        {"점수구간": b,
                         "전체건수": v["total"],
                         "전환율(%)": round(v["conv"] / v["total"] * 100, 1) if v["total"] else 0}
                        for b, v in _lband_stats.items()
                        if v["total"] > 0
                    ])
                    if not _lband_df.empty:
                        fig_lead = go.Figure()
                        fig_lead.add_trace(go.Bar(
                            x=_lband_df["점수구간"], y=_lband_df["전환율(%)"],
                            marker_color=["#27ae60" if v >= 50 else "#e74c3c" if v < 25 else "#f39c12"
                                          for v in _lband_df["전환율(%)"]],
                            text=[f"{v}%" for v in _lband_df["전환율(%)"]],
                            textposition="outside", name="전환율",
                        ))
                        fig_lead.add_trace(go.Scatter(
                            x=_lband_df["점수구간"], y=_lband_df["전환율(%)"],
                            mode="lines+markers",
                            line=dict(color="#8e44ad", width=2.5, dash="dot"),
                            marker=dict(size=8, color="#8e44ad"), name="추세선",
                        ))
                        fig_lead.update_layout(
                            xaxis=dict(title="점수 구간 (total_score 기준)"),
                            yaxis=dict(title="전환율 (%)", range=[0, 110]),
                            legend=dict(orientation="h", y=-0.25),
                            margin=dict(t=20, b=60), height=340,
                        )
                        st.plotly_chart(fig_lead, use_container_width=True)
                        st.caption(f"CRM 매칭 총 {len(_all_scored)}건 기준 (전환성공 {len(_conv_data)}건)")
                    else:
                        st.caption("데이터가 부족합니다.")
                else:
                    st.info("CRM 매칭 데이터가 필요합니다.")

            # ── 전환 성공 케이스 공통점 도출
            with _col_success:
                st.markdown("**🏆 전환 성공 상담의 3가지 공통점**")
                # _conv_data 에서 직접 레코드 추출 (단일 소스)
                _success_recs = [r for r, _ in _conv_data]

                if _success_recs:
                    def _iter_field(rec, key):
                        """analyzed 레코드에서 리스트 필드를 안전하게 순회.
                        값이 list → 그대로, str → 쉼표 분리, 기타 → 빈 iter."""
                        val = rec.get(key)
                        if isinstance(val, list):
                            yield from (str(v).strip() for v in val if str(v).strip())
                        elif isinstance(val, str) and val.strip() not in ("", "[]", "null"):
                            import json as _json
                            try:
                                parsed = _json.loads(val)
                                if isinstance(parsed, list):
                                    yield from (str(v).strip() for v in parsed if str(v).strip())
                                    return
                            except Exception:
                                pass
                            for part in val.split(","):
                                p = part.strip().strip('"').strip("'")
                                if p:
                                    yield p

                    # ① 공통 클로징 멘트 집계
                    _closing_cnt: dict = {}
                    for r in _success_recs:
                        for ph in _iter_field(r, "closing_phrases"):
                            _closing_cnt[ph] = _closing_cnt.get(ph, 0) + 1
                    _top_closing = sorted(_closing_cnt.items(), key=lambda x: x[1], reverse=True)[:3]

                    # ② 결심 직전 고객 발화 패턴
                    _decision_cnt: dict = {}
                    for r in _success_recs:
                        for ds in _iter_field(r, "decision_signals"):
                            _decision_cnt[ds] = _decision_cnt.get(ds, 0) + 1
                    _top_decision = sorted(_decision_cnt.items(), key=lambda x: x[1], reverse=True)[:3]

                    # ③ 공통 강점 키워드
                    _str_cnt: dict = {}
                    for r in _success_recs:
                        for s in _iter_field(r, "strengths"):
                            _str_cnt[s] = _str_cnt.get(s, 0) + 1
                    _top_str = sorted(_str_cnt.items(), key=lambda x: x[1], reverse=True)[:3]

                    st.markdown(
                        f'<div style="background:#f0fff4;border-left:4px solid #27ae60;'
                        f'border-radius:6px;padding:10px 14px;margin:4px 0;">'
                        f'<b>🗣 공통 클로징 멘트</b></div>',
                        unsafe_allow_html=True,
                    )
                    if _top_closing:
                        for ph, cnt in _top_closing:
                            st.markdown(
                                f'<div style="background:#f9fffe;border:1px solid #a8e6c4;'
                                f'border-radius:4px;padding:6px 12px;margin:3px 0;font-size:13px;">'
                                f'💬 {ph} <span style="color:#888;font-size:11px;">({cnt}건)</span></div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("클로징 멘트 데이터 없음 (재분석 필요)")

                    st.markdown(
                        f'<div style="background:#fff9e6;border-left:4px solid #f39c12;'
                        f'border-radius:6px;padding:10px 14px;margin:8px 0 4px;">'
                        f'<b>🔑 결심 직전 고객 발화 패턴</b></div>',
                        unsafe_allow_html=True,
                    )
                    if _top_decision:
                        for ds, cnt in _top_decision:
                            st.markdown(
                                f'<div style="background:#fffdf0;border:1px solid #f5d88e;'
                                f'border-radius:4px;padding:6px 12px;margin:3px 0;font-size:13px;">'
                                f'🙋 {ds} <span style="color:#888;font-size:11px;">({cnt}건)</span></div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("패턴 데이터 없음 (재분석 필요)")

                    st.markdown(
                        f'<div style="background:#f0f4ff;border-left:4px solid #4169E1;'
                        f'border-radius:6px;padding:10px 14px;margin:8px 0 4px;">'
                        f'<b>✨ 공통 상담사 강점</b></div>',
                        unsafe_allow_html=True,
                    )
                    for s, cnt in _top_str[:3]:
                        st.markdown(
                            f'<div style="background:#f5f8ff;border:1px solid #b3c6ff;'
                            f'border-radius:4px;padding:6px 12px;margin:3px 0;font-size:13px;">'
                            f'⭐ {s} <span style="color:#888;font-size:11px;">({cnt}건)</span></div>',
                            unsafe_allow_html=True,
                        )
                    st.caption(f"전환 성공 {len(_success_recs)}건 기준 분석")
                else:
                    st.info(
                        "CRM 매칭 후 결제 완료 케이스가 있으면\n"
                        "공통 패턴을 자동 추출합니다.\n\n"
                        "※ 기존 분석 결과는 🔄 재분석으로 클로징 멘트를 추가 추출할 수 있습니다."
                    )

            st.markdown("---")

        # ════════════════════════════════════════════
        if _crm_load_error:
            st.info(f"📋 CRM 연동 오류 — {_crm_load_error[:100]}")

        if records_filtered:
            # ════════════════════════════════════════════
            # ④-B 상담방식별 Top 5 랭킹
            # ════════════════════════════════════════════
            st.subheader("🏆 상담방식별 Top 5 랭킹")
            _ch_cols = st.columns(3)
            for _ch_key, _ch_col in [("방문", _ch_cols[0]), ("비대면", _ch_cols[1]), ("전화", _ch_cols[2])]:
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
                        _ch_df.index = _ch_df.index + 1
                        st.dataframe(_ch_df, use_container_width=True, hide_index=False,
                                     column_config={
                                         "직원명":   st.column_config.TextColumn("직원명"),
                                         "건수":     st.column_config.NumberColumn("건수", format="%d건"),
                                         "평균점수": st.column_config.ProgressColumn(
                                             "평균점수", min_value=0, max_value=100, format="%.1f점"),
                                     })
                    else:
                        st.caption("해당 방식 데이터 없음")

            st.markdown("---")

            # ════════════════════════════════════════════
            # 기간별 추이
            # ════════════════════════════════════════════
            if len(records_filtered) >= 2:
                t_data = [{"날짜": r["date"], "점수": r["total_score"]}
                          for r in records_filtered if r.get("date")]
                if t_data:
                    t_df  = pd.DataFrame(t_data)
                    t_df["날짜"] = pd.to_datetime(t_df["날짜"])
                    t_agg = t_df.groupby("날짜")["점수"].mean().round(1).reset_index()
                    fig_t = px.line(t_agg, x="날짜", y="점수", markers=True, range_y=[0, 100],
                                    color_discrete_sequence=["royalblue"])
                    fig_t.add_hline(y=85, line_dash="dot", line_color="green",
                                    annotation_text="우수(85점)", annotation_position="bottom right")
                    fig_t.add_hline(y=60, line_dash="dot", line_color="red",
                                    annotation_text="개선필요(60점)", annotation_position="top right")
                    fig_t.update_layout(margin=dict(t=20, b=20), hovermode="x unified")
                    st.subheader("기간별 점수 추이")
                    st.plotly_chart(fig_t, use_container_width=True)
                    st.markdown("---")

            # ════════════════════════════════════════════
            # 분석 결과 테이블
            # ════════════════════════════════════════════
            st.subheader("분석 결과 테이블")
            _tbl_mand_items = st.session_state.user_config.get("mandatory_items", [])
            rows = []
            for r in records_filtered:
                p  = parse_title(r.get("title",""))
                mc = r.get("mandatory_check", {})
                if _tbl_mand_items:
                    mand_parts = []
                    for mi in _tbl_mand_items:
                        raw_val = mc.get(mi) if mc else None
                        if raw_val is None:           done = False
                        elif isinstance(raw_val, dict): done = _coerce_done(raw_val.get("done", False))
                        else:                           done = _coerce_done(raw_val)
                        mand_parts.append(("✅ " if done else "❌ ") + mi)
                    mand_text = ("  |  ".join("❓ " + mi for mi in _tbl_mand_items)
                                 if not mc else "  |  ".join(mand_parts))
                else:
                    mand_text = "-"
                rows.append({
                    "날짜": r["date"], "상담방식": p["mode"], "지사": p["branch"],
                    "직원명": p["staff"], "상담자": p["client"],
                    "총점": r["total_score"], "등급": score_grade(r["total_score"]),
                    "엔진": r.get("engine",""),
                    "흐름": f"{sum(r.get('flow_stages',{}).values())}/{len(r.get('flow_stages',{})) or 6}",
                    "필수 안내": mand_text,
                })

            def _score_color(s):
                if s >= 85: return "#27ae60"
                if s >= 70: return "#2980b9"
                if s >= 60: return "#f39c12"
                return "#e74c3c"

            _tbl_body = ""
            for row in rows:
                sc = row["총점"]; col = _score_color(sc)
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
            st.markdown("---")

            # ════════════════════════════════════════════
            # 필수 안내 준수율 차트
            # ════════════════════════════════════════════
            mand_items = st.session_state.user_config.get("mandatory_items", [])
            mand_analyzed = [r for r in records_filtered if r.get("mandatory_check")]
            if mand_items and mand_analyzed:
                st.subheader("필수 안내 준수율")
                mand_stats = []
                for item in mand_items:
                    done_count = 0
                    for r in mand_analyzed:
                        raw_val = r.get("mandatory_check", {}).get(item)
                        if raw_val is None: continue
                        done_raw = raw_val.get("done", raw_val.get("준수", False)) if isinstance(raw_val, dict) else raw_val
                        if _coerce_done(done_raw): done_count += 1
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
                st.dataframe(mand_df[["항목","준수율(%)","준수","미준수","분석건수"]],
                             use_container_width=True, hide_index=True)
                st.markdown("---")

            # ════════════════════════════════════════════
            # AI 전략 인사이트 — 3가지 관점
            # ════════════════════════════════════════════
            st.subheader("💡 AI 전략 인사이트")

            # ── 공통 데이터 준비 ──
            # 전환 성공 ID 집합: CRM 매칭 + 구입 확인된 레코드
            _insight_success_ids: set = set()
            _insight_fail_ids: set    = set()   # 리드 점수 높은데 미결제
            if not _matched_df.empty and "crm_구입여부" in _matched_df.columns and "id" in _matched_df.columns:
                _matched_df["결제성공_ins"] = _matched_df["crm_구입여부"].apply(_is_purchased)
                _insight_success_ids = set(_matched_df[_matched_df["결제성공_ins"]]["id"].tolist())
                _fail_mask = (~_matched_df["결제성공_ins"])
                _insight_fail_ids = set(_matched_df[_fail_mask]["id"].tolist())

            _ins_success_recs = [r for r in records_filtered if r.get("id") in _insight_success_ids]
            # 전환 실패 중 리드 점수 60점 이상인 건만 (의미 있는 고관심 미전환)
            _ins_fail_recs    = [r for r in records_filtered
                                 if r.get("id") in _insight_fail_ids and r.get("lead_score", 0) >= 60]

            ins_tab1, ins_tab2, ins_tab3 = st.tabs([
                "🏆 결제 전환 성공 패턴",
                "⚠️ 전환 실패 원인",
                "📊 득점 격차 분석",
            ])

            # ── 탭 1: 결제 전환 성공 패턴 ──────────────────────────────────
            with ins_tab1:
                if not _ins_success_recs:
                    st.info("CRM '구입' 확인 레코드가 없습니다. CRM 연동 후 재확인하세요.")
                else:
                    st.caption(f"CRM 구입 확인 {len(_ins_success_recs)}건에서 추출한 전환 성공 패턴")

                    # 클로징 멘트 빈도
                    _cph: dict = {}
                    for r in _ins_success_recs:
                        for ph in r.get("closing_phrases", []):
                            if ph: _cph[ph] = _cph.get(ph, 0) + 1
                    _top_cp = sorted(_cph.items(), key=lambda x: x[1], reverse=True)[:5]

                    # 결심 직전 고객 신호 빈도
                    _dsh: dict = {}
                    for r in _ins_success_recs:
                        for ds in r.get("decision_signals", []):
                            if ds: _dsh[ds] = _dsh.get(ds, 0) + 1
                    _top_ds = sorted(_dsh.items(), key=lambda x: x[1], reverse=True)[:5]

                    # 핵심 설득 키워드
                    _sckw: dict = {}
                    for r in _ins_success_recs:
                        for kw in r.get("counselor_keywords", []):
                            _sckw[kw] = _sckw.get(kw, 0) + 1
                    _top_sckw = sorted(_sckw.items(), key=lambda x: x[1], reverse=True)[:12]

                    col_cp, col_ds = st.columns(2)
                    with col_cp:
                        st.markdown("**💬 핵심 클로징 멘트**")
                        if _top_cp:
                            for ph, cnt in _top_cp:
                                st.markdown(
                                    f'<div style="background:#f0fff4;border-left:4px solid #27ae60;'
                                    f'border-radius:6px;padding:8px 14px;margin:5px 0;'
                                    f'font-size:13px;line-height:1.5;">'
                                    f'💬 {ph} <span style="color:#888;font-size:11px;">({cnt}건)</span>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.info("클로징 멘트 데이터 없음")
                    with col_ds:
                        st.markdown("**🎯 결심 직전 고객 신호**")
                        if _top_ds:
                            for ds, cnt in _top_ds:
                                st.markdown(
                                    f'<div style="background:#fff9e6;border-left:4px solid #f39c12;'
                                    f'border-radius:6px;padding:8px 14px;margin:5px 0;'
                                    f'font-size:13px;line-height:1.5;">'
                                    f'🎯 {ds} <span style="color:#888;font-size:11px;">({cnt}건)</span>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.info("결심 신호 데이터 없음")

                    st.markdown("**🔑 전환 성공 상담사 핵심 키워드**")
                    if _top_sckw:
                        _kw_html = " ".join(
                            f'<span style="background:#e8f4fd;border:1px solid #3498db;'
                            f'border-radius:14px;padding:4px 10px;margin:3px;'
                            f'display:inline-block;font-size:13px;color:#1a5276;">'
                            f'{"🔥" if cnt >= 3 else "✦"} {kw} <b>({cnt})</b></span>'
                            for kw, cnt in _top_sckw
                        )
                        st.markdown(_kw_html, unsafe_allow_html=True)

            # ── 탭 2: 전환 실패 원인 분석 ────────────────────────────────
            with ins_tab2:
                if not _ins_fail_recs:
                    st.info("리드 점수 60점 이상 미전환 레코드가 없습니다.")
                else:
                    st.caption(
                        f"리드 점수 60점↑ 미결제 {len(_ins_fail_recs)}건에서 추출한 망설임 포인트"
                    )

                    # 위험 신호 빈도
                    _rsh: dict = {}
                    for r in _ins_fail_recs:
                        for rs in r.get("risk_signals", []):
                            if rs: _rsh[rs] = _rsh.get(rs, 0) + 1
                    _top_rs = sorted(_rsh.items(), key=lambda x: x[1], reverse=True)[:8]

                    # 고객 주요 키워드 (망설임 관련)
                    _fail_ckw: dict = {}
                    for r in _ins_fail_recs:
                        for kw in r.get("customer_keywords", []):
                            _fail_ckw[kw] = _fail_ckw.get(kw, 0) + 1
                    _top_fail_ckw = sorted(_fail_ckw.items(), key=lambda x: x[1], reverse=True)[:10]

                    # 감정 부정 평균
                    _neg_scores = [r.get("sentiment", {}).get("negative", 0) for r in _ins_fail_recs]
                    _avg_neg = round(sum(_neg_scores) / max(len(_neg_scores), 1) * 100, 1)

                    col_rs, col_fkw = st.columns(2)
                    with col_rs:
                        st.markdown("**⚠️ 결정적 망설임 신호 (위험 신호)**")
                        if _top_rs:
                            for rs, cnt in _top_rs:
                                st.markdown(
                                    f'<div style="background:#fef9e7;border-left:4px solid #e74c3c;'
                                    f'border-radius:6px;padding:7px 12px;margin:4px 0;'
                                    f'font-size:13px;">'
                                    f'⚠️ {rs} <span style="color:#888;font-size:11px;">({cnt}건)</span>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.info("위험 신호 데이터 없음")
                    with col_fkw:
                        st.markdown(f"**💭 고객 주요 키워드** (부정 감정 평균: {_avg_neg}%)")
                        if _top_fail_ckw:
                            _fkw_html = " ".join(
                                f'<span style="background:#fce8e8;border:1px solid #e74c3c;'
                                f'border-radius:14px;padding:4px 10px;margin:3px;'
                                f'display:inline-block;font-size:13px;color:#922b21;">'
                                f'{kw} ({cnt})</span>'
                                for kw, cnt in _top_fail_ckw
                            )
                            st.markdown(_fkw_html, unsafe_allow_html=True)

            # ── 탭 3: 득점 격차 분석 ─────────────────────────────────────
            with ins_tab3:
                if len(records_filtered) < 3:
                    st.info("득점 격차 분석은 3건 이상 분석 후 가능합니다.")
                else:
                    # 항목별 점수 집계
                    _item_scores: dict = {}
                    for r in records_filtered:
                        rubric_r = r.get("rubric_used") or []
                        for k, v in r.get("quality_scores", {}).items():
                            try:
                                fv = float(v)
                            except (TypeError, ValueError):
                                continue
                            배점 = next((row["배점"] for row in rubric_r if row["항목"] == k), 5)
                            환산 = round((fv / 5.0) * 배점, 2)
                            if k not in _item_scores:
                                _item_scores[k] = {"scores": [], "max": 배점}
                            _item_scores[k]["scores"].append(환산)

                    if not _item_scores:
                        st.info("품질 점수 데이터가 없습니다.")
                    else:
                        _gap_rows = []
                        for item, data in _item_scores.items():
                            sc = data["scores"]
                            if len(sc) < 2: continue
                            _max_sc  = max(sc)
                            _min_sc  = min(sc)
                            _avg_sc  = round(sum(sc) / len(sc), 2)
                            _gap     = round(_max_sc - _min_sc, 2)
                            _gap_rows.append({
                                "항목": item,
                                "평균(환산)": _avg_sc,
                                "최고점": _max_sc,
                                "최저점": _min_sc,
                                "격차": _gap,
                                "배점": data["max"],
                            })

                        _gap_df = pd.DataFrame(_gap_rows).sort_values("격차", ascending=False)

                        st.caption("격차가 클수록 개인별 편차가 큰 취약 항목")

                        # 상위 3개 취약 항목 카드
                        _top3_gap = _gap_df.head(3)
                        _gap_cols = st.columns(len(_top3_gap))
                        for idx, (_, row) in enumerate(_top3_gap.iterrows()):
                            with _gap_cols[idx]:
                                st.markdown(
                                    f'<div style="background:#fef9e7;border:1px solid #f39c12;'
                                    f'border-radius:8px;padding:12px;text-align:center;">'
                                    f'<div style="font-weight:700;font-size:14px;color:#d35400;">'
                                    f'{row["항목"]}</div>'
                                    f'<div style="font-size:22px;font-weight:700;margin:6px 0;">'
                                    f'격차 {row["격차"]}점</div>'
                                    f'<div style="font-size:12px;color:#555;">'
                                    f'최고 {row["최고점"]}점 / 최저 {row["최저점"]}점 / '
                                    f'평균 {row["평균(환산)"]}점</div>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                        st.markdown("**전체 항목별 득점 격차표**")
                        st.dataframe(
                            _gap_df[["항목","배점","평균(환산)","최고점","최저점","격차"]]
                            .reset_index(drop=True),
                            use_container_width=True,
                            hide_index=True,
                        )


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
                    # ── 사전 검사: 클라이언트 및 AI 키 확인 ──
                    if not client:
                        st.error("팀블로 API에 연결되어 있지 않습니다. 사이드바에서 재연결하세요.")
                    elif not is_gemini_ready(cfg) and not is_openai_ready(cfg):
                        st.error(
                            "AI 분석 키가 설정되지 않았습니다.  \n"
                            ".env 파일에 `GEMINI_API_KEY` 또는 `OPENAI_API_KEY`를 입력한 뒤 앱을 재시작하세요."
                        )
                    else:
                        try:
                            # 재분석 시 트랜스크립트 캐시 강제 초기화 → 최신 설정으로 프롬프트 재생성
                            st.session_state.transcripts.pop(sel_id, None)
                            with st.spinner("AI 심층 분석 중..."):
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
                            etype_btn = classify_error(e)
                            if etype_btn == "AUTH":
                                st.error(
                                    f"인증 오류: API 키가 만료되었거나 유효하지 않습니다.  \n"
                                    f"`.env` 파일의 키 값을 확인하세요.  \n"
                                    f"상세: `{str(e)[:200]}`"
                                )
                            elif etype_btn == "RATE_LIMIT":
                                st.warning(f"API 호출 한도 초과입니다. 잠시 후 다시 시도하세요.  \n`{str(e)[:120]}`")
                            elif etype_btn == "EMPTY_DATA":
                                st.info(f"이 상담은 분석 가능한 텍스트가 없습니다.  \n`{str(e)[:120]}`")
                            else:
                                st.error(f"분석 실패 [{etype_btn}]: {str(e)[:300]}")

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
