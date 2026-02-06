import os
import time
import json
import ast
from dotenv import load_dotenv

# --- [모듈 임포트] ---
from rag_retriever import get_opponent_party_report, SMOGON_DB, LEAD_STATS
from Battle_Preparing.user_party import my_party

# 계산기 모듈
from Calculator.calculator import run_calculation
from Calculator.speed_checker import check_turn_order
from Calculator.stat_estimator import estimate_stats 
from Calculator.move_loader import get_move_data # [NEW] API기반 기술 로더

# LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# 1. 환경 설정
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY가 .env 파일에 설정되지 않았습니다.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview", 
    temperature=0.1, 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --------------------------------------------------------------------------
# [Helper 0] 토큰 정보 추출 함수
# --------------------------------------------------------------------------
def get_token_info(response):
    """LangChain 응답 객체에서 토큰 사용량을 추출합니다."""
    try:
        usage = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
        elif hasattr(response, 'response_metadata') and 'usage_metadata' in response.response_metadata:
            usage = response.response_metadata['usage_metadata']
            
        if usage:
            return {
                "input_tokens": usage.get('input_tokens', 0),
                "output_tokens": usage.get('output_tokens', 0),
                "total_tokens": usage.get('total_tokens', 0)
            }
    except Exception:
        pass
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

# --------------------------------------------------------------------------
# [Helper 1] 시뮬레이션 실행 함수 (수정됨)
# --------------------------------------------------------------------------
def run_simulation(my_party_data, opponent_list):
    """
    [핵심] 내 포켓몬 vs 상대 주요 선봉의 대면 시뮬레이션 실행
    """
    report = "=== ⚔️ 선봉 대면 시뮬레이션 (Simulation Report) ===\n"
    
    # 1. 상대 선봉 후보 선정 (Top 3)
    sorted_opps = sorted(opponent_list, key=lambda x: LEAD_STATS.get(x, 0), reverse=True)[:3]
    report += f"🎯 상대 유력 선봉 TOP 3: {', '.join(sorted_opps)}\n\n"

    for my_name, my_data in my_party_data.items():
        # 내 포켓몬 스펙 포장
        my_spec = {
            'stats': my_data['stats'],
            'ranks': {}, 
            'item': my_data['item'],
            'status': None,
            'ability': my_data.get('ability'),
            'types': [], 
            'is_terastal': False
        }
        
        # [수정] 내 기술 중 '가장 위력이 높은 기술' 하나 선정
        my_best_move = "Tackle"
        # 비교를 위해 초기값 위력 0 설정
        my_move_spec = {"name": "Tackle", "power": 0, "type": "Normal", "category": "Physical", "priority": 0}
        
        for m in my_data['moves']:
            # API 로더를 통해 정보 가져오기
            info = get_move_data(m)
            
            # 공격 기술이고, 현재 선택된 기술보다 위력이 높으면 교체
            # (break 없이 끝까지 돌려서 가장 센 기술을 찾음)
            if info['power'] > my_move_spec['power']:
                my_best_move = m
                my_move_spec = info
        
        report += f"[{my_name}의 분석]\n"

        for opp_name in sorted_opps:
            # 상대 스펙 추정
            opp_est = estimate_stats(opp_name)
            if not opp_est: continue
            
            opp_spec = {
                'stats': opp_est['stats'],
                'ranks': {},
                'item': None, 
                'status': None,
                'screens': {}
            }
            
            # A. 스피드 확인 (상대 기술 우선도는 0 가정)
            speed_res = check_turn_order(
                my_spec, opp_spec, 
                field_spec={}, 
                my_move_spec=my_move_spec,
                opp_move_spec={'priority':0}
            )
            
            speed_txt = "🚀선공" if speed_res['is_my_turn'] else "🐢후공"
            if speed_res['is_my_turn'] is None: speed_txt = "⚖️동속"
            
            # B. 데미지 확인
            dmg_res = run_calculation(my_spec, opp_spec, my_move_spec, field_spec={})
            ko_txt = dmg_res['damage']['ko_result']
            percent = dmg_res['damage']['percent_range']
            
            report += f"  vs {opp_name}: {speed_txt} | {my_best_move}: {percent} ({ko_txt})\n"
            
        report += "\n"
        
    return report

# --------------------------------------------------------------------------
# [Helper 2] 응답 추출 및 입력 파싱
# --------------------------------------------------------------------------
def extract_clean_content(response):
    try:
        content = ""
        if isinstance(response, dict):
            if 'text' in response: content = response['text']
            elif 'content' in response: content = response['content']
        elif hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    parts.append(item['text'])
                else:
                    parts.append(str(item))
            content = "".join(parts)
            
        # 딕셔너리 형태의 문자열 파싱 시도
        try:
            parsed = ast.literal_eval(str(content))
            if isinstance(parsed, dict) and 'text' in parsed:
                return parsed['text']
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                if 'text' in parsed[0]:
                    return parsed[0]['text']
        except (ValueError, SyntaxError):
            pass
            
        return str(content)
    except Exception as e:
        return f"Error: {e}"
        
def parse_opponent_input(user_input_batch):
    """
    [Batch Process] 여러 파티 정보를 한번에 번역
    Input: "파티1 / 파티2 / ..." (슬래시로 구분된 문자열)
    Returns: (parsed_data_dict, token_usage_dict)
    Output schema: { "party_0": ["Mon1",...], "party_1": ["Mon1",...] }
    """
    
    # 입력 전처리: 슬래시(/)로 구분하여 리스트화
    if isinstance(user_input_batch, list):
        party_list = user_input_batch
    else:
        # 슬래시로 분리하고 빈 항목 제거
        party_list = [p.strip() for p in str(user_input_batch).split('/') if p.strip()]
        
    input_text = "\n".join(party_list)
        
    line_count = len(party_list)
    print(f"🔄 입력된 {line_count}개 파티 정보를 일괄 표준화(Batch Processing) 중입니다...")

    parser_template = """
    당신은 '포켓몬 이름 번역기'입니다. 비용 절감을 위해 배치 처리(Batch Processing)를 수행합니다.
    입력된 데이터는 개행문자(New Line)로 구분된 여러 상대방의 포켓몬 파티입니다.
    각 줄(Line)에 포함된 한국어 포켓몬 이름(약어/별명 포함)을 **Smogon/Showdown 영어 공식 명칭**으로 변환하세요.

    [입력 데이터]
    {user_input}

    [출력 형식 (JSON)]
    - 입력된 줄의 순서대로 "party_0", "party_1"... 형태의 키(Key)를 사용하세요.
    - 값(Value)은 영어 이름 문자열들의 리스트(List)여야 합니다.
    - Markdown 코드 블럭 없이 순수 JSON 객체만 출력하세요.

    예시:
    {{
        "party_0": ["Flutter Mane", "Urshifu-Rapid-Strike", "Dragonite", ...],
        "party_1": ["Gholdengo", "Ogerpon-Wellspring", "Ting-Lu", ...]
    }}
    """
    try:
        response = llm.invoke(parser_template.format(user_input=input_text))
        
        # 토큰 정보 추출
        token_info = get_token_info(response)
        print(f"💰 [Batch Parser] Tokens: I:{token_info['input_tokens']} + O:{token_info['output_tokens']} = {token_info['total_tokens']}")

        content = extract_clean_content(response)
        clean_content = content.replace("```json", "").replace("```python", "").replace("```", "").strip()
        
        parsed_data = {}
        try:
            parsed_data = json.loads(clean_content)
        except:
            try:
                parsed_data = ast.literal_eval(clean_content)
            except Exception as parse_err:
                 print(f"⚠️ 파싱 포맷 에러: {parse_err}")
                 # 실패 시 빈 딕셔너리 반환
                 return {}, token_info
        
        return parsed_data, token_info
        
    except Exception as e:
        print(f"❌ 배치 이름 변환 실패: {e}")
        return {}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

def format_my_party_info():
    if not my_party.team: return "❌ 내 파티 정보 없음"
    text = "=== 🛡️ 내 파티 상세 스펙 (My Team Stats) ===\n"
    for name, data in my_party.team.items():
        stats = data['stats']
        stat_str = f"H{stats['hp']} A{stats['atk']} B{stats['def']} C{stats['spa']} D{stats['spd']} [S{stats['spe']}]"
        moves = ", ".join(data['moves'])
        text += f"[{name}] @ {data['item']} | {data['ability']} | {data['tera_type']} Tera | Stats: {stat_str} | Moves: {moves}\n"
    return text

# --------------------------------------------------------------------------
# [Main Function] 분석 실행
# --------------------------------------------------------------------------
def analyze_entry_strategy(opponent_input):
    """
    [Entry Phase] 배치 처리 지원 (Batch Supported)
    Calculates simulations for ALL parties, then sends ONE prompt to LLM.
    
    Args:
        opponent_input: Raw string (lines of parties) OR List of strings
        
    Returns: 
        (analysis_result_dict, token_usage_dict)
        Output schema: { "party_0": "Report Text...", "party_1": "Report Text..." }
    """
    total_tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    # 1. 입력 파싱 (배치 파서 사용)
    # opponent_input이 이미 딕셔너리라면 파싱 건너뜀 (확장성 고려)
    if isinstance(opponent_input, dict):
        parsed_batch = opponent_input
    else:
        parsed_batch, parse_tokens = parse_opponent_input(opponent_input)
        for k in total_tokens: total_tokens[k] += parse_tokens[k]

    if not parsed_batch: 
        return {}, total_tokens

    party_count = len(parsed_batch)
    print(f"🔍 [Entry Phase] {party_count}개 파티에 대한 시뮬레이션 및 배치 분석 준비 중...")

    # 2. Python 내부 연산 (RAG + Simulation) - 토큰 비용 없음
    # 각 파티별로 Context를 미리 생성하여 텍스트 덩어리로 만듭니다.
    my_team_basic = format_my_party_info()
    
    batch_context_text = ""
    
    for party_id, opp_list in parsed_batch.items():
        # A. 상대 파티 RAG 데이터
        opp_context = get_opponent_party_report(opp_list)
        
        # B. 대면 시뮬레이션 (계산기)
        try:
            sim_report = run_simulation(my_party.team, opp_list)
        except Exception as e:
            sim_report = f"Simulation Error: {e}"
            
        # C. 텍스트 결합
        batch_context_text += f"""
        [[ {party_id} 상세 데이터 ]]
        1. Opponent Team Info:
        {opp_context}
        
        2. Simulation Report:
        {sim_report}
        --------------------------------------------------
        """

    # 3. 배치 프롬프트 설계
    template = """
    당신은 '포켓몬 랭크배틀(3vs3 싱글)' 전문 AI 코치입니다.
    
    아래에는 **사용자의 파티(My Team)** 정보 하나와, **여러 명의 상대방(Opponents)** 데이터가 나열되어 있습니다.
    각 상대방(Key: party_0, party_1...)에 대해 개별적인 승리 전략 리포트를 작성하여 JSON 객체로 반환하세요.

    [My Team Info]
    {my_team_context}

    [Batch Opponent Data]
    {batch_context_text}

    [분석 로직]
    1. **선봉 결정 (Lead Check)**: [3. 시뮬레이션 결과]를 보세요. 상대 유력 선봉(TOP 3)을 상대로 '🚀선공'이면서 '확정 1타'를 내는 포켓몬이 있다면 최고의 선봉입니다.
    2. **스피드 싸움**: 시뮬레이션에서 '🐢후공'이 뜨는 대면은 위험합니다. 기합의띠나 내구 보정이 없다면 피하세요.
    3. **선출 구성**: 선봉을 이길 수 있는 포켓몬 1마리 + 일관성 있는 에이스 1마리 + 쿠션 1마리로 구성하세요.

    [승리 플랜 양식]

    1. **나의 추천 선출**:
       - **세 마리 구성 요약: [포켓몬 이름], [포켓몬 이름], [포켓몬 이름]**
       - **선봉(Lead): [포켓몬 이름]**
         - 선정 이유: **(시뮬레이션 결과 인용 필수)** 예: "상대 딩루 상대로 선공이며, 인파이트로 확정 1타가 나옵니다."
       - **후속(Back): [포켓몬 이름], [포켓몬 이름]**
         - 역할: (에이스 / 쿠션 / 스위퍼)

    2. **상대 예상 선출 (Top 3)**: [이름], [이름], [이름]
       - 이유: (선봉 확률 통계 및 내 파티와의 상성 고려)

    3. **승리 플랜 (Game Plan)**:
       - (초반 운영과 주의해야 할 상대의 테라스탈/도구 변수를 3줄 요약)

    [출력 형식 (JSON Only)]
    각 키(party_N)에 대한 값은 아래 포맷의 문자열이어야 합니다.
    {{
       "party_0": "1. 상대 예상 선출: ... \\n2. 나의 추천 선출: ... \\n3. 승리 플랜: ...",
       "party_1": "1. 상대 예상 선출: ... \\n2. 나의 추천 선출: ... \\n3. 승리 플랜: ...",
       ...
    }}
    **주의**: Markdown 코드 블럭 없이 순수 JSON만 출력하세요.
    """

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    
    try:
        start_time = time.time()
        
        response = chain.invoke({
            "my_team_context": my_team_basic,
            "batch_context_text": batch_context_text
        })
        
        end_time = time.time()
        print(f"⏱️ 배치 분석 완료! (소요 시간: {end_time - start_time:.2f}초)")

        # 토큰 정보 추출
        main_tokens = get_token_info(response)
        print(f"💰 [Strategy Batch] Tokens: I:{main_tokens['input_tokens']} + O:{main_tokens['output_tokens']} = {main_tokens['total_tokens']}")
        
        # 토큰 누적
        for k in total_tokens: total_tokens[k] += main_tokens[k]

        # 결과 파싱
        content = extract_clean_content(response)
        clean_content = content.replace("```json", "").replace("```", "").strip()
        
        result_dict = {}
        try:
            result_dict = json.loads(clean_content)
        except:
            try:
                result_dict = ast.literal_eval(clean_content)
            except Exception as e:
                print(f"⚠️ 배치 결과 JSON 파싱 실패: {e}")
                return {}, total_tokens

        return result_dict, total_tokens

    except Exception as e:
        return {"error": f"❌ Gemini 분석 중 오류 발생: {str(e)}"}, total_tokens
    
def parse_recommended_selection(ai_response_batch):
    """
    [New] 배치 처리된 전략 리포트 딕셔너리에서 선출 정보를 일괄 추출
    Input: { "party_0": "Report...", "party_1": "Report..." }
    Returns: ( { "party_0": {lead, back1, back2}, ... }, token_usage_dict )
    """
    if not ai_response_batch or not isinstance(ai_response_batch, dict):
        return {}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    print("🔄 AI 추천 선출을 일괄 파싱(Batch Parsing)하여 상태에 반영 중...")
    
    # 입력 데이터를 JSON 문자열로 변환하여 프롬프트에 삽입
    input_json_str = json.dumps(ai_response_batch, ensure_ascii=False)

    parser_template = """
    당신은 '포켓몬 선출 리포트 파서'입니다. 배치 처리 모드입니다.
    입력된 JSON 객체는 여러 게임에 대한 분석 리포트(Value)를 담고 있습니다.
    각 리포트 텍스트에서 AI가 추천한 **[나의 선출 포켓몬 3마리]**를 추출하여 구조화된 JSON으로 반환하세요.
    
    규칙:
    1. 반드시 **영어 공식 명칭**만 사용하세요.
    2. 못 찾겠으면 null로 비워두세요.

    [입력 데이터 (JSON)]
    {input_json}

    [출력 형식 (JSON)]
    {{
        "party_0": {{ "lead": "Name", "back1": "Name", "back2": "Name" }},
        "party_1": {{ "lead": "Name", "back1": "Name", "back2": "Name" }},
        ...
    }}
    """
    
    prompt = PromptTemplate.from_template(parser_template)
    chain = prompt | llm
    
    try:
        response = chain.invoke({"input_json": input_json_str})
        
        # 토큰 정보 추출
        token_info = get_token_info(response)
        print(f"💰 [Selection Batch] Tokens: I:{token_info['input_tokens']} + O:{token_info['output_tokens']} = {token_info['total_tokens']}")

        content = extract_clean_content(response)
        clean_json = content.replace("```json", "").replace("```", "").strip()
        
        parsed_result = {}
        try:
            parsed_result = json.loads(clean_json)
        except:
            parsed_result = ast.literal_eval(clean_json)
            
        return parsed_result, token_info
        
    except Exception as e:
        print(f"❌ 배치 선출 파싱 실패: {e}")
        return {}, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
# --------------------------------------------------------------------------
# [실행 예시]
if __name__ == "__main__":
    # [추가된 부분] 테스트를 위해 내 파티를 먼저 로드해야 합니다.
    from Battle_Preparing.party_loader import load_party_from_file
    
    print("📂 [Test Mode] 파티 데이터 로드 중...")
    load_party_from_file("my_team.txt")
    
    if not my_party.team:
        print("❌ 파티 로드 실패. my_team.txt를 확인하세요.")
        exit()

    # 예시 입력 (슬래시로 구분된 여러 파티 정보)
    user_input_batch = """
    날치머, 물라오스, 망나뇽, 물거폰, 미라이돈, 무쇠머리 / 
    딩루, 어써러셔, 라우드본, 랜드로스, 뽀록나, 글라이온 / 
    흑마버드렉스, 모래털가죽, 고릴타, 물라오스, 무쇠머리, 뽀록나 / 
    미라이돈, 엘풍, 물라오스, 무쇠손, 다투곰, 파오젠 / 
    테라파고스, 모래털가죽, 고릴타, 물라오스, 뽀록나, 날치머 / 
    코라이돈, 모래털가죽, 무쇠머리, 뽀록나, 악라오스, 날치머 / 
    백마버드렉스, 키키링, 모래털가죽, 달투곰, 물라오스, 뽀록나 / 
    자마젠타, 땅을기는날개, 고릴타, 무쇠무인, 랜드로스, 뽀록나 / 
    가이오가, 브리두라스, 고릴타, 토네로스, 악라오스, 뽀록나 / 
    자시안, 모래털가죽, 고릴타, 날치머, 물라오스, 무쇠무인
    """
    
    print(f"\n🔍 배치 테스트 입력:\n{user_input_batch}")
    
    # 1. 배치 분석
    results_dict, token_data = analyze_entry_strategy(user_input_batch)
    
    print("\n📊 [Batch Analysis Results]")
    for pid, report in results_dict.items():
        print(f"\n=== {pid} Strategy ===\n{report[:100]}...") # 내용이 기니까 앞부분만 출력
        
    print("\n📊 Total Token Usage in Main Analysis:", token_data)
    
    # 2. 배치 선출 파싱
    selections_dict, sel_tokens = parse_recommended_selection(results_dict)
    
    print("\n📊 [Batch Selection Parsing]")
    print(json.dumps(selections_dict, indent=2))
    print(f"Selection Tokens: {sel_tokens}")