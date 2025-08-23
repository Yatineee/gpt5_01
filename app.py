import os
import re
import json
import time
import requests
import datetime as dt
import streamlit as st
from openai import OpenAI

# ------------------------------
# Config & Clients
# ------------------------------
from openai import OpenAI
import os

from openai import OpenAI
import os

client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=os.environ.get("AIML_API_KEY"),
)

MODEL = os.environ.get("AIML_MODEL", "gpt-5-2025-08-07")  # 或 gpt-5


st.set_page_config(page_title="AI Tutor — Learn & Review", layout="wide")

# 简单持久化（Space 重启会丢失，可扩展为 HF Datasets 或外部DB）
DATA_DIR = "./data"
REV_PATH = f"{DATA_DIR}/reviews.json"
os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(REV_PATH):
    with open(REV_PATH, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

# ------------------------------
# Helpers
# ------------------------------
@st.cache_data(show_spinner=False)
def fetch_github_readme(owner_repo: str):
    """Fetch README.md (or root README) via raw URLs.
    owner_repo: "owner/repo"
    """
    raw_candidates = [
        f"https://raw.githubusercontent.com/{owner_repo}/HEAD/README.md",
        f"https://raw.githubusercontent.com/{owner_repo}/main/README.md",
        f"https://raw.githubusercontent.com/{owner_repo}/master/README.md",
    ]
    for url in raw_candidates:
        r = requests.get(url, timeout=12)
        if r.status_code == 200 and len(r.text) > 32:
            return r.text
    return ""

def split_markdown_units(md: str, max_units: int = 12):
    """粗略按二/三级标题切分为学习单元。"""
    if not md:
        return [{"title": "README", "content": "(README not found)"}]
    # 找到所有 ## 或 ### 段落
    blocks = re.split(r"\n(?=##\s)|\n(?=###\s)", md)
    units = []
    for i, b in enumerate(blocks):
        title_match = re.match(r"^(#{2,3})\s+(.+)", b.strip())
        title = title_match.group(2).strip() if title_match else ("Section " + str(i+1))
        units.append({"title": title, "content": b.strip()})
        if len(units) >= max_units:
            break
    # 若过少，补一个总览
    if len(units) < 2:
        units = [{"title": "Overview", "content": md}]
    return units

# def call_gpt_json(user_prompt: str, system_prompt: str = ""):
#     rsp = client.responses.create(
#         model=MODEL,
#         response_format={"type": "json_object"},
#         input=[
#             {"role": "system", "content": system_prompt or "You are a helpful, expert learning coach."},
#             {"role": "user", "content": user_prompt},
#         ],
#     )
#     text = rsp.output_text
#     try:
#         return json.loads(text) if text else {}
#     except Exception:
#         return {"raw": text}

# def _coerce_json(text: str):
#     # 尝试把返回内容里的 JSON 提取出来（兜底）
#     try:
#         return json.loads(text)
#     except Exception:
#         m = re.search(r"\{.*\}", text, flags=re.S)
#         if m:
#             try: return json.loads(m.group(0))
#             except: pass
#     return {"raw": text}
# def call_gpt_json(user_prompt: str, system_prompt: str = ""):
#     # 1) 尝试 Responses API（如果 Aimlapi/SDK 支持）
#     try:
#         rsp = client.responses.create(
#             model=MODEL,
#             input=[
#                 {"role": "system", "content": system_prompt or "You are a helpful coach."},
#                 {"role": "user", "content": user_prompt},
#             ],
#             # 某些环境不支持这个参数，就会抛 TypeError
#             response_format={"type": "json_object"},
#         )
#         text = getattr(rsp, "output_text", None) or json.dumps(rsp, ensure_ascii=False)
#         return json.loads(text)
#     except TypeError:
#         # 2) 降级到 Chat Completions JSON 模式
#         try:
#             rsp = client.chat.completions.create(
#                 model=MODEL,
#                 messages=[
#                     {"role": "system", "content": system_prompt or "You are a helpful coach."},
#                     {"role": "user", "content": user_prompt},
#                 ],
#                 response_format={"type": "json_object"},
#                 temperature=0.2,
#             )
#             text = rsp.choices[0].message.content
#             return json.loads(text)
#         except Exception as e2:
#             # 3) 最后兜底：让模型仅输出 JSON，然后手动解析
#             prompt = (
#                 "Return a STRICT JSON object only. No prose, no code fences.\n\n" + user_prompt
#             )
#             rsp = client.chat.completions.create(
#                 model=MODEL,
#                 messages=[
#                     {"role": "system", "content": (system_prompt or "You are a helpful coach.") + " Output must be a SINGLE JSON object."},
#                     {"role": "user", "content": prompt},
#                 ],
#                 temperature=0.1,
#             )
#             text = rsp.choices[0].message.content
#             return _coerce_json(text)

# def call_gpt_text(messages):
#     rsp = client.responses.create(model=MODEL, input=messages)
#     return rsp.output_text

# def call_gpt_text(messages):
#     # 优先 Responses；不行就用 Chat Completions
#     try:
#         rsp = client.responses.create(
#             model=MODEL,
#             input=messages,  # [{"role": "...", "content": "..."}]
#         )
#         return rsp.output_text
#     except TypeError:
#         rsp = client.chat.completions.create(
#             model=MODEL,
#             messages=messages,
#             temperature=0.3,
#         )
#         return rsp.choices[0].message.content


# ---- 放在文件顶部附近（导入后）----
import json, re
import streamlit as st

def _extract_text_from_responses_obj(rsp):
    """
    兼容不同 Responses 实现：
    - rsp.output_text
    - rsp.output[*].content[*].text
    - rsp.choices[*].message.content（有些兼容层直接返回 chat 结构）
    - dict/json 场景
    """
    # 1) SDK 对象可能有 output_text
    text = getattr(rsp, "output_text", None)
    if text:
        return text

    # 2) SDK 对象可能能转成 dict
    try:
        d = rsp if isinstance(rsp, dict) else rsp.model_dump()
    except Exception:
        try:
            d = json.loads(str(rsp))
        except Exception:
            d = None

    if isinstance(d, dict):
        # 2a) 标准 Responses 树
        out = d.get("output") or d.get("response") or {}
        # 典型形状：{"output":[{"content":[{"type":"output_text","text":"..."}]}]}
        if isinstance(out, list) and out:
            content = out[0].get("content") if isinstance(out[0], dict) else None
            if isinstance(content, list):
                # 找 text
                for c in content:
                    if isinstance(c, dict):
                        if "text" in c and c["text"]:
                            return c["text"]
                        if c.get("type") in ("output_text","text") and c.get("text"):
                            return c["text"]

        # 2b) 有些兼容层直接返回 chat 结构
        choices = d.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {})
            if isinstance(msg, dict) and msg.get("content"):
                return msg["content"]

        # 2c) 还有些把正文塞在 top-level 的 text / message 里
        for key in ("text","message","content"):
            if isinstance(d.get(key), str) and d[key].strip():
                return d[key]

    # 3) 都没有就空字符串
    return ""

def _coerce_json(text: str):
    """把模型输出尽量解析为 JSON；否则包在 {"raw": "..."}"""
    if not text or not str(text).strip():
        return {"raw": ""}
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", str(text), flags=re.S)
        if m:
            try: return json.loads(m.group(0))
            except: pass
        return {"raw": str(text).strip()}

# ---- 覆盖你的两个调用函数 ----
def call_gpt_json(user_prompt: str, system_prompt: str = ""):
    """
    优先 Responses；若不可用自动降级到 Chat。
    不再强依赖 response_format，以提示词+兜底解析确保 JSON。
    """
    sys = (system_prompt or "You are a helpful coach.") + \
          " Output MUST be a single valid JSON object. No prose, no code fences."

    # 1) 尝试 responses
    try:
        rsp = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user_prompt},
            ],
            # temperature=0.2,
        )
        text = _extract_text_from_responses_obj(rsp)
        data = _coerce_json(text)
        # 方便调试：把原始响应放入 session
        st.session_state._last_api_json = getattr(rsp, "model_dump", lambda: str(rsp))()
        return data
    except TypeError:
        # 某些实现不支持 responses，走 chat
        pass
    except Exception as e:
        # 其他异常再试 chat
        st.info(f"Responses 调用异常，切换 chat: {e}")

    # 2) chat.completions
    rsp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    text = rsp.choices[0].message.content if rsp.choices else ""
    st.session_state._last_api_json = rsp.model_dump() if hasattr(rsp, "model_dump") else rsp
    return _coerce_json(text)

def call_gpt_text(messages):
    """
    文本问答：优先 responses；失败降级 chat。
    messages 形如 [{"role": "...", "content": "..."}]
    """
    try:
        rsp = client.responses.create(model=MODEL, input=messages, temperature=0.3)
        st.session_state._last_api_json = getattr(rsp, "model_dump", lambda: str(rsp))()
        return _extract_text_from_responses_obj(rsp)
    except TypeError:
        pass
    except Exception as e:
        st.info(f"Responses 调用异常，切换 chat: {e}")

    rsp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.3)
    st.session_state._last_api_json = rsp.model_dump() if hasattr(rsp, "model_dump") else rsp
    return rsp.choices[0].message.content if rsp.choices else ""


def save_review(item):
    with open(REV_PATH, "r", encoding="utf-8") as f:
        arr = json.load(f)
    arr.insert(0, item)
    with open(REV_PATH, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)

@st.cache_data(show_spinner=False)
def load_reviews():
    with open(REV_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------------------------
# Sidebar: Survey
# ------------------------------
st.sidebar.header("🎯 Initial")
persona = st.sidebar.multiselect(
    "Who are you（multiple）",
    ["Beginner", "University Student", "High School Student", "Career Changer", "IT Professional", "Data Analyst", "Researcher", "Other"],
)
interests = st.sidebar.multiselect(
    "Areas of Interest（multiple）",
    [
        "Java", "Python", "C++", "C", "C#", "Front-end (HTML/CSS/JS)", "Back-end (Node/Java/.NET)",
        "Data Analysis", "Machine Learning", "AI Agent", "DevOps", "Cloud (Azure/AWS/GCP)"
    ],
)
goals = st.sidebar.multiselect(
    "Your Goals（multiple）",
    ["Expand Employment Skills", "Data Analysis", "Academic Research", "Further Education/Employment", "Hobbies", "LeetCode/Algorithms", "System Design"],
)

with st.sidebar.expander("study time"):
    time_mode = st.radio("Frequency", ["per day", "per week"], horizontal=True)
    hours = st.number_input("Average Study Time (hours)", min_value=0.5, max_value=100.0, value=10.0, step=0.5)

if "profile" not in st.session_state:
    st.session_state.profile = {}

if st.sidebar.button("🧠 generate the plan", use_container_width=True):
    profile = {
        "persona": persona,
        "interests": interests,
        "goals": goals,
        "time": {"mode": time_mode, "hours": hours},
    }
    st.session_state.profile = profile

    # 1) 让 GPT recommend GitHub ebooks + 30 days study plan（JSON）
    user = (
        f"Persona: {', '.join(persona) or '(none)'}\n"
        f"Interests: {', '.join(interests) or '(none)'}\n"
        f"Goals: {', '.join(goals) or '(none)'}\n"
        f"Time: {time_mode} ~ {hours} 小时\n"
    )
    sys = "You are a senior curriculum designer. Recommend high-quality, actively maintained GitHub repos (1-3) that match the user's profile (prefer star>1k, clear README). Then design a 30-day plan."
    prompt = (
        "Based on the user profile below, return JSON with fields:\n" \
        "{\n  \"repos\": [ { \"name\": string, \"url\": string, \"why\": string } ],\n  \"plan30\": [ { \"dayRange\": string, \"milestones\": string[] } ]\n}\n\n" \
        "Profile:\n" + user +
        "Constraints:\n- repos <= 3, concise reasons.\n- Use free resources only.\n- Plan considers the user's time budget."
    )
    with st.spinner("向 GPT‑5 生成建议中…"):
        out = call_gpt_json(prompt, sys)
    st.session_state.reco = out

    if st.checkbox("显示原始返回（debug）"):
        st.write(st.session_state.get("_last_api_json"))


# ------------------------------
# Main Tabs
# ------------------------------
st.title("📚 AI Tutor — Study / Review")

tab1, tab2 = st.tabs(["Study Module", "Review Module"])

# ------------------------------
# Study Module
# ------------------------------
with tab1:
    st.subheader("Recommended Repositories and Learning Units")
    colA, colB = st.columns([1, 1])
    with colA:
        if st.session_state.get("reco"):
            st.json(st.session_state.reco)
        else:
            st.info("Fill out the questionnaire on the left and click 'Generate' to get repository recommendations and a 30-day plan.")

    # Select a GitHub repository
    st.markdown("---")
    repo_input = st.text_input("Select or paste a GitHub repository (owner/repo or full URL)", placeholder="e.g. TheAlgorithms/Python")

    def normalize_repo(s: str):
        if not s:
            return ""
        m = re.search(r"github\.com/([\w.-]+/[\w.-]+)", s)
        return m.group(1) if m else s.strip()

    if st.button("📥 get README and split units", disabled=not repo_input):
        owner_repo = normalize_repo(repo_input)
        with st.spinner("Fetching README.md…"):
            md = fetch_github_readme(owner_repo)
        st.session_state.repo = {"name": owner_repo, "readme": md}
        st.session_state.units = split_markdown_units(md)
        st.success(f"Split into {len(st.session_state.units)} study units")

    if "units" in st.session_state and st.session_state.units:
        unit_titles = [u["title"] for u in st.session_state.units]
        idx = st.selectbox("Select Study Unit", list(range(len(unit_titles))), format_func=lambda i: unit_titles[i])
        unit = st.session_state.units[idx]

        left, right = st.columns([1.4, 1])
        with left:
            st.markdown(f"### 📖 {unit['title']}")
            st.markdown(st.session_state.repo.get("readme")[:2000] if len(unit['content']) < 200 else unit['content'])
            st.caption("The left side displays the warehouse knowledge of this unit (split by the sections of the README).")

        with right:
            st.markdown("### 🤖 GPT‑5 Q&A")
            if "chat" not in st.session_state:
                st.session_state.chat = []  # [{q,a}]

            # 简易对话输入
            q = st.text_area("Your Question", height=120, placeholder="Please provide explanations/examples/practice suggestions based on the content on the left...")
            if st.button("Send Question", disabled=not q):
                # Constructing System Prompt: Use the information on the left as the known context + Request for LeetCode/Knowledge Base suggestions
                system = (
                    "You are a patient, expert CS tutor. Use ONLY the given repo context as known facts when answering; "
                    "if missing, say what is missing and suggest how to find it in the repo. Provide step-by-step guidance. "
                    "After answering, suggest 1-3 relevant LeetCode topics/problems or reputable knowledge-base articles for practice."
                )
                repo_ctx = unit["content"][:6000]
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Repo context (excerpt):\n\n{repo_ctx}\n\nQuestion: {q}"},
                ]
                with st.spinner("GPT‑5 思考中…"):
                    a = call_gpt_text(messages)
                st.session_state.chat.append({"q": q, "a": a, "unit": unit["title"]})

            # Displaying conversation
            for i, turn in enumerate(reversed(st.session_state.chat[-8:])):
                st.markdown(f"**You:** {turn['q']}")
                st.markdown(f"**GPT‑5:** {turn['a']}")
                st.markdown("---")

            # Completing study: Generate summary → Save to review module
            if st.button("✅ Complete Study Unit (Generate Summary and Save)", use_container_width=True):
                # Aggregate current unit-related conversations
                related = [t for t in st.session_state.chat if t["unit"] == unit["title"]]
                qa_text = "\n\n".join([f"Q: {t['q']}\nA: {t['a']}" for t in related])

                summary_prompt = (
                    "Summarize the key takeaways from this study unit. Use the repo context and EMPHASIZE topics covered in the Q&A. "
                    "Return JSON: {\n  \"unit\": string, \"summary\": string, \"keyPoints\": string[], \"followUps\": string[]\n}"
                )
                with st.spinner("Generating study summary…"):
                    j = call_gpt_json(
                        user_prompt=(
                            f"Repo unit title: {unit['title']}\n\nRepo context (excerpt):\n{unit['content'][:6000]}\n\nQ&A:\n{qa_text[:6000]}\n\n"
                            + summary_prompt
                        ),
                        system_prompt="You are a precise note-taker for spaced repetition.",
                    )
                item = {
                    "ts": dt.datetime.utcnow().isoformat() + "Z",
                    "repo": st.session_state.repo.get("name"),
                    "unit": j.get("unit", unit["title"]),
                    "summary": j.get("summary", ""),
                    "keyPoints": j.get("keyPoints", []),
                    "followUps": j.get("followUps", []),
                }
                save_review(item)
                st.success("Saved to review module ✅")

# ------------------------------
# Review Module
# ------------------------------
with tab2:
    st.subheader("🗂️ Study Summary Archive")
    reviews = load_reviews()
    if not reviews:
        st.info("No summaries available yet. Please save one after completing a study unit.")
    else:
        for r in reviews:
            with st.container(border=True):
                st.markdown(f"**Time**: {r['ts']}  |  **Repository**: {r.get('repo','-')}  |  **Unit**: {r.get('unit','-')}")
                if r.get("summary"):
                    st.markdown("**Summary**: " + r["summary"])
                if r.get("keyPoints"):
                    st.markdown("**Key Points**:")
                    st.write("\n".join([f"• {x}" for x in r["keyPoints"]]))
                if r.get("followUps"):
                    st.markdown("**Follow-up Suggestions**:")
                    st.write("\n".join([f"• {x}" for x in r["followUps"]]))
