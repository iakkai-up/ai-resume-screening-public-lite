import json


AI_DISCLAIMER = "本系统仅用于HR辅助筛选，AI分析结果不作为自动录用或自动淘汰依据，最终决策必须由HR人工复核。"


def safe_parse_json(text: str) -> tuple[dict | None, str | None]:
    """
    尽量从大模型输出中解析 JSON。

    有些模型会把 JSON 包在 ```json 代码块里，或在前后加解释文字。
    这里做最小容错：先清理代码块，再截取第一个 { 到最后一个 }。
    """
    if not text or not text.strip():
        return None, "模型返回为空。"

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"无法解析为JSON：{exc}；原始输出：{text[:500]}"

    if not isinstance(data, dict):
        return None, "JSON顶层结构不是对象。"

    return data, None


def get_recommendation(score: int) -> str:
    """根据分数返回辅助推荐等级。"""
    if score >= 85:
        return "强烈建议面试"
    if score >= 70:
        return "建议面试"
    if score >= 55:
        return "备选"
    if score >= 40:
        return "不优先"
    return "暂不推荐"


def normalize_result(
    file_name: str,
    resume_data: dict,
    match_data: dict,
    raw_ai_output: str = "",
) -> dict:
    """把模型输出整理成页面和CSV统一使用的字段。"""
    score = _to_score(match_data.get("score", 0))
    recommendation = str(match_data.get("recommendation") or get_recommendation(score))

    return {
        "文件名": file_name,
        "姓名": _as_text(resume_data.get("name", "未知")),
        "学校": _as_text(resume_data.get("school", "")),
        "专业": _as_text(resume_data.get("major", "")),
        "学历": _as_text(resume_data.get("degree", "")),
        "匹配度": score,
        "推荐等级": recommendation,
        "命中硬性条件": _as_text(match_data.get("matched_must_have", "")),
        "缺失硬性条件": _as_text(match_data.get("missing_must_have", "")),
        "证据摘要": _as_text(match_data.get("evidence", "")),
        "匹配理由": _as_text(match_data.get("match_reasons", "")),
        "风险点": _as_text(match_data.get("risk_points", "")),
        "面试追问": _as_text(match_data.get("interview_questions", "")),
        "原始AI输出": raw_ai_output,
    }


def make_error_result(
    file_name: str,
    error_message: str,
    resume_data: dict | None = None,
    raw_ai_output: str = "",
) -> dict:
    """当某份简历处理失败时，也返回一行可展示结果，避免页面空白。"""
    resume_data = resume_data or {}
    return {
        "文件名": file_name,
        "姓名": _as_text(resume_data.get("name", "未知")),
        "学校": _as_text(resume_data.get("school", "")),
        "专业": _as_text(resume_data.get("major", "")),
        "学历": _as_text(resume_data.get("degree", "")),
        "匹配度": 0,
        "推荐等级": "需人工复核",
        "命中硬性条件": "",
        "缺失硬性条件": "",
        "证据摘要": "",
        "匹配理由": "该候选人的自动分析未完成，请HR人工查看简历。",
        "风险点": error_message,
        "面试追问": "请围绕岗位硬性条件、核心技能和项目经历进行人工追问。",
        "原始AI输出": raw_ai_output,
    }


def get_mock_results() -> list[dict]:
    """稳定的现场演示数据；不调用API，也不依赖上传文件。"""
    return [
        {
            "文件名": "mock_resume_01.pdf",
            "姓名": "张明",
            "学校": "上海交通大学",
            "专业": "计算机科学与技术",
            "学历": "本科",
            "匹配度": 92,
            "推荐等级": "强烈建议面试",
            "命中硬性条件": "Python基础；数据整理；Prompt调试；产品文档",
            "缺失硬性条件": "无明显缺失",
            "证据摘要": "简历包含数据分析 Demo、简历信息抽取实验和 HR 系统需求梳理经历。",
            "匹配理由": "Python、数据分析和机器学习项目经历与岗位核心要求高度匹配，有完整项目交付经验。",
            "风险点": "企业级系统经验仍需面试确认。",
            "面试追问": "请介绍一个你独立完成的数据分析项目；如何处理脏数据和模型效果不稳定的问题？",
            "原始AI输出": "",
        },
        {
            "文件名": "mock_resume_02.docx",
            "姓名": "李佳",
            "学校": "浙江大学",
            "专业": "软件工程",
            "学历": "硕士",
            "匹配度": 84,
            "推荐等级": "建议面试",
            "命中硬性条件": "Python；接口设计；文档编写；模型输出评估",
            "缺失硬性条件": "HR业务场景经验较少",
            "证据摘要": "简历包含招聘流程自动化工具、LLM 输出稳定性评估和后端接口经验。",
            "匹配理由": "后端开发、接口设计和项目协作经验较强，具备较好的工程实现能力。",
            "风险点": "简历中AI相关实践较少，需要确认学习迁移能力。",
            "面试追问": "你如何设计一个简历解析服务？遇到接口失败时如何保证用户体验？",
            "原始AI输出": "",
        },
        {
            "文件名": "mock_resume_03.txt",
            "姓名": "王珂",
            "学校": "南京大学",
            "专业": "信息管理与信息系统",
            "学历": "本科",
            "匹配度": 73,
            "推荐等级": "建议面试",
            "命中硬性条件": "HR系统实习；数据整理；沟通表达",
            "缺失硬性条件": "Prompt调试和工程实现证据偏少",
            "证据摘要": "简历体现 HR 系统实习和数据整理经验，但技术项目深度不足。",
            "匹配理由": "有HR系统实习和数据整理经验，理解业务场景，基础技能符合岗位多数要求。",
            "风险点": "技术深度和复杂项目经验偏弱。",
            "面试追问": "你如何判断一个AI筛选结果是否可信？你会如何向HR解释模型输出？",
            "原始AI输出": "",
        },
        {
            "文件名": "mock_resume_04.pdf",
            "姓名": "陈诺",
            "学校": "华南理工大学",
            "专业": "数据科学与大数据技术",
            "学历": "本科",
            "匹配度": 66,
            "推荐等级": "备选",
            "命中硬性条件": "数据处理；可视化基础",
            "缺失硬性条件": "真实业务项目；Prompt调试；HR场景经验",
            "证据摘要": "简历以课程项目为主，与岗位部分能力相关，但缺少业务落地证据。",
            "匹配理由": "具备数据处理和可视化基础，部分课程项目与岗位相关。",
            "风险点": "缺少真实业务项目和团队协作证据。",
            "面试追问": "请说明你做过的数据可视化项目，以及如何验证数据结论是否可靠。",
            "原始AI输出": "",
        },
        {
            "文件名": "mock_resume_05.docx",
            "姓名": "赵然",
            "学校": "普通本科院校",
            "专业": "工商管理",
            "学历": "本科",
            "匹配度": 48,
            "推荐等级": "不优先",
            "命中硬性条件": "HR实习；招聘流程理解",
            "缺失硬性条件": "Python；数据处理；Prompt调试；系统Demo",
            "证据摘要": "简历体现 HR 实习经历，但缺少岗位要求中的技术能力和 Demo 交付证据。",
            "匹配理由": "有HR实习经历，了解招聘流程，但与岗位要求中的技术能力匹配不足。",
            "风险点": "Python、数据处理和系统实现经验缺失。",
            "面试追问": "你是否有使用自动化工具提升招聘效率的经历？目前掌握哪些数据处理工具？",
            "原始AI输出": "",
        },
    ]


def _to_score(value) -> int:
    """把模型返回的分数转成 0-100 的整数。"""
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        score = 0
    return max(0, min(100, score))


def _as_text(value) -> str:
    """把列表、字典等结果转成适合页面展示的文字。"""
    if value is None:
        return ""
    if isinstance(value, list):
        return "；".join(_as_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)
