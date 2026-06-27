import pandas as pd


EXPORT_COLUMNS = [
    "文件名",
    "姓名",
    "学校",
    "专业",
    "学历",
    "匹配度",
    "推荐等级",
    "命中硬性条件",
    "缺失硬性条件",
    "证据摘要",
    "匹配理由",
    "风险点",
    "面试追问",
]


def results_to_csv(results: list[dict]) -> bytes:
    """把筛选结果导出为 CSV；加 UTF-8 BOM，方便 Excel 正确显示中文。"""
    rows = []
    for item in results:
        row = {column: item.get(column, "") for column in EXPORT_COLUMNS}
        rows.append(row)

    dataframe = pd.DataFrame(rows, columns=EXPORT_COLUMNS)
    csv_text = dataframe.to_csv(index=False)
    return ("\ufeff" + csv_text).encode("utf-8")
