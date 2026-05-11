import pytest

from agent.tools.agent_tools import (
    _filter_exam_format_docs,
    _filter_exam_paper_docs,
    _format_exam_paper_reference,
    _format_search_results,
)


def test_format_search_results_exposes_url_domain_and_snippet():
    docs = [
        {
            "title": "CLOCK页面置换算法详解",
            "link": "https://example.edu.cn/os/clock",
            "snippet": "CLOCK算法使用访问位近似LRU，降低精确LRU的维护开销。",
        }
    ]

    text = _format_search_results("CLOCK 页面置换算法", docs)

    assert "[CLOCK页面置换算法详解](https://example.edu.cn/os/clock)" in text
    assert "来源：example.edu.cn" in text
    assert "摘要：CLOCK算法使用访问位近似LRU" in text


def test_filter_exam_format_docs_removes_unrelated_course_noise():
    docs = [
        {
            "title": "数据库系统概论期末试卷",
            "link": "https://noise.example/db",
            "snippet": "包含数据库三级模式、关系代数和 SQL 查询。",
        },
        {
            "title": "操作系统期末考试题型结构",
            "link": "https://example.edu.cn/os/final",
            "snippet": "操作系统考试常见题型包括选择题、判断题、填空题和简答题。",
        },
    ]

    filtered = _filter_exam_format_docs("操作系统", docs, limit=5)

    assert [doc["title"] for doc in filtered] == ["操作系统期末考试题型结构"]


def test_filter_exam_paper_docs_prioritizes_real_exam_papers():
    docs = [
        {
            "title": "操作系统知识点速记-CSDN博客",
            "link": "https://blog.csdn.net/os-summary",
            "snippet": "整理进程、线程、内存管理等知识点，没有试卷文件。",
        },
        {
            "title": "某大学操作系统期末考试试卷A卷及答案.pdf",
            "link": "https://cs.example.edu.cn/os/final-a.pdf",
            "snippet": "包含选择题、填空题、判断题、简答题和参考答案。",
        },
        {
            "title": "数据库期末考试真题",
            "link": "https://cs.example.edu.cn/db/final.pdf",
            "snippet": "数据库系统概论试卷。",
        },
    ]

    filtered = _filter_exam_paper_docs("操作系统", docs, limit=3)

    assert [doc["title"] for doc in filtered] == ["某大学操作系统期末考试试卷A卷及答案.pdf"]


def test_format_exam_paper_reference_warns_against_copying_real_questions():
    docs = [
        {
            "title": "操作系统期末考试试卷A卷及答案.pdf",
            "link": "https://cs.example.edu.cn/os/final-a.pdf",
            "snippet": "包含选择题、填空题、判断题、简答题和参考答案。",
        }
    ]

    text = _format_exam_paper_reference("操作系统", "操作系统 期末考试 试卷 filetype:pdf", docs)

    assert "【联网真实试卷参考】" in text
    assert "禁止照抄或改写真实试卷原题" in text
    assert "操作系统期末考试试卷A卷及答案.pdf" in text


@pytest.mark.asyncio
async def test_fetch_exam_paper_reference_uses_real_paper_results(monkeypatch):
    from agent.tools import agent_tools

    def fake_duckduckgo(query, num_results=6):
        return [
            {
                "title": "操作系统知识点总结",
                "link": "https://blog.csdn.net/os-summary",
                "snippet": "整理操作系统复习知识点。",
            },
            {
                "title": "操作系统期末考试试卷A卷及答案.pdf",
                "link": "https://cs.example.edu.cn/os/final-a.pdf",
                "snippet": "包含选择题、填空题、判断题、简答题和参考答案。",
            },
        ]

    monkeypatch.setattr(agent_tools, "_run_duckduckgo_results", fake_duckduckgo)

    text = await agent_tools.fetch_exam_paper_reference("操作系统")

    assert "【联网真实试卷参考】" in text
    assert "操作系统期末考试试卷A卷及答案.pdf" in text
    assert "操作系统知识点总结" not in text


@pytest.mark.asyncio
async def test_web_search_falls_back_when_duckduckgo_results_are_filtered(monkeypatch):
    from agent.tools import agent_tools

    def fake_duckduckgo(query, num_results=6):
        return [
            {
                "title": "短视频娱乐内容",
                "link": "https://www.douyin.com/video/1",
                "snippet": "与学习无关的低质结果。",
            }
        ]

    class FakeTavilySearchResults:
        def __init__(self, max_results=5):
            self.max_results = max_results

        def run(self, query):
            return [
                {
                    "title": "CLOCK 页面置换算法课程资料",
                    "url": "https://example.edu/os/clock",
                    "content": "CLOCK 算法通过访问位近似 LRU，降低替换开销。",
                }
            ]

    monkeypatch.setattr(agent_tools, "_run_duckduckgo_results", fake_duckduckgo)
    monkeypatch.setattr("langchain_community.tools.TavilySearchResults", FakeTavilySearchResults)

    text = await agent_tools.web_search.ainvoke({"query": "CLOCK 页面置换算法"})

    assert "example.edu" in text
    assert "CLOCK 页面置换算法课程资料" in text
