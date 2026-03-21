#!/usr/bin/env python3
"""
MCP Server for 期末复习助手
提供联网搜索、知识库查询等功能
"""

import asyncio
import json
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 搜索工具
async def search_web(query: str) -> str:
    """联网搜索"""
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        return result[:2000] if len(result) > 2000 else result
    except Exception as e:
        return f"搜索失败: {str(e)}"

# 搜索试卷格式
async def search_exam_format(course: str) -> str:
    """搜索试卷格式"""
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        result = search.run(f"{course} 期末考试试卷 格式 题型 结构")
        return result[:1500] if len(result) > 1500 else result
    except Exception as e:
        return f"搜索失败: {str(e)}"

# 创建 MCP Server
app = Server("deep-revision-mcp")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用工具"""
    return [
        Tool(
            name="web_search",
            description="联网搜索信息。当需要验证答案、查找默认试卷风格、查询考试规范时使用。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_exam_format",
            description="搜索大学期末考试试卷格式范例。当用户没有上传样卷时，可以搜索该学科的典型试卷格式。",
            inputSchema={
                "type": "object",
                "properties": {
                    "course": {"type": "string", "description": "课程名称"}
                },
                "required": ["course"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """调用工具"""
    if name == "web_search":
        query = arguments.get("query", "")
        result = await search_web(query)
        return [TextContent(type="text", text=result)]

    elif name == "search_exam_format":
        course = arguments.get("course", "")
        result = await search_exam_format(course)
        return [TextContent(type="text", text=result)]

    else:
        return [TextContent(type="text", text=f"未知工具: {name}")]

async def main():
    """启动 MCP Server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
