import json
import os
from pydantic import BaseModel, Field
from typing import List, Dict

from model.factory import chat_model
from utils.logger_handler import logger
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class KnowledgeGraphNode(BaseModel):
    subject: str = Field(description="核心知识点或概念名称")
    relation: str = Field(description="关系或意图，如'定义为'、'需要复习'、'理解错误于'")
    object: str = Field(description="具体的详解或结论")

class SessionMemoryManager:
    def __init__(self, max_recent_turns: int = 3, trash_threshold: int = 4):
        self.max_recent_turns = max_recent_turns
        self.trash_threshold = trash_threshold
        self.persistence_path = os.path.join(os.getcwd(), "data", "sessions.json")
        # {session_id: {"name": "...", "recent": [], "trash": [], "graph": []}}
        self.store: Dict[str, Dict] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        """从磁盘加载会话元数据"""
        if os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, 'r', encoding='utf-8') as f:
                    self.store = json.load(f)
                    logger.info(f"已从 {self.persistence_path} 加载 {len(self.store)} 个会话。")
            except Exception as e:
                logger.error(f"加载会话数据失败: {e}")

    def save_to_disk(self):
        """将当前内存状态落盘"""
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        try:
            with open(self.persistence_path, 'w', encoding='utf-8') as f:
                json.dump(self.store, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存会话数据失败: {e}")

    def _init_session(self, session_id: str, name: str = None):
        if session_id not in self.store:
            self.store[session_id] = {
                "name": name or session_id,
                "recent": [],
                "trash": [],
                "graph": []
            }
            self.save_to_disk()

    def register_session(self, session_id: str, name: str):
        """显式注册/创建新会话"""
        self._init_session(session_id, name)
        # 如果已存在，则更新名称
        self.store[session_id]["name"] = name
        self.save_to_disk()

    def add_message(self, session_id: str, role: str, content: str):
        self._init_session(session_id)
        session = self.store[session_id]
        
        session["recent"].append({"role": role, "content": content})
        
        # 1. 触发滑动窗口挤出机制：当近期消息对（一来一回算1轮，即2条）超过限制
        max_messages = self.max_recent_turns * 2
        if len(session["recent"]) > max_messages:
            # 把最老的两条消息踢入垃圾桶回收站
            kicked_out = session["recent"][:2]
            session["recent"] = session["recent"][2:]
            session["trash"].extend(kicked_out)
            
            # 2. 检查蓄水池是否溢出，触发总结清洗
            if len(session["trash"]) >= self.trash_threshold * 2:
                self._trigger_background_distillation(session_id)
        
        self.save_to_disk()

    def _trigger_background_distillation(self, session_id: str):
        """
        触发提纯，注意在FastAPI中，这部分实际上是在BackgroundTasks里调用的
        为了低耦合，暴露方法供外部异步调用。
        """
        # 将这批垃圾数据深拷贝并清空原始池
        trash_data = list(self.store[session_id]["trash"])
        self.store[session_id]["trash"].clear()
        
        try:
            self._distill_to_graph(session_id, trash_data)
        except Exception as e:
            logger.error(f"[记忆提纯失败] {e}")

    def _distill_to_graph(self, session_id: str, trash_data: List[Dict]):
        parser = JsonOutputParser(pydantic_object=KnowledgeGraphNode)
        prompt = PromptTemplate(
            template="你是知识提炼师。请从以下零散的对话历史中，提炼出学生掌握不牢固的知识点或是复习焦点。如果没有明确有效的知识点，则返回空列表。\n"
                     "对话记录：{history}\n"
                     "请使用如下JSON格式输出，不要包含Markdown标记：\n{format_instructions}\n"
                     "返回类型必须是一个 JSON Array 数组。",
            input_variables=["history"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        
        history_text = "\n".join([f"{item['role']}: {item['content']}" for item in trash_data])
        chain = prompt | chat_model | parser
        
        logger.info(f"正在压缩提纯 {session_id} 的 {len(trash_data)} 条遗忘记录...")
        result = chain.invoke({"history": history_text})
        
        # 将提炼出的纯净晶体写入长线图谱区
        if result and isinstance(result, list):
            self.store[session_id]["graph"].extend(result)
            self.save_to_disk()
            logger.info(f"提纯完成，已为 {session_id} 抽取 {len(result)} 条结构化节点。")

    def get_memory_context(self, session_id: str) -> str:
        """
        组装提示词上下文
        """
        self._init_session(session_id)
        session = self.store[session_id]
        
        context = ""
        # 长时记忆拼装
        if session["graph"]:
            context += "【以往复习知识点图谱备忘】\n"
            for node in session["graph"]:
                context += f"- 概念[{node.get('subject', '')}] : {node.get('relation', '')} -> {node.get('object', '')}\n"
            
        # 短时近期记忆拼装 (这里会被Langchain自身的scratchpad覆盖，也可手动拼装)
        return context

    def get_all_sessions(self) -> List[Dict[str, str]]:
        """返回当前所有活跃的会话列表，包含 id 和展示名称"""
        return [{"id": k, "name": v["name"]} for k, v in self.store.items()]

    def rename_session(self, session_id: str, new_name: str) -> bool:
        if session_id in self.store:
            self.store[session_id]["name"] = new_name
            self.save_to_disk()
            return True
        return False

    def clear_session(self, session_id: str):
        if session_id in self.store:
            del self.store[session_id]
            self.save_to_disk()
            return True
        return False

# 全局单例
memory_manager = SessionMemoryManager()
