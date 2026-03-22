现代 LangChain 架构中，凡是需要强制输出 JSON 的场景，都应该完全抛弃 `StrOutputParser` + 正则解析的做法。
* **修复方向**：
  必须引入 `pydantic`，并改用 `with_structured_output`。例如验证节点：
  ```python
  from pydantic import BaseModel, Field

  class VerifyResult(BaseModel):
      valid: bool = Field(description="验证是否通过")
      score: int = Field(description="分数")
      issues: List[dict] = Field(description="发现的问题列表")
      fix_needed: bool = Field(description="是否需要修复")
      summary: str = Field(description="总结")

  # 在 verify_quiz_node 中调用时：
  chain = prompt | chat_model.with_structured_output(VerifyResult)
  verify_result = await chain.ainvoke(...)
  ```

### 遗留缺陷 3：Prompt 依然缺少强制反偷懒指令

* **错误定位**：
  在 `generate_exam_node` 的 Prompt 中，虽然给出了格式模板，但并没有针对“省略号占位符”的强制约束词。
* **修复方案**：
  在 `<rules>` 标签内必须加上极端的系统级指令（System Prompt 级别的约束）：
  ```text
  - 绝对禁止使用“...”或“等”占位符敷衍！
  - 无论输出有多长，你必须逐字逐句完整生成要求的每一道题。如果要求 10 道题，哪怕耗尽 Token 也必须输出完整的 10 道题的题干和选项。
  - 任何省略行为将被判定为任务失败。
  ```

### 小 Bug 4：`call_llm` 的参数传递与模板不匹配

* **错误定位**：
  在 `verify_exam_node` 中，调用 `call_llm` 时传入了四个参数：
  ```python
  result = await call_llm(
      """... prompt_template ...""",
      topics=", ".join(state['topics']),
      quiz_types=", ".join(state['quiz_types']),
      contexts=contexts_combined,
      exam_paper=state['exam_paper']
  )