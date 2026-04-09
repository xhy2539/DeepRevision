from api.message_protocol import SCHEMA_VERSION, _parse_quiz_content, build_assistant_message


def test_quiz_choice_regression():
    content = """1.（选择题，分值：5分，难度：中等）
下列哪项属于关系型数据库的事务特性？
A. 原子性
B. 异步性
C. 松耦合
D. 幂等性
答案：A
解析：事务满足 ACID 特性。"""
    questions = _parse_quiz_content(content)
    assert len(questions) == 1
    assert questions[0]["stem"].startswith("下列哪项")
    assert questions[0]["analysis"] == questions[0]["explanation"]


def test_quiz_fill_regression():
    content = """1.（填空题，分值：5分，难度：简单）
TCP 三次握手依次是（ ）、（ ）、（ ）。
答案：SYN，SYN-ACK，ACK
解析：三次握手确保双方收发能力正常。"""
    questions = _parse_quiz_content(content)
    assert len(questions) == 1
    assert questions[0]["type"] == "填空题"


def test_quiz_judge_regression():
    content = """1.（判断题，分值：2分，难度：简单）
HTTPS 默认端口是 443。
答案：正确
解析：HTTPS 基于 TLS，默认监听 443。"""
    questions = _parse_quiz_content(content)
    assert len(questions) == 1
    assert questions[0]["answer"] == "正确"


def test_quiz_essay_regression():
    content = """1.（简答题，分值：10分，难度：较难）
简述 CAP 定理并说明其工程权衡。
答案：一致性、可用性、分区容忍性三者不可同时满足。
解析：分布式系统需要按业务做权衡。"""
    questions = _parse_quiz_content(content)
    assert len(questions) == 1
    assert questions[0]["type"] == "简答题"


def test_quiz_mixed_regression():
    content = """1.（选择题，分值：2分，难度：简单）
OSI 共有几层？
A. 5
B. 6
C. 7
D. 8
答案：C
解析：OSI 参考模型共 7 层。

2.（判断题，分值：2分，难度：简单）
IP 协议是面向连接的。
答案：错误
解析：IP 是无连接协议。"""
    questions = _parse_quiz_content(content)
    assert len(questions) == 2


def test_structured_payload_passthrough_and_schema_version():
    structured_result = {
        "kind": "quiz_set",
        "render_mode": "interactive_cards",
        "schema_version": SCHEMA_VERSION,
        "text": "text",
        "payload": {
            "schema_version": SCHEMA_VERSION,
            "questions": [],
            "kind": "quiz_set",
            "render_mode": "interactive_cards",
        },
    }
    message = build_assistant_message("quiz", "fallback", "sid", structured_result=structured_result)
    assert message["schema_version"] == SCHEMA_VERSION
    assert message["payload"]["schema_version"] == SCHEMA_VERSION
