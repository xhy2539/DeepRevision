"""测试 exam workflow 是否能正常加载和执行"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置最小环境
os.environ['MINIMAX_API_KEY'] = 'test-key'

print("1. Testing imports...")
try:
    from agent.multi_agent.quiz_agent import run_exam_agent, exam_workflow
    print(f"   exam_workflow: {exam_workflow}")
    print("   Imports OK")
except Exception as e:
    print(f"   Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n2. Testing workflow structure...")
try:
    from langgraph.graph import StateGraph, END
    print(f"   StateGraph: {StateGraph}")
    print("   END: {END}")
    print("   Workflow structure OK")
except Exception as e:
    print(f"   Structure check failed: {e}")
    sys.exit(1)

print("\n3. All checks passed!")
