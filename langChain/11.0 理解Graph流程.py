from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# ==========================================
# 1. State (状态)：图里流转的“公交车”
# ==========================================
# 我们定义这辆车上只载着一个叫 count 的数字
class CounterState(TypedDict):
    count: int

# ==========================================
# 2. Node (节点)：干活的工人
# ==========================================
# 节点就是一个普通函数。接收公交车(state)，把里面的 count +1，然后放回车上
def add_one_node(state: CounterState):
    current_count = state["count"]
    new_count = current_count + 1
    print(f"👉 [节点执行] 收到数字 {current_count}，执行 +1，变成 {new_count}")
    return {"count": new_count}

# ==========================================
# 3. Conditional Edge (条件边)：智能红绿灯
# ==========================================
# 它负责检查公交车上的状态，决定下一站去哪
def should_continue(state: CounterState):
    if state["count"] >= 3:
        print("   🚦 [红绿灯] 已经数到 3 了，走向 END，下班！\n")
        return END
    else:
        print("   🚦 [红绿灯] 还没到 3，打回原点继续算！\n")
        return "node_a"

# ==========================================
# 4. 画图连线 (把上面的零件拼起来)
# ==========================================
workflow = StateGraph(CounterState)

# 登记工人
workflow.add_node("node_a", add_one_node)

# 规定起点
workflow.add_edge(START, "node_a")

# 规定红绿灯路口：node_a 干完活后，必须经过 should_continue 的检查
workflow.add_conditional_edges("node_a", should_continue)

# 编译成最终程序
app = workflow.compile()

# ==========================================
# 5. 运行！
# ==========================================
print("🚀 启动 LangGraph 状态机...\n")
final_state = app.invoke({"count": 0})  # 给定初始状态为 0
print(f"🎉 任务结束！最终状态: {final_state}")