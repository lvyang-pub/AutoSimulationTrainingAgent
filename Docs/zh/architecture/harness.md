# Harness 隔离机制

## 为什么需要隔离

> 🚧 待填充：Agent 不直接 import Env 代码，改为通过 Harness 调用 →
> 换 Env 不需要改 Agent，换 Agent 不需要改 Env；边界清晰、可测试、可扩展。

## exec.py 如何工作

> 🚧 待填充：Harness/exec.py 位于项目顶层，供所有 Agent 共用；
> 接收「模块路径 + 函数名 + 参数」，动态 import 并调用，返回结构化结果。

## 调用契约

> 🚧 待填充：Env 通过 `observation/` 和 `action/` 声明能提供什么、接受什么操作；
> 函数以「模块路径 + 函数名」被调用，示例：

```python
harness.call("action/configure.py", "set_params",
             params_dict={"env": {"kp": 80.0}})
```
