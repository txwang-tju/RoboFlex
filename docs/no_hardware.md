# 没有硬件，如何继续完成项目

当前仍可完成意图到任务的接口、连续执行对照、异常恢复和可复现实验。采用**真实传感素材＋仿真执行**逐层替换输入，让每条任务都有可追溯的状态与结果。

## 先记录参考闭环

```powershell
python -m pip install -e ".[sim]"
python -m roboflex simulate --scenario success --video --output runs/success-01
python -m roboflex simulate --scenario no-intent --output runs/no-intent-01
python -m roboflex simulate --scenario cancel --output runs/cancel-01
python -m roboflex simulate --scenario network-loss --output runs/network-01
python -m roboflex simulate --scenario target-shift --output runs/shift-01
python -m roboflex simulate --scenario blocked --output runs/blocked-01
```

每次选择新的空目录。`trace.jsonl`记录阶段、本体／物体状态和重新规划；`report.json`记录最终状态、事件与仿真假设；`simulation.mp4`来自同一条执行轨迹。失败记录与成功视频一并保留。

当前默认提供器使用固定语义结果，操作端手部与手套为合成特征，物体位置来自仿真真值。它验证Task Token、接近、抓取、携行、放置及停止链路，不用于宣称真人意图识别成功率。

MuJoCo场景是参考机构：底盘采用全向自由度近似麦轮运动，四轴机械臂使用几何IK，物体靠接近门限和附着约束携带。它没有真实麦轮滚子接触、全臂碰撞、人机接触保护或柔性机械爪力学，不能据此证明鸡蛋无损抓取。

## 替换真实操作端输入

无需小车即可用电脑摄像头或手机录制真人素材。让手和候选物体出现在同一画面，覆盖指向、接近、停顿、撤回、相邻目标与普通自然动作。按用户／会话隔离数据，标注目标ID、无意图、确认、取消和时间范围。

现有摄像头基线可实时观察手部与目标：

```powershell
python -m pip install -e ".[vision]"
python -m roboflex download-model
python -m roboflex camera --model models/hand_landmarker.task
```

默认仍为mock执行。将MediaPipe图像关键点交给 `hand_features()`，将独立手套样本交给 `glove_features()`，通过 `structured_hand_text()`保留来源。没有实体手套时，可以使用明确标记的传感回放检查融合接口，不能将其作为真实传感实测。

先评价目标选择、无意图误触发、相邻目标消歧、确认负担和撤回响应。只展示一次正确意图，不能替代这些对照。

## 连接真实VLM

在[云端说明](cloud_stack.md)中配置实际可用模型。把操作端关键帧、手部描述、机器人视角及当前场景清单发送给服务；模型只选择已有实体与意图状态。固定回放与真实VLM分开统计，保存模型ID、提示版本、输入证据、耗时、费用和原始语义输出。

参考仿真支持 `--provider live`。此时发送的是仿真场景图像及合成操作端特征，可以检查模型服务和任务协议；若要评价真人意图，仍需接入上一步采集的真实操作端素材。

模型响应慢时重新获得当前场景和用户确认，不能用过期图像驱动新动作。远程语义服务失联时，由本地执行器根据租约与状态停止，不能等待云端下发停止指令。

## 接入合作研究执行框架

本项目依托已开展的连续移动操作合作研究。开源参考基线提供可运行接口与对照环境；研究平台承载完整可靠性感知规划和相位控制，并非将合作研究重新描述为尚未开展。

固定同一意图输入、物体、场景与完成条件，将任务映射到活动TaskSet、机器人模型、地图与位姿，再接收带时刻、坐标和相位的底盘／关节参考。检查任务版本、有效期、运动学和关节限制后进入控制。

建议比较逐段停稳基线、当前参考执行、合作研究完整方案，以及去掉观测窗口／补偿余量／相位反馈后的配置。记录完整任务成功、总时长、等待、重新规划、人工介入和物品失控。参考FSM不自动等同于论文的相位加权优化控制。

适配VLA时单独记录模型权重、动作定义、维度、单位、参考坐标与控制频率，先在相同任务下比较其相对几何策略的增益。高层VLM生成文字步骤不表示已经接入VLA。

## 视频生成工具的用途

AI视频可用于家庭产品愿景、工业设计或交互概念，标注“概念演示”。项目技术验证将三个来源分开：

| 来源 | 可以说明 |
|---|---|
| 既有机器人实拍 | 历史平台的实际运动与抓取基础 |
| 从日志对应状态渲染的MuJoCo视频 | 该仿真假设下的接口、轨迹和闭环结果 |
| AI生成／动画概念片 | 未来场景和产品交互设想 |

仿真验收先看日志与物体结果，再看视频；概念片不计入技术成功率。
