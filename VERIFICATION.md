# 验证记录

## 2026-10-20：云端意图与移动操作参考闭环

Windows / Python 3.12.14；MuJoCo 3.16.0、NumPy 2.5.4、Pillow 12.3.0、imageio-ffmpeg 0.6.0。本次未连接真实机器人、未调用付费模型服务、未部署腾讯云资源。

- 完整回归100项：96项通过，4项旧视觉可选测试因未装视觉依赖跳过。新云端与仿真测试均实际运行。
- 本地HTTP鉴权、任务生成与新鲜确认、推理中取消、缓存上限、场景时效与版本拒绝、麦轮映射、A*和规划器契约通过。
- 本地可编辑安装成功，CLI帮助与原有命令保留。新增GitHub Actions配置，远程运行结果以仓库Actions为准。
- `success`：释放后物体稳定在托盘内，COMPLETE，参考仿真时间11.24秒。
- `target-shift`：目标平移0.10米后重规划一次，COMPLETE，11.64秒。
- `cancel`：携物阶段取消，CANCELLED，夹持约束保持。
- `network-loss`：停止刷新心跳后本地租约过期，CANCELLED，夹持约束保持。
- `no-intent`：IDLE，未启动取物；`blocked`：真实规划检查拒绝受阻路线，FAULT。
- 成功情景视频从实际仿真状态渲染，与报告及逐周期日志一同保存在 `media/simulation-reference.mp4` 和 `examples/reference_run/`。

上述时间是仿真时间，不是实机、网络或模型响应性能。合成操作端与回放VLM、真值空间定位、全向底盘和抓取附着近似均在输出中标注。参考A*＋IK＋阶段执行不替代合作研究的完整优化器及相位加权控制器；不验证柔性夹持力学和真实意图准确率。

## 历史验证

2026-10-19提交副本复核：Python 3.12最小环境运行42项测试，38项通过、4项视觉可选测试因未安装视觉依赖跳过；其余控制、几何和TCP测试通过。离线HTML含8类控制器回放，页面播放与情景切换另作检查。以下保留既有完整依赖环境的历史验证记录，不混作本次新增实机实验。

2026-10-12，Windows / Python 3.12；仅仿真与生成图像测试，没有操作摄像头或真实机械臂。

- 自动测试：42 项全部通过，无跳过。详细结果见 `verification-tests.txt`。
- 实际加载 Google Hand Landmarker 模型并完成生成图像推理；ArUco 目标 ID 检测通过。
- 相机位姿：已知投影的合成标记重建、非法变换和分辨率不匹配测试通过。
- 本地 TCP 完整取放闭环已运行至 COMPLETE；取消抢占、看门狗、错误 ACK 与拒绝停止测试通过。
- Python 可编辑安装、标记图生成、Windows 启动脚本、JSONL 回放与离线统计已运行。
- 实机的舵机标定、通信固件、夹持效果和家庭使用性能尚待硬件联调。

| 合成情景 | 预期及实际最终状态 |
|---|---|
| success | COMPLETE |
| ambiguous | OBSERVING |
| no-intent | OBSERVING |
| cancel | CANCELLED |
| withdraw | CANCELLED |
| stale | CANCELLED |
| target-shift | CANCELLED |
| stale-glove | CONFIRMING |

合成情景只验证程序行为，不代表真实用户的意图识别准确率或物理抓取成功率。
