# 验证记录

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
