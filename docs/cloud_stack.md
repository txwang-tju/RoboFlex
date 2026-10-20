# 云端意图与移动操作工程

本页对应 `roboflex/stack/`，解释手套、腾讯云与小车之间的数据契约和部署方法。完整项目及合作研究机制见[系统方案](system_architecture.md)；无硬件实验见[开发路线](no_hardware.md)。

## 数据链路与职责

1. 操作端上传带采集时刻、设备会话和序号的视频／手套观测。MediaPipe只从图像提取手部关键点；柔性弯曲与IMU数据独立编码，不能作为MediaPipe图像输入。
2. 云端保留最新观测窗口，结合动作阶段变化触发VLM。VLM接收关键帧、结构化手部文本和已有场景实体，输出语义目标建议。
3. TaskCompiler检查明确确认、实体身份、空间绑定、版本和时效，形成一次 `PICK_PLACE` 的Task Token。模型不产生舵机角、夹持力或凭空构造三维坐标。
4. 本地执行接收任务并重新核验，以参考基线或合作研究规划器生成运动。设备／物体反馈形成完成证据，通信ACK本身不代表取物成功。

VLM输出目标，VLA动作候选接口负责机器人动作语义，两者不能互换。当前开源参考执行以几何规划完成取放；合作研究的连续全身优化器通过适配接口衔接，不在这里伪造相同实现。

## 观测窗口

`LatestStreams`为 `operator`、`robot`、`glove` 各保留一个最新包。设备会话变化需明确重新连接；同一会话的序号和采集时刻必须递增。默认参考门限为观测年龄0.75秒、流间偏差0.15秒，实际部署需依据测量调整。

```json
{
  "stream": "glove",
  "session": "glove-boot-01",
  "seq": 101,
  "captured_at_s": 1792470000.0,
  "payload": {
    "bends": [0.1, 0.2, 0.8, 0.8, 0.7],
    "gyro_dps": [1.2, -0.7, 3.1]
  }
}
```

示例时间仅说明格式，不能直接作为当前请求。弯曲值先按用户和传感器标定归一化到0–1，陀螺仪使用度／秒。陀螺仪不直接提供稳定绝对朝向，更不能直接将操作端手朝向变成远端物体位置。

跨主机采用同步UTC秒并记录时钟不确定性；单机控制租约采用本地时基。不能比较两台设备各自的单调时钟。Wi-Fi和4G的到达时刻不等于传感采集时刻，必须同时记录并检查排队与延迟。

## camera-stream接入

依据 [camera-stream 0.3.8](https://pypi.org/project/camera-stream/0.3.8/) 的实际接口，先订阅主题，再读取最新帧。下面代码运行在能访问推流端的接收主机；相机服务依照该包的Linux设备配置启动。

```python
from camera_stream import StreamClient

with StreamClient("tcp://robot-private-ip:5555") as client:
    camera = client.subscribe("base_camera/color", warm_up_timeout=5)
    camera.wait_for_state("ONLINE", timeout=5)
    frame = camera.read(timeout=1)
    if frame is not None:
        bgr_image = frame.image
        sequence = frame.sequence
        captured_utc_ns = frame.header["captured_utc_ns"]
```

本仓库 `CameraStreamSource` 将上述帧转换为统一观测包。这里不使用不存在的 `get_latest_frame()` 接口。相机主题由实际配置确定；BGR图像编码成发给模型的JPEG前需正确转换颜色。

camera-stream以可信网络中的ZeroMQ推流为基础。4G回传需私有网络／受认证隧道或专用网关，不能将原始PUB端口直接当成公网安全视频服务。`captured_monotonic_ns`仅属于采集主机的时基；跨设备使用同步后的UTC和不确定性记录。

## 模型与HTTP接口

```powershell
python -m roboflex cloud --provider replay --host 127.0.0.1 --port 8766
```

默认回放提供器返回固定契约样例，标记为 `fixture_replay_NOT_live_VLM`。真实模型通过环境变量设置：

```powershell
$env:ROBOFLEX_VLM_BASE_URL = "https://your-model-endpoint.example/v1"
$env:ROBOFLEX_VLM_MODEL = "your-available-vision-model-id"
$env:ROBOFLEX_VLM_API_KEY = "your-api-key"
python -m roboflex cloud --provider live
```

`CompatibleVLM`当前使用支持图像的Chat Completions兼容消息格式；其他服务需对应的提供器适配。项目名称不决定API模型ID，实际部署记录型号、接口、提示版本和费用。密钥仅保存在运行环境。

| 路径 | 内容 |
|---|---|
| `GET /health` | 健康状态与当前提供器标签 |
| `POST /v1/intent` | 接收 `window` 和 `scene`，产生建议并尝试任务编译 |
| `POST /v1/tasks` | 用 `proposal_id`、新鲜 `window` 和 `scene`确认已有建议 |
| `POST /v1/cancel` | 使待处理与进行中的语义推理代次失效 |

`window`包含 `captured_at_s`、`hand_text`、明确布尔值 `confirmed`、`evidence_ids`以及最多8张内嵌JPEG／PNG关键帧。`scene`包含采集时刻、物体与交接点清单、版本和时钟不确定性。实体带ID、标签、世界坐标、位置来源及版本。完整字段以 `contracts.py` 和 `validate_window()` 为准。

模型建议只允许 `intent`、`object_id`、`destination_id`、`confidence`、`reason`。意图为 `NO_INTENT`、`CLARIFY`、`CANCEL`或 `PICK_PLACE`；未知字段、编造的实体和过期绑定不能进入执行。

### 慢推理后的新鲜确认

VLM耗时可能超过观测有效期。服务保留有限时长的语义建议并返回 `AWAITING_FRESH_CONFIRMATION`，而非将旧图像重新盖上当前时间戳。

正确流程是：接收 `proposal_id` → 重新采集画面与空间状态 → 检查用户仍选择同一目标并获得当前确认 → 发送 `/v1/tasks` → 小车接收后再次检查任务与目标版本。无有效确认时保持等待。取消会使旧 `proposal_id`失效。

仿真客户端在此流程中使用新的合成确认窗口，以检查协议；这不代表已经采集真人确认。

## 腾讯云部署布局

可将模型适配与任务网关部署在腾讯云CVM／容器上，将视频接收服务放在同一私网。推荐连接关系：

```text
操作端／小车 ──认证隧道或TLS网关── 云端视频与传感入口
                                    ↓
                              意图服务／任务管理
                                    ↓
小车本地执行器 ←──受认证任务通道── Task Token
       ↑
本地反馈、停止、控制租约与设备保护
```

本仓库没有替用户创建云资源或证明云服务已经上线。实际部署时设置 `ROBOFLEX_ACCESS_TOKEN`，在反向代理终止TLS，限制来源、请求大小和连接数量。服务绑定非回环地址时必须配置访问令牌；远端客户端仅接受HTTPS。推理接口一次运行一个请求，多余请求返回429，采集侧保留最新窗口而非无限排队。

低频语义推理与本地高频控制分开。50 Hz是当前参考仿真的控制周期设置，不是4G远程控制频率或真实平台实测值。云端取消接口不能替代小车本地停止、物理急停和夹持保持机制。

## 空间与硬件适配

`roboflex.stack.planner_bridge.planning_request`将已绑定Task Token、Scene、底盘与四轴状态转换为合作研究规划器的输入契约，包含机器人模型、地图版本、步骤前置条件与完成条件。`validate_reference`校验返回轨迹的身份、单位、有效期、维度和关节界限；`t_s`相对接收校验时刻，短时轨迹不能超过有效期，后续片段需要更新。此适配层不包含研究优化器源码，通过数据校验也不意味着已经通过扫掠体碰撞检查或获准驱动电机。

离线图像可用 `python -m roboflex describe-hand --image hand.jpg --glove-json examples/glove_sample.json --model models/hand_landmarker.task`生成描述，需先安装vision依赖并下载模型。示例手套JSON是合成值，请用实际标定测量替换。此命令只处理素材，不下发任务。

普通RGB视频、文本描述及IMU本身不会生成可信三维地图。参考仿真使用 `simulation_truth`；实际平台应选择实测标记、注册RGB-D或经过验证的地图，并提供相机／底盘／工具变换。不同视角的用户手部与小车目标不能只凭图像位置直接对齐。

当前硬件按ESP32、柔性弯曲＋IMU、树莓派、广和通4G、麦轮、舵机与柔性机械爪组织。历史照片中的ESP8266与控制器保持原样标注。柔性机械爪的被动顺应性不能充当力反馈，抓持是否成立需使用平台实际具备的视觉或设备证据。

上真实设备前，适配轮径和车体尺寸、轮序／转向、连杆／工具偏置、舵机数学角与驱动命令映射，以及停止与设备状态协议。未完成这些参数时保留仿真执行。
