# 参考、合作研究与方法来源

知意随行结合GPT-6语义理解、意图驱动任务组织及连续移动操作的规划控制。以下列出合作研究与工具出处，便于查阅技术机制和部署条件。

1. **连续移动操作合作研究：Synergizing Efficiency and Reliability for Continuous Mobile Manipulation。** [公开预印本](https://arxiv.org/abs/2604.05430)、[全文](https://arxiv.org/html/2604.05430v1)、[公开演示](https://www.bilibili.com/video/BV1YWP4zxEQD)。本项目以合作团队成果为研究基础，围绕意图驱动取物开展衔接与应用。可靠性感知的时空全身规划、观测窗口、补偿余量与相位依赖切换控制据此展开，方法保留原始研究署名。项目将上述机制衔接到意图驱动的取物任务。连续操作相关论文尚未见刊，相关专利进展随正式文件更新。
2. **GPT-6。** 本项目中承担图像与用户意图理解，输出目标或任务步骤。[官方模型文档入口](https://developers.openai.com/api/docs/models)。实际部署记录具体模型标识与配置。
3. **Shared Autonomy via Hindsight Optimization for Teleoperation and Teaming。** [论文](https://arxiv.org/abs/1706.00155)。目标不确定条件下的共享自主方法参照，支持目标推断与用户控制权的设计讨论。
4. **MediaPipe Hand Landmarker。** [官方Python文档](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)。随附工程的手部关键点工具。
5. **OpenCV ArUco。** [官方文档](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)。随附工程的目标标记与位姿工具。
6. **LeRobot／SmolVLA。** [官方文档](https://huggingface.co/docs/lerobot/en/smolvla)。动作策略接口与适配方法参考，实际采用型号由部署配置记录。
7. **家庭养老应用环境。** 长乐区[“十五五”规划纲要](https://www.fzcl.gov.cn/xjwz/zwgk/zfxxgk/fdzdgknr/gmjjhshfzghzxghqyghjxgzc/202607/t20260709_5344593.htm)、[基本养老服务体系建设实施方案](https://www.fzcl.gov.cn/xjwz/ztzl/jczwgk/sdly/ylfw/gkqk/ylfwtyzc/202410/t20241021_4911826.htm)。用于居家养老、辅具服务与场景合作的背景说明。

手势交互与嵌入式基础沿用项目材料中的远程手势抓取小车、RT-Thread手势移动抓取及EIT 2022相关研究，支撑传感输入与设备控制的系统衔接。

原论文、研究平台、图片和代码保留相应署名与许可。随附工程的模型与第三方依赖按各自官方条款配置，项目代码由团队复核及调试。
