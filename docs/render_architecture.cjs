// Editable SVG source. Render with Node + sharp: node docs/render_architecture.cjs
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const out = path.resolve(__dirname, '../assets');
const el = [];
const esc = s => String(s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const text = (x,y,s,size=23,color='#18324D',weight=400,anchor='start') => el.push(`<text x="${x}" y="${y}" fill="${color}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}">${esc(s)}</text>`);
const lines = (x,y,ss,size=22,color='#496275',gap=33,weight=400) => ss.forEach((s,i)=>text(x,y+i*gap,s,size,color,weight));
const rect = (x,y,w,h,fill,stroke='#DCE6EC',r=16,dash='') => el.push(`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="2"${dash?` stroke-dasharray="${dash}"`:''}/>`);
const box = (x,y,w,h,title,ss,opts={}) => { rect(x,y,w,h,opts.fill||'#FFFFFF',opts.stroke||'#D8E5ED'); text(x+24,y+40,title,25,opts.titleColor||'#075789',700); lines(x+24,y+77,ss,opts.size||21,opts.color||'#496275',opts.gap||31); };
const arrow = (d,color='#11A397',dash=false) => el.push(`<path d="${d}" fill="none" stroke="${color}" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"${dash?' stroke-dasharray="8 7"':''} marker-end="url(#${color==='#CE8A26'?'amber':color==='#758BA0'?'grey':'teal'})"/>`);
const tag=(x,y,label,color='#075789')=>{text(x,y,label,18,color,600);};
el.push(`<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1510" viewBox="0 0 1800 1510"><defs><marker id="teal" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0L9 4.5L0 9Z" fill="#11A397"/></marker><marker id="amber" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0L9 4.5L0 9Z" fill="#CE8A26"/></marker><marker id="grey" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0L9 4.5L0 9Z" fill="#758BA0"/></marker></defs><style>text{font-family:'Microsoft YaHei','Noto Sans CJK SC',sans-serif}</style><rect width="1800" height="1510" fill="#F8FBFD"/>`);
text(40,55,'知意随行 RoboFlex',34,'#075789',700);
text(40,105,'从人的意图，到连续移动操作',38,'#18324D',700);
text(1760,55,'项目总体架构',23,'#617A8D',600,'end');
text(1760,101,'感知语义在云端 · 运动反馈在本地',23,'#617A8D',400,'end');
rect(40,150,490,1110,'#F1F6FA','#D8E5ED',22);
rect(570,150,660,1110,'#EDF8F7','#CAE7E3',22);
rect(1270,150,490,1110,'#F1F6FA','#D8E5ED',22);
text(65,195,'01  使用者端',27,'#075789',700);
text(600,195,'02  腾讯云 · 多模态意图与任务',27,'#087F77',700);
text(1295,195,'03  机器人端 · 本地闭环',27,'#075789',700);
// Main boxes and source devices.
box(70,250,430,180,'传感手套',[
  '柔性弯曲传感器 + IMU / 陀螺仪',
  'ESP32 采集姿态 · Wi-Fi 上行',
  '标定后输出关节弯曲与运动特征'
]);
box(70,500,430,150,'使用者 RGB 视频',[
  '手部动作、物体关系、候选目标',
  '图像与传感数据分别采集、再对齐'
]);
box(70,740,430,150,'人的决策参与',[
  '低负担确认目标与目的地',
  '允许不操作、消歧与撤回'
]);
box(70,1000,430,200,'交互原则',[
  '不依赖语音指令',
  '从连续行为推断候选意图',
  'VLM 不直接输出电机命令',
  '用户确认后才编译可执行任务'
],{fill:'#EAF2F8',stroke:'#D4E2EC',size:21,gap:31});
box(610,230,580,100,'多源同步与最新帧缓存',[
  '时间戳 / 会话 / 序号 / 时效校验'
],{fill:'#FFFFFF'});
box(610,375,580,140,'多模态特征与结构化描述',[
  'MediaPipe：仅从图像提取手部关键点',
  '柔性 + IMU：独立标定后转为特征文本'
]);
box(610,560,580,120,'事件触发 VLM · 模型可配置',[
  '关键帧 + 手部特征 → 意图候选 / 置信度'
]);
box(610,725,580,160,'Task Token 编译与空间绑定',[
  '确认 + 目标 / 目的地 ID + 版本 + 有效期',
  '绑定 world 坐标下的米制位姿',
  '检查观测时效、场景版本与执行可行性'
]);
box(610,930,580,270,'合作研究 · 慢层全局规划',[
  '可靠性感知的时空全身轨迹优化',
  '感知时间保障 · 补偿余量 · 连续任务',
  '输出底盘与机械臂的全身参考轨迹',
  '接收状态变化与失败事件，在线重规划'
],{fill:'#E7F4F2',stroke:'#8ACFC5',gap:34});
tag(634,1170,'合作研究基础已完成；联合优化器通过适配接口接入','#087F77');
box(1310,230,410,170,'机器人视觉回传',[
  '相机 + camera-stream',
  '最新帧优先 · 广和通 4G',
  '公网经安全隧道 / 鉴权网关'
],{size:20,gap:30});
box(1310,455,410,180,'场景定位与机器人状态',[
  '深度 / 标记标定 / 已验证地图',
  '物体位姿、目标 ID、坐标系',
  'RGB 本身不等于米制 3D 地图'
],{size:20,gap:31});
box(1310,685,410,130,'独立本地停止链路',[
  '急停 / 看门狗 / 人工接管',
  '断网与云端延迟不阻塞停止'
],{fill:'#FFF6E7',stroke:'#E7C795',titleColor:'#A66D1D',size:20,gap:29});
box(1310,865,410,210,'合作研究 · 相位加权反馈',[
  '按阶段平滑调整代价权重：',
  '轨迹跟踪 / 任务误差 / 平滑性',
  '局部反馈补偿 · 安全约束',
  '参考控制节拍 50 Hz'
],{fill:'#E7F4F2',stroke:'#8ACFC5',size:20,gap:30});
box(1310,1120,410,110,'执行硬件',[
  '树莓派 · 麦轮底盘 · 舵机 · 柔性爪'
],{size:19});
// Directed flows, with short labels at the channel boundary.
arrow('M500 290 L610 270'); tag(518,252,'姿态');
arrow('M500 550 L550 550 L550 310 L610 310'); tag(514,530,'RGB');
arrow('M900 330 L900 375');
arrow('M900 515 L900 560');
arrow('M900 680 L900 725');
arrow('M900 885 L900 930');
arrow('M500 800 L610 800'); tag(512,785,'确认');
arrow('M1310 285 L1190 285'); tag(1200,268,'视频');
arrow('M1310 550 L1250 550 L1250 840 L1190 840');
tag(1238,675,'场景','#617A8D');
arrow('M1190 985 L1310 985'); tag(1201,970,'轨迹');
arrow('M1515 1075 L1515 1120');
arrow('M1720 1175 L1740 1175 L1740 1035 L1720 1035','#758BA0');
tag(1622,1104,'状态反馈','#617A8D');
arrow('M1515 815 L1515 865','#CE8A26');
arrow('M1310 1050 L1246 1050 L1246 1105 L1190 1105','#758BA0');
tag(1203,1135,'事件','#617A8D');
// Current public-repository reference backend separated from completed research.
rect(40,1290,1720,120,'#FFFFFF','#C9DDE7',18);
text(64,1328,'当前开源无硬件验证',25,'#075789',700);
text(370,1328,'同一 Task Token → MuJoCo 参考后端 → 动力学步进 / 日志 / 渲染视频',24,'#18324D',600);
text(64,1370,'参考执行：A* + IK + 阶段执行；输入可用回放或在线 VLM；空间场景由仿真真值提供。',22,'#496275');
text(64,1447,'实现口径',21,'#075789',700);
text(195,1447,'云端契约 / 感知接口 / 仿真参考执行已实现；合作研究联合优化器与硬件驱动按接口对接。',21,'#496275');
text(64,1482,'物理边界',21,'#A66D1D',700);
text(195,1482,'柔性爪以被动顺应缓冲接触，当前无力反馈；参考仿真不验证柔性爪抓蛋安全。',21,'#496275');
el.push('</svg>');
fs.mkdirSync(out,{recursive:true});
const svg = el.join('\n');
fs.writeFileSync(path.join(out,'architecture-cloud-edge.svg'),svg,'utf8');
sharp(Buffer.from(svg)).png().toFile(path.join(out,'architecture-cloud-edge.png')).then(()=>console.log('Rendered architecture SVG + PNG'));
