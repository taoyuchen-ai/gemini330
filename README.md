# gemini330 — AI 体态监测与矫正系统

中学生课堂坐姿/读写姿势对视力与脊柱影响的研究系统。
Orbbec Gemini335 深度相机 + MediaPipe BlazePose 33 关键点 + Dotii-Display 桌面视觉提醒点。

## 架构

- `src/camera_capture.py` — pyorbbecsdk aligned depth+color 30 fps（ADR-0002）
- `src/pose_inference.py` — MediaPipe BlazePose 33 关键点（ADR-0005）
- `src/posture_classify.py` — 6 类坐姿纯函数分类（ADR-0005）
- `src/depth_distance.py` — 头-桌面距离（前倾读写 vs 趴桌，ADR-0002）
- `src/statistics.py` — 1 秒窗移动众数 + 不良占比（ADR-0006）
- `src/reminder_policy.py` — 30 秒持续 + 60 秒间隔（ADR-0007）
- `src/dotii_reminder.py` — Dotii-Display HTTP fail/idle 表情（ADR-0004）
- `src/storage.py` — SQLite WAL 每分钟落盘（ADR-0007）
- `src/exceptions.py` — 相机重连/离座/API 失联/MediaPipe 超时（ADR-0007）
- `src/reports.py` — 个人/班级日报
- `src/pipeline.py` — 主入口串联
- `config/thresholds.yaml` — 阈值与实验参数（ADR-0005）
- `docs/adr/` — 9 个架构决策记录
- `CONTEXT.md` — 术语表

## 姿势分类（6 类）

| 标签 | 类别 | 说明 |
|------|------|------|
| `sit_upright` | 可接受 | 坐正 |
| `forward_read` | 可接受 | 前倾读写 |
| `desk_lying` | **不良** | 趴桌 |
| `bend` | **不良** | 弯腰 |
| `lean_back` | 中性 | 后仰（休息） |
| `turn_side` | 中性 | 转头侧身 |
| `absence` | — | 离座（不计分母） |
| `missing` | — | 缺失帧（不计分母） |

不良占比 = (趴桌帧 + 弯腰帧) / (有效帧 − 离座 − 缺失)，不插值。

## 部署

1. Windows 台式机 + 独立 GPU（MediaPipe GPU 推理 30 fps）。
2. 接 Orbbec Gemini335（USB3），装 `pyorbbecsdk`。
3. Dotii-Display（ESP32-S3-Touch-AMOLED-1.75）连同一局域网，HTTP API `127.0.0.1:8787`。
4. `pip install -e .`（含 mediapipe / opencv-python / pyorbbecsdk / requests / pyyaml / numpy / scipy）。

> **复用提示**：Dotii-Display 原仓库无 LICENSE，复用前需联系作者 [ZeroOne000011](https://github.com/ZeroOne000011) 确认授权。

## 现场标定（每位被试，校准期 ~1 周）

1. 被试按校准姿势坐正、前倾读写、趴桌、弯腰、后仰、转头各采样若干帧。
2. 实测躯干-大腿角、躯干后仰/前倾角、头偏航、躯干侧偏，替换 `config/thresholds.yaml` 中 `rgb.*` 占位值。
3. 桌面参考深度：相机视野对准被试座位后，记录桌面参考像素深度填 `desk_depth_m`（运行参数）。

## 运行

```bash
python -c "from src.pipeline import run_session, load_thresholds; ..." # 详见 pipeline.run_session 签名
```

`run_session` 接受已 `start()` 的 camera/pose、Storage、thresholds、participant_id、phase、desk_depth_m、dotii_show。返回个人日报 dict。

实验阶段：`baseline`（基线 2 周，无提醒）→ `intervention`（干预 4 周，开提醒）→ `followup`（跟踪 2 周，关提醒验证内化效应）。

## 数据保留

- 原始 33 关键点坐标分析完成 30 日内删除。
- 日报聚合保留 1 学期。
- 不存任何图像。

## 测试

```bash
pytest -q
```

纯函数模块（posture_classify / statistics / reminder_policy / depth_distance / storage / dotii_reminder / reports）TDD；pipeline 用 fake camera/pose/clock 集成测试。

## 伦理与知情同意

班主任签同意书「不口头提醒坐姿」；任课教师不知情；按视力分层招募；退出者分析。详见 `docs/adr/0009-teacher-coordination-and-participant-recruitment.md`。
