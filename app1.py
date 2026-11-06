"""Automatic six-clip AIBUS-Cloud dashboard. Run: python app1.py (port 9529)."""
import copy
from concurrent.futures import ThreadPoolExecutor
import html
import hashlib
import json
import logging
import math
import os
import random
from pathlib import Path
import threading
import subprocess
import shutil
import time
import uuid

# Gradio creates HTTP clients on import. Ignore unsupported SOCKS proxies only
# when their optional transport is absent; local serving needs no such proxy.
import importlib.util
if importlib.util.find_spec("socksio") is None:
    for _key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        if os.environ.get(_key, "").lower().startswith(("socks://", "socks5://", "socks5h://", "socks4://")):
            os.environ.pop(_key)
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

ROOT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = os.getenv("BUVFM_CHECKPOINT_PATH", str(ROOT_DIR / "ckpts/vjepa_full/best_vjepa_model9639(paper).pt"))
RISK_NAMES = ("LOW RISK", "MEDIUM RISK", "HIGH RISK")
RISK_COLORS = ("#16a085", "#e9a323", "#eb5969")
LOG = logging.getLogger("app1")
CACHE_DIR = Path(os.getenv("BUVFM_CACHE_DIR", str(ROOT_DIR / ".cache/app1_results"))).resolve()


def file_stamp(path):
    path = Path(path).resolve()
    try:
        stat = path.stat()
        return [str(path), stat.st_size, stat.st_mtime_ns]
    except OSError:
        return [str(path), None, None]


class ClipResultCache:
    """Persistent per-stage results; a missing render invalidates Stage 1 only."""

    def __init__(self, source):
        self.source = source
        # Include inference source and model stamps so edits invalidate old results.
        common = [file_stamp(source), file_stamp(ROOT_DIR / "pipeline_utils.py")]
        stage1 = common + [file_stamp(ROOT_DIR / name) for name in (
            "distill/det/output/ckpts/distilled.pt",
            "distill/cls/output/ckpts/best_finetune_distilled.pt",
            "distill/quality_con.py")]
        cloud = common + [file_stamp(CHECKPOINT_PATH), file_stamp(ROOT_DIR / "buildmodel.py"),
                          file_stamp(ROOT_DIR / "utils.py")]
        self.stage1_dir = self.directory("stage1", stage1)
        self.prediction_dir = self.directory("prediction", cloud)

    @staticmethod
    def directory(stage, stamps):
        key = hashlib.sha256(json.dumps(stamps, sort_keys=True).encode()).hexdigest()
        return CACHE_DIR / stage / key

    @staticmethod
    def read(directory):
        try:
            return json.loads((directory / "result.json").read_text())
        except (OSError, ValueError):
            return None

    @staticmethod
    def write(directory, result):
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f"{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(json.dumps(result, allow_nan=False))
            temporary.replace(directory / "result.json")
        finally:
            temporary.unlink(missing_ok=True)

    def load_stage1(self):
        result = self.read(self.stage1_dir)
        if not isinstance(result, dict):
            return None
        try:
            if (result["status"] not in ("pass", "stop") or not result["frame_results"]
                    or result["classification"]["total_frames"] <= 0):
                return None
            video = Path(result["annotated_video_path"])
            if not video.is_file() or not video.stat().st_size:
                return None
        except (KeyError, TypeError, OSError):
            return None
        return result

    def save_stage1(self, result):
        self.stage1_dir.mkdir(parents=True, exist_ok=True)
        target = self.stage1_dir / "annotated.mp4"
        temporary = self.stage1_dir / f"{uuid.uuid4().hex}.mp4"
        try:
            shutil.copyfile(result["annotated_video_path"], temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        saved = dict(result, annotated_video_path=str(target))
        self.write(self.stage1_dir, saved)
        return saved

    def load_prediction(self):
        result = self.read(self.prediction_dir)
        try:
            if result["predicted_class"] not in (0, 1, 2):
                return None
            probs = [result["probs"][f"class_{c}"] for c in range(3)]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in probs):
                return None
            if not math.isclose(sum(probs), 1, abs_tol=.01):
                return None
        except (KeyError, TypeError):
            return None
        return result


def case_clip_paths(directory):
    cropped = [directory / f"{i}_cropped.mp4" for i in range(6)]
    if all(path.is_file() for path in cropped):
        return cropped
    numbered = [directory / f"{i}.mp4" for i in range(6)]
    if all(path.is_file() for path in numbered):
        return numbered
    # Some exports keep the last left clip and right clips' original names.
    left = [p for p in numbered[:3] if p.is_file()] + sorted(directory.glob("Left__*.mp4"))
    right = [p for p in numbered[3:] if p.is_file()] + sorted(directory.glob("Right__*.mp4"))
    if len(left) == len(right) == 3:
        return left + right
    raise ValueError(f"Case {directory.name} must contain three left and three right clips")


def discover_cases():
    case_root = ROOT_DIR / "assets" / "selected_videos"
    cases = {directory.name: case_clip_paths(directory)
             for directory in sorted(case_root.glob("Patient_*")) if directory.is_dir()}
    if not cases:
        raise ValueError(f"No patient cases found in {case_root}")
    return cases


CASES = discover_cases()
DEFAULT_CASE = next(iter(CASES))
CLIP_PATHS = CASES[DEFAULT_CASE]
CLIP_DIR = CLIP_PATHS[0].parent


def playback_video_path(path):
    """Lossless fast-start copy for browsers; inference keeps the source file."""
    if not path.is_file():
        return path
    target = path.parent / ".playback" / path.name
    if target.is_file() and target.stat().st_mtime_ns >= path.stat().st_mtime_ns:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{path.stem}.{os.getpid()}.tmp.mp4")
    try:
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path),
                        "-map", "0", "-c", "copy", "-movflags", "+faststart", str(temporary)],
                       check=True, capture_output=True, timeout=30)
        temporary.replace(target)
        return target
    except (OSError, subprocess.SubprocessError):
        LOG.exception("Could not prepare fast-start playback for %s", path)
        temporary.unlink(missing_ok=True)
        return path


class ClipService:
    """One resident model and one active batch shared by concurrent viewers."""

    def __init__(self, clip_paths=None, engine=None):
        self.clip_paths = list(CLIP_PATHS if clip_paths is None else clip_paths)
        self.engine = engine
        self.lock = threading.Lock()
        self.batch_lock = engine.batch_lock if engine else threading.Lock()
        self.ready = engine.ready if engine else threading.Event()
        self.started = False
        self.pipeline = None
        self.stage1_workers = []
        self.state = {
            "model": "loading", "inferring": False, "model_detail": "正在加载 AIBUS-Cloud 权重与 Stage 1 模型…",
            "phase": "Waiting for Models", "completed": 0, "done": False,
            "error": None, "risk": None, "processed_frames": 0, "processing_seconds": 0.0,
            "clips": [{"name": p.name, "status": "Waiting for Clips from Edge-side", "received": False,
                       "video": None, "annotated_video": None, "frame_results": None,
                       "detail": "", "prediction": None,
                       "error": None} for p in self.clip_paths],
        }
        self.empty_clips = copy.deepcopy(self.state["clips"])

    def snapshot(self):
        with self.lock:
            state = copy.deepcopy(self.state)
        if self.engine:
            engine_state = self.engine.snapshot()
            state.update(model=engine_state["model"], model_detail=engine_state["model_detail"])
        return state

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def update_clip(self, index, **values):
        with self.lock:
            self.state["clips"][index].update(values)

    def preload(self):
        try:
            # The original loaders use checkpoint paths relative to the repo.
            os.chdir(ROOT_DIR)
            from pipeline_utils import PipelineRunner, Stage1Filter
            self.pipeline = PipelineRunner(CHECKPOINT_PATH)
            self.pipeline._ensure_stage3()
            # YOLO predictors carry mutable state; each concurrent clip owns one.
            self.stage1_workers = [self.pipeline.stage1]
            self.stage1_workers.extend(Stage1Filter() for _ in CLIP_PATHS[1:])
            LOG.info("Models ready: Stage 1=%s, YOLO=%s, AIBUS-Cloud=%s",
                     self.pipeline.stage1.device, self.pipeline.stage1.yolo_device,
                     self.pipeline.stage3.device)
            self.update(model="running", model_detail="")
        except Exception as exc:
            LOG.exception("Model initialization failed")
            self.update(model="error", model_detail=f"模型加载失败：{exc}",
                        error=f"模型加载失败：{exc}", phase="Model Loading Failed", done=True)
        finally:
            self.ready.set()

    def start(self):
        with self.lock:
            if self.started:
                if not self.state["done"] or self.state["model"] == "error":
                    return
                # A fresh page replays real reception/inference with warm models.
                self.state.update(phase="Waiting for Clips from Edge-side", completed=0, done=False,
                                  error=None, risk=None, inferring=False, processed_frames=0, processing_seconds=0.0,
                                  cache_presenting=False,
                                  clips=copy.deepcopy(self.empty_clips))
            self.started = True
        threading.Thread(target=self._run, name="clip-inference", daemon=True).start()

    def _run(self):
        self.update(phase="Waiting for Processing")
        self._process_batch()

    def _process_batch(self):
        # Prepare cached lossless web copies, then publish the entire batch.
        playback_paths = [playback_video_path(path) for path in self.clip_paths]
        caches = [ClipResultCache(path) for path in self.clip_paths]
        cached_stage1 = [cache.load_stage1() for cache in caches]
        cached_predictions = [cache.load_prediction() for cache in caches]
        if all(s1 is not None and (s1["status"] == "stop" or prediction is not None)
               for s1, prediction in zip(cached_stage1, cached_predictions)):
            for state in self.cached_display_states(playback_paths, cached_stage1, cached_predictions):
                self.update(**state)
            LOG.info("All six clips restored from cache; no inference needed")
            return
        with self.lock:
            for clip, path, playback in zip(self.state["clips"], self.clip_paths, playback_paths):
                clip.update(received=True, detail="")
                if path.is_file():
                    clip.update(video=str(playback), status="accept")
                else:
                    clip.update(status="File Missing", error=f"Video not found: {path}")
            self.state["phase"] = "All Clips Received · Waiting for Models"
        # Restore before waiting on model loading: complete caches need no model.
        for index, s1 in enumerate(cached_stage1):
            if s1 is None:
                continue
            prediction = ({"predicted_class": 0, "confidence": None, "probs": None}
                          if s1["status"] == "stop" else cached_predictions[index])
            with self.lock:
                self.state["clips"][index].update(
                    annotated_video=s1["annotated_video_path"], frame_results=s1["frame_results"],
                    prediction=prediction, status=("No Lesions" if s1["status"] == "stop"
                                                  else "Lesions Detected" if prediction else "Stage 1 Cached"))
                self.state["completed"] += 2 if prediction is not None else 1
                self.state["processed_frames"] += s1["classification"]["total_frames"]
        if all(clip["prediction"] is not None for clip in self.snapshot()["clips"]):
            self._finish_batch(0.0)
            LOG.info("All six clips restored from cache; no inference needed")
            return
        with self.batch_lock:
            self._infer_missing(caches, cached_stage1, cached_predictions)

    def cached_display_states(self, playback_paths, stage1_results, predictions):
        """Keep each viewer's cache presentation independent of other requests."""
        state = self.snapshot()
        state.update(clips=copy.deepcopy(self.empty_clips), cache_presenting=True,
                     phase="Waiting for Clips from Edge-side", completed=0, done=False,
                     error=None, risk=None, inferring=False, processed_frames=0, processing_seconds=0.0)
        for index, playback in enumerate(playback_paths):
            state["clips"][index].update(received=True, video=str(playback), status="accept")
            state["phase"] = f"Receiving Clip {index + 1:02d} / 06"
            yield copy.deepcopy(state)
            time.sleep(random.uniform(0.5, 1.0))
        for index, s1 in enumerate(stage1_results):
            state["clips"][index].update(
                annotated_video=s1["annotated_video_path"], frame_results=s1["frame_results"],
                status="No Lesions" if s1["status"] == "stop" else "Lesions Detected")
            state["phase"] = f"Clip {index + 1:02d} / 06 · Edge-side Detection & Classification"
            state["completed"] += 1
            state["processed_frames"] += s1["classification"]["total_frames"]
            yield copy.deepcopy(state)
            time.sleep(random.uniform(0.5, 1.0))
        for index, (s1, prediction) in enumerate(zip(stage1_results, predictions)):
            if s1["status"] == "stop":
                prediction = {"predicted_class": 0, "confidence": None, "probs": None}
            state["clips"][index]["prediction"] = prediction
            state["phase"] = f"Clip {index + 1:02d} / 06 · Risk Stratification"
            state["completed"] += 1
            yield copy.deepcopy(state)
            time.sleep(random.uniform(0.5, 1.0))
        state.update(done=True, risk=max(c["prediction"]["predicted_class"] for c in state["clips"]),
                     phase="All Clips Processed")
        yield copy.deepcopy(state)

    def _infer_missing(self, caches, cached_stage1, cached_predictions):
        self.ready.wait()
        if self.snapshot()["model"] == "error":
            engine_state = self.engine.snapshot() if self.engine else self.snapshot()
            self.update(done=True, error=engine_state["error"], phase="Model Loading Failed")
            return
        if self.engine:
            self.pipeline = self.engine.pipeline
            self.stage1_workers = self.engine.stage1_workers
        started = time.perf_counter()
        self.update(inferring=True, phase="Left Breast · Edge-side Detection & Classification")

        def process_clip(index):
            path = self.clip_paths[index]
            stage1_completed = cached_stage1[index] is not None
            try:
                LOG.info("Clip %02d parallel processing started", index + 1)
                self.update_clip(index, status="Rendering Edge-side Results", detail="")
                s1 = cached_stage1[index]
                if s1 is None:
                    s1 = self.stage1_workers[index].run(str(path))
                    try:
                        s1 = caches[index].save_stage1(s1)
                    except (OSError, TypeError, ValueError):
                        LOG.exception("Could not cache Stage 1 for %s", path)
                    with self.lock:
                        self.state["processed_frames"] += s1["classification"]["total_frames"]
                        self.state["completed"] += 1
                        self.state["processing_seconds"] = time.perf_counter() - started
                        self.state["clips"][index].update(
                            annotated_video=s1["annotated_video_path"],
                            frame_results=s1["frame_results"], detail="")
                stage1_completed = True
                if s1["status"] == "stop":
                    prediction = {"predicted_class": 0, "confidence": None, "probs": None}
                    status = "No Lesions"
                else:
                    self.update_clip(index, status="AIBUS-Cloud Infering")
                    # Evaluation-only cloud weights are shared across workers.
                    prediction = cached_predictions[index]
                    if prediction is None:
                        prediction = self.pipeline.stage3.predict(str(path))
                        try:
                            caches[index].write(caches[index].prediction_dir, prediction)
                        except (OSError, TypeError, ValueError):
                            LOG.exception("Could not cache risk prediction for %s", path)
                    status = "Lesions Detected"
                self.update_clip(index, prediction=prediction, status=status)
                LOG.info("Clip %02d parallel processing finished: class_%s",
                         index + 1, prediction["predicted_class"])
            except Exception as exc:
                LOG.exception("Clip %s failed", path)
                self.update_clip(index, status="Inference Failed", error=str(exc), detail=str(exc))
            finally:
                with self.lock:
                    self.state["completed"] += 1 if stage1_completed else 2
                    self.state["processing_seconds"] = time.perf_counter() - started

        # Complete all three left clips before scheduling the right breast.
        for side, start in (("Left Breast", 0), ("Right Breast", 3)):
            self.update(phase=f"{side} · Edge-side Detection & Classification")
            clips = self.snapshot()["clips"]
            indices = [i for i in range(start, start + 3)
                       if not clips[i]["error"] and clips[i]["prediction"] is None]
            with ThreadPoolExecutor(max_workers=3, thread_name_prefix=f"{side.split()[0].lower()}-clip") as executor:
                list(executor.map(process_clip, indices))
        self._finish_batch(time.perf_counter() - started)

    def _finish_batch(self, seconds):
        clips = self.snapshot()["clips"]
        if any(c["error"] or c["prediction"] is None for c in clips):
            self.update(done=True, inferring=False, phase="Processing Finished · Some Clips Failed",
                        error="Some clips failed. Overall risk is unavailable; see clip errors and server logs.",
                        processing_seconds=seconds)
        else:
            risk = max(c["prediction"]["predicted_class"] for c in clips)
            self.update(done=True, inferring=False, risk=risk, phase="All Clips Processed", completed=12,
                        processing_seconds=seconds)


# One shared model engine; each case keeps independent videos and predictions.
service = ClipService()
case_services = {DEFAULT_CASE: service}
case_services_lock = threading.Lock()
viewer_tokens = {}
viewer_tokens_lock = threading.Lock()


def get_case_service(case_name):
    if case_name not in CASES:
        raise ValueError("Unknown case")
    with case_services_lock:
        if case_name not in case_services:
            case_services[case_name] = ClipService(CASES[case_name], engine=service)
        return case_services[case_name]


def release_viewer(session_id):
    with viewer_tokens_lock:
        viewer_tokens.pop(session_id, None)


STATE_COLORS = {
    "waiting": ("#b77916", "#fff7e6"),
    "receiving": ("#0891b2", "#e8f8fc"),
    "screening": ("#2563eb", "#edf3ff"),
    "grading": ("#7c3aed", "#f3edff"),
    "complete": ("#159b80", "#e8f8f1"),
    "negative": ("#64748b", "#f0f4f8"),
    "failed": ("#dc4c60", "#fff0f2"),
}


def status_kind(text, error=None):
    if error or "失败" in text or "缺失" in text or "Failed" in text:
        return "failed"
    if "未检出" in text or text == "No Lesions":
        return "negative"
    if "完成" in text or text in ("Lesions Detected", "All Clips Processed"):
        return "complete"
    if "分级" in text or "Risk Stratification" in text or text == "AIBUS-Cloud Infering":
        return "grading"
    if "识别中" in text or "检测 / 分类" in text or "Edge-side Detection & Classification" in text or text == "Rendering Edge-side Results":
        return "screening"
    if "正在接收" in text or text.startswith("Receiving Clip"): 
        return "receiving"
    return "waiting"


def state_style(kind):
    color, tint = STATE_COLORS[kind]
    return f"--state-color:{color};--state-tint:{tint}"


def progress_html(state, max_percent=100):
    received = sum(c["received"] and not c["error"] for c in state["clips"])
    finished = sum(c["prediction"] is not None for c in state["clips"])
    # Six receptions + twelve processing steps. Only confirmed work advances it.
    percent = round((received + state["completed"]) / 18 * 100)
    percent = min(percent, max_percent)
    if state["error"]:
        percent = min(percent, 99)
    kind = status_kind(state["phase"], state["error"])
    raw_count = sum(c["video"] is not None for c in state["clips"])
    detected_count = sum(c["annotated_video"] is not None for c in state["clips"])
    return f'''<div class="progress-card state-{kind}" style="{state_style(kind)}"
    data-presentation-id="{state.get('presentation_id', '')}" data-raw-count="{raw_count}"
    data-detected-count="{detected_count}" data-probability-count="{finished}">
    <div class="progress-title"><span class="phase-badge">{html.escape(state['phase'])}</span><b>{percent}%</b></div>
    <div class="progress-track" role="progressbar" aria-label="Clip Reception and Processing Progress" aria-valuenow="{percent}" aria-valuemin="0" aria-valuemax="100"><span style="width:{percent}%"></span></div>
    <div class="progress-meta"><span>Received {received} / 6</span><span>Processed {finished} / 6</span><span>AIBUS-Edge → AIBUS-Cloud</span></div></div>'''


def model_html(state):
    if state["model"] == "loading":
        mode, label = "loading", "Loading"
    elif state["model"] == "error":
        mode, label = "error", "Load Failed"
    elif state["inferring"]:
        mode, label = "inferring", "Inferring"
    elif state["error"]:
        mode, label = "error", "Inference Error"
    else:
        mode, label = "ready", "Ready"
    frames = state["processed_frames"]
    seconds = state["processing_seconds"]
    speed = f"{frames / seconds:.1f} frames/s" if seconds > 0 else "—"
    finished = sum(c["prediction"] is not None for c in state["clips"])
    return f'''<div class="model-card"><div class="eyebrow">MODEL STATUS</div>
    <div class="model-heading"><span>AIBUS-Cloud</span><span class="model-badge {mode}"><i></i>{label}</span></div>
    <div class="engine-row"><span>Processing Speed</span><b>{speed}</b></div>
    <div class="engine-row"><span>Frames Processed</span><b>{frames} frames</b></div>
    <div class="engine-row"><span>Clips Completed</span><b>{finished} / 6 clips</b></div></div>'''



def gauge_html(risk=None, error=None, clips=None, show_breasts=True, pointer_duration=1.4):
    segments = []
    for start, end, color in [(180, 120, RISK_COLORS[0]), (120, 60, RISK_COLORS[1]), (60, 0, RISK_COLORS[2])]:
        def point(deg):
            rad = math.radians(deg)
            return 170 + 128 * math.cos(rad), 160 - 128 * math.sin(rad)
        x1, y1 = point(start)
        x2, y2 = point(end)
        segments.append(f'<path d="M{x1},{y1} A128,128 0 0 1 {x2},{y2}" stroke="{color}" stroke-width="25" fill="none"/>')
    needle = ""
    if risk is not None:
        rotation = (30, 90, 150)[risk]
        needle = f'''<g class="risk-needle-anchor" transform="translate(170 160)">
        <g class="risk-needle" transform="rotate({rotation})"
        style="--needle-angle:{rotation}deg;--pointer-duration:{pointer_duration}s">
        <line x1="0" y1="0" x2="-105" y2="0" stroke="#173c47" stroke-width="5" stroke-linecap="round"/>
        </g></g><circle cx="170" cy="160" r="9" fill="#173c47"/>'''

    title = ("LOW RISK", "MEDIUM RISK", "HIGH RISK")[risk] if risk is not None else ("暂无法评估" if error else "Pending")
    color = RISK_COLORS[risk] if risk is not None else "#71868d"
    breast_rows = []
    for name, start in (("Left Breast", 0), ("Right Breast", 3)):
        side_clips = clips[start:start + 3] if clips else []
        side_color = "#71868d"
        if any(c["error"] for c in side_clips):
            side_label = "Unavailable"
        elif len(side_clips) == 3 and all(c["prediction"] is not None for c in side_clips):
            side_risk = max(c["prediction"]["predicted_class"] for c in side_clips)
            side_label = ("LOW RISK", "MEDIUM RISK", "HIGH RISK")[side_risk]
            side_color = RISK_COLORS[side_risk]
        else:
            side_label = "Pending"
        breast_rows.append(f'<div class="breast-risk-row"><span>{name}</span>'
                           f'<b style="color:{side_color}">{side_label}</b></div>')
    message = f'<p class="error-message">{html.escape(error)}</p>' if error else ""
    ready_class = " risk-ready" if risk is not None else ""
    return f'''<div class="risk-card{ready_class}" style="--risk-color:{color}"><div class="eyebrow">OVERALL RISK</div>
    <svg viewBox="0 0 340 210" role="img" aria-label="{title}半圆风险仪表盘">{''.join(segments)}{needle}
    <text x="52" y="198" text-anchor="middle">LOW</text><text x="170" y="20" text-anchor="middle">Medium</text><text x="288" y="198" text-anchor="middle">HIGH</text></svg>
    <div class="risk-value" style="color:{color}">{title}</div>
    {'<div class="breast-risk-summary">' + ''.join(breast_rows) + '</div>' if show_breasts else ''}
    {message}</div>'''


def stage1_chart_html(clip):
    """Compact version of app.py's valid-frame / YOLO chart."""
    frames = clip["frame_results"]
    if not frames:
        message = clip["error"] or "Rendering Edae-side Results"
        return f'<div class="chart-placeholder">{html.escape(message)}</div>'
    width, height, left, right, top, bottom = 400, 115, 31, 12, 12, 23
    plot_w, plot_h = width - left - right, height - top - bottom
    first, last = frames[0]["frame_idx"], frames[-1]["frame_idx"]

    def x(value):
        return left + (value - first) / max(last - first, 1) * plot_w

    def y(value):
        return top + (1 - value) * plot_h

    # Keep the valid-frame shading from the original chart, merging runs.
    ranges, start = [], None
    for i, frame in enumerate(frames):
        valid = frame["yolo_score"] > .5 and 1 - frame["mobilenet_score"] > .5
        if valid and start is None:
            start = i
        if start is not None and (not valid or i == len(frames) - 1):
            end = i if valid else i - 1
            x0 = max(left, x(frames[start]["frame_idx"]) - plot_w / max(len(frames), 1) / 2)
            x1 = min(width - right, x(frames[end]["frame_idx"]) + plot_w / max(len(frames), 1) / 2)
            ranges.append(f'<rect x="{x0:.2f}" y="{top}" width="{max(x1-x0, 1):.2f}" height="{plot_h}" fill="#22c55e" opacity=".10"/>')
            start = None
    curves = []
    for key, color in (("mobilenet_score", "#2563eb"), ("yolo_score", "#10b981")):
        points = ' '.join(f'{x(f["frame_idx"]):.2f},{y(1-f[key] if key == "mobilenet_score" else f[key]):.2f}' for f in frames)
        curves.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linejoin="round"/>')
    grid = ''.join(f'<line x1="{left}" x2="{width-right}" y1="{y(v)}" y2="{y(v)}" stroke="#dfe8ec" stroke-dasharray="3 3"/><text x="24" y="{y(v)+3}" text-anchor="end">{v:g}</text>' for v in (0, .5, 1))
    return f'''<div class="mini-chart"><div class="chart-legend"><span style="color:#2563eb">━ Valid</span><span style="color:#10b981">━ YOLO confidence</span><span>Threshold 0.5 · Light green: Valid frames</span></div>
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Per-frame Valid and YOLO scores, threshold 0.5">{''.join(ranges)}{grid}{''.join(curves)}
    <text x="{left}" y="{height-5}">{first}</text><text x="{width-right}" y="{height-5}" text-anchor="end">{last}</text><text x="{width/2}" y="{height-5}" text-anchor="middle">Frame</text></svg></div>'''


def probability_chart_html(index, clip):
    pred = clip["prediction"]
    if pred is None or pred["probs"] is None:
        if clip["error"]:
            message = clip["error"]
        elif pred is not None:
            message = "No lesions were detected in Edge-side"
        else:
            message = "AIBUS-Cloud Infering"
        return f'<div class="prob-card"><div class="chart-placeholder">{html.escape(message)}</div></div>'
    bars = []
    for c in range(3):
        value = float(pred["probs"][f"class_{c}"])
        x, bar_h = 63 + c * 105, value * 130
        bars.append(f'<rect x="{x}" y="{167-bar_h}" width="50" height="{bar_h}" rx="5" fill="{RISK_COLORS[c]}"/><text x="{x+25}" y="{158-bar_h}" text-anchor="middle" class="bar-value">{value:.1%}</text><text x="{x+25}" y="190" text-anchor="middle">{RISK_NAMES[c]}</text>')
    grid = ''.join(f'<line x1="42" x2="365" y1="{167-v*130}" y2="{167-v*130}" stroke="#e6edef"/><text x="35" y="{171-v*130}" text-anchor="end">{v:.0%}</text>' for v in (0, .5, 1))
    risk = pred["predicted_class"]
    return f'''<div class="prob-card"><div class="prob-result" style="color:{RISK_COLORS[risk]}">{RISK_NAMES[risk]}</div>
    <svg viewBox="0 0 390 220" role="img" aria-label="Clip {index+1:02d} 的 AIBUS-Cloud 三分类概率">{grid}{''.join(bars)}</svg></div>'''


CSS = """
html {overflow-y:scroll}
.gradio-container {width:100%!important;max-width:1920px!important;box-sizing:border-box!important;margin:auto!important;background:#f5f9fa!important;color:#173c47!important}
.gradio-container > .main {width:100%!important;max-width:100%!important;box-sizing:border-box!important}
.gradio-container > .main > .wrap,.gradio-container main.contain {width:100%!important;min-width:0!important;box-sizing:border-box!important}
#risk-sidebar {grid-column:2;grid-row:3 / span 2;display:flex!important;flex-direction:column!important;gap:14px!important;align-self:start!important;min-width:0!important}
#case-selector {width:100%!important;max-width:none!important;margin:0!important;box-sizing:border-box!important}
.eyebrow {font-size:11px;letter-spacing:1.8px;color:#789199;font-weight:700}
.progress-card,.model-card,.risk-card {background:white;border:1px solid #e0eaed;border-radius:18px;padding:23px;box-shadow:0 5px 20px #173c4705}
.progress-card {padding:10px 14px;border-radius:12px}
.progress-title {display:flex;align-items:center;justify-content:space-between;font-size:13px;margin:0 0 6px;line-height:1.3}.progress-title b {color:var(--state-color)}
.phase-badge {color:var(--state-color);background:var(--state-tint);border-radius:6px;padding:3px 8px;font-weight:600}
.progress-track {height:14px;background:#eaf1f2;border-radius:8px;overflow:hidden}.progress-track span {display:block;height:100%;background:var(--state-color);transition:width .5s,background .3s;position:relative;overflow:hidden}
.progress-track span:after {content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent,#ffffff80,transparent);animation:stream 1.8s infinite}
.progress-meta {display:flex;justify-content:space-between;margin-top:5px;color:#789199;font-size:11px;line-height:1.3}
#batch-progress {padding:0!important;margin:0!important}#batch-progress .html-container {padding:0!important}
.gradio-container main.contain > .column {gap:8px!important}
.clip-status.state-badge {display:inline-block;align-self:flex-start;color:var(--state-color);background:var(--state-tint);padding:3px 6px;border-radius:5px;line-height:1.4}
#model-sidebar {position:sticky;top:18px;align-self:flex-start}.model-heading {font-size:27px;font-weight:700;display:flex;justify-content:space-between;align-items:center;margin:14px 0}
.model-badge {font-size:12px;display:flex;gap:7px;align-items:center;padding:7px 12px;border-radius:20px;background:#e9f8f3;color:#159b80}
.model-badge i {width:7px;height:7px;background:currentColor;border-radius:50%}.model-badge.loading {background:#fff5de;color:#c79122}.loading i,.active {animation:pulse 1.5s infinite}.model-badge.error {color:#df4c60;background:#fff0f2}
.model-card p,.risk-card p {font-size:13px;color:#71868d;line-height:1.7}.engine-row {display:flex;justify-content:space-between;border-top:1px solid #eef3f4;padding-top:14px;margin-top:14px;font-size:12px;color:#71868d}.engine-row b {color:#31545e}
.risk-card {text-align:center}.risk-card .eyebrow {text-align:left}.risk-card svg {width:100%;max-width:360px;margin-top:26px;overflow:visible}.risk-card svg text {font-size:18px;font-weight:600;fill:#000}.risk-value {font-size:29px;font-weight:750;margin-top:-8px}.risk-rule {font-size:12px;line-height:1.9;color:#8ca0a6;background:#f6f9fa;border-radius:10px;padding:12px;margin-top:20px}
.risk-needle {transform-box:view-box;transform-origin:0px 0px;transform:rotate(var(--needle-angle));animation:risk-pointer-sweep var(--pointer-duration,1.4s) cubic-bezier(.22,.75,.25,1) both}
@keyframes risk-pointer-sweep {from {transform:rotate(0deg)}to {transform:rotate(var(--needle-angle))}}
.risk-ready .risk-value {animation:risk-result-in .5s ease-out .8s both}
.risk-ready {border-color:var(--risk-color);animation:risk-result-glow 1.8s ease-in-out infinite}
@keyframes risk-result-in {from {opacity:0;transform:translateY(5px)}to {opacity:1;transform:translateY(0)}}
@keyframes risk-result-glow {
    0%,100% {border-color:color-mix(in srgb,var(--risk-color) 35%,white);box-shadow:0 0 0 1px color-mix(in srgb,var(--risk-color) 15%,transparent),0 0 4px color-mix(in srgb,var(--risk-color) 12%,transparent)}
    50% {border-color:var(--risk-color);box-shadow:0 0 0 2px color-mix(in srgb,var(--risk-color) 70%,transparent),0 0 18px color-mix(in srgb,var(--risk-color) 50%,transparent),inset 0 0 8px color-mix(in srgb,var(--risk-color) 10%,transparent)}
}
.clip-card {border:1px solid #e0eaed!important;border-radius:16px!important;background:white!important;padding:12px!important;animation:receive .65s ease-out both;box-shadow:0 4px 14px #173c4705}
.clip-info {padding:9px 10px}.clip-info h3 {font-size:22px;margin:6px 0 10px}.clip-info p {font-size:12px;color:#789199;line-height:1.7}.clip-status {font-size:13px;color:#209f92}.risk-tag {font-size:12px;margin-left:12px}.probabilities {display:flex;gap:14px;font-size:11px;color:#789199;flex-wrap:wrap}.probabilities b {color:#31545e}
video {object-fit:contain!important;background:#101c24!important} footer {display:none!important}.error-message {color:#dc4c60!important}
#queue-note {color:#789199;font-size:13px;padding:8px} @keyframes receive {from {opacity:0}to {opacity:1}}
@keyframes stream {from {transform:translateX(-100%)}to {transform:translateX(100%)}} @keyframes pulse {50% {opacity:.45}}
.column-heading {padding:15px 12px;background:#eaf3f4;border-radius:12px;color:#173c47}.column-heading b {font-size:16px}.column-heading p {font-size:12px;color:#789199;margin:7px 0 0}
.clip-label {font-size:15px;font-weight:700;padding:5px 0 8px}.clip-label span {font-size:11px;color:#789199;font-weight:400;margin-left:8px}
.stage-detail {font-size:11px;color:#789199;line-height:1.6}.mini-chart svg,.prob-card svg {display:block;width:100%;height:auto}.mini-chart svg text,.prob-card svg text {font-size:11px;fill:#71868d}.prob-card svg .bar-value {fill:#31545e;font-weight:700;font-size:13px}
.chart-legend {display:flex;gap:9px;flex-wrap:wrap;font-size:9px;color:#8ca0a6;margin:3px 0}.chart-placeholder {min-height:100px;display:flex;align-items:center;justify-content:center;padding:16px;color:#000;background:#f6f9fa;border-radius:10px;font-size:12px;line-height:1.9;text-align:center;overflow-wrap:anywhere}
.prob-card {padding:12px 5px}.prob-result {font-size:17px;font-weight:700;margin:12px 0 3px}
@media(max-width:800px) {#model-sidebar {position:static}.progress-meta {font-size:10px}.clip-info {padding:4px}}
#visualization-rows {min-width:0!important}.six-clip-row {display:grid!important;grid-template-columns:repeat(6,minmax(0,1fr))!important;gap:5px!important;align-items:start!important}
.six-clip-row > .clip-card {min-width:0!important;width:100%!important;max-width:100%!important;padding:8px!important;gap:6px!important;box-sizing:border-box!important;overflow:hidden}
.square-video {width:100%!important;aspect-ratio:1/1!important;min-height:0!important;height:auto!important;padding:0!important;position:relative!important;overflow:hidden!important}
.square-video > div {height:100%!important;min-height:0!important}.square-video video {position:absolute!important;inset:0!important;width:100%!important;height:100%!important;object-fit:contain!important}
.clip-label {font-size:12px;padding:0 0 4px}.clip-label span {font-size:10px;margin-left:4px}.clip-status {font-size:11px}.stage-detail {font-size:10px}.chart-legend {font-size:8px;gap:4px}.column-heading {padding:10px 14px}.column-heading p {margin-top:3px}.prob-card {padding:5px 0}.prob-card .eyebrow {font-size:9px;letter-spacing:.6px}.prob-result {font-size:13px;margin:8px 0 0}.prob-card .chart-placeholder {font-size:10px;padding:8px;min-height:110px}
@media(max-width:1100px) {#dashboard-layout {flex-direction:column!important}#model-sidebar {position:static;width:100%!important;flex-direction:row!important}#model-sidebar > div {flex:1}#visualization-rows {width:100%!important}.six-clip-row {gap:5px!important}.six-clip-row > .clip-card {padding:5px!important}}
#visualization-rows {gap:8px!important}.stage-line {display:grid!important;grid-template-columns:32px minmax(0,1fr)!important;gap:6px!important;align-items:stretch!important;width:100%!important}
.stage-label {position:relative!important;min-width:0!important;min-height:0!important;padding:0!important;width:32px!important;align-self:stretch!important;overflow:visible!important;background:#eaf3f4!important;border-radius:8px!important}
.stage-label .stage-label {position:static!important;width:auto!important;background:transparent!important}
#processed-clips > .clip-card {gap:3px!important}
.stage-summary,.stage-chart {padding:0!important;margin:0!important;min-height:0!important}
.stage-summary .html-container,.stage-chart .html-container {padding:0!important}
.stage-summary .stage-detail {line-height:1.35;margin:1px 0 0}
.stage-chart .chart-legend {margin:0 0 1px;line-height:1.25}
.rotated-stage-title {position:absolute;left:50%;top:50%;transform:translate(-50%,-50%) rotate(-90deg);white-space:nowrap;font-size:13px;font-weight:700;color:#31545e}
#dashboard-layout {gap:16px!important}.six-clip-row {align-items:stretch!important}.clip-card {border-radius:10px!important}.clip-label {padding:0!important}.six-clip-row .block.html-container {padding:0!important}
#dashboard-layout {display:grid!important;grid-template-columns:minmax(0,1fr) clamp(300px,22vw,380px)!important;column-gap:14px!important;row-gap:16px!important;align-items:stretch!important;width:100%!important}
#visualization-rows,#model-sidebar {display:contents!important}
#visualization-rows > .stage-line {grid-column:1;min-width:0!important}
#breast-group-headings {grid-column:1;grid-row:1;padding:0!important;margin:0!important;min-width:0!important;min-height:0!important}
#breast-group-headings .html-container {padding:0!important}
.breast-group-headings {display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:5px;padding-left:38px}
.breast-group-headings span {grid-column:span 3;text-align:center;padding:3px 6px;line-height:18px;background:#eaf3f4;border:1px solid #dbe8eb;border-radius:8px;font-size:15px;font-weight:700;color:#31545e}
@media(max-width:1100px) {.breast-group-headings {gap:5px}}
#visualization-rows > .stage-line:nth-child(2) {grid-row:2}
#visualization-rows > .stage-line:nth-child(3) {grid-row:3}
#visualization-rows > .stage-line:nth-child(4) {grid-row:4;min-height:270px!important}
#model-status-panel {grid-column:2;grid-row:2}
#overall-risk-panel {grid-column:2;grid-row:3}
#model-status-panel,#overall-risk-panel {padding:0!important;margin:0!important;min-width:0!important;min-height:0!important;height:100%!important;align-self:stretch!important}
#model-status-panel .html-container,#overall-risk-panel .html-container,
#model-status-panel .gradio-style,#overall-risk-panel .gradio-style {padding:0!important;height:100%!important;box-sizing:border-box!important}
#model-status-panel .model-card,#overall-risk-panel .risk-card {height:100%;box-sizing:border-box;padding:18px;border:1px solid #b9d4d9;box-shadow:0 8px 26px #173c4714}
#model-status-panel .model-heading {font-size:22px;margin:10px 0;gap:8px;white-space:nowrap}
#model-status-panel .model-badge {display:inline-flex;padding:5px 9px;font-size:12px;gap:5px;flex-shrink:0}
.model-badge.inferring {background:#edf3ff;color:#2563eb}.model-badge.inferring i {animation:pulse 1.5s infinite}
#model-status-panel .model-card p {margin:6px 0;font-size:12px;line-height:1.5}
#model-status-panel .engine-row {padding-top:9px;margin-top:9px;font-size:14px;color:#49636b}
#model-status-panel .engine-row b {font-size:16px;color:#173c47}
#model-status-panel .eyebrow,#overall-risk-panel .eyebrow {font-size:13px;letter-spacing:1.5px;color:#31545e}
#model-status-panel .model-card {background:linear-gradient(145deg,#fff,#f0f9f8);border-top:3px solid #159b80}
#overall-risk-panel .risk-card svg {max-width:290px;margin-top:18px}
#overall-risk-panel .risk-rule {margin-top:10px}
/* Let the gauge use both result rows without stretching the detection row. */
#overall-risk-panel {grid-row:3 / span 2;height:auto!important;align-self:start!important}
#overall-risk-panel .html-container,#overall-risk-panel .gradio-style,
#overall-risk-panel .risk-card {height:auto!important}
/* Reserve identical slots while receiving, screening and displaying results. */
.stage-summary {height:auto!important;min-height:24px!important;flex:0 0 auto!important}
.stage-chart {height:84px!important;flex:0 0 84px!important}
.stage-summary .html-container,.stage-chart .html-container {box-sizing:border-box;height:100%}
.stage-chart .chart-placeholder {height:84px;min-height:0;padding:8px;box-sizing:border-box}
.probability-slot {height:160px!important;min-height:160px!important;flex:0 0 160px!important;padding:0!important}
.probability-slot .html-container {padding:0!important}
.clip-card {position:relative}
.clip-card.presentation-pending > :not(.presentation-wait) {visibility:hidden!important}
.presentation-wait {display:none!important}
.presentation-pending > .presentation-wait {display:flex!important;position:absolute;inset:8px;min-width:0;flex-direction:column;align-items:center;justify-content:center;gap:8px;padding:12px;box-sizing:border-box;border-radius:8px;background:linear-gradient(145deg,#f0f7f8,#f8fbfc);color:#31545e;text-align:center;overflow:hidden}
.wait-spinner {width:22px;height:22px;flex:none;box-sizing:border-box;border:2px solid #d5e7ea;border-top-color:#219eaa;border-radius:50%;animation:wait-spin 1.2s linear infinite}
.wait-label {font-size:11px;font-weight:600;line-height:1.5}
.wait-detail {font-size:10px;line-height:1.5;color:#789199}
.wait-stream {width:70%;max-width:110px;height:3px;background:#dcebed;border-radius:4px;overflow:hidden}
.wait-stream::after {content:'';display:block;width:45%;height:100%;border-radius:4px;background:#66b6bf;animation:wait-flow 1.6s ease-in-out infinite}
@keyframes wait-spin {to {transform:rotate(360deg)}}
@keyframes wait-flow {from {transform:translateX(-110%)}to {transform:translateX(330%)}}
.risk-card .risk-value {height:46px;line-height:46px;font-size:36px;text-align:center!important;width:100%;margin-left:auto;margin-right:auto}
#overall-risk-panel .risk-card svg {display:block;margin-left:auto;margin-right:auto}
.breast-risk-summary {display:grid;gap:8px;margin-top:10px}
.breast-risk-row {display:flex;justify-content:space-between;align-items:center;gap:6px;padding:11px 0;border-top:1px solid #dfeaec;font-size:15px;color:#173c47}
.breast-risk-row b {white-space:nowrap}
#model-status-panel .model-card p {height:36px;line-height:18px}
@media(prefers-reduced-motion:reduce) {* {animation:none!important;transition:none!important}}
"""


AUTOPLAY_JS = """
(() => {
    window.aibusVideoSync?.stop();
    const controller = new AbortController();
    const options = {capture: true, signal: controller.signal};
    const prepared = new WeakSet();
    const expectedSeeks = new WeakMap();
    const pairs = new Map();
    let stopped = false;
    let timer;
    let scanFrame;
    const play = video => {
        video.muted = video.defaultMuted = true;
        return video.play().catch(() => {});
    };
    const seek = (video, time) => {
        if (Math.abs(video.currentTime - time) < 0.015) return;
        expectedSeeks.set(video, time);
        video.currentTime = time;
    };
    const prepare = video => {
        video.muted = video.defaultMuted = true;
        if (prepared.has(video)) return;
        prepared.add(video);
        video.autoplay = video.playsInline = true;
        video.loop = false;
        video.preload = 'auto';
        video.removeAttribute('loop');
        for (const attr of ['muted', 'autoplay', 'playsinline']) {
            video.setAttribute(attr, '');
        }
        if (video.readyState >= 2) play(video);
    };
    const scan = () => {
        document.querySelectorAll('.square-video video').forEach(prepare);
        const rawCards = document.querySelectorAll('#raw-clips > .clip-card');
        const detectedCards = document.querySelectorAll('#processed-clips > .clip-card');
        rawCards.forEach((card, index) => {
            const raw = card.querySelector('video');
            const detected = detectedCards[index]?.querySelector('video');
            if (!raw || !detected) {
                pairs.delete(index);
                return;
            }
            const signature = raw.currentSrc + '|' + detected.currentSrc;
            const old = pairs.get(index);
            // Buffering/seeking must not discard a pair and restart its playback.
            if (raw.readyState < 2 || detected.readyState < 2) {
                if (old && (old.raw !== raw || old.detected !== detected || old.signature !== signature)) pairs.delete(index);
                return;
            }
            if (old?.raw === raw && old?.detected === detected && old.signature === signature) return;
            pairs.set(index, {raw, detected, signature, lastCorrection:0});
            seek(raw, 0);
            seek(detected, 0);
            detected.playbackRate = raw.playbackRate;
            play(raw);
            play(detected);
        });
    };
    const pairFor = video => [...pairs.values()].find(pair => pair.raw === video || pair.detected === video);
    const handle = event => {
        const video = event.target;
        if (!(video instanceof HTMLVideoElement) || !video.closest('.square-video')) return;
        if (['loadeddata', 'canplay', 'emptied'].includes(event.type)) {
            scan();
            return;
        }
        const pair = pairFor(video);
        if (!pair) return;
        const other = video === pair.raw ? pair.detected : pair.raw;
        if (event.type === 'ended' || (event.type === 'pause' && video.ended)) {
            other.pause();
            seek(other, other.duration);
        } else if (event.type === 'play') {
            if (other.paused) play(other);
        } else if (event.type === 'pause') {
            if (!other.paused) other.pause();
        } else if (event.type === 'seeking') {
            const expected = expectedSeeks.get(video);
            if (expected !== undefined && Math.abs(video.currentTime - expected) < 0.04) return;
            expectedSeeks.delete(video);
            seek(other, video.currentTime);
        } else if (event.type === 'seeked') {
            expectedSeeks.delete(video);
        } else if (event.type === 'ratechange') {
            if (video === pair.detected && pair.adjustingRate) {
                pair.adjustingRate = false;
                return;
            }
            if (other.playbackRate !== video.playbackRate) other.playbackRate = video.playbackRate;
        }
    };
    for (const event of ['loadeddata', 'canplay', 'emptied', 'play', 'pause', 'ended', 'seeking', 'seeked', 'ratechange']) {
        document.addEventListener(event, handle, options);
    }
    const observer = new MutationObserver(() => {
        if (scanFrame) return;
        scanFrame = requestAnimationFrame(() => {scanFrame = null; scan();});
    });
    observer.observe(document.body, {childList: true, subtree: true, attributes: true, attributeFilter: ['src']});
    const tick = () => {
        if (stopped) return;
        for (const pair of pairs.values()) {
            const {raw, detected} = pair;
            if (raw.readyState < 2 || detected.readyState < 2) continue;
            if (raw.ended || detected.ended) {
                raw.pause();
                detected.pause();
                continue;
            }
            if (raw.paused !== detected.paused) {
                const paused = raw.paused ? raw : detected;
                const playing = raw.paused ? detected : raw;
                seek(paused, playing.currentTime);
                play(paused);
            }
            if (raw.paused || detected.paused || raw.seeking || detected.seeking) continue;
            const drift = raw.currentTime - detected.currentTime;
            const now = performance.now();
            if (Math.abs(drift) > 0.25 && now - pair.lastCorrection > 800) {
                seek(detected, raw.currentTime);
                pair.lastCorrection = now;
            } else {
                // Smooth small differences instead of repeatedly flushing the decoder.
                const adjustment = Math.abs(drift) > 0.025 ? Math.max(-0.08, Math.min(0.08, drift * 0.5)) : 0;
                const rate = raw.playbackRate * (1 + adjustment);
                if (Math.abs(detected.playbackRate - rate) > 0.01) {
                    pair.adjustingRate = true;
                    detected.playbackRate = rate;
                }
            }
        }
    };
    window.aibusVideoSync = {
        stop() {stopped = true; clearInterval(timer); cancelAnimationFrame(scanFrame); observer.disconnect(); controller.abort();}
    };
    scan();
    tick();
    timer = setInterval(tick, 100);
})();
"""


# A later file may decode first; reveal only the contiguous ready prefix.
AUTOPLAY_JS += """
(() => {
    window.aibusPresentation?.stop();
    let frame;
    const scan = () => {
        const marker = document.querySelector('#batch-progress .progress-card');
        const raw = [...document.querySelectorAll('#raw-clips > .clip-card')];
        const detected = [...document.querySelectorAll('#processed-clips > .clip-card')];
        const probabilities = [...document.querySelectorAll('#probability-clips > .clip-card')];
        const reveal = (cards, requested, ready, waiting) => {
            let prefix = 0;
            for (let i = 0; i < Math.min(requested, cards.length); i++) {
                if (!ready(cards[i])) break;
                prefix++;
            }
            cards.forEach((card, i) => {
                let placeholder = card.querySelector(':scope > .presentation-wait');
                if (!placeholder) {
                    placeholder = document.createElement('div');
                    placeholder.className = 'presentation-wait';
                    placeholder.setAttribute('role', 'status');
                    placeholder.innerHTML = '<span class="wait-spinner" aria-hidden="true"></span><span class="wait-label"></span><span class="wait-detail"></span><span class="wait-stream" aria-hidden="true"></span>';
                    card.appendChild(placeholder);
                }
                const pending = i >= prefix;
                card.classList.toggle('presentation-pending', pending);
                placeholder.setAttribute('aria-hidden', String(!pending));
                if (pending) {
                    const [label, detail] = waiting(i);
                    const title = placeholder.querySelector('.wait-label');
                    const subtitle = placeholder.querySelector('.wait-detail');
                    if (title.textContent !== label) title.textContent = label;
                    if (subtitle.textContent !== detail) subtitle.textContent = detail;
                }
            });
            return prefix;
        };
        const videoReady = card => {
            const video = card.querySelector('video');
            return !!video?.getAttribute('src') && video.readyState >= 2;
        };
        const rawCount = Number(marker?.dataset.rawCount || 0);
        const detectedCount = Number(marker?.dataset.detectedCount || 0);
        const probabilityCount = Number(marker?.dataset.probabilityCount || 0);
        const visibleRaw = reveal(raw, rawCount, videoReady, i =>
            i < rawCount ? ['Receiving clip', 'Reading clip data…']
                         : ['Awaiting clip', 'Awaiting clip data from AIBUS-Edge…']);
        const visibleDetected = reveal(detected, visibleRaw === 6 ? detectedCount : 0, videoReady, i =>
            visibleRaw < 6 ? ['Detection & Classification', 'Awaiting clip reception…']
            : i < detectedCount ? ['Detection & Classification', 'Retrieving detection and classification results…']
                                : ['Detection & Classification', 'Awaiting detection and classification results…']);
        reveal(probabilities, visibleDetected === 6 ? probabilityCount : 0, () => true, () =>
            visibleDetected < 6 ? ['Risk Stratification', 'Awaiting detection and classification results…']
                               : ['Risk Stratification', 'Retrieving class probabilities…']);
    };
    const schedule = () => {
        if (frame) return;
        frame = requestAnimationFrame(() => {frame = null; scan();});
    };
    const observer = new MutationObserver(schedule);
    observer.observe(document.body, {childList:true, subtree:true, attributes:true,
        attributeFilter:['src','data-raw-count','data-detected-count','data-probability-count','data-presentation-id']});
    document.addEventListener('loadeddata', schedule, true);
    document.addEventListener('emptied', schedule, true);
    const timer = setInterval(scan, 100);
    window.aibusPresentation = {stop() {
        observer.disconnect(); clearInterval(timer); cancelAnimationFrame(frame);
        document.removeEventListener('loadeddata', schedule, true);
        document.removeEventListener('emptied', schedule, true);
    }};
    scan();
})();
"""


def build_demo():
    with gr.Blocks(title="AIBUS-Cloud · Clip 风险评估", fill_width=True) as demo:
        viewer = gr.State(value=lambda: uuid.uuid4().hex, delete_callback=release_viewer)
        progress = gr.HTML(progress_html(service.snapshot()), elem_id="batch-progress")
        raw_videos, processed_videos, details, charts, probabilities = [], [], [], [], []
        with gr.Row(equal_height=False, elem_id="dashboard-layout"):
            with gr.Column(scale=10, min_width=0, elem_id="visualization-rows"):
                gr.HTML('<div class="breast-group-headings"><span>Left Breast</span><span>Right Breast</span></div>',
                        elem_id="breast-group-headings")
                with gr.Row(elem_classes=["stage-line"]):
                    gr.HTML('<div class="rotated-stage-title">01 · Edge-side representative clip</div>', elem_classes=["stage-label"])
                    with gr.Row(elem_classes=["six-clip-row"], elem_id="raw-clips"):
                        for i in range(6):
                            with gr.Column(min_width=0, elem_classes=["clip-card"]):
                                raw_videos.append(gr.Video(
                                    label=f"Clip {i+1:02d} 原始视频", interactive=False,
                                    autoplay=True, loop=False, show_label=False,
                                    elem_classes=["square-video"]))
                with gr.Row(elem_classes=["stage-line"]):
                    gr.HTML('<div class="rotated-stage-title">02 · Edge-side Detection & Classification</div>', elem_classes=["stage-label"])
                    with gr.Row(elem_classes=["six-clip-row"], elem_id="processed-clips"):
                        for i in range(6):
                            with gr.Column(min_width=0, elem_classes=["clip-card"]):
                                processed_videos.append(gr.Video(
                                    label=f"Clip {i+1:02d} 检测分类视频", interactive=False,
                                    autoplay=True, loop=False, show_label=False,
                                    elem_classes=["square-video"]))
                                details.append(gr.HTML('<div class="stage-detail">accept</div>', elem_classes=["stage-summary"]))
                                charts.append(gr.HTML(stage1_chart_html(service.empty_clips[i]), elem_classes=["stage-chart"]))
                with gr.Row(elem_classes=["stage-line"]):
                    gr.HTML('<div class="rotated-stage-title">03 · Cloud-side Risk Stratification</div>', elem_classes=["stage-label"])
                    with gr.Row(elem_classes=["six-clip-row"], elem_id="probability-clips"):
                        for i in range(6):
                            with gr.Column(min_width=0, elem_classes=["clip-card"]):
                                probabilities.append(gr.HTML(probability_chart_html(i, service.empty_clips[i]), elem_classes=["probability-slot"]))
            with gr.Column(scale=2, min_width=230, elem_id="model-sidebar"):
                model = gr.HTML(model_html(service.snapshot()), elem_id="model-status-panel")
                with gr.Column(min_width=0, elem_id="risk-sidebar"):
                    gauge = gr.HTML(gauge_html(), elem_id="overall-risk-panel")
                    case_selector = gr.Dropdown(choices=list(CASES), value=DEFAULT_CASE,
                                                label="选择病例", interactive=True, elem_id="case-selector")
        outputs = [progress, model, gauge, *raw_videos, *processed_videos,
                   *details, *charts, *probabilities]

        def watch(case_name, session_id):
            token = uuid.uuid4().hex
            with viewer_tokens_lock:
                viewer_tokens[session_id] = token
            selected_service = get_case_service(case_name)
            # Explicitly clear old sources before starting a new presentation.
            empty = selected_service.snapshot()
            empty.update(clips=copy.deepcopy(selected_service.empty_clips), completed=0,
                         phase="Waiting for Clips from Edge-side", done=False, risk=None,
                         error=None, inferring=False, processed_frames=0, processing_seconds=0.0,
                         presentation_id=token)
            reset = {progress: progress_html(empty), model: model_html(empty), gauge: gauge_html()}
            for i, clip in enumerate(empty["clips"]):
                reset[raw_videos[i]] = None
                reset[processed_videos[i]] = None
                reset[details[i]] = '<div class="stage-detail">accept</div>'
                reset[charts[i]] = stage1_chart_html(clip)
                reset[probabilities[i]] = probability_chart_html(i, clip)
            yield reset
            time.sleep(0.2)
            with viewer_tokens_lock:
                if viewer_tokens.get(session_id) != token:
                    return
            caches = [ClipResultCache(path) for path in selected_service.clip_paths]
            s1_results = [cache.load_stage1() for cache in caches]
            predictions = [cache.load_prediction() for cache in caches]
            replay = None
            if all(s1 is not None and (s1["status"] == "stop" or pred is not None)
                   for s1, pred in zip(s1_results, predictions)):
                playback_paths = [playback_video_path(path) for path in selected_service.clip_paths]
                replay = selected_service.cached_display_states(playback_paths, s1_results, predictions)
            else:
                selected_service.start()
            previous = empty
            while True:
                with viewer_tokens_lock:
                    if viewer_tokens.get(session_id) != token:
                        return
                state = next(replay) if replay is not None else selected_service.snapshot()
                state["presentation_id"] = token
                if state != previous:
                    # Publish the final gauge first; 100% follows its pointer animation.
                    changes = {progress: progress_html(state, max_percent=99), model: model_html(state)}
                    if (previous is None
                            or (state["risk"], state["error"]) != (previous["risk"], previous["error"])
                            or any((clip["prediction"], clip["error"]) != (old["prediction"], old["error"])
                                   for clip, old in zip(state["clips"], previous["clips"]))):
                        gauge_clips = None if state.get("cache_presenting") and not state["done"] else state["clips"]
                        changes[gauge] = gauge_html(state["risk"], state["error"], gauge_clips)
                    for i, clip in enumerate(state["clips"]):
                        old = previous["clips"][i] if previous else None
                        if old is None or clip["video"] != old["video"]:
                            changes[raw_videos[i]] = clip["video"]
                        if old is None or clip["annotated_video"] != old["annotated_video"]:
                            changes[processed_videos[i]] = clip["annotated_video"]
                        if old != clip:
                            active = " active" if (clip["status"].endswith("中")
                                                  or clip["status"] in ("Rendering Edge-side Results", "AIBUS-Cloud Infering")) else ""
                            kind = status_kind(clip["status"], clip["error"])
                            detail_html = (f'<div class="stage-detail">{html.escape(clip["detail"])}</div>'
                                           if clip["detail"] else "")
                            changes[details[i]] = (f'<div class="clip-status state-badge state-{kind}{active}" style="{state_style(kind)}">{html.escape(clip["status"])}</div>'
                                                   + detail_html)
                            changes[probabilities[i]] = probability_chart_html(i, clip)
                        if old is None or clip["frame_results"] != old["frame_results"]:
                            changes[charts[i]] = stage1_chart_html(clip)
                    with viewer_tokens_lock:
                        if viewer_tokens.get(session_id) != token:
                            return
                    yield changes
                    previous = state
                if state["done"]:
                    if state["risk"] is not None and not state["error"]:
                        time.sleep(1.6)
                        with viewer_tokens_lock:
                            if viewer_tokens.get(session_id) != token:
                                return
                        yield {progress: progress_html(state)}
                    return
                if replay is None:
                    time.sleep(0.3)

        initial_watch = demo.load(watch, inputs=[case_selector, viewer], outputs=outputs,
                                  concurrency_limit=None, show_progress="hidden")
        case_selector.change(watch, inputs=[case_selector, viewer], outputs=outputs,
                             cancels=[initial_watch], concurrency_limit=None,
                             trigger_mode="multiple", show_progress="hidden")
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    threading.Thread(target=service.preload, name="model-preload", daemon=True).start()
    demo = build_demo()
    demo.queue().launch(server_name="0.0.0.0", server_port=9529, share=False,
                        allowed_paths=[str(CACHE_DIR), *{str(paths[0].parent) for paths in CASES.values()}], css=CSS,
                        theme=gr.themes.Soft(primary_hue="teal"), js=AUTOPLAY_JS, footer_links=[])
