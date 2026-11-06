"""Single-clip AIBUS-Cloud dashboard. Run with /opt/conda/bin/python app2.py."""
import logging
import threading
import time

from app1 import CASES, DEFAULT_CASE, CHECKPOINT_PATH, CSS, ROOT_DIR, gauge_html, gr

LOG = logging.getLogger("app2")
VIDEO_PATH = CASES[DEFAULT_CASE][0]


class SingleClipService:
    def __init__(self):
        self.lock = threading.Lock()
        self.started = False
        self.state = {"phase": "Loading Models", "risk": None, "error": None, "done": False}

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def start(self):
        with self.lock:
            if self.started:
                return
            self.started = True
        threading.Thread(target=self.run, name="single-clip-inference", daemon=True).start()

    def run(self):
        try:
            if not VIDEO_PATH.is_file():
                raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
            from pipeline_utils import PipelineRunner
            pipeline = PipelineRunner(CHECKPOINT_PATH)
            pipeline._ensure_stage3()
            LOG.info("Models ready: Stage 1=%s, YOLO=%s, AIBUS-Cloud=%s",
                     pipeline.stage1.device, pipeline.stage1.yolo_device, pipeline.stage3.device)
            self.update(phase="Edge-side Detection & Classification")
            result = pipeline.run_stage1(str(VIDEO_PATH))
            if result["status"] == "stop":
                risk = 0
            else:
                self.update(phase="AIBUS-Cloud Inferring")
                risk = pipeline.stage3.predict(str(VIDEO_PATH))["predicted_class"]
            self.update(phase="Risk Stratification Complete", risk=risk, done=True)
            LOG.info("%s classification complete: class_%s", VIDEO_PATH.name, risk)
        except Exception as exc:
            LOG.exception("Single clip inference failed")
            self.update(phase="Inference Failed", error=str(exc), done=True)


service = SingleClipService()

APP2_CSS = CSS + """
.gradio-container {max-width:720px!important;padding:24px!important}
#single-dashboard {gap:18px!important}
#single-video {width:100%!important;max-width:460px!important;aspect-ratio:1/1;margin:0 auto!important}
#single-video video {object-fit:contain!important}
#single-risk {padding:0!important;width:100%;max-width:460px;margin:0 auto!important}
#single-risk .html-container {padding:0!important}
#single-risk .risk-card {padding:24px;border:1px solid #b9d4d9;box-shadow:0 8px 26px #173c4714}
#single-risk .risk-card svg {display:block;max-width:310px;margin:14px auto 8px}
#single-risk .eyebrow {font-size:14px;color:#31545e}
#single-status {text-align:center;color:#31545e;font-size:14px;min-height:24px}
#single-risk .risk-ready .risk-value {animation-duration:1s;animation-delay:1.4s}
#single-risk .risk-ready {animation-duration:3s}
"""


AUTOPLAY_JS = """
(() => {
    window.aibusSingleVideoAutoplay?.abort();
    window.aibusSingleVideoObserver?.disconnect();
    const controller = new AbortController();
    window.aibusSingleVideoAutoplay = controller;
    const prepare = () => {
        const video = document.querySelector('#single-video video');
        if (!video) return;
        video.muted = video.defaultMuted = true;
        video.autoplay = video.loop = video.playsInline = true;
        for (const name of ['muted', 'autoplay', 'loop', 'playsinline']) {
            if (!video.hasAttribute(name)) video.setAttribute(name, '');
        }
        if (video.paused && video.readyState >= 2) video.play().catch(() => {});
    };
    const observer = new MutationObserver(prepare);
    observer.observe(document.body, {childList:true, subtree:true, attributes:true, attributeFilter:['src']});
    window.aibusSingleVideoObserver = observer;
    document.addEventListener('canplay', event => {
        if (event.target.matches?.('#single-video video')) prepare();
    }, {capture:true, signal:controller.signal});
    prepare();
})();
"""


def build_demo():
    with gr.Blocks(title="AIBUS-Cloud · Single Clip Risk", fill_width=True) as demo:
        with gr.Column(elem_id="single-dashboard"):
            gr.Video(value=str(VIDEO_PATH), show_label=False, interactive=False,
                     autoplay=True, loop=True, elem_id="single-video")
            gauge = gr.HTML(gauge_html(show_breasts=False, pointer_duration=2.4), elem_id="single-risk")
            status = gr.HTML("Loading Models", elem_id="single-status")

        def watch():
            service.start()
            yield gauge_html(show_breasts=False, pointer_duration=2.4), "AIBUS-Cloud Inferring"
            time.sleep(1)
            previous = None
            while True:
                state = service.snapshot()
                if state != previous:
                    yield gauge_html(state["risk"], state["error"], show_breasts=False, pointer_duration=2.4), state["phase"]
                    previous = state
                if state["done"]:
                    return
                time.sleep(0.3)

        demo.load(watch, outputs=[gauge, status], concurrency_limit=None, show_progress="hidden")
    return demo


if __name__ == "__main__":
    import os
    os.chdir(ROOT_DIR)
    logging.basicConfig(level=logging.INFO)
    build_demo().queue().launch(server_name="0.0.0.0", server_port=9528, share=False,
                               allowed_paths=[str(VIDEO_PATH.parent)], css=APP2_CSS,
                               theme=gr.themes.Soft(primary_hue="teal"), js=AUTOPLAY_JS, footer_links=[])
