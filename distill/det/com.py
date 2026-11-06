"""Comparable foundation-model -> YOLO detection distillation.

Run from the repository root with python -m distill.det.com --help.
`all` means the eight /home/lx/baselines foundation models. V-JEPA and
`none` (supervised YOLO control) are optional additional experiments.
Each experiment runs in a fresh process, including Ultralytics-managed DDP.
"""
from __future__ import annotations

import argparse
from functools import partial
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import ultralytics
from torch import nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import init_seeds

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ("uscl", "ultrafedfm", "echocare", "usfm", "openus", "usf_mae", "ultrasam", "busgen")
CHANNELS = dict(zip(BASELINES, (512, 768, 2048, 768, 768, 768, 256, 256)))
CHANNELS.update(vjepa=1408, none=0)
DEFAULT_CKPTS = {
    "uscl": "ckpts/USCL.pth", "ultrafedfm": "ckpts/UltraFedFM.pth",
    "echocare": "ckpts/echocare_encoder.pth", "usfm": "ckpts/USFM_latest.pth",
    "openus": "ckpts/openus_cpt0150.pth",
    "usf_mae": "ckpts/USF-MAE_full_pretrain_43dataset_100epochs.pt",
    "ultrasam": "ckpts/UltraSam.pth",
    "busgen": "BUSGen/Checkpoints/Medical_BUSGen_6cls/ckpt_80_.pt",
}
# Teacher lives outside the student: excluded from optimizer, EMA and checkpoints.
_TEACHERS = {}


def configure_nms(backend):
    """Use Ultralytics' GPU PyTorch NMS when torchvision CUDA is incompatible."""
    if backend == "torch":
        import torchvision.ops
        from ultralytics.utils.nms import TorchNMS
        # Both AMP's temporary detector and validation access these entrypoints.
        torchvision.ops.nms = TorchNMS.nms
        torchvision.ops.boxes.nms = TorchNMS.nms


@contextmanager
def baseline_imports(root):
    """Disambiguate the two repositories' incompatible `buildmodel` modules."""
    old_path = sys.path[:]
    previous = sys.modules.pop("buildmodel", None)
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.modules.pop("buildmodel", None)
        if previous is not None:
            sys.modules["buildmodel"] = previous
        sys.path[:] = old_path


def baseline_module(root):
    key = "_det_com_baseline_model"
    if key not in sys.modules:
        path = Path(root) / "detection/model.py"
        spec = importlib.util.spec_from_file_location(key, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def teacher_config():
    value = os.environ.get("DET_COM_CONFIG")
    if not value:
        raise RuntimeError("Training needs DET_COM_CONFIG; launch using python -m distill.det.com")
    return json.loads(value)


def get_teacher(config, device):
    key = (config["teacher"], config["teacher_ckpt"], config["baseline_root"],
           config["teacher_res"], str(device))
    if key not in _TEACHERS:
        name = config["teacher"]
        if name == "vjepa":
            from distill.det.buildmodel import load_teacher
            teacher = load_teacher(config["teacher_ckpt"], resolution=config["teacher_res"], device=device)
        else:
            with baseline_imports(config["baseline_root"]):
                module = baseline_module(config["baseline_root"])
                teacher = module.build_backbone(
                    name, config["teacher_ckpt"], config["teacher_res"],
                    last_only=name in {"uscl", "echocare", "openus"},
                ).to(device)
                if name == "openus" and device.type == "cpu":
                    # Same scan math using the repository's PyTorch reference
                    # backend; Triton/selective-scan CUDA kernels need a GPU.
                    from vmamba_models import dino_vmamba, csm_triton
                    # Its wrappers enter torch.cuda.device even for CPU input.
                    dino_vmamba.cross_scan_fn = lambda x, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, **kw: csm_triton.CrossScanF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)
                    dino_vmamba.cross_merge_fn = lambda x, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, **kw: csm_triton.CrossMergeF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)
                    for layer in teacher.modules():
                        if hasattr(layer, "forward_corev2"):
                            layer.forward_core = partial(
                                layer.forward_corev2, selective_scan_backend="torch",
                                scan_force_torch=True,
                            )
        teacher.eval().requires_grad_(False)
        _TEACHERS[key] = teacher
    return _TEACHERS[key]


@torch.no_grad()
def teacher_features(config, images):
    teacher = get_teacher(config, images.device)
    # Match the existing detector/KD pipeline's ImageNet normalization.
    if config["teacher"] == "vjepa":
        from distill.det.buildmodel import extract_teacher_2d_features
        return torch.cat([extract_teacher_2d_features(
            teacher, chunk, teacher_res=config["teacher_res"]
        ) for chunk in images.split(config["teacher_batch"])])
    images = F.interpolate(images, (config["teacher_res"],) * 2,
                           mode="bilinear", align_corners=False)
    mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    # Chunk only the frozen teacher; student sees the same full training batch.
    chunks = images.split(config["teacher_batch"])
    features = [teacher((chunk - mean) / std) for chunk in chunks]
    if any(not isinstance(x, torch.Tensor) or x.ndim != 4 for x in features):
        raise RuntimeError("Teacher must return spatial BCHW features")
    return torch.cat(features)


class ComparisonDetectionModel(DetectionModel):
    """A normal YOLO with a registered, training-only feature projection."""

    def configure_kd(self, config):
        self.kd_config = dict(config)
        if config["teacher"] == "none":
            return
        # Probe before optimizer/DDP/EMA setup, preserving BN statistics.
        training = self.training
        self.eval()
        with torch.no_grad():
            x = torch.zeros(1, 3, 64, 64)
            for layer in self.model[:config["hook_layer"] + 1]:
                x = layer(x)
        self.train(training)
        # Separate RNG stream: teacher/adapter size cannot change student init.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["seed"])
            self.kd_adapter = nn.Conv2d(x.shape[1], CHANNELS[config["teacher"]], 1)

    def loss(self, batch, preds=None):
        config = getattr(self, "kd_config", {"teacher": "none"})
        if config["teacher"] == "none":
            return super().loss(batch, preds)
        if not self.training or preds is not None:
            loss, items = super().loss(batch, preds)
            return loss, torch.cat((items, items.new_zeros(2)))
        captured = []
        hook = self.model[config["hook_layer"]].register_forward_hook(
            lambda module, args, output: captured.append(output)
        )
        try:
            preds = self.predict(batch["img"])
        finally:
            hook.remove()
        loss, items = super().loss(batch, preds)
        student = self.kd_adapter(captured[0]).float()
        # Teacher is instantiated outside augmentation RNG stream as well.
        target = teacher_features(config, batch["img"]).float()
        target = F.interpolate(target, student.shape[-2:], mode="bilinear", align_corners=False)
        mse = F.mse_loss(student, target)
        cosine = (1 - F.cosine_similarity(student.flatten(1), target.flatten(1), dim=1)).mean()
        kd = config["alpha"] * mse + config["beta"] * cosine
        # Ultralytics sums its 3 batch-scaled components before backward.
        # Add KD once, regardless of whether that version returns scalar/vector.
        loss = loss + kd * len(batch["img"]) / loss.numel()
        return loss, torch.cat((items, torch.stack((mse.detach(), cosine.detach()))))


def load_lesion_weights(model, weights):
    """Preserve the pretrained lesion output when reducing a two-class head."""
    source = weights["model"] if isinstance(weights, dict) else weights
    if source.model[-1].nc not in (1, 2):
        raise ValueError("Lesion initialization requires a one/two-class checkpoint (class 0 = lesion)")
    model.load(weights)
    with torch.no_grad():
        for attr in ("cv3", "one2one_cv3"):
            if not hasattr(model.model[-1], attr):
                continue
            for dst, src in zip(getattr(model.model[-1], attr), getattr(source.model[-1], attr)):
                dst[-1].weight.copy_(src[-1].weight[:1])
                dst[-1].bias.copy_(src[-1].bias[:1])
    model.lesion_config = {"lesion_class": 0, "normal_class": 1,
                           "normal_training_labels": "removed; images retained as negatives"}


class ComparisonTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        configure_nms(teacher_config().get("nms_backend", "torchvision"))
        super().__init__(*args, **kwargs)
        # Keep this experiment's outputs local, independent of machine-wide
        # MLflow/W&B/TensorBoard integration settings. Also applies in DDP.
        for event, callbacks in self.callbacks.items():
            self.callbacks[event] = [callback for callback in callbacks
                if not callback.__module__.startswith("ultralytics.utils.callbacks.")
                or callback.__module__ == "ultralytics.utils.callbacks.base"]

    def get_model(self, cfg=None, weights=None, verbose=True):
        config = teacher_config()
        init_seeds(config["seed"], deterministic=True)
        model = ComparisonDetectionModel(cfg, nc=self.data["nc"],
                                        ch=self.data.get("channels", 3), verbose=verbose)
        if config.get("lesion_only") and self.data["nc"] != 1:
            raise ValueError("--lesion_only requires filtered single-class data; use distill.det.lesion prepare")
        if weights is not None:
            if config.get("lesion_only"):
                load_lesion_weights(model, weights)
            else:
                model.load(weights)
        model.configure_kd(config)
        return model

    def get_validator(self):
        validator = super().get_validator()
        if teacher_config()["teacher"] != "none":
            self.loss_names = (*self.loss_names, "kd_mse", "kd_cosine")
        return validator

    def _setup_train(self):
        super()._setup_train()
        config = teacher_config()
        if config["teacher"] != "none":
            # Loading before the first batch avoids consuming augmentation RNG.
            with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
                get_teacher(config, self.device)

    def export_student(self):
        """Portable detector without the training-only KD projection."""
        from copy import deepcopy
        model = deepcopy(self.ema.ema).cpu()
        for attribute in ("kd_adapter", "kd_config"):
            if hasattr(model, attribute):
                delattr(model, attribute)
        model.__class__ = DetectionModel
        model.criterion = None
        return model

    def save_epoch_inference(self):
        """Rank 0 saves one checkpoint + one inference JSON per epoch."""
        config = teacher_config()
        epoch = self.epoch + 1
        checkpoint = self.save_dir / f"epoch_{epoch:03d}.pt"
        report_path = self.save_dir / f"epoch_{epoch:03d}.json"
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({"model": self.export_student().half(), "epoch": self.epoch,
                    "train_args": vars(self.args)}, temporary)
        temporary.replace(checkpoint)
        command = [sys.executable, "-m", "distill.det.lesion", "infer",
                   "--model_path", str(checkpoint), "--data", config["eval_data"],
                   "--imgsz", str(self.args.imgsz), "--batch", str(config["eval_batch"]),
                   "--conf", str(config["eval_conf"]), "--iou", str(config["eval_iou"]),
                   "--device", "cpu" if self.device.type == "cpu" else "0",
                   "--nms_backend", config["nms_backend"],
                   "--output_json", str(report_path)]
        if not config.get("lesion_only"):
            command.append("--no-normal_completion")
        # This is ordinary single-GPU inference. Inherited torchrun ranks would
        # make Ultralytics initialize its dataset code as a distributed worker.
        environment = dict(os.environ)
        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR",
                    "MASTER_PORT", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"):
            environment.pop(key, None)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        result = json.loads(report_path.read_text())
        result.update(epoch=epoch, checkpoint=str(checkpoint),
                      training_metrics={str(k): float(v) for k, v in self.metrics.items()})
        selection = {"protocol": "strict_box_iou", "metric": "lesion_AP50_95", "mode": "max"}
        result["selection"] = selection
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result))
        temporary.replace(report_path)
        best_path = self.save_dir / "best.json"
        score = result["reports"][selection["protocol"]]["metrics"][selection["metric"]]
        previous = json.loads(best_path.read_text()) if best_path.exists() else None
        if previous is None or score > previous["reports"][selection["protocol"]]["metrics"][selection["metric"]]:
            temporary = best_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result))
            temporary.replace(best_path)
        print(f"Epoch {epoch} inference saved: {report_path}; lesion_AP50_95={score:.6f}", flush=True)

    def save_model(self):
        super().save_model()
        if self.best_fitness == self.fitness:
            torch.save({"model": self.export_student().float(), "train_args": vars(self.args)},
                       self.wdir / "best_student.pt")
        if teacher_config().get("epoch_inference", False):
            self.save_epoch_inference()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--teachers", nargs="+", default=["all"], choices=["all", *CHANNELS])
    p.add_argument("--baseline_root", type=Path, default=Path("/home/lx/baselines"))
    p.add_argument("--teacher_ckpt", type=Path, help="Override checkpoint for a single teacher")
    p.add_argument("--vjepa_ckpt", type=Path, default=ROOT / "ckpts/vjepa_full/best_vjepa_model9639(paper).pt")
    p.add_argument("--data", type=Path, default=Path("/home/lx/dataset/6cls_detection/data.yaml"))
    p.add_argument("--lesion_only", action="store_true", help="Use prepared class-0-only labels and transfer pretrained lesion classifier")
    p.add_argument("--epoch_inference", action=argparse.BooleanOptionalAction, default=None,
                   help="Save epoch_NNN.pt/json plus best.json; enabled by default for --lesion_only, optional for two-class training")
    p.add_argument("--eval_data", type=Path, help="Original two-class YAML; defaults to prepared dataset manifest source")
    p.add_argument("--eval_batch", type=int, default=16)
    p.add_argument("--eval_conf", type=float, default=.25)
    p.add_argument("--eval_iou", type=float, default=.7)
    p.add_argument("--model", default=str(ROOT / "yolo11m.pt"))
    p.add_argument("--output_dir", type=Path, default=ROOT / "distill/det/output")
    p.add_argument("--run_name", default="comparison")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=256)
    p.add_argument("--batch", type=int, default=128, help="Global batch across all GPUs")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--teacher_res", type=int, default=224)
    p.add_argument("--teacher_batch", type=int, default=4, help="Frozen teacher microbatch per GPU")
    p.add_argument("--hook_layer", type=int, default=6)
    p.add_argument("--alpha", type=float, default=1.)
    p.add_argument("--beta", type=float, default=.5)
    p.add_argument("--optimizer", choices=["AdamW", "SGD"], default="AdamW")
    p.add_argument("--lr0", type=float, default=.001)
    p.add_argument("--weight_decay", type=float, default=.0005)
    p.add_argument("--warmup_epochs", type=float, default=3.)
    p.add_argument("--warmup_bias_lr", type=float, default=.1)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--nms_backend", choices=["torchvision", "torch"], default="torchvision",
                   help="torch uses Ultralytics GPU PyTorch NMS, avoiding torchvision's compiled CUDA NMS")
    p.add_argument("--fraction", type=float, default=1., help="Training subset for smoke runs")
    p.add_argument("--dry_run", action="store_true", help="Print plans/check paths without loading models")
    p.add_argument("--smoke", action="store_true", help="CPU teacher forward + student backward, no dataset/training")
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def plans(args):
    names = list(dict.fromkeys(n for name in args.teachers for n in (BASELINES if name == "all" else [name])))
    if args.teacher_ckpt and (len(names) != 1 or names == ["none"]):
        raise ValueError("--teacher_ckpt requires one named teacher")
    if args.batch < 1 or args.teacher_batch < 1 or args.epochs < 1:
        raise ValueError("batch, teacher_batch and epochs must be positive")
    if args.teacher_res % 32 or args.imgsz % 32:
        raise ValueError("imgsz and teacher_res must be divisible by 32")
    if not 0 < args.fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    if args.hook_layer < 0 or args.hook_layer > 10:
        raise ValueError("hook_layer must be a YOLO11 backbone layer in [0,10]")
    if Path(args.run_name).name != args.run_name or args.run_name in {"", ".", ".."}:
        raise ValueError("run_name must be a single directory name")
    epoch_inference = args.lesion_only if args.epoch_inference is None else args.epoch_inference
    eval_data = args.eval_data
    if epoch_inference and not args.smoke:
        if eval_data is None and not args.lesion_only:
            eval_data = args.data
        if eval_data is None:
            manifest = args.data.parent / "manifest.json"
            if not manifest.is_file():
                raise ValueError("Pass --eval_data ORIGINAL_TWO_CLASS_YAML or use a prepared dataset with manifest.json")
            eval_data = Path(json.loads(manifest.read_text())["source"])
        if not eval_data.is_file():
            raise FileNotFoundError(eval_data)
        import yaml
        if len(yaml.safe_load(eval_data.read_text())["names"]) != 2:
            raise ValueError("--eval_data must be the original two-class evaluation YAML")
        if args.eval_batch < 1 or not 0 < args.eval_conf < 1 or not 0 < args.eval_iou <= 1:
            raise ValueError("Require eval_batch>0, 0<eval_conf<1, 0<eval_iou<=1")
    result = []
    for name in names:
        ckpt = (args.teacher_ckpt or (args.vjepa_ckpt if name == "vjepa" else
                args.baseline_root / DEFAULT_CKPTS[name])) if name != "none" else None
        if ckpt and not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        if name in BASELINES and not (args.baseline_root / "detection/model.py").is_file():
            raise FileNotFoundError(args.baseline_root / "detection/model.py")
        config = {"teacher": name, "teacher_ckpt": str(ckpt.resolve()) if ckpt else "",
                  "baseline_root": str(args.baseline_root.resolve()),
                  **{k: getattr(args, k) for k in ("teacher_res", "teacher_batch", "hook_layer", "alpha", "beta", "seed", "nms_backend", "lesion_only")}}
        config.update(epoch_inference=epoch_inference,
                      eval_data=str(eval_data.resolve()) if eval_data else None,
                      eval_batch=args.eval_batch, eval_conf=args.eval_conf, eval_iou=args.eval_iou)
        output = args.output_dir.resolve() / args.run_name / name
        if output.exists() and not args.dry_run and not args.smoke:
            raise FileExistsError(f"Refusing to overwrite experiment: {output}; choose a new --run_name")
        result.append((config, output))
    if not args.smoke and not args.data.is_file():
        raise FileNotFoundError(args.data)
    return result


def smoke(config, args):
    """Exercise real pretrained teacher, trainable projection, detector loss and export."""
    from copy import deepcopy
    from types import SimpleNamespace
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT
    torch.set_num_threads(2)
    init_seeds(args.seed)
    base = YOLO(args.model).model
    model = ComparisonDetectionModel(base.yaml, nc=1 if args.lesion_only else 2, verbose=False)
    if args.lesion_only:
        load_lesion_weights(model, base)
    else:
        model.load(base)
    model.configure_kd(config)
    model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    before = model.kd_adapter.weight.detach().clone() if hasattr(model, "kd_adapter") else None
    batch = {"img": torch.rand(2, 3, 64, 64), "batch_idx": torch.tensor([0., 1.]),
             "cls": torch.tensor([[0.], [0. if args.lesion_only else 1.]]), "bboxes": torch.tensor([[.5, .5, .3, .3]] * 2)}
    loss, items = model(batch)
    assert torch.isfinite(loss).all(), (loss, items)
    loss.sum().backward()
    optimizer.step()
    if before is not None:
        assert model.kd_adapter.weight.grad is not None
        assert not torch.equal(before, model.kd_adapter.weight)
        assert all(p.grad is None for t in _TEACHERS.values() for p in t.parameters())
    clone = deepcopy(model).eval()
    with torch.no_grad():
        clone(batch["img"])
        _, val_items = clone.loss(batch)
    assert val_items.numel() == items.numel()
    print(json.dumps({"teacher": config["teacher"], "smoke": "passed", "loss_items": items.tolist(),
                      "adapter_updated": before is not None}))


def main():
    args = parse_args()
    experiments = plans(args)
    if args.dry_run:
        for config, output in experiments:
            print(json.dumps({**config, "output": str(output), "training": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, ensure_ascii=False))
        return
    if not args._worker:
        # Fresh process per teacher prevents retained GPU allocations/import conflicts.
        for config, _ in experiments:
            command = [sys.executable, "-m", "distill.det.com"]
            argv = sys.argv[1:]
            command.extend(argv)
            command.extend(["--teachers", config["teacher"], "--_worker"])
            for option in ("data", "baseline_root", "output_dir", "vjepa_ckpt"):
                command.extend([f"--{option}", str(getattr(args, option).resolve())])
            if args.eval_data:
                command.extend(["--eval_data", str(args.eval_data.resolve())])
            if args.teacher_ckpt:
                command.extend(["--teacher_ckpt", str(args.teacher_ckpt.resolve())])
            if Path(args.model).is_file():
                command.extend(["--model", str(Path(args.model).resolve())])
            subprocess.run(command, cwd=ROOT, check=True)
        return
    config, output = experiments[0]
    os.environ["DET_COM_CONFIG"] = json.dumps(config)
    # Used by Ultralytics' generated DDP script and pickled model class.
    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    if args.smoke:
        smoke(config, args)
        return
    output.mkdir(parents=True)
    overrides = {
        "data": str(args.data.resolve()), "model": args.model, "epochs": args.epochs,
        "imgsz": args.imgsz, "batch": args.batch, "device": args.device, "workers": args.workers,
        "seed": args.seed, "deterministic": True, "project": str(output.parent),
        "name": output.name, "exist_ok": True, "optimizer": args.optimizer,
        "lr0": args.lr0, "lrf": .01, "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs, "warmup_bias_lr": args.warmup_bias_lr,
        "cos_lr": True, "patience": 0, "nbs": args.batch,
        "amp": args.amp, "save_period": -1, "fraction": args.fraction,
        "mosaic": 1., "close_mosaic": 10, "mixup": 0., "copy_paste": 0.,
        "hsv_h": .015, "hsv_s": .7, "hsv_v": .4, "degrees": 0.,
        "translate": .1, "scale": .5, "shear": 0., "perspective": 0.,
        "flipud": 0., "fliplr": .5, "multi_scale": 0., "compile": False,
        "val": True, "plots": False,
    }
    (output / "comparison.json").write_text(json.dumps({"teacher": config, "training": overrides,
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "kd_batch_scaled": True}, indent=2))
    trainer = ComparisonTrainer(overrides=overrides)
    trainer.train()


# Never pickle __main__ classes: both direct script and module execution work.
if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from distill.det.com import main as entrypoint
    entrypoint()
