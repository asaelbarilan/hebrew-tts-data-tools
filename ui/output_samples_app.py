import argparse
import io
import json
import random
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
from datasets import Dataset, DatasetDict, load_from_disk


def load_dataset_any(path: str):
    ds_path = Path(path)
    if ds_path.is_file():
        return Dataset.from_file(str(ds_path))
    if not ds_path.exists():
        raise FileNotFoundError(f"Path not found: {ds_path}")

    # HF save_to_disk format
    try:
        return load_from_disk(str(ds_path))
    except Exception:
        pass

    # Common alternative: directory containing raw.arrow
    raw_arrow = ds_path / "raw.arrow"
    if raw_arrow.exists():
        return Dataset.from_file(str(raw_arrow))

    # Another common alternative: save_to_disk under a "raw" subdir
    raw_subdir = ds_path / "raw"
    if raw_subdir.exists():
        return load_from_disk(str(raw_subdir))

    raise FileNotFoundError(
        f"Directory {ds_path} is neither a Dataset/DatasetDict directory nor contains raw.arrow/raw/"
    )


def pick_split(data_obj, split_name: str | None):
    if isinstance(data_obj, DatasetDict):
        if split_name and split_name in data_obj:
            return data_obj[split_name], split_name
        first_split = next(iter(data_obj.keys()))
        return data_obj[first_split], first_split
    return data_obj, "train"


def decode_audio(item):
    audio = item.get("audio")
    if not audio:
        return None
    if isinstance(audio, dict):
        audio_bytes = audio.get("bytes")
        if audio_bytes:
            arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            return (sr, arr)
        audio_path = audio.get("path")
        if audio_path and Path(audio_path).exists():
            arr, sr = sf.read(audio_path, dtype="float32")
            return (sr, arr)
    return None


def build_app(dataset, split_name: str):
    total = len(dataset)

    def get_item(idx: int):
        if total == 0:
            return None, "Dataset is empty", "", "", "{}"
        idx = max(0, min(int(idx), total - 1))
        item = dataset[idx]
        audio = decode_audio(item)
        text = item.get("text", "")
        raw_text = item.get("raw_text", "")
        metadata = json.dumps(item.get("metadata", {}), ensure_ascii=False, indent=2)
        info = f"Split: {split_name} | Sample {idx + 1}/{total}"
        return audio, info, text, raw_text, metadata

    def next_item(idx):
        return (min(int(idx) + 1, total - 1),) + get_item(min(int(idx) + 1, total - 1))

    def prev_item(idx):
        return (max(int(idx) - 1, 0),) + get_item(max(int(idx) - 1, 0))

    def random_item():
        if total == 0:
            return (0,) + get_item(0)
        idx = random.randint(0, total - 1)
        return (idx,) + get_item(idx)

    with gr.Blocks(title="Prepared Dataset Sample Player") as demo:
        gr.Markdown("## Prepared Dataset Sample Player")
        gr.Markdown("Listen to generated samples and inspect text/metadata.")

        current_idx = gr.State(0)
        with gr.Row():
            audio = gr.Audio(label="Audio", type="numpy")
            with gr.Column():
                info = gr.Textbox(label="Info", interactive=False)
                text = gr.Textbox(label="Text", interactive=False, lines=3)
                raw_text = gr.Textbox(label="Raw Text", interactive=False, lines=3)
        metadata = gr.Code(label="Metadata", language="json")

        with gr.Row():
            btn_prev = gr.Button("Previous")
            btn_rand = gr.Button("Random")
            btn_next = gr.Button("Next")

        with gr.Row():
            jump = gr.Number(label="Jump to index", value=0, precision=0)
            btn_jump = gr.Button("Go")

        demo.load(get_item, [current_idx], [audio, info, text, raw_text, metadata])
        btn_prev.click(prev_item, [current_idx], [current_idx, audio, info, text, raw_text, metadata])
        btn_next.click(next_item, [current_idx], [current_idx, audio, info, text, raw_text, metadata])
        btn_rand.click(random_item, [], [current_idx, audio, info, text, raw_text, metadata])
        btn_jump.click(
            lambda x: (int(x),) + get_item(int(x)),
            [jump],
            [current_idx, audio, info, text, raw_text, metadata],
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UI player for prepared dataset samples")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/CrowdRecital_ivritai",
        help="Path to prepared dataset (save_to_disk folder or .arrow file)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split name if loading a DatasetDict",
    )
    args = parser.parse_args()

    ds_obj = load_dataset_any(args.data_path)
    ds, used_split = pick_split(ds_obj, args.split)
    app = build_app(ds, used_split)
    app.launch(server_name="127.0.0.1", server_port=7861, share=False)
