import sys
import os
import random
from pathlib import Path
import gradio as gr
import pandas as pd

from normalizer.hebrew_tts_normalizer import load_word_replacements, normalize_tts_text, TTSNormalizeOptions, load_word_replacements_tsv
from datasets import load_from_disk, Dataset

# Setup Paths
CURRENT_DIR = Path(__file__).parent.absolute()
DATA_PATH = Path(os.getenv("DATA_PATH", str(CURRENT_DIR / "data" / "sample")))
# Load Resources
print(f"Loading dataset from {DATA_PATH}...")
try:
    if DATA_PATH.is_file():
        # Load single arrow file
        dataset = Dataset.from_file(str(DATA_PATH))
    else:
        # Load directory (HuggingFace disk format)
        dataset = load_from_disk(str(DATA_PATH))
        
    print(f"Dataset loaded. Size: {len(dataset)}")
except Exception as e:
    print(f"Failed to load dataset: {e}")
    # Fallback/Debug mode?
    dataset = []

# Load replacements
word_replacements = {}
word_replacements = load_word_replacements()

# Options configurations
OPTIONS_CONSERVATIVE = TTSNormalizeOptions(
    apply_word_replacements=False,
    expand_numbers=False,
    keep_punctuations=True,
    attach_punctuations_to_token=True
)

OPTIONS_EXPANDED = TTSNormalizeOptions(
    apply_word_replacements=True,
    expand_numbers=True,
    keep_punctuations=True,
    attach_punctuations_to_token=True,
    stt_compat_mode=False,
    remove_parentheses=False  # Keep parentheses content as per user preference
)

def get_sample(index):
    if not dataset or index < 0 or index >= len(dataset):
        return None, "Index out of bounds or dataset empty", "", "", ""
    
    item = dataset[int(index)]
    
    # Handle Audio
    audio_path = None
    
    # Check 'audio_path' column first (F5-TTS custom format)
    if 'audio_path' in item and item['audio_path']:
        audio_path = item['audio_path']
    # Check 'audio' column (HF format)
    elif 'audio' in item:
        audio_val = item['audio']
        if isinstance(audio_val, str):
            audio_path = audio_val
        elif isinstance(audio_val, dict) and 'path' in audio_val:
            audio_path = audio_val['path']

    # Resolve relative paths
    if audio_path and not os.path.isabs(audio_path):
        candidates = [
            CURRENT_DIR / audio_path,
            CURRENT_DIR / "data" / "CrowdRecital" / audio_path,
            Path(audio_path)
        ]
        found = False
        for c in candidates:
            if c.exists():
                audio_path = str(c)
                found = True
                break
        if not found:
            audio_path = None # File not found

    raw_text = item.get('raw_text', '') or item.get('text', '')
    
    # Normalize
    norm_conservative = normalize_tts_text(raw_text, options=OPTIONS_CONSERVATIVE, word_replacements=word_replacements)
    norm_expanded = normalize_tts_text(raw_text, options=OPTIONS_EXPANDED, word_replacements=word_replacements)
    
    return audio_path, raw_text, norm_conservative, norm_expanded, f"Sample {index} / {len(dataset)}"

def next_sample(idx):
    if not dataset: return (0, None, "", "", "", "")
    new_idx = min(idx + 1, len(dataset) - 1)
    return (new_idx,) + get_sample(new_idx)

def prev_sample(idx):
    if not dataset: return (0, None, "", "", "", "")
    new_idx = max(idx - 1, 0)
    return (new_idx,) + get_sample(new_idx)

def random_sample():
    if not dataset: return (0, None, "", "", "", "")
    new_idx = random.randint(0, len(dataset) - 1)
    return (new_idx,) + get_sample(new_idx)

def get_batch_view(start_idx, count=20):
    """Generate a CSV-like dataframe for a batch of items"""
    if not dataset:
        return pd.DataFrame(columns=["Index", "Error"])
        
    data = []
    end_idx = min(start_idx + count, len(dataset))
    
    for i in range(start_idx, end_idx):
        item = dataset[i]
        raw = item.get('text', '')
        norm = normalize_tts_text(raw, options=OPTIONS_EXPANDED, word_replacements=word_replacements)
        changed = "YES" if raw != norm else "no"
        data.append([i, raw, norm, changed])
        
    return pd.DataFrame(data, columns=["Index", "Raw Text", "Normalized (Expanded)", "Changed?"])

def process_text_only(text):
    if not text:
        return "", ""
    
    # Clean up surrounding quotes if user pasted a Python string
    text = text.strip()
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        text = text[1:-1]
        
    norm_cons = normalize_tts_text(text, options=OPTIONS_CONSERVATIVE, word_replacements=word_replacements)
    norm_exp = normalize_tts_text(text, options=OPTIONS_EXPANDED, word_replacements=word_replacements)
    return norm_cons, norm_exp

with gr.Blocks(title="Hebrew Normalizer Auditor") as demo:
    gr.Markdown("## 🎧 Hebrew TTS Normalizer Auditor")
    gr.Markdown("Review raw text vs normalized text against the actual audio.")
    
    with gr.Tabs():
        with gr.Tab("Dataset Audit"):
            current_index = gr.State(0)
            
            with gr.Row():
                with gr.Column(scale=1):
                    audio_player = gr.Audio(label="Original Audio", type="filepath")
                    sample_info = gr.Textbox(label="Info", interactive=False)
                    
                with gr.Column(scale=2):
                    txt_raw = gr.Textbox(label="Raw Text (from dataset)", interactive=False, lines=2)
                    txt_cons = gr.Textbox(label="Conservative Normalization (Safe)", interactive=False, lines=2)
                    txt_exp = gr.Textbox(label="Expanded Normalization (Numbers/Abbrev)", interactive=False, lines=2)
            
            with gr.Row():
                btn_prev = gr.Button("⬅️ Previous")
                btn_rand = gr.Button("🎲 Random")
                btn_next = gr.Button("➡️ Next")
            
            gr.Markdown("### 📋 Batch View (Click index to load above)")
            with gr.Row():
                num_start = gr.Number(label="Start Index", value=0, precision=0)
                btn_load_batch = gr.Button("Load Batch Table")
            
            table_view = gr.Dataframe(
                headers=["Index", "Raw Text", "Normalized (Expanded)", "Changed?"],
                interactive=False,
                wrap=True
            )

            # Actions
            btn_prev.click(prev_sample, [current_index], [current_index, audio_player, txt_raw, txt_cons, txt_exp, sample_info])
            btn_next.click(next_sample, [current_index], [current_index, audio_player, txt_raw, txt_cons, txt_exp, sample_info])
            btn_rand.click(random_sample, [], [current_index, audio_player, txt_raw, txt_cons, txt_exp, sample_info])
            
            # Load initial sample
            demo.load(get_sample, [current_index], [audio_player, txt_raw, txt_cons, txt_exp, sample_info])
            
            # Batch view
            btn_load_batch.click(get_batch_view, [num_start], table_view)
            
            # Add a jump box
            with gr.Row():
                jump_idx = gr.Number(label="Jump to Index", precision=0)
                btn_jump = gr.Button("Go")
            
            btn_jump.click(lambda x: (x,)+get_sample(x), [jump_idx], [current_index, audio_player, txt_raw, txt_cons, txt_exp, sample_info])

        with gr.Tab("Text Playground"):
            gr.Markdown("### ✍️ Test Normalizer on Custom Text")
            with gr.Row():
                input_text = gr.Textbox(label="Input Text", lines=3, placeholder="Enter Hebrew text here...")
                btn_process = gr.Button("Normalize", variant="primary")
            
            with gr.Row():
                out_cons = gr.Textbox(label="Conservative Normalization", lines=3, interactive=False)
                out_exp = gr.Textbox(label="Expanded Normalization", lines=3, interactive=False)
            
            btn_process.click(process_text_only, [input_text], [out_cons, out_exp])


if __name__ == "__main__":
    print("Starting Gradio app...")
    allowed_paths = [
        str(CURRENT_DIR.parent), # F5-TTS root (allows access to ../data)
        str(CURRENT_DIR / "data"), 
        "C:\\Users\\Asael\\PycharmProjects\\F5-TTS\\data" # Explicit fallback
    ]
    # Use 127.0.0.1 for Windows compatibility
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True, allowed_paths=allowed_paths)
