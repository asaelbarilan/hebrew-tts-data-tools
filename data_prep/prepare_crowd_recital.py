import os
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datasets.arrow_writer import ArrowWriter
from importlib.resources import files
import io
from typing import Optional, List, Tuple, Dict
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence
import subprocess
import soundfile as sf


def is_valid_text_for_tts(text: str) -> bool:
    """
    Validate if text is suitable for TTS training.
    """
    if not text or len(text.strip()) < 5:
        return False
    
    # Check for LaTeX/math expressions
    latex_indicators = ['\\displaystyle', '\\frac', '\\sum', '\\int', '\\alpha', '\\beta', '\\gamma']
    if any(indicator in text for indicator in latex_indicators):
        return False
    
    # Check for excessive numbers or symbols
    text_chars = len(text)
    if text_chars == 0:
        return False
    
    digit_ratio = sum(c.isdigit() for c in text) / text_chars
    if digit_ratio > 0.3:
        return False
    
    # Check for repetitive patterns
    if len(set(text.split())) < 3:  # At least 3 unique words
        return False
    
    return True


def clean_transcript(transcript):
    """
    Remove any unwanted characters, LaTeX expressions, and filter out non-natural text.
    """
    # Remove LaTeX/math expressions
    latex_patterns = [
        r'\\displaystyle\s*\{[^}]*\}',  # \displaystyle{k}
        r'\\frac\{[^}]*\}\{[^}]*\}',    # \frac{a}{b}
        r'\\sum\s*\{[^}]*\}',           # \sum{k}
        r'\\[a-zA-Z]+\s*\{[^}]*\}',     # Any LaTeX command with braces
        r'\\[a-zA-Z]+',                 # Any LaTeX command
    ]
    
    cleaned = transcript
    for pattern in latex_patterns:
        cleaned = re.sub(pattern, '', cleaned)
    
    # Remove excessive punctuation and special characters
    cleaned = re.sub(r'[^\w\s\u0590-\u05FF\u2000-\u206F\u3000-\u303F\uFF00-\uFFEF.,!?;:()]', '', cleaned)
    
    # Remove extra whitespace
    cleaned = " ".join(cleaned.split())
    
    # Filter out very short or very long text
    if len(cleaned) < 5 or len(cleaned) > 500:
        return ""
    
    # Filter out text with too many numbers or special characters
    if sum(c.isdigit() for c in cleaned) > len(cleaned) * 0.3:
        return ""
    
    return cleaned


def load_metadata(session_dir: Path) -> Optional[dict]:
    """Load metadata from session directory."""
    metadata_path = session_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_path}")
        return None
        
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metadata from {metadata_path}: {str(e)}")
        return None


def split_audio_on_silence(audio: AudioSegment, min_silence_len: int = 500, silence_thresh: int = -40) -> List[Tuple[AudioSegment, int, int]]:
    """Split audio on silence and return segments with their start/end times."""
    # Detect silence
    silence_ranges = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    
    # Convert silence ranges to segment boundaries
    segments = []
    start = 0
    for silence_start, silence_end in silence_ranges:
        segment = audio[start:silence_start]
        if len(segment) > 400:  # Minimum 0.4 seconds
            segments.append((segment, start, silence_start))
        start = silence_end
    
    # Add the last segment
    if start < len(audio):
        segment = audio[start:]
        if len(segment) > 400:  # Minimum 0.4 seconds
            segments.append((segment, start, len(audio)))
    
    return segments


def align_text_with_segments(text: str, segments: List[Tuple[AudioSegment, int, int]]) -> List[Tuple[AudioSegment, str, int, int]]:
    """Align text with audio segments using simple word count ratio."""
    words = text.split()
    total_duration = sum(len(seg[0]) for seg in segments)
    
    aligned_segments = []
    current_word_idx = 0
    
    for segment, start_ms, end_ms in segments:
        segment_duration = len(segment)
        segment_ratio = segment_duration / total_duration
        num_words = max(1, int(len(words) * segment_ratio))
        
        segment_text = " ".join(words[current_word_idx:current_word_idx + num_words])
        current_word_idx += num_words
        
        aligned_segments.append((segment, segment_text, start_ms, end_ms))
    
    return aligned_segments


def load_aligned_transcript(session_dir: Path) -> Optional[dict]:
    """Load aligned transcript from session directory."""
    transcript_path = session_dir / "transcript.aligned.json"
    if not transcript_path.exists():
        print(f"Aligned transcript not found: {transcript_path}")
        return None
        
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading aligned transcript from {transcript_path}: {str(e)}")
        return None


def parse_vtt_time(time_str: str) -> float:
    """Convert VTT time format (HH:MM:SS.mmm) to seconds."""
    h, m, s = time_str.split(':')
    s, ms = s.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def load_vtt_transcript(session_dir: Path) -> Optional[List[Dict]]:
    """Load and parse VTT transcript from session directory."""
    vtt_path = session_dir / "transcript.vtt"
    if not vtt_path.exists():
        print(f"VTT transcript not found: {vtt_path}")
        return None
        
    try:
        with open(vtt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Parse VTT content
        segments = []
        current_segment = None
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('NOTE'):
                continue
                
            # Check for timestamp line
            if '-->' in line:
                if current_segment:
                    segments.append(current_segment)
                start_time, end_time = line.split('-->')
                current_segment = {
                    'start': parse_vtt_time(start_time.strip()),
                    'end': parse_vtt_time(end_time.strip()),
                    'text': ''
                }
            elif current_segment and line:
                # Append text to current segment
                current_segment['text'] += ' ' + line
                current_segment['text'] = current_segment['text'].strip()
                
        if current_segment:
            segments.append(current_segment)
            
        return segments
        
    except Exception as e:
        print(f"Error loading VTT transcript from {vtt_path}: {str(e)}")
        return None


def split_long_segment(audio: AudioSegment, start_ms: int, end_ms: int, text: str, 
                      per_segment_scores: List[dict]) -> List[Tuple[AudioSegment, int, int, str, float]]:
    """Split a long audio segment using silence detection and improved text alignment."""
    # Extract the audio segment
    segment = audio[start_ms:end_ms]
    duration = len(segment) / 1000  # Convert to seconds
    
    # If segment is too short, skip it entirely
    if duration < 4:
        return []
    
    # If segment is in good range, keep it as is
    if duration <= 30:
        quality_score = next(
            (score["probability"] for score in per_segment_scores
             if abs(score["start"] - start_ms/1000) < 0.1 and abs(score["end"] - end_ms/1000) < 0.1),
            0.0
        )
        return [(segment, start_ms, end_ms, text, quality_score)]
    
    # Use silence detection to find natural breaks in the audio
    silence_segments = split_audio_on_silence(segment, min_silence_len=300, silence_thresh=-35)
    
    # If we found good silence-based segments, use them
    if len(silence_segments) > 1:
        return _align_text_with_silence_segments(
            audio, start_ms, silence_segments, text, per_segment_scores
        )
    
    # Fallback to text-based splitting with improved logic
    return _split_by_text_breaks(audio, start_ms, end_ms, text, per_segment_scores)


def _align_text_with_silence_segments(audio: AudioSegment, start_ms: int, 
                                    silence_segments: List[Tuple[AudioSegment, int, int]], 
                                    text: str, per_segment_scores: List[dict]) -> List[Tuple[AudioSegment, int, int, str, float]]:
    """Align text with silence-detected segments using word-based distribution."""
    words = text.split()
    total_duration = sum(len(seg[0]) for seg in silence_segments)
    
    result = []
    current_word_idx = 0
    
    for segment, seg_start, seg_end in silence_segments:
        segment_duration = len(segment)
        segment_ratio = segment_duration / total_duration
        num_words = max(1, int(len(words) * segment_ratio))
        
        # Ensure we don't exceed available words
        if current_word_idx + num_words > len(words):
            num_words = len(words) - current_word_idx
        
        if num_words <= 0:
            continue
            
        segment_text = " ".join(words[current_word_idx:current_word_idx + num_words])
        current_word_idx += num_words
        
        # Calculate actual start/end times in original audio
        actual_start = start_ms + seg_start
        actual_end = start_ms + seg_end
        
        # Get quality score for this segment
        quality_score = next(
            (score["probability"] for score in per_segment_scores
             if abs(score["start"] - actual_start/1000) < 0.1 and abs(score["end"] - actual_end/1000) < 0.1),
            0.0
        )
        
        # Only add if duration is good and we have text
        if len(segment) / 1000 >= 4 and segment_text.strip():
            result.append((
                segment,
                actual_start,
                actual_end,
                segment_text,
                quality_score
            ))
    
    return result


def _split_by_text_breaks(audio: AudioSegment, start_ms: int, end_ms: int, text: str, 
                         per_segment_scores: List[dict]) -> List[Tuple[AudioSegment, int, int, str, float]]:
    """Split text using multiple break patterns for better segmentation."""
    # Define multiple break patterns for Hebrew text
    break_patterns = [
        r'[.!?]',      # Sentence endings
        r'[;:]',       # Semi-colon and colon
        r'[,،]',       # Comma (English and Arabic)
        r'\s+',        # Multiple spaces
    ]
    
    # Split text using the most appropriate break pattern
    segments = []
    for pattern in break_patterns:
        potential_segments = [s.strip() for s in re.split(pattern, text) if s.strip()]
        if len(potential_segments) > 1:
            segments = potential_segments
            break
    
    # If no good breaks found, split by words
    if len(segments) <= 1:
        words = text.split()
        # Try to create segments of roughly equal word count
        target_words_per_segment = max(1, len(words) // 3)  # Aim for 3 segments
        segments = []
        for i in range(0, len(words), target_words_per_segment):
            segment_words = words[i:i + target_words_per_segment]
            if segment_words:
                segments.append(" ".join(segment_words))
    
    if not segments:
        return []
    
    # Calculate timing using word-based estimation (more accurate than character-based)
    words = text.split()
    total_duration = (end_ms - start_ms) / 1000
    words_per_second = len(words) / total_duration if total_duration > 0 else 1
    
    result = []
    current_start = start_ms
    current_word_idx = 0
    
    for segment_text in segments:
        segment_words = segment_text.split()
        segment_word_count = len(segment_words)
        
        # Estimate duration for this segment
        estimated_duration = segment_word_count / words_per_second
        current_end = current_start + int(estimated_duration * 1000)
        
        # Ensure we don't exceed the original segment bounds
        if current_end > end_ms:
            current_end = end_ms
        
        # Get quality score for this segment
        quality_score = next(
            (score["probability"] for score in per_segment_scores
             if abs(score["start"] - current_start/1000) < 0.1 and abs(score["end"] - current_end/1000) < 0.1),
            0.0
        )
        
        # Extract audio segment
        segment_audio = audio[current_start:current_end]
        segment_duration = len(segment_audio) / 1000
        
        # Only add if duration is good and we have meaningful text
        if segment_duration >= 4 and len(segment_text.strip()) > 5:
            result.append((
                segment_audio,
                current_start,
                current_end,
                segment_text,
                quality_score
            ))
        
        current_start = current_end
        current_word_idx += segment_word_count
    
    return result


def process_session(session_dir: Path, save_dir: str) -> List[dict]:
    """Process a single session directory and return list of samples with duration and quality filtering. Concatenation is currently disabled."""
    try:
        metadata = load_metadata(session_dir)
        if not metadata:
            return []
            
        # Load both VTT and aligned transcripts
        vtt_segments = load_vtt_transcript(session_dir)
        aligned_transcript = load_aligned_transcript(session_dir)
        
        if not vtt_segments or not aligned_transcript:
            return []
            
        audio_path = session_dir / "audio.mka"
        if not audio_path.exists():
            print(f"Audio file not found: {audio_path}")
            return []
        
        quality_score = metadata.get("quality_score", 0.0)
        if quality_score < 0.65:
            print(f"Skipping due to low quality score: {quality_score}")
            return []
        
        audio = AudioSegment.from_file(str(audio_path))
        wav_dir = Path(save_dir) / "wavs"
        wav_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Extract all valid segments from VTT
        all_segments = []
        for i, vtt_segment in enumerate(vtt_segments):
            start_ms = int(vtt_segment['start'] * 1000)
            end_ms = int(vtt_segment['end'] * 1000)
            text = vtt_segment['text'].strip()
            
            if not text:
                continue
                
            # Clean and validate text
            cleaned_text = clean_transcript(text)
            if not cleaned_text or not is_valid_text_for_tts(cleaned_text):
                continue
                
            # Split long segments using natural text breaks
            segments = split_long_segment(
                audio, 
                start_ms, 
                end_ms, 
                cleaned_text,
                metadata.get("per_segment_quality_scores", [])
            )
            
            # Add segment info for processing
            for j, (segment_audio, seg_start_ms, seg_end_ms, seg_text, seg_quality) in enumerate(segments):
                # Only keep segments with valid duration and quality
                duration = len(segment_audio) / 1000
                if  4 <= duration <= 26: #seg_quality >= 0.65 and
                    all_segments.append({
                        'audio': segment_audio,
                        'text': seg_text,
                        'duration': duration,
                        'quality': seg_quality,
                        'original_start': seg_start_ms,
                        'original_end': seg_end_ms,
                        'vtt_start': vtt_segment['start'],
                        'vtt_end': vtt_segment['end'],
                        'segment_id': f"{session_dir.name}_seg{i}_{j}"
                    })
        
        samples = []
        # Export only valid segments as individual files
        for segment in all_segments:
            wav_path = wav_dir / f"{segment['segment_id']}.wav"
            segment['audio'].export(str(wav_path), format="wav")
            # Check with soundfile
            try:
                sf.info(str(wav_path))
            except Exception as e:
                print(f"[WARNING] soundfile could not open {wav_path}: {e}")
            sample = {
                "audio_path": str(wav_path),
                "text": segment['text'],
                "duration": segment['duration'],
                "original_path": str(audio_path),
                "start_ms": segment['original_start'],
                "end_ms": segment['original_end'],
                "quality_score": segment['quality'],
                "vtt_start": segment['vtt_start'],
                "vtt_end": segment['vtt_end'],
                "sample_type": "single"
            }
            samples.append(sample)
        
        # Concatenation logic is commented out for now to simplify and ensure only valid segments are included.
        # If you want to maximize data usage by combining short segments, uncomment the following block:
        # if short_segments:
        #     concatenated_samples = _concatenate_short_segments(
        #         short_segments, audio_path, wav_dir, session_name
        #     )
        #     samples.extend(concatenated_samples)
        
        # Long segment splitting is also not needed since we already filter by duration
        # for segment in long_segments:
        #     further_split = _split_very_long_segment(
        #         segment, audio_path, wav_dir
        #     )
        #     samples.extend(further_split)
        
        return samples
        
    except Exception as e:
        print(f"Error processing session {session_dir}: {str(e)}")
        return []


def _concatenate_short_segments(short_segments: List[dict], audio_path: Path, wav_dir: Path, session_name: str) -> List[dict]:
    """Concatenate short segments with silences to create samples in the 4-26 second range."""
    samples = []
    
    # Create silence segments of different lengths
    silence_05s = AudioSegment.silent(duration=500)  # 0.5 seconds
    silence_1s = AudioSegment.silent(duration=1000)  # 1 second
    
    current_group = []
    current_duration = 0
    group_id = 0
    
    for segment in short_segments:
        # Calculate total duration if we add this segment
        potential_duration = current_duration + segment['duration']
        
        # If adding this segment would exceed 26 seconds, finalize current group
        if potential_duration > 26 and current_group:
            if current_duration >= 4:  # Only create sample if group is long enough
                concatenated_sample = _create_concatenated_sample(
                    current_group, silence_05s, silence_1s, audio_path, wav_dir, 
                    session_name, group_id
                )
                samples.append(concatenated_sample)
                group_id += 1
            
            # Start new group
            current_group = [segment]
            current_duration = segment['duration']
        else:
            # Add to current group
            current_group.append(segment)
            current_duration = potential_duration
    
    # Process the last group
    if current_group and current_duration >= 4:
        concatenated_sample = _create_concatenated_sample(
            current_group, silence_05s, silence_1s, audio_path, wav_dir, 
            session_name, group_id
        )
        samples.append(concatenated_sample)
    
    return samples


def _create_concatenated_sample(segments: List[dict], silence_05s: AudioSegment, silence_1s: AudioSegment,
                               audio_path: Path, wav_dir: Path, session_name: str, group_id: int) -> dict:
    """Create a concatenated sample from multiple short segments with silences."""
    # Concatenate audio segments with silences
    concatenated_audio = AudioSegment.empty()
    concatenated_text = []
    
    for i, segment in enumerate(segments):
        # Add segment
        concatenated_audio += segment['audio']
        concatenated_text.append(segment['text'])
        
        # Add silence between segments (but not after the last one)
        if i < len(segments) - 1:
            # Use longer silence for longer gaps, shorter for shorter gaps
            if segment['duration'] > 2:
                concatenated_audio += silence_1s
            else:
                concatenated_audio += silence_05s
    
    # Save concatenated audio
    wav_path = wav_dir / f"{session_name}_concat_{group_id}.wav"
    concatenated_audio.export(str(wav_path), format="wav")
    
    # Calculate average quality score
    avg_quality = sum(seg['quality'] for seg in segments) / len(segments)
    
    return {
        "audio_path": str(wav_path),
        "text": " ".join(concatenated_text),
        "duration": len(concatenated_audio) / 1000,
        "original_path": str(audio_path),
        "start_ms": segments[0]['original_start'],
        "end_ms": segments[-1]['original_end'],
        "quality_score": avg_quality,
        "vtt_start": segments[0]['vtt_start'],
        "vtt_end": segments[-1]['vtt_end'],
        "sample_type": "concatenated",
        "num_segments": len(segments)
    }


def _split_very_long_segment(segment: dict, audio_path: Path, wav_dir: Path) -> List[dict]:
    """Further split very long segments that somehow got through."""
    samples = []
    
    # Split into chunks of ~20 seconds with overlap
    chunk_duration = 20  # seconds
    overlap = 2  # seconds
    
    audio = segment['audio']
    total_duration = len(audio) / 1000
    
    if total_duration <= 26:
        # If somehow it's now in range, just save it
        wav_path = wav_dir / f"{segment['segment_id']}_fixed.wav"
        audio.export(str(wav_path), format="wav")
        
        sample = {
            "audio_path": str(wav_path),
            "text": segment['text'],
            "duration": segment['duration'],
            "original_path": str(audio_path),
            "start_ms": segment['original_start'],
            "end_ms": segment['original_end'],
            "quality_score": segment['quality'],
            "vtt_start": segment['vtt_start'],
            "vtt_end": segment['vtt_end'],
            "sample_type": "long_split"
        }
        samples.append(sample)
    else:
        # Split into overlapping chunks
        start_ms = 0
        chunk_id = 0
        
        while start_ms < len(audio):
            end_ms = min(start_ms + int(chunk_duration * 1000), len(audio))
            chunk_audio = audio[start_ms:end_ms]
            chunk_duration_sec = len(chunk_audio) / 1000
            
            if chunk_duration_sec >= 4:  # Only save if long enough
                wav_path = wav_dir / f"{segment['segment_id']}_chunk{chunk_id}.wav"
                chunk_audio.export(str(wav_path), format="wav")
                
                # Estimate text portion (rough approximation)
                text_ratio = chunk_duration_sec / total_duration
                words = segment['text'].split()
                start_word = int(chunk_id * len(words) * (chunk_duration / total_duration))
                end_word = min(start_word + int(len(words) * text_ratio), len(words))
                chunk_text = " ".join(words[start_word:end_word])
                
                sample = {
                    "audio_path": str(wav_path),
                    "text": chunk_text,
                    "duration": chunk_duration_sec,
                    "original_path": str(audio_path),
                    "start_ms": segment['original_start'] + start_ms,
                    "end_ms": segment['original_start'] + end_ms,
                    "quality_score": segment['quality'],
                    "vtt_start": segment['vtt_start'],
                    "vtt_end": segment['vtt_end'],
                    "sample_type": "long_split"
                }
                samples.append(sample)
            
            start_ms = end_ms - int(overlap * 1000)  # Overlap
            chunk_id += 1
    
    return samples


def main():
    result = []
    duration_list = []
    vocab_set = set()

    # Convert all .mka files to .wav in the dataset directory (recursively)
    print("Converting .mka files to .wav (if needed)...")
    for root, dirs, files in os.walk(dataset_dir_path):
        for file in files:
            if file.endswith('.mka'):
                mka_path = os.path.join(root, file)
                wav_path = os.path.splitext(mka_path)[0] + '.wav'
                if not os.path.exists(wav_path):
                    print(f"Converting {mka_path} -> {wav_path}")
                    try:
                        subprocess.run([
                            'ffmpeg', '-y', '-i', mka_path, wav_path
                        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    except Exception as e:
                        print(f"Error converting {mka_path}: {e}")

    # Set the path to the top-level dataset folder
    dataset_dir = Path(dataset_dir_path)
    session_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    print(f"Found {len(session_dirs)} recording sessions.")

    # Process sessions in parallel
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = [executor.submit(process_session, session_dir, save_dir) for session_dir in session_dirs]

    for future in tqdm(futures, total=len(futures), desc="Aggregating sessions"):
        samples = future.result()
        for sample in samples:
            result.append(sample)
            duration_list.append(sample["duration"])
            vocab_set.update(list(sample["text"]))

    executor.shutdown()

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"\nSaving processed dataset to {save_dir} ...")

    # Save samples to an Arrow file
    arrow_path = Path(save_dir) / "raw.arrow"
    with ArrowWriter(path=str(arrow_path)) as writer:
        for sample in tqdm(result, desc="Writing to raw.arrow"):
            writer.write(sample)

    # Save durations to a JSON file
    duration_json_path = Path(save_dir) / "duration.json"
    with open(duration_json_path, "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # Save vocabulary to a text file (one character per line)
    vocab_txt_path = Path(save_dir) / "vocab.txt"
    with open(vocab_txt_path, "w", encoding="utf-8") as f:
        for char in sorted(vocab_set):
            f.write(char + "\n")

    print(f"\nFor {dataset_name}:")
    print(f"  Sample count: {len(result)}")
    print(f"  Vocabulary size: {len(vocab_set)}")
    print(f"  Total duration: {sum(duration_list) / 3600:.2f} hours")


if __name__ == "__main__":
    max_workers = 8  # Adjust based on your CPU capabilities

    # Name for the dataset (used for naming the output directory)
    # Fix: Use base dataset name without tokenizer suffix for consistency
    dataset_name = "CrowdRecital"  # Remove tokenizer suffix

    # Set the path to your dataset folder (where each session folder is stored)
    dataset_dir_path = os.getenv("DATASET_DIR", r"C:\Users\Asael\PycharmProjects\crowd-recital")

    # Define where to save the preprocessed data.
    from importlib.resources import files

    # Fix: Use consistent path construction
    save_dir = os.path.abspath(os.path.join("data", dataset_name))
    
    print(f"\nPreparing dataset {dataset_name}, saving to {save_dir}...\n")
    main()
