from dataclasses import dataclass
import json
from random import shuffle, uniform
import logging
from pathlib import Path
from typing import Iterator
import uuid
import statistics

from tqdm import tqdm
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder
from torchcodec import AudioSamples
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    Audio as AudioColumnType,
    Value as ValueColumnType,
    Features,
)
from huggingface_hub import DatasetCard, DatasetCardData, upload_file
from f5_tts.train.datasets.heb_norm.hebrew_tts_normalizer import (
    normalize_tts_text,
    load_word_replacements,
    TTSNormalizeOptions,
)


@dataclass
class SegmentWords:
    start: float
    end: float
    word: str
    probability: float
    tokens: list[int]


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[SegmentWords | dict]
    seek: float | None = 0.0
    tokens: list[int] = None
    temperature: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None

    def has_words(self) -> bool:
        return bool(self.words and len(self.words) > 0)

    def __post_init__(self):
        self.words = [
            SegmentWords(**w) if isinstance(w, dict) else w for w in self.words
        ]


logger = logging.getLogger(__name__)


def _load_data_manifest(
    input_folder: Path,
    audio_filename_glob: str,
    segments_glob: str,
    metadata_glob: str,
):
    segments_files = list(input_folder.glob(segments_glob))
    audio_files = []
    metadata_files = []
    for segments_file in segments_files:
        # find the audio file that has matches the glob and within the same directory
        search_within_folder = segments_file.parent
        found_audio_files = list(search_within_folder.glob(audio_filename_glob))
        # expect only one audio file
        assert (
            len(found_audio_files) == 1
        ), f"Expected 1 audio file, found {len(found_audio_files)} for {segments_file} (taking first)"
        audio_files.extend(found_audio_files[:1])
        # expect only one metadata file
        found_metadata_files = list(search_within_folder.glob(metadata_glob))
        assert (
            len(found_metadata_files) == 1
        ), f"Expected 1 metadata file, found {len(found_metadata_files)} for {segments_file} (taking first)"
        metadata_files.extend(found_metadata_files[:1])
    return list(zip(audio_files, segments_files, metadata_files))


from subprocess import CalledProcessError, run, PIPE, DEVNULL

import numpy as np

TARGET_SAMPLE_RATE = 24_000


def load_audio_in_target_format(file: str, sr: int = TARGET_SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary
    Shamelessly stolen from https://github.com/openai/whisper/blob/main/whisper/audio.py
    Thanks OpenAI :)

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def get_segment_word_scores(segment: Segment) -> list[float]:
    """
    Get the word scores for a segment.
    This is a helper function to extract the word scores from a segment.
    """
    if not segment.has_words:
        return []

    # Extract word scores from the segment
    word_scores = []
    for word in segment.words:
        if hasattr(word, "probability"):
            word_scores.append(word.probability)
    return word_scores


def calculate_median_quality_score(scores: list[float]) -> float:
    """
    Calculate the median quality score for a list of scores.
    This is a helper function to calculate the median quality score for a list of scores.
    """
    # Calculate the median probability of all words in the segment
    quality_score = float(np.median(scores)) if scores else 0.0
    return quality_score


def calculate_segments_quality_score(segments: list[Segment]) -> float:
    if not segments:
        return 0.0

    """Calculate the quality score based on the median word probabilities for a single segment."""
    try:
        all_word_probs = []
        for segment in segments:
            all_word_probs.extend(get_segment_word_scores(segment))
        # Calculate the median probability of all words in the segment
        quality_score = calculate_median_quality_score(all_word_probs)
        return quality_score

    except Exception:
        return 0.0


class DurationController:
    """
    Adaptive controller for managing slice durations to achieve target average/median duration.
    Each process maintains its own controller instance to adapt based on local statistics.
    """

    def __init__(
        self, min_duration: float, max_duration: float, target_duration: float
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.target_duration = target_duration
        self.completed_durations: list[float] = []
        self.rejected_attempts = 0

    def get_next_target_duration(self) -> float:
        """
        Adaptively calculate the next slice target duration based on running statistics.

        Returns:
            Target duration for the next slice in seconds
        """
        # Bootstrap phase: use random durations around target initially
        if len(self.completed_durations) < 5:
            return uniform(
                max(self.min_duration, self.target_duration * 0.7),
                min(self.max_duration, self.target_duration * 1.3),
            )

        # Calculate current average
        current_avg = statistics.mean(self.completed_durations)

        # Adaptive adjustment: compensate for deviation from target
        avg_error = self.target_duration - current_avg

        # Dampening factor to reduce oscillation (don't overcorrect)
        dampening = 0.5
        adjusted_target = self.target_duration + (avg_error * dampening)

        # Add controlled randomness to maintain distribution variance
        variance_factor = 0.35  # 35% variance around target
        variance = self.target_duration * variance_factor
        next_target = uniform(adjusted_target - variance, adjusted_target + variance)

        # Clamp to valid range
        return max(self.min_duration, min(self.max_duration, next_target))

    def record_completed_slice(self, duration: float):
        """Record a successfully completed slice duration."""
        self.completed_durations.append(duration)

    def record_rejected_slice(self):
        """Record when a slice is rejected due to quality filtering."""
        self.completed_durations.pop()


def generate_slices(
    input_segments: list[Segment],
    audio_duration: float,
    max_duration: float,
    per_segment_quality_threshold: float = 0,
    duration_controller: DurationController = None,
):
    next_slice_start = 0
    curr_input_segment_idx = 0
    slices = []
    while next_slice_start < audio_duration:
        slice_start = next_slice_start

        # Ensure current segment exists
        # and validate it's duration.
        if curr_input_segment_idx < len(input_segments):
            curr_input_segment_duration = (
                input_segments[curr_input_segment_idx].end
                - input_segments[curr_input_segment_idx].start
            )
            # If the first segment to work on is too long for a single slice or of 0 length we must skip it.
            if (
                curr_input_segment_duration > max_duration
                or curr_input_segment_duration == 0
            ):
                # skip if any segment ahead
                if curr_input_segment_idx + 1 < len(input_segments):
                    next_slice_start = input_segments[curr_input_segment_idx + 1].start
                    curr_input_segment_idx += 1
                # or break since nothing more to work on
                else:
                    next_slice_start = audio_duration

                continue

        curr_slice_source_segment_idxs = []
        curr_slice_source_segments = []
        curr_slice_segments = []
        curr_slice = {
            "segments": curr_slice_segments,
            "seek": slice_start,
            "duration": 0.0,
        }
        slices.append(curr_slice)

        # Determine target duration for this slice
        if duration_controller:
            current_slice_target = duration_controller.get_next_target_duration()
        else:
            # Fallback to max_duration (original behavior)
            current_slice_target = max_duration

        # normal slice length is the expected slice hop - but this could be overridden below. See comments.
        next_slice_start = slice_start + current_slice_target
        # clip the slice end to the total audio duration
        slice_end = min(next_slice_start, audio_duration)

        # While more segments to work on and the current segment start is within the slice
        while (
            curr_input_segment_idx < len(input_segments)
            and input_segments[curr_input_segment_idx].start < slice_end
        ):
            curr_input_segment = input_segments[curr_input_segment_idx]

            # track the source segments used in this slice for quality analysis after slice completion
            curr_slice_source_segments.append(curr_input_segment)
            curr_slice_source_segment_idxs.append(curr_input_segment_idx)

            # Add it to the slice (start is always relative)
            slice_segment = {"start": max(0, curr_input_segment.start - slice_start)}
            curr_slice_segments.append(slice_segment)

            # Clip the segment end to the entire audio duration (prevent tiny overflow)
            curr_input_segment_end = min(curr_input_segment.end, audio_duration)

            # If this input segment fully fits within the slice, include it.
            if curr_input_segment_end <= slice_end:
                slice_segment["end"] = min(
                    max_duration, curr_input_segment_end - slice_start
                )  # relative to slice
                slice_segment["text"] = curr_input_segment.text
                slice_segment["word_scores"] = get_segment_word_scores(
                    curr_input_segment
                )
                # entire segment is included - advance to next input segment
                curr_input_segment_idx += 1
            else:
                # The segment does NOT fully fit in this slice. We do NOT include partial segments.
                # Remove the last (partial) entries we added for this segment.
                curr_slice_source_segments.pop()
                curr_slice_source_segment_idxs.pop()
                curr_slice_segments.pop()

                # Calculate the actual duration of this slice (from slice_start to end of last included segment)
                if curr_slice_source_segment_idxs:
                    last_segment_end = min(
                        input_segments[curr_slice_source_segment_idxs[-1]].end,
                        audio_duration,
                    )
                    curr_slice["duration"] = last_segment_end - slice_start
                else:
                    # No segments in this slice, duration is 0
                    curr_slice["duration"] = 0.0

                # Start the next slice at the start of the segment that did not fit.
                # Do not advance curr_input_segment_idx so the same segment will be considered
                # at the beginning of the next slice.
                next_slice_start = curr_input_segment.start

                # Close this slice (we won't try to include this partial segment here)
                break

        # If we didn't break out of the segment loop, calculate the actual slice duration
        # (slice was closed due to reaching slice_end or end of segments)
        if curr_slice["duration"] == 0.0 and curr_slice_source_segment_idxs:
            last_segment_end = min(
                input_segments[curr_slice_source_segment_idxs[-1]].end, audio_duration
            )
            curr_slice["duration"] = last_segment_end - slice_start

        # Record the ACTUAL slice duration for adaptive control feedback
        # This must happen here (not in generate_examples_from_slices) because:
        # 1. The actual duration differs from target due to segment boundaries
        # 2. We need to adapt based on what we're actually producing, not what succeeds later
        if duration_controller and curr_slice["duration"] > 0:
            duration_controller.record_completed_slice(curr_slice["duration"])

        # Slice Quality Control
        slice_quality_score = calculate_segments_quality_score(
            curr_slice_source_segments
        )

        # Check if the slice quality is below threshold to abandon it and force a new slice
        if (
            curr_slice_source_segments
            and slice_quality_score < per_segment_quality_threshold
        ):
            # This slice is suspected as low quality

            # Look for a segment with good quality to start the next slice
            # skip the first segment in the slice (otherwise we probably are going
            # to just repeat the same slice)
            found_good_segment = False
            for seg_idx_within_slice, seg_of_slice in enumerate(
                curr_slice_source_segments
            ):
                if seg_idx_within_slice == 0:
                    continue

                segment_score = calculate_segments_quality_score([seg_of_slice])

                if segment_score >= per_segment_quality_threshold:
                    # Found a good enough segment, start next slice from here
                    next_slice_start = seg_of_slice.start
                    curr_input_segment_idx = curr_slice_source_segment_idxs[
                        seg_idx_within_slice
                    ]
                    found_good_segment = True
                    break

            # If no good segment found, start from the end of the last checked segment
            if not found_good_segment:
                next_segment_idx_after_slice_segments = (
                    curr_slice_source_segment_idxs[-1] + 1
                )
                # if any segment ahead
                if next_segment_idx_after_slice_segments < len(input_segments):
                    next_slice_start = input_segments[
                        next_segment_idx_after_slice_segments
                    ].start
                    curr_input_segment_idx = next_segment_idx_after_slice_segments
                # or there are more segments - stop slicing
                else:
                    next_slice_start = audio_duration

            # Clear the current slice content as we're abandoning it
            curr_slice_segments.clear()

            # Record that we rejected this slice due to quality
            if duration_controller:
                duration_controller.record_rejected_slice()

    return slices


def merge_slice_segments(
    slices: list[dict], merge_below_gap_threshold: float = 0.3
) -> list[dict]:
    """
    Merge segments within each slice that are close together.

    Args:
        slices: List of slices, each containing a list of segments
        merge_below_gap_threshold: Merge segments if gap between them is less than this threshold (in seconds)

    Returns:
        List of slices with merged segments
    """
    if not slices:
        return slices

    result_slices = []

    for slice_data in slices:
        # Create a new slice with the same properties as the original, but copy it to avoid modifying the original
        new_slice = {
            key: value for key, value in slice_data.items() if key != "segments"
        }
        new_slice["segments"] = []

        segments = slice_data.get("segments", [])

        # If no segments or only one segment, no merging needed
        if len(segments) <= 1:
            new_slice["segments"] = [segment.copy() for segment in segments]
            result_slices.append(new_slice)
            continue

        # Create a copy of segments to process
        result_segments = [segment.copy() for segment in segments]

        # Process segments in reverse order
        i = len(result_segments) - 1
        while i > 0:  # Stop at index 1 (second segment)
            current_segment = result_segments[i]
            prev_segment = result_segments[i - 1]

            # Check if we can merge the current segment with the previous one
            can_merge = False

            # Current segment must have start, end, and text to be mergeable
            # Note: No "end" cases means an open-only slice where the last segment
            # mark a segment which could not end within the same slice. we need
            # to keep it as is.
            if all(key in current_segment for key in ["start", "end", "text"]):
                # Calculate the gap between segments
                gap = current_segment["start"] - prev_segment["end"]

                # Check if the gap is small enough
                if gap < merge_below_gap_threshold:
                    can_merge = True

            if can_merge:
                # Merge current segment into previous segment
                prev_segment["end"] = current_segment["end"]
                prev_segment["text"] = prev_segment["text"] + current_segment["text"]

                # Remove the current segment as it's now merged
                result_segments.pop(i)

            # Move to previous segment
            i -= 1

        # Add all processed segments to the new slice
        new_slice["segments"] = result_segments
        result_slices.append(new_slice)

    return result_slices


def get_slice_audio_samples(
    audio_decoder: AudioDecoder, slice, slice_length
) -> AudioSamples:
    audio_start_sec = slice["seek"]
    audio_samples = audio_decoder.get_samples_played_in_range(
        audio_start_sec, audio_start_sec + slice_length
    )
    return audio_samples


word_replacement_cache = None
cleanup_options = TTSNormalizeOptions(
    apply_word_replacements=True,
    expand_numbers=True,
    keep_punctuations=True,
    attach_punctuations_to_token=True,
    stt_compat_mode=False,
    remove_parentheses=False
)


def cleanup_text(text: str) -> str:
    global word_replacement_cache
    if not word_replacement_cache:
        word_replacement_cache = load_word_replacements()
    return normalize_tts_text(text, options=cleanup_options, word_replacements=word_replacement_cache)


def generate_examples_from_slices(
    slices,
    audio_file: str,
    metadata: dict,
    copy_metadata_fields: list[str] = [],
) -> Iterator[dict]:
    source_id = metadata.get("source_id", "unknown")
    source_entry_id = metadata.get("source_entry_id", str(uuid.uuid4()))
    logger.debug(f"Generating dataset from {source_id}/{source_entry_id}")

    # No slices - nothing to do
    if not slices:
        logger.debug(f"No slices in {source_id}/{source_entry_id}")
        return None

    # At least one segments we can work on is expected
    if next(iter([seg for s in slices for seg in s["segments"]]), None) is None:
        logger.debug(f"No segments in {source_id}/{source_entry_id}")
        return None

    prev_example = None
    audio_decoder = AudioDecoder(
        str(audio_file), sample_rate=TARGET_SAMPLE_RATE, num_channels=1
    )
    for slice in slices:
        if slice["segments"]:
            try:
                slice_text = ""
                for segment in slice["segments"]:
                    if "text" in segment:
                        slice_text += segment["text"]
                all_word_scores = [
                    score
                    for segment in slice["segments"]
                    for score in segment.get("word_scores", [])
                ]
                segments_quality_score = calculate_median_quality_score(all_word_scores)
                slice_duration = slice.get("duration", 0.0)
                slice_audio_samples = get_slice_audio_samples(
                    audio_decoder, slice, slice_duration
                )
                audio_encoder = AudioEncoder(
                    slice_audio_samples.data, sample_rate=TARGET_SAMPLE_RATE
                )
                example = {
                    "audio": {
                        "bytes": audio_encoder.to_tensor(format="mp3")
                        .numpy()
                        .tobytes(),
                        "path": source_entry_id,
                    },
                    "text": cleanup_text(slice_text),
                    "raw_text": slice_text,
                    "metadata": {
                        "seek": float(slice["seek"]),
                        "duration": slice_duration,
                        "source": source_id,
                        "entry_id": source_entry_id,
                        "quality_score": segments_quality_score,
                    },
                    "has_prev": False,
                    "prev_text": "",
                }
                if prev_example:
                    example["prev_text"] = prev_example["text"]
                    example["has_prev"] = True
                if copy_metadata_fields:
                    for field_to_copy in copy_metadata_fields:
                        example["metadata"][field_to_copy] = metadata.get(
                            field_to_copy, None
                        )
                yield example
                prev_example = example
            except Exception as e:
                prev_example = None
                logger.error(
                    f"Error processing slice seek {float(slice['seek']):.2f} in {source_id}:{source_entry_id}: {e}"
                )
                if "Could not push packet to decoder" in str(e):
                    # we cannot recover from this
                    break
        else:
            prev_example = None

    logger.debug(f"Done with samples from {source_id}/{source_entry_id}")


def parse_exclude_filter(filter_str: str) -> tuple[str, str, str]:
    """
    Parse a filter string in format 'field_path:eq:value' into components.

    Args:
        filter_str: Filter string like 'source_id:eq:youtube'

    Returns:
        Tuple of (field_path, operator, value)
    """
    parts = filter_str.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid filter format: {filter_str}. Expected 'field_path:eq:value'"
        )
    field_path, operator, value = parts
    if operator != "eq":
        raise ValueError(f"Unsupported operator: {operator}. Only 'eq' is supported")
    return field_path, operator, value


def get_nested_value(obj: dict, field_path: str):
    """
    Get a value from a nested dictionary using dot notation.

    Args:
        obj: Dictionary to traverse
        field_path: Dot-separated path like 'metadata.source_id'

    Returns:
        The value at the specified path, or None if path doesn't exist
    """
    keys = field_path.split(".")
    current = obj
    try:
        for key in keys:
            if isinstance(current, dict):
                current = current[key]
            else:
                return None
        return current
    except (KeyError, TypeError):
        return None


def should_exclude_entry(
    metadata: dict, exclude_filters: list[tuple[str, str, str]]
) -> bool:
    """
    Check if an entry should be excluded based on metadata filters.

    Args:
        metadata: Metadata dictionary
        exclude_filters: List of parsed filter tuples (field_path, operator, value)

    Returns:
        True if the entry should be excluded, False otherwise
    """
    if not exclude_filters:
        return False

    # OR logic: exclude if ANY filter matches
    for field_path, operator, value in exclude_filters:
        actual_value = get_nested_value(metadata, field_path)
        if actual_value is not None and str(actual_value) == value:
            return True
    return False


def prepare_training_dataset(
    input_folder: Path,
    max_duration: float = 30,
    min_duration: float = 5.0,
    target_duration: float = None,
    max_source_entries: int = None,
    audio_filename_glob: str = "audio.*",
    segments_filename_glob: str = "transcript.*.json",
    metadata_glob: str = "metadata.json",
    num_proc: int = 1,
    per_proc_per_chunk_size: int = 10,
    per_sample_quality_threshold: float = 0,
    per_segment_quality_threshold: float = 0,
    copy_metadata_fields: list[str] = [],
    exclude_filters: list[str] = [],
) -> Dataset:
    """
    Prepare captioned datasets from the input folder.
    Produce audio slices and corresponding text including previous text when available
    Returns a HuggingFace Dataset. Splitting (if needed) should be applied outside this function.
    """
    input_folder = Path(input_folder)
    input_manifest = _load_data_manifest(
        input_folder,
        segments_glob=f"**/{segments_filename_glob}",
        audio_filename_glob=audio_filename_glob,
        metadata_glob=metadata_glob,
    )

    # sort the input_manifest by audio_duration (asc)
    # so we get chunks that take about the same time in parallel.
    # each entry is (audio_file, segments_file, metadata_file)
    # need to load the metadata_file (it's a json)
    # get the "duration" field (fallback is session_duration field) from the json
    # sort by that value
    def _get_manifest_duration(entry):
        metadata_file = entry[2]
        try:
            with open(metadata_file, "r") as f:
                md = json.load(f)
            duration = md.get("duration", md.get("session_duration", None))
            if duration is None:
                return 0.0
            return float(duration)
        except Exception:
            logger.warning(
                f"Could not read duration from metadata {metadata_file}, using 0.0"
            )
            return 0.0

    input_manifest = sorted(input_manifest, key=_get_manifest_duration)

    # Limit the number of source entries to process
    if max_source_entries:
        input_manifest = input_manifest[:max_source_entries]

    # Aim for reasonable entries per worker within each chunk
    manifest_processing_chunk_size = num_proc * per_proc_per_chunk_size

    def examples_from_entry_generator(input_manifest_shards):
        # Create a duration controller instance for this process
        duration_controller = None
        if target_duration is not None:
            duration_controller = DurationController(
                min_duration=min_duration,
                max_duration=max_duration,
                target_duration=target_duration,
            )

        yielded_at_least_one = False
        for audio_file, segments_data_file, metadata_file in input_manifest_shards:
            try:
                # Load metadata first to check exclusion filters
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                # Check if entry should be excluded based on metadata filters
                if should_exclude_entry(metadata, exclude_filters):
                    logger.debug(
                        f"Excluding sample {audio_file} based on metadata filters"
                    )
                    continue

                # Load captions
                segments = []
                with open(segments_data_file, "r") as segs_file:
                    segs_json = json.load(segs_file)
                    segments: list[Segment] = [
                        Segment(**s) if isinstance(s, dict) else s
                        for s in segs_json["segments"]
                    ]

                # Get sample quality score
                sample_quality_score = metadata.get("quality_score", None)

                if (
                    sample_quality_score is not None
                    and per_sample_quality_threshold > 0
                    and sample_quality_score < per_sample_quality_threshold
                ):
                    logger.debug(
                        f"Skipping sample {audio_file} with quality score {sample_quality_score} (threshold: {per_sample_quality_threshold})"
                    )
                    continue

                # Load Audio (streams output from an FFMPEG process for memory efficiency)
                audio_decoder = AudioDecoder(
                    str(audio_file), sample_rate=TARGET_SAMPLE_RATE, num_channels=1
                )
                audio_duration = audio_decoder.metadata.duration_seconds

                # Create slices of the captions with the intended slice
                slices = generate_slices(
                    segments,
                    audio_duration,
                    max_duration,
                    per_segment_quality_threshold,
                    duration_controller=duration_controller,
                )

                slices = merge_slice_segments(slices)

                # Generate the dataset
                for example in generate_examples_from_slices(
                    slices,
                    audio_file,
                    metadata,
                    copy_metadata_fields,
                ):
                    yielded_at_least_one = True
                    yield example

                if not yielded_at_least_one:
                    # ensure at least one (empty) example is returned from this generator
                    # to overcome Dataset "from_generator" bug
                    yield {"text": ""}
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")

    input_manifest_chunks = [
        input_manifest[i : i + manifest_processing_chunk_size]
        for i in range(0, len(input_manifest), manifest_processing_chunk_size)
    ]

    # Why? Dataset.from_generator does not properly release memory from the generator
    # after completion. To avoid OOM, we:
    # 1. Generate multiple smaller datasets in chunks
    # 2. Let each chunk's generator get GC'd after completion
    # 3. Concatenate the memory-mapped datasets at the end
    # This trades off some disk I/O for better memory usage, while still
    # maintaining parallel generation within each chunk.
    all_datasets = []
    for input_manifest_chunk in tqdm(
        input_manifest_chunks, desc="Generating input manifest chunks"
    ):
        try:
            dataset_features = Features(
                {
                    "audio": AudioColumnType(),
                    "text": ValueColumnType(dtype="string"),
                    "raw_text": ValueColumnType(dtype="string"),
                    "metadata": {
                        "seek": ValueColumnType(dtype="float32"),
                        "duration": ValueColumnType(dtype="float32"),
                        "source": ValueColumnType(dtype="string"),
                        "entry_id": ValueColumnType(dtype="string"),
                        "quality_score": ValueColumnType(dtype="float32"),
                    },
                    "has_prev": ValueColumnType(dtype="bool"),
                    "prev_text": ValueColumnType(dtype="string"),
                }
            )
            if copy_metadata_fields:
                for field_to_copy in copy_metadata_fields:
                    dataset_features["metadata"][field_to_copy] = ValueColumnType(
                        dtype="string"
                    )
            generator_kwargs = {}
            if num_proc > 1:
                generator_kwargs["num_proc"] = num_proc
            generated_dataset = Dataset.from_generator(
                examples_from_entry_generator,
                gen_kwargs={"input_manifest_shards": list(input_manifest_chunk)},
                features=dataset_features,
                **generator_kwargs,
            ).filter(
                lambda example: example["text"]
            )  # filter out empty examples
        except ValueError as e:
            if "corresponds to no data" in str(e):
                logger.info("Skipping dataset creation because no data was found.")
                continue
            else:
                raise  # Re-raise unexpected errors

        all_datasets.append(generated_dataset)

    if not all_datasets:
        return None

    examples_dataset = concatenate_datasets(all_datasets)
    examples_dataset = examples_dataset.cast_column(
        "audio", AudioColumnType(sampling_rate=TARGET_SAMPLE_RATE)
    )

    return examples_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CLI to prepare a training dataset from the ivrit.ai normalized datasets"
    )
    parser.add_argument(
        "input_folder",
        help="Path to the folder containing audio, transcript, and metadata data in the normalized structure",
    )
    parser.add_argument(
        "--max_source_entries",
        type=int,
        default=None,
        help="Maximum number of source entries to process",
    )
    parser.add_argument(
        "--audio_filename_glob", default="audio.*", help="Glob pattern for audio files"
    )
    parser.add_argument(
        "--segments_filename_glob",
        default="transcript.*.json",
        help="Glob pattern for segments files",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=30.0,
        help="Maximum duration for output samples in seconds (default: 30)",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=5.0,
        help="Minimum duration for output samples in seconds (default: 5)",
    )
    parser.add_argument(
        "--target_duration",
        type=float,
        default=None,
        help="Target average/median duration for output samples in seconds. "
        "When specified, enables adaptive duration control to aim for this value. "
        "If not specified, samples will fill up to max_duration (original behavior).",
    )
    parser.add_argument(
        "--slice_length",
        type=int,
        default=None,
        help="[DEPRECATED] Use --max_duration instead. Kept for backward compatibility.",
    )
    parser.add_argument(
        "--validation_split_size",
        type=float,
        default=0,
        help="Split size for evaluation (between 0 and 1)",
    )
    parser.add_argument(
        "--num_proc", type=int, default=1, help="Number of processes to use"
    )
    parser.add_argument(
        "--per_proc_per_chunk_size",
        type=int,
        default=10,
        help=(
            "Number of entries per process per chunk. "
            "This is a memory usage consideration. This number times the number of processes will define the "
            "amount of memory kept around during the generation of a sub-dataset. "
            "If each sample is large (minutes of audio), this number should be decreased. "
            "If each sample is small (seconds of audio), this number can be increased to increase parallelism efficiency. "
        ),
    )
    parser.add_argument(
        "--per_sample_quality_threshold",
        type=float,
        default=0,
        help="Quality threshold for per-sample quality filtering (0-1 below this threshold the entire sample is dropped)",
    )
    parser.add_argument(
        "--per_segment_quality_threshold",
        type=float,
        default=0,
        help="Quality threshold for per-segment quality filtering (0-1 below this threshold a segment and it's surrounding slice are dropped)",
    )
    parser.add_argument(
        "--copy_metadata_fields",
        nargs="*",
        default=[],
        help="specify dataset specific metadata fields to copy into output segments from souce entries",
    )
    parser.add_argument(
        "--exclude_by_md",
        nargs="+",
        default=[],
        help="Exclude entries by metadata property filters in format 'field_path:eq:value'. Multiple filters are joined with OR logic.",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Push the dataset to the hub"
    )
    parser.add_argument(
        "--output_dataset_name",
        type=str,
        help="Name of the dataset, Omit to not store any dataset (dry-run)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset to set in dataset info",
    )
    parser.add_argument(
        "--dataset_license_file",
        type=str,
        help="A license file to upload as the dataset license",
    )
    parser.add_argument(
        "--dataset_version",
        type=str,
        help="Version of the dataset to set in dataset info",
    )
    parser.add_argument(
        "--dataset_card_language",
        type=str,
        help="Language of the dataset for the dataset card",
    )
    parser.add_argument(
        "--dataset_card_license",
        type=str,
        help="License of the dataset for the dataset card",
    )
    parser.add_argument(
        "--dataset_card_language_creators",
        type=str,
        nargs="+",
        help="Language creators type for the dataset card",
    )
    parser.add_argument(
        "--dataset_card_task_categories",
        type=str,
        nargs="+",
        help="Task categories for the dataset card",
    )
    parser.add_argument(
        "--dataset_card_pretty_name", type=str, help="Pretty name for the dataset card"
    )
    parser.add_argument(
        "--push_as_public", action="store_true", help="Push the dataset as public"
    )
    parser.add_argument(
        "--clear_output_dataset_cache_files",
        action="store_true",
        help="Clear the HF cache for the output dataset on disk",
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO", help="Log level of the general logger."
    )

    args = parser.parse_args()

    # Configure Logger
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % args.log_level)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Parse exclude filters
    parsed_exclude_filters = []
    if args.exclude_by_md:
        for filter_str in args.exclude_by_md:
            try:
                parsed_filter = parse_exclude_filter(filter_str)
                parsed_exclude_filters.append(parsed_filter)
            except ValueError as e:
                logger.error(f"Invalid exclude filter '{filter_str}': {e}")
                raise

    # Handle backward compatibility for slice_length
    if args.slice_length is not None:
        logger.warning("--slice_length is deprecated, use --max_duration instead")
        if (
            args.max_duration == 30.0
        ):  # Only override if user didn't set max_duration explicitly
            args.max_duration = float(args.slice_length)

    # Prepare the dataset
    output_dataset = prepare_training_dataset(
        input_folder=args.input_folder,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
        target_duration=args.target_duration,
        max_source_entries=args.max_source_entries,
        audio_filename_glob=args.audio_filename_glob,
        segments_filename_glob=args.segments_filename_glob,
        num_proc=args.num_proc,
        per_proc_per_chunk_size=args.per_proc_per_chunk_size,
        per_sample_quality_threshold=args.per_sample_quality_threshold,
        per_segment_quality_threshold=args.per_segment_quality_threshold,
        copy_metadata_fields=args.copy_metadata_fields,
        exclude_filters=parsed_exclude_filters,
    )

    if output_dataset:
        if args.dataset_name:
            output_dataset.info.dataset_name = args.dataset_name
        if args.dataset_version:
            output_dataset.info.version = args.dataset_version

        # Create dataset card if any of the card-related arguments are provided
        dataset_card = None
        if any(
            [
                args.dataset_card_language,
                args.dataset_card_license,
                args.dataset_card_language_creators,
                args.dataset_card_task_categories,
                args.dataset_card_pretty_name,
            ]
        ):
            card_data = DatasetCardData(
                language=args.dataset_card_language,
                license=args.dataset_card_license,
                language_creators=args.dataset_card_language_creators,
                task_categories=args.dataset_card_task_categories,
                pretty_name=args.dataset_card_pretty_name,
            )
            dataset_card = DatasetCard.from_template(
                card_data, template_path="assets/ivritai_dataset_card_template.md"
            )

        if args.validation_split_size > 0:
            # If a validation split is requested, split the dataset in main
            assert (
                args.validation_split_size < 1.0
            ), "validation_split_size must be a float between 0 and 1"
            temp = output_dataset.train_test_split(test_size=args.validation_split_size)
            output_dataset = DatasetDict({"train": temp["train"], "eval": temp["test"]})

        if args.output_dataset_name:
            if args.push_to_hub:
                if not args.push_as_public:
                    logger.warning("Pushing the dataset to the hub as private")
                output_dataset.push_to_hub(
                    args.output_dataset_name, private=not args.push_as_public
                )
                # Push dataset card if it was created
                if dataset_card:
                    dataset_card.push_to_hub(
                        repo_id=args.output_dataset_name, repo_type="dataset"
                    )

                if (
                    args.dataset_license_file
                    and Path(args.dataset_license_file).exists()
                ):
                    upload_file(
                        path_or_fileobj=args.dataset_license_file,
                        repo_id=args.output_dataset_name,
                        path_in_repo="LICENSE",
                        repo_type="dataset",
                    )
            else:
                output_dataset.save_to_disk(args.output_dataset_name)
                # Save dataset card if it was created
                if dataset_card:
                    logger.warning(
                        "Dataset card will be saved locally since push_to_hub is not enabled"
                    )
                    dataset_card.save(f"{args.output_dataset_name}/README.md")

            # report the created dataset sizes per split
            if isinstance(output_dataset, DatasetDict):
                for split, ds in output_dataset.items():
                    logger.info(f"{split}: {ds.num_rows} samples")
            else:
                logger.info(f"Dataset created with {output_dataset.num_rows} samples")

        if args.clear_output_dataset_cache_files and output_dataset:
            logger.info("Clearing output dataset cache files")
            output_dataset.cleanup_cache_files()
    else:
        logger.warning("No dataset was created")
