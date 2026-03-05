#!/usr/bin/env python3
"""
Script to analyze and plot the duration distribution of audio samples in a dataset.
"""

from datasets import load_from_disk
import statistics
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Plot duration distribution from a prepared dataset"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/CrowdRecital_ivritai",
        help="Path to dataset directory (load_from_disk format)",
    )
    args = parser.parse_args()

    # Load the dataset
    ds_name = args.data_path
    print(f"Loading dataset from '{ds_name}'...")
    dataset = load_from_disk(ds_name)
    dataset = dataset.select_columns(["metadata"])

    # Extract durations from metadata
    durations = []
    for sample in tqdm(dataset):
        duration = sample["metadata"]["duration"]
        durations.append(duration)
    dataset = None

    # Calculate statistics
    num_samples = len(durations)
    max_duration = max(durations)
    min_duration = min(durations)
    avg_duration = statistics.mean(durations)
    median_duration = statistics.median(durations)
    stdev_duration = statistics.stdev(durations) if num_samples > 1 else 0.0

    # Print statistics
    print("\n" + "=" * 50)
    print("DURATION DISTRIBUTION STATISTICS")
    print("=" * 50)
    print(f"Number of samples: {num_samples}")
    print(f"Maximum duration:  {max_duration:.2f}s")
    print(f"Minimum duration:  {min_duration:.2f}s")
    print(f"Average duration:  {avg_duration:.2f}s")
    print(f"Median duration:   {median_duration:.2f}s")
    print(f"Std deviation:     {stdev_duration:.2f}s")
    print("=" * 50 + "\n")

    # Create histogram using ASCII art (no external dependencies)
    print("HISTOGRAM (Duration Distribution)")
    print("-" * 50)

    # Determine bin edges
    num_bins = 20
    bin_width = (max_duration - min_duration) / num_bins
    bins = [0] * num_bins

    # Fill bins
    for duration in durations:
        bin_idx = int((duration - min_duration) / bin_width)
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
        bins[bin_idx] += 1

    # Find max count for scaling
    max_count = max(bins)
    bar_width = 40  # characters

    # Print histogram
    for i, count in enumerate(bins):
        bin_start = min_duration + i * bin_width
        bin_end = bin_start + bin_width
        bar_length = int((count / max_count) * bar_width) if max_count > 0 else 0
        bar = "█" * bar_length
        print(f"{bin_start:6.2f}-{bin_end:6.2f}s │{bar} {count}")

    print("-" * 50)
    print(f"\nHistogram saved to: duration_histogram.png")

    # Try to create a proper histogram image if matplotlib is available
    try:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.hist(durations, bins=30, edgecolor="black", alpha=0.7)
        plt.xlabel("Duration (seconds)")
        plt.ylabel("Frequency")
        plt.title(f"Audio Duration Distribution (n={num_samples})")
        plt.grid(True, alpha=0.3)

        # Add statistics text box
        stats_text = f"Mean: {avg_duration:.2f}s\nMedian: {median_duration:.2f}s\nStd: {stdev_duration:.2f}s"
        plt.text(
            0.98,
            0.97,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig("duration_histogram.png", dpi=150)
        print("✓ PNG histogram created successfully!")

    except ImportError:
        print("Note: matplotlib not available - only ASCII histogram shown above")
        print("To generate PNG histogram, install matplotlib: uv add matplotlib")


if __name__ == "__main__":
    main()
