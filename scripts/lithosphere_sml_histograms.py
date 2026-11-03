#%% Import

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm

if __package__:
    from .lithosphere_global_map import (
        discover_checkpoints,
        load_or_download_sml,
        sml_cache_file,
        sml_derivative_limit_nt_per_minute,
        sml_limit_nt,
        supermag_userid)
else:
    from lithosphere_global_map import (
        discover_checkpoints,
        load_or_download_sml,
        sml_cache_file,
        sml_derivative_limit_nt_per_minute,
        sml_limit_nt,
        supermag_userid)


#%% Configuration

input_base = Path('/media/bing/LaCie/EZIE_lithosphere_l3')
output_file = sml_cache_file.parent / 'sml_histograms.png'
sml_match_tolerance_seconds = 60.0
histogram_bins = 80


#%% Match checkpoint times to SML minutes

def read_checkpoint_times(path):
    """Read valid observation timestamps from one complete checkpoint."""

    missing_time = np.iinfo(np.int64).min
    with Dataset(path) as dataset:
        if getattr(dataset, 'processing_complete', 0) != 1:
            return None
        if 'time' not in dataset.variables:
            raise ValueError(f'{path} is missing required variable: time')
        time = np.asarray(
            np.ma.filled(dataset.variables['time'][:], missing_time),
            dtype=np.int64).ravel()
    return time[time != missing_time]


def nearest_sml_indices(observation_time_ns, sml_time_seconds,
                        tolerance_seconds=60.0):
    """Return nearest SML indices and the match mask for observation times."""

    observation_time_ns = np.asarray(observation_time_ns, dtype=np.int64).ravel()
    sml_time_seconds = np.asarray(sml_time_seconds, dtype=np.int64).ravel()
    if sml_time_seconds.size == 0 or np.any(np.diff(sml_time_seconds) <= 0):
        raise ValueError('SML time must be a non-empty increasing array')

    observation_seconds = observation_time_ns/1_000_000_000
    right = np.searchsorted(sml_time_seconds, observation_seconds)
    right = np.clip(right, 0, sml_time_seconds.size-1)
    left = np.clip(right-1, 0, sml_time_seconds.size-1)
    use_right = (
        np.abs(sml_time_seconds[right]-observation_seconds) <
        np.abs(sml_time_seconds[left]-observation_seconds))
    nearest = np.where(use_right, right, left)
    matched = (
        np.abs(sml_time_seconds[nearest]-observation_seconds) <=
        tolerance_seconds)
    return nearest, matched


def collect_data_sml(
        checkpoint_directory=input_base, sml_cache_path=sml_cache_file,
        userid=supermag_userid, refresh_sml=False,
        tolerance_seconds=sml_match_tolerance_seconds, show_progress=True):
    """Collect unique SML minutes covered by complete checkpoint observations."""

    checkpoint_directory = Path(checkpoint_directory)
    sml_cache_path = Path(sml_cache_path)
    paths = [
        path for path in discover_checkpoints(checkpoint_directory)
        if path.resolve() != sml_cache_path.resolve()]
    processed_files = 0
    incomplete_files = 0
    observation_count = 0
    observation_time_chunks = []

    loop = tqdm(
        paths, desc='Reading checkpoint times',
        disable=not show_progress)
    for path in loop:
        observation_time = read_checkpoint_times(path)
        if observation_time is None:
            incomplete_files += 1
            continue

        observation_time_chunks.append(observation_time)
        observation_count += observation_time.size
        processed_files += 1
        loop.set_postfix(files=processed_files, observations=observation_count)

    if not observation_time_chunks:
        raise ValueError('No complete checkpoints contain observation times')

    observation_times = np.concatenate(observation_time_chunks)
    unique_times, time_counts = np.unique(
        observation_times, return_counts=True)
    start_ns = unique_times[0]
    stop_ns = unique_times[-1]
    sml_time, sml, sml_derivative = load_or_download_sml(
        sml_cache_path, start_ns, stop_ns,
        userid=userid, refresh=refresh_sml)

    nearest, matched = nearest_sml_indices(
        unique_times, sml_time,
        tolerance_seconds=tolerance_seconds)
    covered_minutes = np.zeros(sml_time.size, dtype=bool)
    covered_minutes[nearest[matched]] = True
    matched_observation_count = np.sum(time_counts[matched])

    selected = np.flatnonzero(covered_minutes)
    if selected.size == 0:
        raise ValueError('No checkpoint observations matched the SML cache')

    selected_sml = np.asarray(sml[selected], dtype=float)
    selected_derivative = np.asarray(sml_derivative[selected], dtype=float)
    finite_sml = np.isfinite(selected_sml)
    finite_derivative = np.isfinite(selected_derivative)
    jointly_finite = finite_sml & finite_derivative
    quiet = (
        jointly_finite &
        (np.abs(selected_sml) < sml_limit_nt) &
        (np.abs(selected_derivative) < sml_derivative_limit_nt_per_minute))

    summary = {
        'discovered_file_count': len(paths),
        'complete_time_file_count': processed_files,
        'processed_file_count': processed_files,
        'incomplete_file_count': incomplete_files,
        'observation_count': observation_count,
        'unique_observation_time_count': unique_times.size,
        'matched_observation_count': matched_observation_count,
        'unique_minute_count': selected.size,
        'finite_sml_minute_count': np.count_nonzero(finite_sml),
        'finite_derivative_minute_count': np.count_nonzero(finite_derivative),
        'quiet_minute_count': np.count_nonzero(quiet),
        'quiet_minute_fraction': np.count_nonzero(quiet)/selected.size}
    return selected_sml[finite_sml], selected_derivative[finite_derivative], summary


#%% Plot

def plot_sml_histograms(
        path, sml, sml_derivative, summary, bins=histogram_bins,
        sml_limit_nt=sml_limit_nt,
        derivative_limit_nt_per_minute=sml_derivative_limit_nt_per_minute):
    """Plot the SML and dSML/dt distributions at data-covered minutes."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sml = np.asarray(sml, dtype=float)
    sml_derivative = np.asarray(sml_derivative, dtype=float)
    if sml.size == 0 or sml_derivative.size == 0:
        raise ValueError('Histogram inputs must contain finite values')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    axes[0].hist(sml, bins=bins, color='0.35', edgecolor='white', linewidth=0.4)
    axes[0].axvline(-sml_limit_nt, color='tab:red', linestyle='--')
    axes[0].axvline(sml_limit_nt, color='tab:red', linestyle='--')
    axes[0].set_xlabel('SML [nT]')
    axes[0].set_ylabel('unique data-covered minutes')
    axes[0].set_title('SML distribution')

    axes[1].hist(
        sml_derivative, bins=bins,
        color='0.35', edgecolor='white', linewidth=0.4)
    axes[1].axvline(
        -derivative_limit_nt_per_minute,
        color='tab:red', linestyle='--')
    axes[1].axvline(
        derivative_limit_nt_per_minute,
        color='tab:red', linestyle='--')
    axes[1].set_xlabel('dSML/dt [nT/min]')
    axes[1].set_ylabel('unique data-covered minutes')
    axes[1].set_title('SML temporal derivative')

    quiet_percent = 100*summary['quiet_minute_fraction']
    fig.suptitle(
        f"EZIE data coverage: {summary['unique_minute_count']:,} unique minutes; "
        f"{quiet_percent:.1f}\\% pass both limits")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def build_sml_histograms(
        checkpoint_directory=input_base, output_path=output_file,
        sml_cache_path=sml_cache_file, userid=supermag_userid,
        refresh_sml=False, tolerance_seconds=sml_match_tolerance_seconds,
        bins=histogram_bins, show_progress=True):
    sml, derivative, summary = collect_data_sml(
        checkpoint_directory=checkpoint_directory,
        sml_cache_path=sml_cache_path,
        userid=userid,
        refresh_sml=refresh_sml,
        tolerance_seconds=tolerance_seconds,
        show_progress=show_progress)
    plot_sml_histograms(
        output_path, sml, derivative, summary, bins=bins)
    return sml, derivative, summary


#%% Command-line entry point

def positive_integer(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('value must be at least 1')
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot SML distributions at EZIE checkpoint times')
    parser.add_argument('--input', type=Path, default=input_base,
                        help='root directory containing orbit checkpoints')
    parser.add_argument('--output', type=Path, default=output_file,
                        help='output histogram image')
    parser.add_argument('--sml-cache', type=Path, default=sml_cache_file)
    parser.add_argument('--supermag-userid', default=supermag_userid)
    parser.add_argument('--bins', type=positive_integer, default=histogram_bins)
    parser.add_argument('--refresh-sml', action='store_true',
                        help='redownload SML even if the cache covers the data')
    return parser.parse_args()


def main():
    args = parse_args()
    _, _, summary = build_sml_histograms(
        checkpoint_directory=args.input,
        output_path=args.output,
        sml_cache_path=args.sml_cache,
        userid=args.supermag_userid,
        refresh_sml=args.refresh_sml,
        bins=args.bins)
    print(
        f"Wrote {args.output} from {summary['processed_file_count']} files and "
        f"{summary['unique_minute_count']} unique data-covered minutes")


if __name__ == '__main__':
    main()
