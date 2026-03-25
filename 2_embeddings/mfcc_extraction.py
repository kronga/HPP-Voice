import os

import librosa
from tqdm import tqdm
import soundfile as sf
import multiprocessing
from functools import partial
import pandas as pd
from typing import Tuple, Optional, Dict, List


def process_single_file(file_info, input_dir, output_dir, target_sr=16000, top_db=30):
    """
    Process a single audio file, removing silence and optionally resampling.

    Args:
    file_info (tuple): Tuple containing (root, file) path information.
    input_dir (str): Base input directory.
    output_dir (str): Output directory where processed file will be saved.
    target_sr (int): Target sampling rate for resampling.
    top_db (float): The threshold (in decibels) below reference to consider as silence.

    Returns:
    tuple: (output_path, duration_info) where duration_info is a dict with file info and durations,
           or (None, None) if processing failed.
    """
    root, file = file_info
    file_path = os.path.join(root, file)

    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=target_sr)
        original_duration = len(y) / sr

        # Trim silence from beginning and end
        y_trimmed, trim_indices = librosa.effects.trim(y, top_db=top_db)
        trimmed_duration = len(y_trimmed) / sr

        # Calculate how much silence was removed
        silence_removed = original_duration - trimmed_duration

        # Determine output file name
        if root == input_dir:
            # If file is directly in input_dir, use original filename
            base_name = os.path.splitext(file)[0]
        else:
            # If file is in a subdirectory, use reg_code_date format
            path_parts = root.split(os.sep)
            reg_code = path_parts[-2]  # Get the second-to-last folder as reg_code
            date = os.path.splitext(file)[0].replace("_", "")
            base_name = f"{reg_code}_{date}"

        # Generate output file path with original extension
        output_file = f"{base_name}{os.path.splitext(file)[1]}"
        output_path = os.path.join(output_dir, output_file)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save the processed (trimmed) audio
        sf.write(output_path, y_trimmed, target_sr)

        # Prepare duration info for CSV
        duration_info = {
            'input_file': file_path,
            'output_file': output_path,
            'original_duration_seconds': round(original_duration, 3),
            'trimmed_duration_seconds': round(trimmed_duration, 3),
            'silence_removed_seconds': round(silence_removed, 3),
            'silence_percentage': round((silence_removed / original_duration) * 100, 2) if original_duration > 0 else 0,
            'target_sample_rate': target_sr,
            'top_db_threshold': top_db
        }

        # print(f"Processed: {file} | Original: {original_duration:.2f}s ? Trimmed: {trimmed_duration:.2f}s")

        return output_path, duration_info
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None, None


def save_duration_csv(duration_data: List[Dict], csv_path: str):
    """
    Save the collected duration data to a CSV file.

    Args:
    duration_data (List[Dict]): List of duration info dictionaries from processing
    csv_path (str): Path where the CSV file should be saved
    """
    if not duration_data:
        print("No duration data to save.")
        return

    df = pd.DataFrame(duration_data)
    df.to_csv(csv_path, index=False)

    # Print summary statistics
    total_files = len(df)
    total_original_duration = df['original_duration_seconds'].sum()
    total_trimmed_duration = df['trimmed_duration_seconds'].sum()
    total_silence_removed = df['silence_removed_seconds'].sum()
    avg_silence_percentage = df['silence_percentage'].mean()

    print(f"\n=== PROCESSING SUMMARY ===")
    print(f"Total files processed: {total_files}")
    print(
        f"Total original duration: {total_original_duration:.2f} seconds ({total_original_duration / 60:.2f} minutes)")
    print(f"Total trimmed duration: {total_trimmed_duration:.2f} seconds ({total_trimmed_duration / 60:.2f} minutes)")
    print(f"Total silence removed: {total_silence_removed:.2f} seconds ({total_silence_removed / 60:.2f} minutes)")
    print(f"Average silence percentage: {avg_silence_percentage:.2f}%")
    print(f"CSV saved to: {csv_path}")


def process_audio_batch(input_dir, output_dir, target_sr=16000, top_db=30, csv_path=None):
    """
    Process a batch of audio files and save duration statistics to CSV.

    Args:
    input_dir (str): Directory containing audio files to process
    output_dir (str): Directory to save processed files
    target_sr (int): Target sampling rate
    top_db (float): Silence threshold in dB
    csv_path (str): Path to save CSV file. If None, uses output_dir/processing_log.csv

    Returns:
    List[str]: List of successfully processed file paths
    """
    if csv_path is None:
        csv_path = os.path.join(output_dir, "processing_log.csv")

    # Collect all audio files
    audio_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aiff')
    file_list = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(audio_extensions):
                file_list.append((root, file))

    print(f"Found {len(file_list)} audio files to process...")

    # Process files and collect results
    processed_files = []
    duration_data = []

    for file_info in file_list:
        result_path, duration_info = process_single_file(
            file_info, input_dir, output_dir, target_sr, top_db
        )

        if result_path is not None and duration_info is not None:
            processed_files.append(result_path)
            duration_data.append(duration_info)

    # Save CSV with duration data
    save_duration_csv(duration_data, csv_path)

    return processed_files


def process_full_audio_files(input_dir, output_dir, target_sr=16000, top_db=30, num_workers=None, csv_path=None):
    """
    Process all audio files in the specified directory and its subdirectories using multiprocessing,
    removing silence and optionally resampling.

    Args:
    input_dir (str): Directory containing the original audio files.
    output_dir (str): Directory where processed audio files will be saved.
    target_sr (int): Target sampling rate for resampling.
    top_db (float): The threshold (in decibels) below reference to consider as silence.
    num_workers (int): Number of worker processes to use. If None, uses CPU count - 1.
    csv_path (str): Path to save CSV file. If None, uses output_dir/processing_log.csv

    Returns:
    List[str]: List of successfully processed file paths
    """
    import multiprocessing
    from functools import partial
    from tqdm import tqdm

    # Supported audio file extensions
    supported_extensions = [".flac", ".wav", ".mp3"]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set CSV path if not provided
    if csv_path is None:
        csv_path = os.path.join(output_dir, "processing_log.csv")

    # If num_workers is not specified, use CPU count - 1 (leave one core free)
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    # Collect all files to process
    files_to_process = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.endswith(ext) for ext in supported_extensions):
                files_to_process.append((root, file))

    print(f"Found {len(files_to_process)} audio files to process using {num_workers} workers")

    # Create a partial function with fixed parameters
    process_fn = partial(
        process_single_file,
        input_dir=input_dir,
        output_dir=output_dir,
        target_sr=target_sr,
        top_db=top_db
    )

    # Process files in parallel
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_fn, files_to_process),
            total=len(files_to_process),
            desc="Processing audio files"
        ))

    # Separate successful results and duration data
    processed_files = []
    duration_data = []

    for result in results:
        if result[0] is not None and result[1] is not None:  # Both output_path and duration_info exist
            output_path, duration_info = result
            processed_files.append(output_path)
            duration_data.append(duration_info)

    # Save CSV with duration data
    save_duration_csv(duration_data, csv_path)

    print(f"Successfully processed {len(processed_files)} out of {len(files_to_process)} files")
    print(f"All processed audio files saved in {output_dir}")
    print(f"Processing log saved to: {csv_path}")

    return processed_files


if __name__ == "__main__":
    input_dir = "/net/mraid20/export/genie/LabData/Data/10K/aws_lab_files/voice"
    output_dir = "/net/mraid20/export/genie/LabData/Analyses/DeepVoiceFolder/trimmed_topDB40_aggregated_audio_full_length_2025May"
    
    # Process full audio files without segmentation using multiprocessing
    process_full_audio_files(
        input_dir=input_dir,
        output_dir=output_dir,
        target_sr=16000,  # Target sampling rate - set to None to keep original       
        num_workers=30, # Set to None to use CPU count - 1, or specify a number
        top_db=40,  # Adjust this value to control silence detection sensitivity
        csv_path=os.path.join(output_dir, "processing_log.csv")
    )
