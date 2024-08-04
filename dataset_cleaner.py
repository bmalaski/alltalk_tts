import shutil
from collections import Counter
from multiprocessing import Pool

import librosa
import numpy as np
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.datasets.formatters import *
from matplotlib import pylab as plt


def formatter(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "myspeaker"
    with open(txt_file, "r", encoding="utf-8-sig") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[1]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def load_item(item):
    text = item["text"].strip()
    file_name = item["audio_file"].strip()
    audio, sr = librosa.load(file_name, sr=None)
    audio_len = len(audio) / sr
    text_len = len(text)
    return file_name, text, text_len, audio, audio_len


def plot_text_length_vs_mean_audio_duration(text_vs_avg):
    plt.title("text length vs mean audio duration")
    plt.scatter(list(text_vs_avg.keys()), list(text_vs_avg.values()))
    plt.show()


def plot_text_length_vs_std(text_vs_std):
    plt.title("text length vs STD")
    plt.scatter(list(text_vs_std.keys()), list(text_vs_std.values()))
    plt.show()


def plot_distribution_of_text_lengths(lengths):
    plt.hist(lengths, bins=20, edgecolor='black')  # Adjust the number of bins as needed
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Text Lengths in TTS Dataset')
    plt.show()


def plot_speech_duration_per_character(points, mean, std_dev):
    plt.figure(figsize=(10, 8))
    plt.hist(points, bins=30, edgecolor='black', linewidth=1.2)
    plt.axvline(mean, color='white', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean:.5f}')
    plt.axvline(mean + std_dev, color='orange', linestyle='dashed', linewidth=1.5,
                label=f'Standard Deviation: {std_dev:.5f}')
    plt.axvline(mean - std_dev, color='orange', linestyle='dashed', linewidth=1.5)
    plt.axvline(mean + 2 * std_dev, color='red', linestyle='dashed', linewidth=1.5,
                label=f'Standard Deviation: {std_dev:.5f}')
    plt.axvline(mean - 2 * std_dev, color='red', linestyle='dashed', linewidth=1.5)
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Speech Duration Per Character for All Items')
    plt.grid(True)
    plt.show()


def plot_pie_chart_of_audio_proportions(remaining_length, excluded_length):
    sizes = [remaining_length, excluded_length]
    labels = ['Remaining Data', 'Excluded Data']
    colors = ['lightblue', 'lightcoral']
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Proportion of seconds of audio kept after cull')
    plt.show()


def clean_data_set(training_dir):
    NUM_PROC = 4
    DATASET_CONFIG = BaseDatasetConfig(
        formatter="ljspeech", meta_file_train="metadata.csv", path=training_dir
    )

    train_samples, eval_samples = load_tts_samples(DATASET_CONFIG, eval_split=True, formatter=formatter)
    if eval_samples is not None:
        items = train_samples + eval_samples
    else:
        items = train_samples

    print(" > Number of audio files: {}".format(len(items)))
    print(items[0])

    wav_files = []
    for item in items:
        wav_file = item["audio_file"].strip()
        wav_files.append(wav_file)
        if not os.path.exists(wav_file):
            print(wav_file)

    c = Counter(wav_files)
    print([item for item, count in c.items() if count > 1])

    if NUM_PROC == 1:
        data = []
        for m in tqdm(items):
            data += [load_item(m)]
    else:
        with Pool(NUM_PROC) as p:
            data = list(tqdm(p.imap(load_item, items), total=len(items)))

    w_count = Counter()
    for item in tqdm(data):
        text = item[1].lower().strip()
        for word in text.split():
            w_count[word] += 1
    print(" > Number of words: {}".format(len(w_count)))

    text_vs_durs = {}
    text_len_counter = Counter()
    lengths = []
    for item in tqdm(data):
        text = item[1].lower().strip()
        text_len = len(text)
        text_len_counter[text_len] += 1
        lengths.append(text_len)
        audio_len = item[-1]
        try:
            text_vs_durs[text_len] += [audio_len]
        except:
            text_vs_durs[text_len] = [audio_len]

    text_vs_avg = {}
    text_vs_median = {}
    text_vs_std = {}
    for key, durs in text_vs_durs.items():
        text_vs_avg[key] = np.mean(durs)
        text_vs_median[key] = np.median(durs)
        text_vs_std[key] = np.std(durs)

    sec_per_chars = []
    for item in data:
        text = item[1]
        dur = item[-1]
        sec_per_char = dur / len(text)
        sec_per_chars.append(sec_per_char)

    mean = np.mean(sec_per_chars)
    std = np.std(sec_per_chars)
    print(mean)
    print(std)

    plot_text_length_vs_mean_audio_duration(text_vs_avg)
    plot_text_length_vs_std(text_vs_std)
    plot_distribution_of_text_lengths(lengths)

    cleaned_data = []
    durs_per_char = []
    for each in data:
        durs_per_char.append(each[-1] / each[2])
    durs_mean = np.mean(durs_per_char)
    durs_sd = np.std(durs_per_char)

    points = durs_per_char
    mean = durs_mean
    std_dev = durs_sd

    plot_speech_duration_per_character(points, mean, std_dev)

    minimum_duration = 0.7
    maximum_duration = 13.0
    maximum_sds = 2.5

    cleaned_data = []
    shorties = []
    longies = []
    misfits = []
    for item in data:
        item_perchar_dur = item[-1] / item[2]
        difference = abs(item_perchar_dur - durs_mean)
        item_zscore = difference / durs_sd
        item = item + (item_zscore,)
        if item[-2] < minimum_duration:
            shorties.append(item)
        elif item[-2] > maximum_duration:
            longies.append(item)
        elif item_zscore > maximum_sds:
            misfits.append(item)
        else:
            cleaned_data.append(item)

    excluded = shorties + longies + misfits

    print(
        f"found {len(shorties)} short items and {len(longies)} long items and {len(misfits)} items whose length conformed but whose per-char duration exceeded {maximum_sds} standard deviations from the mean. Excluding {len(shorties) + len(longies) + len(misfits)} items")

    ranked_shorts = sorted(shorties, key=lambda x: x[-2])
    if len(ranked_shorts) > 0:
        print(
            f"Duration of shortest item excluded for being too short: {ranked_shorts[0][-2]} Text from shortest item excluded for being too short: {ranked_shorts[0][1]}")
        print(ranked_shorts[0])
        print(
            f"Duration of longest item excluded for being too short: {ranked_shorts[-1][-2]} Text from shortest item excluded for being too short: {ranked_shorts[-1][1]}")
        print(ranked_shorts[-1])

    ranked_misfits = sorted(misfits, key=lambda x: x[-1])[::-1]
    if len(ranked_misfits) > 0:
        print(
            f"Duration of worst item excluded for having too much variance: {ranked_misfits[0][-2]} and its text: {ranked_misfits[0][1]}")
        print(ranked_misfits[0])
        print(
            f"Duration of best item excluded for having too much variance: {ranked_misfits[-1][-2]} and its text: {ranked_misfits[-1][1]}")
        print(ranked_misfits[-1])

    text_vs_durs = {}
    text_len_counter = Counter()
    lengths = []
    for item in tqdm(cleaned_data):
        text = item[1].lower().strip()
        text_len = len(text)
        text_len_counter[text_len] += 1
        lengths.append(text_len)
        audio_len = item[-2]
        try:
            text_vs_durs[text_len] += [audio_len]
        except:
            text_vs_durs[text_len] = [audio_len]

    text_vs_avg = {}
    text_vs_median = {}
    text_vs_std = {}
    for key, durs in text_vs_durs.items():
        text_vs_avg[key] = np.mean(durs)
        text_vs_median[key] = np.median(durs)
        text_vs_std[key] = np.std(durs)

    sec_per_chars = []
    for item in cleaned_data:
        text = item[1]
        dur = item[-2]
        sec_per_char = dur / len(text)
        sec_per_chars.append(sec_per_char)

    mean = np.mean(sec_per_chars)
    std = np.std(sec_per_chars)
    print(mean)
    print(std)

    plot_text_length_vs_mean_audio_duration(text_vs_avg)
    plot_text_length_vs_std(text_vs_std)
    plot_distribution_of_text_lengths(lengths)

    durs_per_char = []
    for each in cleaned_data:
        durs_per_char.append(each[-2] / each[2])
    durs_mean = np.mean(durs_per_char)
    durs_sd = np.std(durs_per_char)

    points = durs_per_char
    mean = durs_mean
    std_dev = durs_sd

    plot_speech_duration_per_character(points, mean, std_dev)

    remaining_length = sum([each[-2] for each in cleaned_data])
    excluded_length = sum([each[-2] for each in excluded])
    remaining_length, excluded_length

    plot_pie_chart_of_audio_proportions(remaining_length, excluded_length)

    def seconds_to_hms(seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        return hours, minutes, remaining_seconds

    hours, minutes, remaining_seconds = seconds_to_hms(remaining_length)
    print(f"Remaining length: {hours} hours, {minutes} minutes, and {remaining_seconds} seconds.")

    hours, minutes, remaining_seconds = seconds_to_hms(excluded_length)
    print(f"Excluded length: {hours} hours, {minutes} minutes, and {remaining_seconds} seconds.")

    speaker_reference = '/kaggle/input/janeeyre/wavs/jane_eyre_01_f000015.wav'
    audio, sr = librosa.load(os.path.join(training_dir, 'wavs', speaker_reference), sr=None)

    rawLength = [x[-2] for x in cleaned_data]

    print(f"seconds: {max(rawLength)}")
    print(f"maximum audio file length: {max(rawLength) * sr}")

    print(f"speaker reference file length: {len(audio)}")

    out_dir = '/kaggle/working/'

    excluded_files = set([each[0].split('/')[-1].split('.')[0] for each in excluded])

    dropped = 0
    sanitised_rows = []
    with open(os.path.join(training_dir, 'metadata.csv'), 'r') as file:
        csv_reader = csv.reader(file, delimiter='|')
        for row in csv_reader:
            if row[0] not in excluded_files:
                sanitised_rows.append(row)
            else:
                dropped += 1

    assert dropped == len(excluded_files)

    with open(os.path.join(out_dir, 'metadata.csv'), 'w', encoding='utf-8') as file:
        csv_writer = csv.writer(file, delimiter='|')
        for row in sanitised_rows:
            csv_writer.writerow(row)

    wavs_dir = os.path.join(out_dir, 'wavs')
    os.makedirs(wavs_dir, exist_ok=True)

    files = os.listdir(os.path.join(training_dir, 'wavs'))

    for file in files:
        source_file = os.path.join(training_dir, 'wavs', file)
        destination_file = os.path.join(wavs_dir, file)
        shutil.copy2(source_file, destination_file)

    deleted_files = 0
    for file_name in excluded_files:
        file_path = os.path.join(wavs_dir, file_name + '.wav')
        try:
            os.remove(file_path)
            deleted_files += 1
        except OSError as e:
            print(f'Error deleting file {file_path}: {e}')

    assert deleted_files == dropped
    print(f'deleted {deleted_files} wavs')