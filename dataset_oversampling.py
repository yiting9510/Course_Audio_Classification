import os
import json

import librosa
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoFeatureExtractor
import pandas as pd


def load_label(label_path, class_dict):
    df = pd.read_csv(label_path, sep="\s+", header=None)
    df[0] = df[0].astype(float)
    df[1] = df[1].astype(float)
    df[2] = df[2].apply(lambda x: class_dict[x])
    stage_starts = df[0].tolist()
    stage_ends = df[1].tolist()
    stage_labels = df[2].tolist()

    return stage_starts, stage_ends, stage_labels


class AudioDataset(Dataset):
    def __init__(self, label_files, class_dict, config, oversample=False):

        if config["cache_dir"]:
            self.use_cache = True
        else:
            self.use_cache = False
        self.audio_files = [label_path.replace(".txt", ".wav").replace("label", "audio") for label_path in label_files]
        self.label_files = label_files
        self.cache_dir = config["cache_dir"]
        self.segment_length = config["segment_length"]
        self.stride = config["stride"]
        self.mod = config["mod"]
        self.pairs = []
        self.label_sequence_dict = {}
        self.oversample = oversample

        for audio_file, label_file in zip(self.audio_files, self.label_files):
            stage_starts, stage_ends, stage_labels = load_label(label_file, class_dict)

            class_end = stage_ends[-1]
            segment_start = stage_starts[0]
            segment_end = segment_start + self.segment_length
            segment_num = int((stage_ends[-1] - self.segment_length) / self.stride)
            for _ in range(segment_num):
                self.pairs.append(
                    (audio_file, segment_start, min(segment_end, class_end))
                )
                segment_start += self.stride
                segment_end += self.stride

            label_sequence = []
            for stage_start, stage_end, stage_label in zip(
                stage_starts, stage_ends, stage_labels
            ):
                label_sequence += [stage_label] * int(
                    (stage_end - stage_start) * SAMPLING_RATE / MODEL_DOWNSAMPLE_RATE
                )
            self.label_sequence_dict[audio_file] = torch.tensor(
                label_sequence, dtype=torch.long
            )

        if self.oversample:
            self._oversample()

    def _oversample(self):
        # 收集所有片段的標籤
        labels = [self.label_sequence_dict[pair[0]][int(pair[1] * SAMPLING_RATE / MODEL_DOWNSAMPLE_RATE): int(pair[2] * SAMPLING_RATE / MODEL_DOWNSAMPLE_RATE)] for pair in self.pairs]
        
        # 計算每個類別的樣本數量
        label_counts = torch.bincount(torch.cat(labels))
        
        # 找到最多樣本數量的類別
        max_count = label_counts.max().item()

        # 新增 pairs 清單以存放重複的樣本
        new_pairs = []
        
        # 對每個類別進行處理
        for label in range(len(label_counts)):
            # 如果該類別的樣本數量少於最大樣本數量
            if label_counts[label] < max_count:
                # 找出該類別的所有片段
                label_pairs = [pair for pair in self.pairs if self.label_sequence_dict[pair[0]][int(pair[1] * SAMPLING_RATE / MODEL_DOWNSAMPLE_RATE): int(pair[2] * SAMPLING_RATE / MODEL_DOWNSAMPLE_RATE)].mode().values.item() == label]
                
                # 計算需要重複的次數
                multiplier = (max_count // 3) // label_counts[label].item()
                
                # 重複該類別的片段
                new_pairs.extend(label_pairs * (multiplier - 1))
        
        # 將重複的樣本加入到 pairs 清單中
        self.pairs.extend(new_pairs)


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        audio_file, start, end = self.pairs[index]

        start_index, end_index = int(start * SAMPLING_RATE), int(end * SAMPLING_RATE)

        if self.use_cache:
            filename = os.path.splitext(os.path.basename(audio_file))[0]
            cache_path = os.path.join(self.cache_dir, filename, f"{start}-{end}.wav")
            assert os.path.exists(cache_path)
            audio, _ = librosa.load(cache_path, sr=SAMPLING_RATE)
        else:
            audio, _ = librosa.load(audio_file, sr=SAMPLING_RATE)
            audio = audio[start_index:end_index]

        label_start_index = int(start_index / MODEL_DOWNSAMPLE_RATE)
        label_end_index = int(end_index / MODEL_DOWNSAMPLE_RATE)
        label = self.label_sequence_dict[audio_file][label_start_index:label_end_index]
        if self.mod == "audio_cls":
            label = torch.mode(label).values

        return audio, label

    def create_cache(self):
        cache_dir = self.cache_dir
        last_audio_file = None
        for audio_file, start, end in tqdm(self.pairs, total=len(self.pairs)):
            filename = os.path.splitext(os.path.basename(audio_file))[0]
            cache_path = os.path.join(cache_dir, filename, f"{start}-{end}.wav")
            if os.path.exists(cache_path):
                continue
            if audio_file != last_audio_file:
                os.makedirs(os.path.join(cache_dir, filename), exist_ok=True)
                wav, _ = librosa.load(audio_file, sr=SAMPLING_RATE)
                last_audio_file = audio_file
            start_index, end_index = int(start * SAMPLING_RATE), int(
                end * SAMPLING_RATE
            )
            wav_segment = wav[start_index:end_index]
            sf.write(cache_path, wav_segment, SAMPLING_RATE)


class TrainAudioDataset(AudioDataset):
    def __init__(self, label_files, class_dict, config, oversample=False):
        super().__init__(
            label_files, class_dict, config=config, oversample=oversample
        )


class InferAudioDataset(Dataset):
    def __init__(self, label_file, class_dict, config):
        self.audio_file = label_file
        self.segment_length = config["segment_length"]
        self.stride = config["stride"]

        self.audio, _ = librosa.load(label_file, sr=SAMPLING_RATE)
        duration = self.audio.shape[0] / SAMPLING_RATE
        segment_num = int((duration - self.segment_length) // self.stride + 1)
        self.pairs = []
        segment_start, segment_end = 0, self.segment_length
        for _ in range(segment_num):
            a, b = segment_start, min(segment_end, duration)
            self.pairs.append((a, b))
            segment_start += self.stride
            segment_end += self.stride

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        start, end = self.pairs[index]

        start_index, end_index = int(start * SAMPLING_RATE), int(end * SAMPLING_RATE)
        audio = self.audio[start_index:end_index]
        pred_start_i = int(start * SAMPLING_RATE / MODEL_DOWNSAMPLE_RATE)
        pred_end_i = int(end * SAMPLING_RATE / MODEL_DOWNSAMPLE_RATE)
        return audio, pred_start_i, pred_end_i - 1

    def collate_fn(self, batch):
        audios, start_indexes, end_indexes = zip(*batch)
        audios = feature_extractor(
            audios, sampling_rate=SAMPLING_RATE, padding=True, return_tensors="pt"
        )
        start_indexes = torch.tensor(start_indexes)
        end_indexes = torch.tensor(end_indexes)
        return audios, start_indexes, end_indexes


def collate_fn(batch):
    audios, labels = zip(*batch)
    audios = feature_extractor(
        audios, sampling_rate=SAMPLING_RATE, padding=True, return_tensors="pt"
    )
    if labels[0].ndim == 2:
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
    else:
        labels = torch.stack(labels)
    return audios, labels


SAMPLING_RATE = 16000
MODEL_DOWNSAMPLE_RATE = 320
class_names = [
    "Introduction_or_Opening",
    "Lecture_or_Presentation",
    "Break_or_Transition",
    "Conclusion_or_Summary",
    "Others",
]
inv_class_dict = {i: label for i, label in enumerate(class_names)}
short_class_names = ["Opening", "Lecture", "Break", "Conclusion", "Others"]
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")


def get_dataloader(config, split_num=0):
    assert config["num_class"] in [2, 5]

    label_file = config["split_file"]
    with open(label_file, "r") as f:
        all_split = json.load(f)

    split = all_split[str(split_num)]
    train_label_list = split["train"]
    valid_label_list = split["valid"]
    test_label_list = split["test"]

    if config["num_class"] == 2:
        class_dict = {
            "Introduction_or_Opening": 0,
            "Lecture_or_Presentation": 0,
            "Break_or_Transition": 1,
            "Conclusion_or_Summary": 0,
            "Others": 1,
        }
        class_names = ["Lecture_or_Presentation", "Others"]
        short_class_names = ["Lecture", "Others"]
        inv_class_dict = {0: "Lecture_or_Presentation", 1: "Others"}

    elif config["num_class"] == 5:
        class_dict = {
            "Introduction_or_Opening": 0,
            "Lecture_or_Presentation": 1,
            "Break_or_Transition": 2,
            "Conclusion_or_Summary": 3,
            "Others": 4,
        }
        class_names = ["Introduction_or_Opening", "Lecture_or_Presentation", "Break_or_Transition", "Conclusion_or_Summary", "Others"]
        short_class_names = ["Opening", "Lecture", "Break", "Conclusion", "Others"]
        inv_class_dict = {i: label for i, label in enumerate(class_names)}

    print(f"Number of train files: {len(train_label_list)}")
    print(f"Number of valid files: {len(valid_label_list)}")
    print(f"Number of test files: {len(test_label_list)}")

    train_dataset = TrainAudioDataset(
        train_label_list,
        class_dict=class_dict,
        config=config,
        oversample=True,
    )
    valid_dataset = AudioDataset(valid_label_list, class_dict=class_dict, config=config)
    test_dataset = AudioDataset(test_label_list, class_dict=class_dict, config=config)

    if config['cache_dir']:
        print("Creating cache ...")
        os.makedirs(config['cache_dir'], exist_ok=True)
        train_dataset.create_cache()
        valid_dataset.create_cache()
        test_dataset.create_cache()

    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=config["workers"],
                              pin_memory=True,
                              persistent_workers=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=collate_fn,
                              num_workers=config["workers"],
                              pin_memory=True,
                              persistent_workers=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=collate_fn,
                             num_workers=config["workers"],
                             pin_memory=True,
                             persistent_workers=True)

    return train_loader, valid_loader, test_loader


def get_inference_dataloader(audio_path, config):
    if config["num_class"] == 2:
        class_dict = {
            "Introduction_or_Opening": 0,
            "Lecture_or_Presentation": 0,
            "Break_or_Transition": 1,
            "Conclusion_or_Summary": 0,
            "Others": 1,
        }
        class_names = ["Lecture_or_Presentation", "Others"]
        short_class_names = ["Lecture", "Others"]
        inv_class_dict = {0: "Lecture", 1: "Others"}

    elif config["num_class"] == 5:
        class_dict = {
            "Introduction_or_Opening": 0,
            "Lecture_or_Presentation": 1,
            "Break_or_Transition": 2,
            "Conclusion_or_Summary": 3,
            "Others": 4,
        }
        class_names = ["Introduction_or_Opening", "Lecture_or_Presentation", "Break_or_Transition", "Conclusion_or_Summary", "Others"]
        short_class_names = ["Opening", "Lecture", "Break", "Conclusion", "Others"]
        inv_class_dict = {i: label for i, label in enumerate(class_names)}
    infer_dataset = InferAudioDataset(audio_path, class_dict, config=config)
    infer_loader = DataLoader(
        infer_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=infer_dataset.collate_fn,
        num_workers=config["workers"],
        pin_memory=True,
        persistent_workers=True,
    )

    return infer_loader
