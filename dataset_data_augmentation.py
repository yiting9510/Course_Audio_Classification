import os
import json

import librosa
import soundfile as sf
import torch
from audiomentations import AddGaussianSNR, Compose, TimeStretch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoFeatureExtractor
import pandas as pd


def load_label(label_path, class_dict):
    # 載入標註，轉換為三個 list
    # 段落起點, 段落終點, 類別
    # print(class_dict)
    # print("正在處理標籤檔案：", label_path)
    df = pd.read_csv(label_path, sep="\s+", header=None)
    df[0] = df[0].astype(float)
    df[1] = df[1].astype(float)
    df[2] = df[2].apply(lambda x: class_dict[x])
    stage_starts = df[0].tolist()
    stage_ends = df[1].tolist()
    stage_labels = df[2].tolist()

    return stage_starts, stage_ends, stage_labels


# 建立 Dataset 的類別，它負責將準備好的音檔和標註轉換成模型能接受的格式
# Custom dataset 官方英文教學: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class AudioDataset(Dataset):
    # pytorch 的 Dataset 類別至少需要定義 __init__、__len__ 和 __getitem__ 函式
    def __init__(self, label_files, class_dict, config, transform=None):

        # print("Init audio dataset!")
        """
        Args:
            audio_files (list): 音檔路徑的 list
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
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
        # transform 用於定義預處理操作，例如 data augmentation
        self.transform = transform
        self.pairs = []
        self.label_sequence_dict = {}
        # 將資料轉換成上述格式
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

        

    def __len__(self):
        # 說明此資料集有幾筆資料，len(self.pairs) 代表此資料集的 segment 數量
        return len(self.pairs)

    def __getitem__(self, index):
        # 此函式要輸出音檔片段的 numpy array 以及轉換成整數的標註

        # 取得聲音文件的檔名和標籤
        audio_file, start, end = self.pairs[index]

        # 將秒數轉換為取樣點數
        start_index, end_index = int(start * SAMPLING_RATE), int(end * SAMPLING_RATE)

        # 讀取聲音文件
        # 讀取分割好的音檔
        if self.use_cache:
            filename = os.path.splitext(os.path.basename(audio_file))[0]
            cache_path = os.path.join(self.cache_dir, filename, f"{start}-{end}.wav")
            assert os.path.exists(cache_path)
            audio, _ = librosa.load(cache_path, sr=SAMPLING_RATE)
        # 讀取完整的音檔
        else:
            audio, _ = librosa.load(audio_file, sr=SAMPLING_RATE)
            audio = audio[start_index:end_index]

        # 如果有定義 transform，則應用 transform
        if self.transform:
            audio = self.transform(samples=audio, sample_rate=SAMPLING_RATE)

        label_start_index = int(start_index / MODEL_DOWNSAMPLE_RATE)
        label_end_index = int(end_index / MODEL_DOWNSAMPLE_RATE)
        label = self.label_sequence_dict[audio_file][label_start_index:label_end_index]
        if self.mod == "audio_cls":
            # 眾數
            label = torch.mode(label).values

        return audio, label

    def create_cache(self):
        cache_dir = self.cache_dir
        # os.makedirs(cache_dir, exist_ok=True)
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
            # 分割音檔的格式為: CACHE_DIR \\ 音檔名稱 \\ 起始秒數-結束秒數.wav
            sf.write(cache_path, wav_segment, SAMPLING_RATE)


class TrainAudioDataset(AudioDataset):
    # 重新定義建構式
    def __init__(self, label_files, class_dict, config, transform=None):
        # 調用父類別的建構式
        super().__init__(
            label_files, class_dict, config=config, transform=transform
        )

        # for audio_file, segment_start, segment_end in self.pairs:
        #     class_multiplier = class_multiplier[aux_label]
        #     duplicated_pair = [
        #         (audio_file, segment_start, segment_end, aux_label)
        #     ] * class_multiplier
        #     self.pairs.extend(duplicated_pair)


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

        # 將秒數轉換為取樣點數
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


# 從 Dataset 取出的資料會經果此函式整理，輸出型別為 pytorch tensor 的模型輸入以及標註
def collate_fn(batch):
    # 將 batch 中的每個元素取出
    audios, labels = zip(*batch)

    # 使用 huggingface 的 feature_extractor，將一個批次的音檔轉換為 pytorch tensor
    # 輸出的 shape 為 (batch_size, max_length)
    # sampling_rate 是設定輸入音檔的採樣率
    # max_length 是音檔的 tensor 最大長度
    # truncation 為 True 代表大於 max_length 的音檔將從後截斷
    # padding 為 True 代表長度不及 max_length 的 tensor ，將補零到 max_length
    # return_tensors 為 "pt" 代表指定 feature_extractor 會反還 pytorch tensor
    audios = feature_extractor(
        audios, sampling_rate=SAMPLING_RATE, padding=True, return_tensors="pt"
    )

    # 指定將 labels 轉換為 pytorch tensor
    # 填充 labels 到同一長度
    if labels[0].ndim == 2:
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
    else:
        labels = torch.stack(labels)
    return audios, labels


SAMPLING_RATE = 16000
MODEL_DOWNSAMPLE_RATE = 320
# 訓練時 label 必須是 0~(NUM_CLASSES-1) 的整數
# 這邊設定字典將自訂的類別名稱轉換為整數
class_names = [
    "Introduction_or_Opening",
    "Lecture_or_Presentation",
    "Break_or_Transition",
    "Conclusion_or_Summary",
    "Others",
]
inv_class_dict = {i: label for i, label in enumerate(class_names)}
short_class_names = ["Opening", "Lecture", "Break", "Conclusion", "Others"]
# 用來平衡不平衡資料的類別權重
# class_multiplier = {
#     "Introduction_or_Opening": 25,
#     "Lecture_or_Presentation": 0,
#     "Break_or_Transition": 26,
#     "Conclusion_or_Summary": 20,
#     "Others": 8
# }
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

    # 訓練資料集使用以下的資料增強手法
    augment = Compose(
        [
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            AddGaussianSNR(min_snr_db=5.0, max_snr_db=40.0, p=0.5),
        ]
    )

    # 使用分割好的音檔建立資料集
    train_dataset = TrainAudioDataset(
        train_label_list,
        class_dict=class_dict,
        config=config,
        transform=augment,
    )
    valid_dataset = AudioDataset(valid_label_list, class_dict=class_dict, config=config)
    test_dataset = AudioDataset(test_label_list, class_dict=class_dict, config=config)

    # 事先分割成短音檔，提升訓練時載入資料的速度
    # 會使用額外空間，要注意空間夠不夠
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
