import random
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import json
from typing import Dict, List


def split_data(
    dev_label_list: List[str], valid_ratio: float, num_class: int, num_split: int
) -> Dict[int, Dict[str, List[str]]]:
    file_label_set = {}
    duration = {}
    if num_class == 2:
        class_dict = {
            "Introduction_or_Opening": 0,
            "Lecture_or_Presentation": 0,
            "Break_or_Transition": 1,
            "Conclusion_or_Summary": 0,
            "Others": 1,
        }
    elif num_class == 5:
        class_dict = {
            "Introduction_or_Opening": 0,
            "Lecture_or_Presentation": 1,
            "Break_or_Transition": 2,
            "Conclusion_or_Summary": 3,
            "Others": 4,
        }

    for i, label_path in enumerate(dev_label_list):
        df = pd.read_csv(label_path, sep="\s+", header=None)
        duration[i] = df.iloc[-1, 1]
        file_label_set[i] = df.iloc[:, 2].apply(lambda x: class_dict[x]).unique()

    indices = list(range(len(dev_label_list)))
    splits = {}
    expect_duraion_ratio = (1 - valid_ratio) / valid_ratio
    while len(splits) < num_split:
        random.shuffle(indices)
        train_indices = indices[: int(len(indices) * (1-valid_ratio))]
        valid_indices = indices[int(len(indices) * (1-valid_ratio)) :]

        train_label_set = set()
        for i in train_indices:
            train_label_set = train_label_set.union(file_label_set[i])
        valid_label_set = set()
        for i in valid_indices:
            valid_label_set = valid_label_set.union(file_label_set[i])
        if len(train_label_set) != num_class or len(valid_label_set) != num_class:
            continue

        # total duration of train
        train_duration = sum([duration[i] for i in train_indices])
        valid_duration = sum([duration[i] for i in valid_indices])
        duration_ratio = train_duration / valid_duration
        if expect_duraion_ratio - 0.5 < duration_ratio < expect_duraion_ratio + 0.5:
            train_label_list = [str(dev_label_list[i]) for i in train_indices]
            valid_label_list = [str(dev_label_list[i]) for i in valid_indices]
            splits[len(splits)] = {
                "train": train_label_list,
                "valid": valid_label_list,
                "duration_ratio": round(duration_ratio, 3),
            }

    return splits


parser = ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--num_class", type=int, required=True)
parser.add_argument("--valid_ratio", type=float, default=0.2)
parser.add_argument("--split_num", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

# predifined test files
test_files = [
    "(list_8)〔高二物化〕線上直播課程(00_01_29_40)",
    "(list_5)〔高一英文〕線上直播課程(00_00_27_45)",
    "Label1(酷課雲公民停課不停學)",
    "X2Download.app - [學盟文教]國九總複習師資-弘理社會(李偉歷史) (128 kbps)",
    "(list_4)〔高一國文〕線上直播課程(00_00_51_59)",
    "List_4【112統測直播】英文",
    "List_2【111統測重點複習課程– 共同英文】",
    "List_5【112統測直播】設計群專二 基礎圖學",
    "停課不停學〔高一物理〕線上直播課程 (1)",
    "List_9【111統測重點複習課程– 數學C】",
]

if __name__ == "__main__":
    assert args.num_class in [2, 5]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    label_dir = Path(args.data_dir) / "label"
    label_list = sorted(label_dir.glob("*.txt"))

    test_label_list: List[str] = []
    for test_file in test_files:
        if label_dir / f"{test_file}.txt" in label_list:
            test_label_list.append(str(label_dir / f"{test_file}.txt"))
        else:
            assert False, f"Test file {test_file} not found"

    # the rest of the files are dev files, split them into train and valid
    dev_label_list = [
        str(label) for label in label_list if str(label) not in test_label_list
    ]
    splits = split_data(
        dev_label_list, args.valid_ratio, args.num_class, args.split_num
    )
    for i, split in enumerate(splits):
        splits[i]["test"] = [str(label) for label in test_label_list]

    # write json
    filename = f"{args.num_class}-class-split.json"
    with open(args.output_dir / filename, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=4, ensure_ascii=False)
