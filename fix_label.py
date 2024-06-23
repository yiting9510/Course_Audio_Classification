from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import librosa


parser = ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/mnt/ext_disk/dhf/data(1)")

args = parser.parse_args()


if __name__ == "__main__":
    label_dir = Path(args.data_dir) / "label"
    label_list = sorted(label_dir.glob("*.txt"))
    audio_dir = Path(args.data_dir) / "audio"
    audio_list = sorted(audio_dir.glob("*.wav"))
    fixed_label_dir = Path(args.data_dir) / "label_fixed"
    fixed_label_dir.mkdir(parents=True, exist_ok=True)
    for label_path, audio_path in zip(label_list, audio_list):
        label_df = pd.read_csv(label_path, header=None, sep="\s+")
        label_duration = label_df.iloc[-1, 1]
        audio_duration = librosa.get_duration(path=audio_path)
        filename = audio_path.stem
        if audio_duration != label_duration:
            print(
                filename,
                "audio_duration < label_duration",
                audio_duration - label_duration,
            )
            label_df.iloc[-1, 1] = audio_duration
        
        # 6 digits
        label_df.iloc[:, 0] = label_df.iloc[:, 0].apply(lambda x: f"{x:.6f}")
        label_df.iloc[:, 1] = label_df.iloc[:, 1].apply(lambda x: f"{x:.6f}")
        label_df.to_csv(
            fixed_label_dir / label_path.stem, header=None, index=None, sep="\t"
        )
