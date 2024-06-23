from pathlib import Path
import random
import warnings

import torch
from sklearn.metrics import f1_score
from torch import nn, optim
from tqdm import tqdm

import wandb
from  dataset_data_augmentation import get_dataloader, short_class_names
from model import ClassAudioClsModel, ClassAudioTaggingModel

warnings.filterwarnings("ignore")


# 訓練使用的函式
def train(model, data_loader):
    # 訓練時要將模型切換為訓練模式

    model.train()
    loss_list = []
    y_true, y_pred = [], []

    print(
        f"Starting iter of data_loader with total length of {len(data_loader)} items"
    )

    for i, (inputs,
            labels) in enumerate(tqdm(data_loader, total=len(data_loader))):
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type,
                            dtype=torch.float16, # use torch.bfloat16 if possible
                            enabled=config['use_amp']):
            input_values, attention_mask = inputs.input_values.to(
                device), inputs.attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_values, attention_mask)
            if config['mod'] == "seq_labling":
                labels = labels[:, :outputs.shape[1]]
                loss = criterion(outputs.permute(0, 2, 1), labels)
            else:
                loss = criterion(outputs, labels)

        y_true.append(labels)
        if config['mod'] == "seq_labling":
            y_pred.append(outputs.argmax(2))  # 取機率最大的類別爲預測結果
        else:
            y_pred.append(outputs.argmax(1))
        loss_list.append(loss.item())
        scaler.scale(loss).backward()
        if (i + 1) % config["grad_accum_steps"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=config['max_norm'])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            # wandb log
            wandb.log({
                'lr': optimizer.param_groups[0]['lr'],
            })

    y_pred = torch.cat(y_pred).flatten().cpu()
    y_true = torch.cat(y_true).flatten().cpu()
    loss = sum(loss_list) / len(loss_list)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    acc = (y_true == y_pred).sum().item() / len(y_true)
    wandb.log({
        'train_acc': acc,
        'train_loss': loss,
        'train_macro_f1': f1_macro,
        'train_micro_f1': f1_micro,
        'epoch': epoch
    })
    return loss, acc, f1_macro, f1_micro


# 驗證與測試使用的函式
@torch.no_grad()
def test(model, data_loader, validation=True):
    # 切換為驗證模式
    model.eval()
    # 驗證模型不用計算梯度，使用 torch.no_grad() 代表以下的程式運行不需要計算梯度(gradient)
    loss_list = []
    y_true, y_pred = [], []
    for inputs, labels in tqdm(data_loader, total=len(data_loader)):
        with torch.autocast(device_type=device.type,
                            dtype=torch.float16, # use torch.bfloat16 if possible
                            enabled=config['use_amp']):
            input_values, attention_mask = inputs.input_values.to(
                device), inputs.attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_values, attention_mask)
            if config['mod'] == "seq_labling":
                labels = labels[:, :outputs.shape[1]]
                loss = criterion(outputs.permute(0, 2, 1), labels)
            else:
                loss = criterion(outputs, labels)
        y_true.append(labels)
        if config['mod'] == "seq_labling":
            y_pred.append(outputs.argmax(2))  # 取機率最大的類別爲預測結果
        else:
            y_pred.append(outputs.argmax(1))
        loss_list.append(loss.item())

    y_pred = torch.cat(y_pred).flatten().cpu()
    y_true = torch.cat(y_true).flatten().cpu()
    loss = sum(loss_list) / len(loss_list)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    acc = (y_true == y_pred).sum().item() / len(y_true)
    if validation:
        wandb.log({
            'val_acc': acc,
            'val_loss': loss,
            'val_macro_f1': f1_macro,
            'val_micro_f1':f1_micro,
            'epoch': epoch
        })
    else:
        wandb.log({'test_acc': acc, 'test_loss': loss, "test_macro_f1": f1_macro, "test_micro_f1": f1_micro})

        wandb.log({
            "conf_mat":
            wandb.plot.confusion_matrix(probs=None,
                                        y_true=y_true.numpy(),
                                        preds=y_pred.numpy(),
                                        class_names=short_class_names)
        })

    return loss, acc, f1_macro, f1_micro


config = {
    'use_amp': True,
    'max_norm': 5,
    'workers': 4,  # cpu 數量
}

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--split_file", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--cache_dir", type=str, default=None)
parser.add_argument("--num_class", type=int, default=5)
parser.add_argument("--splits", type=int, default=5)

parser.add_argument("--model_checkpoint", type=str, required=True)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--segment_length", type=int, default=30)
parser.add_argument("--stride", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--grad_accum_steps", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-4)

# seed
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mod", type=str, default="seq-label")

if __name__ == "__main__":
    args = parser.parse_args()
    exp_dir = Path(args.output_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    # merge config
    # iterate over all the arguments
    for k, v in vars(args).items():
        config[k] = v

    assert config['mod'] in ["audio_cls", "seq_labling"]
    # 有 GPU 時就使用 GPU
    # 注意：模型與輸入必須在同個 device 上才能進行預測，模型預測和正確答案也必須在同個 device 才能計算 loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # group
    for split_num in range(config["splits"]):
        # set seed
        random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        train_loader, valid_loader, test_loader = get_dataloader(config, split_num)

        if config['mod'] == "audio_cls":
            tagging_model = ClassAudioClsModel(
                config["model_checkpoint"], num_classes=config["num_class"]).to(device)
        elif config['mod'] == "seq_labling":
            tagging_model = ClassAudioTaggingModel(
                config["model_checkpoint"], num_classes=config["num_class"]).to(device)

        optimizer = optim.Adam(
            tagging_model.parameters(),
            lr=config['lr'],
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], steps_per_epoch=len(train_loader)//config["grad_accum_steps"], epochs=config['num_epochs'])

        # 用在加速 Automatic Mixed Precision https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])
        # 定義損失函數
        # 這裡使用 CrossEntropyLoss，因為我們的模型是將每個類別的機率當作預測結果
        # 給予各個類別的權重，讓模型更專注在預測少數類別
        # if config['mod'] == "audio_cls":
        #     class_weight = torch.tensor([1.0, 1.0]).to(device)
        # else:
        #     class_weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
        criterion = nn.CrossEntropyLoss(
            # weight=class_weight  # 設定各類別的權重
        )

        # !wandb login d6b5146b32f8d1aada959ea889f74b2d9b9f74c5
        wandb.init(
            project="Class stage classification",
            # name=config['exp_name'],
            group=config['exp_name'],
            config=config,
            name=f"split_{split_num}",
            # mode="disabled"
        )

        best_f1 = 0
        log_file = exp_dir / 'training_logs.txt'
        best_state_dict_path = exp_dir / "best.pt"

        # with open(log_file, 'w') as file:
        #     file.write("go\n")
        for epoch in tqdm(range(config['num_epochs'])):
            print("Hopping in train")
            train_loss, train_acc, train_macro_f1, train_micro_f1 = train(tagging_model, train_loader)
            log = f'訓練集 Loss: {train_loss:.4f} Accuracy: {train_acc:.2f} Macro F1 score: {train_macro_f1:.2f} Micro F1 score: {train_micro_f1:.2f}\n'
            print(log)
            # with open(log_file, 'a') as file:
            #     file.write(log)
            val_loss, val_acc, val_macro_f1, val_micro_f1 = test(tagging_model, valid_loader)
            log = f'驗證集 Loss: {val_loss:.4f} Accuracy: {val_acc:.2f} Macro F1 score: {val_macro_f1:.2f} Micro F1 score: {val_micro_f1:.2f}\n'
            print(log)
            # with open(log_file, 'a') as file:
            #     file.write(log)
            state_dict_path = exp_dir / f"Epoch_{epoch}_loss_{val_loss:.3f}_f1_{val_macro_f1:.3f}.pt"
            torch.save(tagging_model.state_dict(), state_dict_path)
            if epoch == 0 or best_f1 < val_macro_f1:
                torch.save(tagging_model.state_dict(), best_state_dict_path)
                best_f1 = val_macro_f1

        # 在測試集上評估模型
        tagging_model.load_state_dict(torch.load(best_state_dict_path))
        print("Testing")
        test_loss, test_acc, test_macro_f1, test_micro_f1 = test(tagging_model,
                                            test_loader,
                                            validation=False)
        log = f'測試集 Loss: {test_loss:.4f} Accuracy: {test_acc:.2f} F1 score: {test_macro_f1:.2f} {test_micro_f1:.2f}\n'
        # print(log)
        # with open(log_file, 'a') as file:
        #     file.write(log)
        wandb.finish()
