# 載入 Transformer 模型
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForAudioFrameClassification

class ClassAudioClsModel(nn.Module):
    # pytorch 的模型類別至少需要定義 __init__ 和 forward 函式
    def __init__(self, model_checkpoint, num_classes):
        super().__init__()
        self.num_classes = num_classes
        config = AutoConfig.from_pretrained(model_checkpoint)
        config.use_weighted_layer_sum = True
        model = AutoModelForAudioFrameClassification.from_config(config)
        model.freeze_feature_encoder()
        model.freeze_base_model()
        model.classifier = nn.Sequential()
        self.model = model
        hidden_size = model.config.hidden_size


        # 在 transformer 模型後加三層 LSTM
        # 768 是 feature_extractor 的輸出維度
        self.lstm = nn.LSTM(hidden_size, 512, num_layers=3, batch_first=True)
        # attention pooling
        self.W = nn.Linear(512, 1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, input_values, attention_mask):
        # input_values: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len)
        # output: (batch_size, 1, num_classes)
        x = self.model(input_values=input_values,
                       attention_mask=attention_mask).logits
        x, _ = self.lstm(x)
        softmax = nn.functional.softmax
        att_w = self.W(x).squeeze(-1)
        att_w = softmax(att_w, dim=1).unsqueeze(-1)
        x = torch.sum(x * att_w, dim=1)
        x = self.classifier(x)
        x = F.softmax(x, dim=1)  # 讓各類別的數值轉為機率(總和為一)

        return x

class ClassAudioTaggingModel(nn.Module):
    # pytorch 的模型類別至少需要定義 __init__ 和 forward 函式
    def __init__(self, model_checkpoint, num_classes):
        super().__init__()
        self.num_classes = num_classes
        config = AutoConfig.from_pretrained(model_checkpoint)
        config.use_weighted_layer_sum = True
        model = AutoModelForAudioFrameClassification.from_config(config)
        model.freeze_feature_encoder()
        model.freeze_base_model()
        model.classifier = nn.Sequential()
        self.model = model
        hidden_size = model.config.hidden_size
        # 在 transformer 模型後加三層 LSTM
        # 768 是 feature_extractor 的輸出維度
        self.lstm = nn.LSTM(hidden_size, 512, num_layers=3, batch_first=True)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, input_values, attention_mask):
        # input_values: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len)
        # output: (batch_size, (seq_len//320)-1, num_classes)
        x = self.model(input_values=input_values,
                       attention_mask=attention_mask).logits
        x, _ = self.lstm(x)
        x = self.classifier(x)
        x = F.softmax(x, dim=2)  # 讓各類別的數值轉為機率(總和為一)

        return x


if __name__ == "__main__":
    model_checkpoint = "microsoft/wavlm-base-plus"
    num_classes = 5
    # 輸入尺寸 (160000)
    # 輸出尺寸 (499)
    # 公式為 (160000 // 320)-1
    input_values = torch.randn(1, 160000)  # 將輸入數據轉換為 float32
    attention_mask = torch.ones(1, 160000, dtype=torch.bool)
    model = ClassAudioClsModel(model_checkpoint, num_classes)
    output = model(input_values, attention_mask)
    print(output.shape)

    model = ClassAudioTaggingModel(model_checkpoint, num_classes)
    output = model(input_values, attention_mask)
    print(output.shape)
