import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertTokenizer, BertModel


class ImgEncoder(nn.Module):
    def __init__(self, embed_size):
        """(1) Load the pretrained model as you want.
           (2) Replace final fc layer (score values from the ImageNet)
               with new fc layer (image feature).
           (3) Normalize feature vector.
        """
        super(ImgEncoder, self).__init__()
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features  # input size of feature vector
        model.fc = nn.Identity()  # remove last fc layer

        self.model = model  # loaded model without last fc layer
        self.fc = nn.Linear(in_features, embed_size)  # feature vector of image

    def forward(self, image):
        """Extract feature vector from image vector.
        """
        with torch.no_grad():
            img_feature = self.model(image)  # [batch_size, resnet18_fc=512]

        img_feature = self.fc(img_feature)  # [batch_size, embed_size]

        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        img_feature = img_feature.div(l2_norm)  # l2-normalized feature vector

        return img_feature


class QstEncoder(nn.Module):
    def __init__(self, embed_size):
        super(QstEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
        self.bert = BertModel.from_pretrained('./bert-base-uncased')
        self.fc = nn.Linear(768, embed_size)  # Adjust output size as needed

    def forward(self, input_indices):
        input_text_list = [self.tokenizer.decode(indices.tolist()) for indices in input_indices]
        inputs = self.tokenizer(input_text_list, return_tensors='pt', padding=True, truncation=True, max_length=128,
                                add_special_tokens=True)
        inputs = {key: value.to(self.bert.device) for key, value in inputs.items()}
        outputs = self.bert(**inputs)
        bert_output = outputs.last_hidden_state.mean(dim=1)
        output_embedding = self.fc(bert_output)
        output_embedding = output_embedding.to(input_indices.device)
        return output_embedding


class VqaModel(nn.Module):

    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size):

        super(VqaModel, self).__init__()
        self.img_encoder = ImgEncoder(embed_size)
        self.qst_encoder = QstEncoder(embed_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    def forward(self, img, qst):

        img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
        qst_feature = self.qst_encoder(qst)                     # [batch_size, embed_size]
        combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size=1000]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc2(combined_feature)           # [batch_size, ans_vocab_size=1000]

        return combined_feature