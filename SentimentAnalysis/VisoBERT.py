from transformers import AutoModel, AutoTokenizer
import torch

if __name__ == '__main__':
    model = AutoModel.from_pretrained('uitnlp/visobert')
    tokenizer = AutoTokenizer.from_pretrained('uitnlp/visobert')

    encoding = tokenizer('hào quang rực rỡ', return_tensors='pt')

    with torch.no_grad():
        output = model(**encoding)