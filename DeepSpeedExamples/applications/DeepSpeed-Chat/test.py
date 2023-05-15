from typing import Any
import datasets
from torch.utils.data import Subset
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.utils.data import Dataset, DataLoader


# class Datasets(Dataset):
#     def __init__(self, split) -> None:
#         super().__init__()
#         self.dataset = datasets.load_from_disk("/home/WangXu/huggingface_data/Hello-Simple/miracl-zh-queries-22-12")[split]
    
#     def __getitem__(self, index) -> Any:

#         return super().__getitem__(index)

# print(f'Hello-Simple/miracl-zh-queries-22-12 has {" ".join(list(dataset.keys()))}')
# dataset = dataset['train']
# dataset = dataset.select(range(5))
# print(dataset)

# tokenizer = AutoTokenizer.from_pretrained('/home/WangXu/huggingface_model/chatYuan-our')
# print(tokenizer)
# vocab = tokenizer.get_vocab()
# print(f'vocab length is {len(vocab)}')

# res = [sent[0] for sent in dataset['human_answers']]

# dataset_out = tokenizer.batch_encode_plus(batch_text_or_text_pairs=res, 
#                                           max_length=150, 
#                                           truncation=True,
#                                           padding='max_length', 
#                                           return_attention_mask=True,
#                                           return_length=True,
#                                           return_special_tokens_mask=True,
#                                           return_tensors='pt', 
#                                           add_special_tokens=True)
# # res = tokenizer.batch_decode(dataset_out['input_ids'])
# # print(res)

# model = AutoModel.from_pretrained('/home/WangXu/huggingface_model/chatYuan-our', trust_remote_code=True)
# print(model)


# dataset = datasets.load_dataset('BelleGroup/train_2M_CN')
# dataset.save_to_disk('/home/WangXu/huggingface_data/belle2M/')
# dataset = datasets.load_from_disk('/home/WangXu/huggingface_data/belle2M')['train']
# print(dataset)

from transformers import AutoModel, AutoTokenizer, AutoConfig
model = AutoModel.from_pretrained('mosaicml/mpt-7b')
tokenier = AutoTokenizer.from_pretrained('mosaicml/mpt-7b')
config = AutoConfig.from_pretrained('mosaicml/mpt-7b')
print(model)
print(tokenier)
print(config)

