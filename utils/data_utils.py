import os
import random
import torch
import sys
from datasets import load_dataset
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

def get_calib_train_data(name, tokenizer, nsamples, seqlen=2048, seed=3, batch_size=1, dataset_cache_dir=None):
    import random
    random.seed(seed)
    cache_file = (
        f"cache/{name}_{nsamples}_{seqlen}_{seed}_{batch_size}.pt"
    )
    nsamples += 1 #############################
    if not os.path.exists("cache"):
        os.makedirs("cache")
    if os.path.exists(cache_file):
        traindataset = torch.load(cache_file)
        return traindataset
    if name == "c4":
        traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
        tot_text = "\n\n".join(traindata["text"])
    elif name == "ptb":
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
        tot_text = "\n\n".join(traindata["sentence"])
    elif name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=dataset_cache_dir)
        tot_text = "\n\n".join(traindata["text"])
    else:
        raise NotImplementedError
    traindataset = []
    for s in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            s = s - 1
            continue
        if s % batch_size == 0:
            if s != 0:
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
            inp = trainenc.input_ids[:, :seqlen]
        else:
            inp = torch.cat((inp, trainenc.input_ids[:, :seqlen]), dim=0)
    torch.save(traindataset, cache_file)
    return traindataset



def get_wikitext2(nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=dataset_cache_dir)
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir=dataset_cache_dir)

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation', cache_dir=dataset_cache_dir)

    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
    # traindata = load_dataset('json', data_files={'train': 'utils/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset("json", data_files="utils/c4-validation.json")['train']

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 



def get_ptb_new(nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test', cache_dir=dataset_cache_dir)

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
    valdata = load_dataset("json", data_files="utils/c4-validation.json")['train']

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, tokenizer)
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, tokenizer)
        return get_c4(nsamples, seed, seqlen, tokenizer)
    
    
    
def get_test_data(name, tokenizer, seq_len=2048, batch_size = 4):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)
    ####
    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)
    ####
    if 'wikitext2' in name:
        test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'ptb' in name:
        test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    elif 'c4' in name:
        test_data = load_dataset("json", data_files="utils/c4-validation.json")['train']
        test_dataset = process_data(test_data[0:2000], tokenizer, seq_len, 'text')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def get_train_loader_for_causal_lm(dataset_name, tokenizer, seq_len=2048, batch_size=4, nsamples=None, seed=3):
    """
    加载并处理用于Causal LM训练的数据，从数据集中随机抽取独立的文档样本。
    """
    
    print(f"Loading '{dataset_name}' train split...")
    if 'wikitext' in dataset_name:
        train_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        field_name = 'text'
    elif 'ptb' in dataset_name:
        train_data = load_dataset('ptb_text_only', 'penn_treebank', split='train')
        field_name = 'sentence'
    elif 'c4' in dataset_name:
        # 注意：此处假设C4数据集已在本地准备好
        # train_data = load_dataset('json', data_files={'train': './allenai/c4/en/c4-train.00000-of-01024.json.gz'}, split='train')
        train_data = load_dataset("json", data_files="utils/c4-train.json")['train']
        field_name = 'text'
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported in this example.")

    # 1. 对数据集进行一次性随机洗牌
    # print(f"Shuffling dataset with seed {seed}...")
    shuffled_data = train_data.shuffle(seed=seed)
    
    print(f"Processing shuffled dataset to find samples with at least {seq_len} tokens...")
    
    train_chunks = []
    num_skipped = 0

    # 2. 高效地遍历一次洗牌后的数据集
    for sample in shuffled_data:
        text = sample[field_name] # 使用正确的 field_name
        if not text:
            num_skipped += 1
            continue
            
        token_ids_list = tokenizer.encode(text)

        if len(token_ids_list) >= seq_len:
            # 截取后，需要保持 (1, seq_len) 的形状以匹配DataLoader
            chunk = torch.tensor(token_ids_list[:seq_len]).unsqueeze(0) 
            train_chunks.append(chunk)
        else:
            num_skipped += 1
        
        if nsamples is not None and len(train_chunks) >= nsamples:
            print(f"\nReached the target number of samples: {nsamples}.")
            break

    print(f"Processing finished. Found {len(train_chunks)} samples.")

    class CausalLMDataset(Dataset):
        def __init__(self, data_chunks):
            # 将 list of tensors 合并成一个大的 tensor
            self.data_chunks = torch.cat(data_chunks, dim=0)

        def __getitem__(self, index):
            return {
                "input_ids": self.data_chunks[index],
                "labels": self.data_chunks[index].clone()
            }

        def __len__(self):
            return len(self.data_chunks)

    train_dataset = CausalLMDataset(train_chunks)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print("Train loader created successfully.")
    return train_loader


# 定义一个 Alpaca 格式化模板
ALPACA_PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)

def get_train_loader_for_alpaca(dataset_path, tokenizer, seq_len=512, batch_size=4, nsamples=None, seed=3):
    """
    加载并处理用于 Alpaca 指令微调的数据。
    """
    
    print(f"Loading Alpaca-style dataset from '{dataset_path}'...")
    # 1. 加载 JSON 格式的数据集
    train_data = load_dataset('json', data_files=dataset_path, split='train')
    
    shuffled_data = train_data.shuffle(seed=seed)
    
    print(f"Processing shuffled dataset for instruction-tuning...")
    
    all_input_ids = []
    all_labels = []

    # 2. 遍历数据集，格式化并分别进行分词
    for sample in shuffled_data:
        # 根据模板格式化 prompt
        prompt_text = ALPACA_PROMPT_TEMPLATE.format(instruction=sample['instruction'], input=sample['input'])
        response_text = sample['output']

        # 分别对 prompt 和 response 进行分词
        # 注意：encode 不会自动添加 BOS/EOS token，这取决于你的 tokenizer 配置
        prompt_token_ids = tokenizer.encode(prompt_text)
        response_token_ids = tokenizer.encode(response_text, add_special_tokens=False) # 回答部分不应添加特殊token
        
        # 将 prompt 和 response 的 token 拼接起来
        input_ids = prompt_token_ids + response_token_ids

        # 为 response 添加结束符
        input_ids.append(tokenizer.eos_token_id)
        
        # 如果总长度超过 seq_len，则截断
        if len(input_ids) > seq_len:
            input_ids = input_ids[:seq_len]

        # 3. 创建掩码后的 labels
        labels = list(input_ids) # 复制 input_ids
        prompt_len = len(prompt_token_ids)
        
        # 将 prompt 部分的 labels 设置为 -100
        for i in range(prompt_len):
            labels[i] = -100
            
        # 如果截断导致 response 完全被切掉，则跳过这个样本
        if all(x == -100 for x in labels):
            continue

        all_input_ids.append(torch.tensor(input_ids).unsqueeze(0))
        all_labels.append(torch.tensor(labels).unsqueeze(0))
        
        if nsamples is not None and len(all_input_ids) >= nsamples:
            print(f"\nReached the target number of samples: {nsamples}.")
            break

    print(f"Processing finished. Found {len(all_input_ids)} samples.")

    class InstructionDataset(Dataset):
        def __init__(self, input_ids_list, labels_list):
            self.input_ids = input_ids_list
            self.labels = labels_list

        def __getitem__(self, index):
            # 返回 input_ids 和已经处理好的 labels
            return {
                "input_ids": self.input_ids[index],
                "labels": self.labels[index]
            }

        def __len__(self):
            return len(self.input_ids)

    train_dataset = InstructionDataset(all_input_ids, all_labels)
    # 注意：对于不同长度的序列，通常需要一个自定义的 data_collator 来进行填充(padding)
    # 这里为了简化，我们假设所有序列都被截断或过滤到了相同长度
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print("Instruction-tuning train loader created successfully.")
    return train_loader