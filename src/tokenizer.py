import torch
import json
import ftfy

class Tokenizer:

    """
        Word level tokenizer
    """

    def __init__(self, freq_threshold = 5):

        """
            freq_threshold (default value = 5): define the minimum frequence that a word needs to reach in order to be stored 
            in the Tokenizer libraries, otherwise they will be labeled as <UNK>
        """

        self.word2idx = {
            "<PAD>": 0,
            "<NWL>": 1,
            "<UNK>": 2,
            "<NWS>": 3  
        }  # word : idx

        self.idx2word = {
            0: "<PAD>",
            1: "<NWL>",
            2: "<UNK>",
            3: "<NWS>"
        }  # idx : word

        self.word_freq = {}
        self.lens_dist = []

        self.freq_threshold = freq_threshold

        self.trans_table = str.maketrans({
            '.': ' . ', 
            ',': ' , ', 
            '!': ' ! ',
            '?': ' ? ', 
            ':': ' : ', 
            ';': ' ; ', 
            "'": " ' ", 
            '"': ' " ', 
            '-': ' - ', 
            '–': ' – ', 
            '—': ' — ', 
            '(': ' ( ', 
            ')': ' ) ',  
            '[': ' [ ', 
            ']': ' ] ', 
            '{': ' { ',
            '}': ' } ',
            '…': ' … ',
            '\n': ' <NWL> '  
        })

    def replace_punctuation(self, str_to_replace: str) -> str:
        cleaned_text = ftfy.fix_text(str_to_replace)
        cleaned_text = cleaned_text.replace("â€", '"').replace("â", "")
        cleaned_text = cleaned_text.replace("``", ' " ').replace("`", " ' ")
        return cleaned_text.lower().translate(self.trans_table)

    def create_dicts(self):
        for token, freq in self.word_freq.items():
            if freq >= self.freq_threshold and token not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[token] = idx
                self.idx2word[idx] = token

    def populate_dict(self, dataloader):
        for batch in dataloader:
            for batch_row in batch['text']:
                word_count = 0
                formatted_batch_row = self.replace_punctuation(batch_row)                    
                    
                for token in formatted_batch_row.split():
                    self.word_freq[token] = self.word_freq.get(token, 0) + 1
                    word_count += 1

                self.lens_dist.append(word_count)

        self.L = int(torch.ceil(torch.quantile(torch.tensor(self.lens_dist, dtype=torch.float32), 0.95)) + 1)
        self.create_dicts()

    def __len__(self):
        return len(self.word2idx)

    def encode(self, text_to_encode: str):
        formatted_text = "<NWS> " + self.replace_punctuation(text_to_encode)                 
        encoded_list = [ self.word2idx.get(token, self.word2idx["<UNK>"]) for token in formatted_text.split() ]

        if len(encoded_list) > self.L:
            return encoded_list[:self.L]

        padding = [self.word2idx["<PAD>"]] * (self.L - len(encoded_list))
        return encoded_list + padding

    def batch_encode(self, batch):
        return torch.tensor([ self.encode(sentence) for sentence in batch['text']])

    def decode(self, idxs):
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.tolist()
        return " ".join([self.idx2word.get(idx, "<UNK>") for idx in idxs if idx != 0])


    def store(self, filepath="./states/tokenizer/vocab.json"):
        to_store = {
            "word2idx": self.word2idx,
            "L": self.L
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(to_store, f, ensure_ascii=False, indent=4)
            

    def load(self, filepath="./states/tokenizer/vocab.json"):
        with open(filepath, 'r', encoding='utf-8') as f:
            to_load = json.load(f)  
                
            self.L = to_load.get("L")
            self.word2idx = to_load.get("word2idx")
            self.idx2word = { value:key for key, value in to_load["word2idx"].items() }
