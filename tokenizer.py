PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
FATHA_TOKEN = 4
DAMMA_TOKEN = 5
KASRA_TOKEN = 6
TANWEEN_FATHA_TOKEN = 7
TANWEEN_DAMMA_TOKEN = 8
TANWEEN_KASRA_TOKEN = 9
SHADA_TOKEN = 10
SUKOON_TOKEN = 11
MADDA_TOKEN = 12

from utils import *
from tqdm import tqdm
from collections import Counter

class Tokenizer:
    def __init__(self, character_level=False):
        self.token2index = {"<PAD>": PAD_TOKEN, "<SOS>": SOS_TOKEN,
                            "<EOS>": EOS_TOKEN, "<UNK>": UNK_TOKEN, 
                            "َ": FATHA_TOKEN, "ُ": DAMMA_TOKEN, 
                            "ِ": KASRA_TOKEN, "ً": TANWEEN_FATHA_TOKEN, 
                            "ٌ": TANWEEN_DAMMA_TOKEN, "ٍ": TANWEEN_KASRA_TOKEN,
                            "ّ": SHADA_TOKEN, "ْ": SUKOON_TOKEN, "ٓ": MADDA_TOKEN}
        self.token2count = Counter()
        self.index2token = {SOS_TOKEN: "<SOS>", 
                           EOS_TOKEN: "<EOS>", 
                           UNK_TOKEN: "<UNK>", 
                           PAD_TOKEN: "<PAD>", 
                           FATHA_TOKEN: "َ", 
                           DAMMA_TOKEN: "ُ", 
                           KASRA_TOKEN: "ِ", 
                           TANWEEN_FATHA_TOKEN: "ً", 
                           TANWEEN_DAMMA_TOKEN: "ٌ",
                           TANWEEN_KASRA_TOKEN: "ٍ",
                           SHADA_TOKEN: "ّ",
                           SUKOON_TOKEN: "ْ",
                           MADDA_TOKEN: "ٓ"}
        self.n_tokens = 13

        self.character_level = character_level
        if self.character_level:
            for i in range(1569, 1611):
                self.add_word(chr(i))

    def add_sentence(self, sentence: str):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word: str):
        if self.character_level:
            for c in word:
                self.add(c)
        else:
            self.add(word)

    def add(self, token: str):
        if token not in self.token2index:
            self.token2index[token] = self.n_tokens
            self.token2count[token] += 1
            self.index2token[self.n_tokens] = token
            self.n_tokens += 1

    def build_tokenizer_table(self, train_docs: list[Document]):
        """
        Build a tokenizer from a list of sentences
        """
        for doc in tqdm(train_docs):

            # remove any non Arabic characters and harakat
            clean = strip_harakat(clean_sentence(doc.page_content))

            # skip one word or empty sentences
            if len(clean.split()) <= 1:
                continue
            # skip one character sentence
            if arabic_char_length(clean) <= 1:
                continue

            self.add_sentence(clean)

    def seperate_words_harakat(self, sentence: str) -> tuple[str, str]:
        """
        Remove harakat from a sentence and return the sentence and the harakat
        """
        if len(sentence) == 0:
            return "", ""
        
        clean_sent = strip_harakat(sentence)
        if len(clean_sent) == 0:
            raise ValueError("Sentence has no Arabic characters", {sentence})
        
        before_c = sentence[0]

        # List that contains the harakat of each character
        harkaat = []
        for next_c in sentence[1:]:
            if is_arabic_char(before_c):
                # next character to a letter is a harakah append the harakah
                if is_harakah(next_c):
                    harkaat.append(next_c)
                # next character to a letter is not a harakah append UNK, becuase every letter must have a harakah
                elif is_arabic_char(next_c):
                    harkaat.append(self.index2token[UNK_TOKEN])
                elif not is_harakah(next_c):
                    harkaat.append(self.index2token[UNK_TOKEN])

            # Shaddah is a special case because a harakah can comes after or before it.
            elif is_shaddah(before_c) and is_harakah(next_c):
                harkaat.append(next_c)                

            elif before_c == " ":
                harkaat.append(self.index2token[PAD_TOKEN])

            # print(f"next_c: {next_c} is space: {next_c == " "}",)
            # print(f"harakat: {harkaat}")
            before_c = next_c


        return clean_sent, " ".join(harkaat)
    

    def get_pair(self, sentence: str, encoded=True):
        """
        Creates pair of input words and targets (harakat) from a sentence
        """
        sentence = clean_sentence(sentence)
        input_sentence, target_sentence = self.seperate_words_harakat(sentence)
        if encoded:
            input_sentence = self.encode(input_sentence)
            target_sentence = self.encode(target_sentence)
    
        return input_sentence, target_sentence
        
    def encode(self, text) -> list[int]:
        """
        Encode a sentence or tokens to indices make sure to add <SOS> and <EOS> tokens
        """
        input_ids = [SOS_TOKEN] 
        text_tokens = [self.token2index[word] if word in self.token2index else UNK_TOKEN for word in text.split()]
        input_ids.extend(text_tokens)
        input_ids.append(EOS_TOKEN)
        return input_ids
            
    def decode(self, indices: list[int]) -> str:
        """
        Decode a list of indices to words
        """
        return ' '.join([self.index2token[i] for i in indices])