EOS_TOKEN = 1
SOS_TOKEN = 0
UNK_TOKEN = 2
PAD_TOKEN = 3
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
    def __init__(self, max_length=256):
        self.word2index = {"<SOS>": 0, "<EOS>": 1, "<UNK>": 2, "<PAD>": 3, "َ": 4, "ُ": 5, "ِ": 6, "ً": 7, "ٌ": 8, "ٍ": 9, "ّ": 10, "ْ": 11, "ٓ": 12}
        self.word2count = Counter()
        self.index2word = {SOS_TOKEN: "<SOS>", 
                           EOS_TOKEN: "<EOS>", 
                           UNK_TOKEN: "<UNK>", 
                           PAD_TOKEN: "<PAD>", 
                           FATHA_TOKEN: "<FATHA>", 
                           DAMMA_TOKEN: "<DAMMA>", 
                           KASRA_TOKEN: "<KASRA>", 
                           TANWEEN_FATHA_TOKEN: "<TANWEEN_FATHA>", 
                           TANWEEN_DAMMA_TOKEN: "<TANWEEN_DAMMA>",
                           TANWEEN_KASRA_TOKEN: "<TANWEEN_KASRA>",
                           SHADA_TOKEN: "<SHADA>",
                           SUKOON_TOKEN: "<SUKOON>",
                           MADDA_TOKEN: "<MADDA>"}
        self.n_words = 13

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        self.word2index[word] = self.n_words
        self.word2count[word] += 1
        self.index2word[self.n_words] = word
        self.n_words += 1

    def build_tokenizer_table(self, train_docs):
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

        # First character must be an Arabic character, if not maybe there is a splitting issue or the dataset is corrupted.
        # if is_harakah(before_c):
        #     raise ValueError("First character is a harakah!", {before_c})
        # if not is_arabic_char(before_c):
        #     raise ValueError("First character is not an Arabic character!", {before_c})

        # List that contains the harakat of each character
        harkaat = []
        count_shaddah = 0
        for next_c in sentence[1:]:
            if is_arabic_char(before_c):
                # next character to a letter is a harakah append the harakah
                if is_harakah(next_c):
                    harkaat.append(next_c)
                # next character to a letter is not a harakah append UNK, becuase every letter must have a harakah
                else:
                    harkaat.append(self.index2word[UNK_TOKEN])
            # Shaddah is a special case because a harakah comes after it sometimes.
            elif is_shaddah(before_c) and is_harakah(next_c):
                count_shaddah += 1
                harkaat.append(next_c)
            # space between words append PAD
            elif next_c == ' ':
                harkaat.append(self.index2word[PAD_TOKEN])

            before_c = next_c
        
        # Last character may not have a harakah
        if not is_harakah(next_c):
            harkaat.append(self.index2word[UNK_TOKEN])
        
        char_length = arabic_char_length(clean_sent)
        assert char_length+count_shaddah == len(harkaat), f"Arabic character length != harkaat length {clean_sent} != {harkaat}"
        return clean_sent, " ".join(harkaat)
    

    def get_pair(self, sentence, encoded=True):
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
        text_tokens = [self.word2index[word] if word in self.word2index else UNK_TOKEN for word in text.split()]
        input_ids.extend(text_tokens)
        input_ids.append(EOS_TOKEN)
        return input_ids
            
    def decode(self, indices):
        """
        Decode a list of indices to words
        """
        return ' '.join([self.index2word[i] for i in indices])