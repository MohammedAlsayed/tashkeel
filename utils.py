# imports the necessary modules
import os
from collections import Counter



def get_file_list(dir: str) -> dict:
    directory = {}
    for root, dirs, files in os.walk(dir):
        file_list = []
        for file in files:
            # skip hidden files
            if file[0] == '.':
                continue
            file_list.append(os.path.join(root, file))
        directory[root] = file_list
    return directory

def is_valid_word(word: str) -> bool:
    """
    Check if a word is valid should be only Arabic and full stop or colon. 
    No other characters are allowed like numbers or brackets ..etc
    """
    has_arabic = False
    for c in word:
        if is_arabic_char(c):
            has_arabic = True
        if not is_arabic_char(c) and not is_special_char(c):
            return False
    return has_arabic

def is_arabic_word(word: str) -> bool:
    """
    Check if a word has only Arabic characters
    """
    for c in word:
        if not is_arabic_char(c):
            return False
    return True

def is_arabic_char(char: str) -> bool:
    """
    Check if a character is in the Arabic unicode range
    """
    return ord(char) >= 1569 and ord(char) <= 1610

def is_special_char(char: str)-> bool:
    """
    Check if a character is a punctuation mark or colon
    """
    return char in ['.', ':', '!']

def is_arabic_num(char: str)-> bool:
    """
    Check if a character is Arabic number ١-٩
    """
    return ord(char) >= 1632 and ord(char) <= 1641

def clean_sentence(sentence: str) -> str:
    """
    Remove non Arabic characters from a sentence
    """
    cleaned_sentence = clean_words(sentence.split())
    cleaned_sentence = ' '.join(cleaned_sentence)
    return cleaned_sentence

def clean_words(words: list[str]) -> list[str]:
    """
    Remove non Arabic characters from a word
    """
    cleaned = []
    for word in words:
        cleaned_word = ''
        for c in word:
            if is_arabic_char(c) or is_harakah(c):
                cleaned_word += c
        if cleaned_word:
            cleaned.append(cleaned_word)
    return cleaned

def is_harakah(char: str) -> bool:
    """
    Check if a character is a harakah
    """
    if len(char) != 1:
        return False
    return ord(char) >= 1611 and ord(char) <= 1619

def is_shaddah(char: str) -> bool:
    """
    Check if a character is a shaddah
    """
    if len(char) != 1:
        return False
    return ord(char) == 1617

def has_any_diacritics(word: str) -> bool:
    """
    Check if a word has any diacritics
    """
    for c in word:
        if is_harakah(c):
            return True
    return False

def strip_harakat(sentence: str) -> str:
    return ''.join([c for c in sentence if not is_harakah(c)])

def arabic_char_length(sentence: str) -> int:
    """
    Count the number of Arabic characters in a sentence
    """
    words = sentence.split()
    length = 0
    length += sum(len(w) for w in words)
    length += len(words) - 1 # spaces between words
    return length

def shakkel(sentence: str, harakat:str)-> str:
    """
    Add harakat to a sentence
    """
    shakkel = ''
    # remove start and end tokens from the sentence and harakat 
    sentence = sentence[6:-6]
    harakat = harakat[6:-6].split()

    for s in sentence:
        shakkel += s
        harka = ""
        if len(harakat) > 0 and harakat[0] == '<PAD>' and s != ' ':
            continue
        if s == ' ':
            # keep poping until we reach a space in the harakat
            while len(harakat) > 0 and harakat[0] != '<PAD>':
                harakat.pop(0)
            harakat.pop(0)
        if len(harakat) > 0 and is_arabic_char(s):
            harka = harakat.pop(0)
            if is_harakah(harka):
                shakkel += harka
        if len(harakat) > 0 and is_shaddah(harka):
            harka = harakat.pop(0)
            shakkel += harka
        
    return shakkel

def get_word_statistics(count_dict: Counter):
    total = 0
    count_ar_words = 0
    count_non_ar_words = 0
    count_diacritics = 0
    count_no_diacritics = 0  

    for word in count_dict:
        total += count_dict[word]
        if is_valid_word(word):
            count_ar_words += count_dict[word]
            if has_any_diacritics(word):
                count_diacritics += count_dict[word]
            else:
                count_no_diacritics += count_dict[word]
        else:
            count_non_ar_words += count_dict[word]
                        
    print(f"""Total words: {total}
Arabic words: {100*count_ar_words/total}%
Diacritics in AR words: {100*count_diacritics/count_ar_words}%
No diacritics in AR words: {100*count_no_diacritics/count_ar_words}$
Non Arabic words: {count_non_ar_words/total}""")