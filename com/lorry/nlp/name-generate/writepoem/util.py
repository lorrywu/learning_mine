#coding:utf-8
import unidecode
import string
import random
import time
import math
import torch
from torch.autograd import Variable
import re
import torch.utils.data as data


def convert_file(filename, new_file = None):
    """
    read un-unicode-encoding data
    :param filename:
    :param new_file:
    :return:
    """
    r_file = open(filename, "r")
    lines = r_file.read().split("\n")
    if new_file is None:
        new_file = filename + "1"
    w_file = open(new_file, "w")
    for line in lines:
        newline = re.sub(r"[0-9a-zA-Z,：.!?]+", r"", line)
        w_file.write(newline + "\n")
    w_file.close()

class Lang(object):
    def __init__(self, file_name = None, name = None):
        self.name = name
        self.trimmed = False #是否去掉了一些不常用的词
        self.char2index = {'\n': 0} #词-》索引
        self.char2count = {}
        self.index2char = {0: "\n"} #索引-》词
        self.n_chars = 1 #收录到词典中的词里出现的最少出现次数
        self.text_length = 0;
        self.text = ""
        self.lines = []
        if file_name is not None:
            self.load(file_name)


    def index_chars(self, line):
        """
        从句子收录词到字典
        :param line:
        :return:
        """
        for char in line :
            self.index_char(char)

    def index_char(self, char):
        """
        收录一个词到词典
        :param char:
        :return:
        """
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1

    def load(self, file_name):
        """
        从文件加载语料库
        :param file_name:
        :return:
        """
        lines = open(file_name).read().strip().split("\n")
        for i in range(len(lines)):
            line = unicode(lines[i]).strip()
            #lns = line.split(u"，|。|？|！")
            lns = re.split(u'[，。？！]', line)
            for ln in lns:
                if len(ln) > 0:
                    if u"：" in ln:
                        ln = ln.split(u"：")[1]
                    ln = ln.replace("［", "").replace("］", "").replace("《", "").replace("》", "").replace("）", "").replace("（", "")
                    self.index_chars(ln)
                    self.text_length += len(ln)
                    self.text += ln
                    self.lines.append(ln)


    def trim(self, min_count):
        """
        把语料库里出现的次数低于min_count的词删除
        :param min_count:
        :return:
        """
        if self.trimmed: return
        self.trimmed = True

        keep_chars = []
        for k, v in self.char2count.items():
            if v >= min_count:
                keep_chars.append(k)

        print('keep_chars %s / %s = %.4f' % (
            len(keep_chars), len(self.char2index), len(keep_chars) / len(self.char2index)
        ))

        # Reinitialize dictionaries
        self.char2index = {'\n':0}    # 词-> 索引
        self.char2count = {}
        self.index2char = {0:'\n'} # 索引 -> 词
        self.n_chars = 1 # 默认收录到词典里的在训练库里出现的最少次数

        for char in keep_chars:
            self.index_char(char)

    def indexes(self, line):
        """
        根据lang返回一个句子的张量
        :param line:
        :return:
        """
        indexes = []
        for char in line:
            if self.char2index.get(char) is None: #不存在该词
                self.index_char(char)
            indexes.append(self.char2index[char])
        return indexes

    def show_info(self):
        print "收录{0}个字(字符)".format(self.n_chars)


def read_file(filename):
    lang = Lang(filename)
    global  n_characters
    n_characters = len(lang.char2index)
    return lang

def char_tensor(lang, string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = lang.char2index[string[c]]
    return Variable(tensor)

def time_since(since):
    s = time.time() - float(since)
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

if __name__ == "__main__":
    ln = u"佳人绝代有佳人，幽居在空谷。自云良家子，零落依草木。"
    lns = re.split(u'[［］《》，。？！]', ln)
    for l in lns:
        print l


class MyDataSet(data.Dataset):
    def __init__(self, trains, labels):
        self.trains = trains
        self.labels = labels

    def __getitem__(self, item):
        train, label = self.trains[item], self.labels[item]
        return train, label

    def __len__(self):
        return len(self.trains)

