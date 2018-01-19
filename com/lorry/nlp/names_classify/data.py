#coding:utf-8

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import unicodedata
import string
import torch

def findFiles(path):
    """
    find files
    :param path:
    :return:
    """
    return glob.glob(path)

fileNames = findFiles('/Users/lorry/Documents/python/pytorch_learn_be/my_data/data-names/names/*.txt')
#print(fileNames)

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters)
#print(n_letters)

#transfer unicode string to plain ascii
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn'and c in all_letters
    )

#build the category_lines dictionary
category_lines = {}
all_categories = []

def readLines(fileName):
    lines = open(fileName, encoding="utf-8").read().strip().split("\n")
    return [unicodeToAscii(line) for line in lines]

def fillLanMap():
    for fileName in fileNames:
        category = fileName.split("/")[-1].split(".")[0]
        all_categories.append(category)
        lines = readLines(fileName)
        category_lines[category] = lines
    return category_lines, all_categories

fillLanMap()
#print(category_lines["English"][:2])

#find letter index from all letters
def letterToIndex(letter):
    return all_letters.find(letter)

#just for demonstration, turn a letter into a <1 * n_letters> Tensor
def letterToTensor(letter):
    """
    tern letter to
    :param letter:
    :return:
    """
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    """
    transfer line to a tensor
    :param line:
    :return:
    """
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor