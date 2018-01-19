#coding:utf-8
import json

def get_word_emb(file_name = "/Users/lorry/Documents/python/pytorch_learn_be/my_data/wordemb/wordemb"):
    lines = open(file_name).read().strip().split("\n")
    dict = {}
    for line in lines:
        line = unicode(line)
        ln = line.split("	")
        if len(ln) == 2:
            json_object = json.loads(ln[1])
            vec = list(json_object["vec"])
            dict[unicode(ln[0])] = vec
    return dict


if __name__ == "__main__":
    dict = get_word_emb()
    print len(dict[u"一"])


