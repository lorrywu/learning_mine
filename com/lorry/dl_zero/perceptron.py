#coding:utf-8

"""
learning dl, 感知器,第一课
"""
from functools import reduce


class Perceptron(object):
    def __init__(self, input_num, activator):
        """
        初始化感知器,设置输入参数的个数,及激活函数
        :param input_num: 一个样本的输入的维度
        :param activator:
        """
        self.activator = activator
        #权重向量初始化为0
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        """
        打印学习到的权重,偏置项
        :return:
        """
        return 'weights\t:%s\nbias\t:%ff\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        """
        输入向量,输出感知器的计算结果
        :param input_vec: 输入的向量, input_num * 1
        :return:
        """
        #把input_vec[x1, x2,..., xn]和weights[w1, w2, w3..., wn] 打包在一起
        #变成[(x1, w1), (x2, w2),..., (xn, wn)]
        #然后用map函数计算[x1 * w1, x2 * w2, ..., xn * wn]
        #最后用 reduce求和
        #实现f(wx + b)

        package_x_w = zip(input_vec, self.weights)

        total = 0;
        for x, w in package_x_w:
            total += x * w
        total += self.bias
        return self.activator(total)




    def train(self, input_vecs, label, iteration, rate):
        """
        输入训练数据: 一组向量,每个向量对应的label, 训练轮数, 学习率
        :param input_vec:
        :param label:
        :param iteration:
        :param rate:
        :return:
        """
        for i in range(iteration):
            self._one_iteration(input_vecs, label, rate)


    def _one_iteration(self, input_vecs, labels, rate):
        """
        一次迭代,把所有的训练数据过一遍
        :param input_vec:
        :param labels:
        :param rate:
        :return:
        """
        #把输入和输出打包在一起, 成为样本的列表[(input_vec1, label1), (input_vec2, label2),..., (input_vecn, lableln)]
        #每个训练样本是(input_veci, labeli)
        samples = zip(input_vecs, labels)
        #对每个样本,按照按照感知器规则更新权重
        for (input_vec, label) in samples:
            #计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        """
        按照感知器规则更新权重
        :param input_vec:
        :param output:
        :param label:
        :param rate:
        :return:
        """
        #把input_vec[x1, x2,..., xn]和weights[w1, w2,...wn]打包在一起
        #变成[(x1, w1), [x2, w2]...[xn, wn]]
        #然后利用感知器规则更新权重
        delta = label - output
        package_x_w = zip(input_vec, self.weights)
        index = 0
        for x, w in package_x_w:
            weight =  w + rate * delta * x
            self.weights[index] = weight
            index = index + 1

        #更新bias
        self.bias += rate * delta



def f(x):
    """
    定义激活函数
    :param x:
    :return:
    """
    return 1 if x > 0 else 0

def get_training_dataset():
    """
    基于and 真值表构建训练数据
    :return:
    """
    #构建训练数据
    #输入向量列表
    input_vecs = [[1, 1], [0,0], [1, 0], [0, 1]]
    #labels [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
    labels = [1, 0, 0, 0]
    return input_vecs, labels

def train_and_percetron():
    """
    使用and真值表训练感知器
    :return:
    """

    #创建感知器,输入参数个数为2 (因为and是元函数), 激活函数为f
    p = Perceptron(2, f)

    #训练, 迭代10轮, 学习速率为0.1
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)

    #返回训练好的感知器
    return p

if __name__ == '__main__':
    #训练and感知器
    and_percetion = train_and_percetron()
    #打印训练获得的权重
    print(and_percetion)
    # 测试
    print('1 and 1 = %d' % and_percetion.predict([1, 1]))
    print('0 and 0 = %d' % and_percetion.predict([0, 0]))
    print('1 and 0 = %d' % and_percetion.predict([1, 0]))
    print('0 and 1 = %d' % and_percetion.predict([0, 1]))
