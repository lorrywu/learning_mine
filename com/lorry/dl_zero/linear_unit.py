#coding:utf-8

from  com.lorry.dl.perceptron import Perceptron

#定义激活函数f
f = lambda x:x

class LinearUnit(Perceptron):
    def __init__(self, input_num):
        Perceptron.__init__(self, input_num, f)


def get_training_dataset():
    """
    构造5个例子,以收入数据为例
    :return:
    """
    #输入向量列表,假设每一项是工作年限
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    #标签
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels

def train_linear_unit():
    """
    使用数据训练线性单元
    :return:
    """

    #创建感知器, 输入参数的特征数为1
    lu = LinearUnit(1)
    #训练,迭代10轮,学习速度为0.01
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    #返回训练好的线性单元
    return lu

if __name__ == '__main__':
    #训练单元
    linear_unit = train_linear_unit()
    print(linear_unit)
    # 测试
    print('Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))
    print('Work 15 years, monthly salary = %.2f' % linear_unit.predict([15]))
    print('Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5]))
    print('Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3]))