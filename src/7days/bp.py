#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import random
from numpy import *


def sigmoid(inX):
    '''
    Desc:
        激活函数，我们这里选用 sigmoid 函数
    Args:
        inX ---- 输入向量
    Returns:
        1.0 / (1 + exp(-inX)) --- 经过激活函数作用过后得到的数
    '''
    return 1.0 / (1 + exp(-inX))


class Node(object):
    '''
    Desc:
        神经网络中的节点类
    Args:
        object --- 节点对象
    Returns：
        None
    '''
    def __init__(self, layer_index, node_index):
        '''
        Desc:
            构造节点对象
        Args:
            self
            layer_index --- 节点所属的层的索引，表示输出层，隐藏层还是输出层
            node_index --- 节点的索引，对应的层的第几个节点
        Returns:
            None
        '''
        self.layer_index = layer_index # 层索引
        self.node_index = node_index   # 节点索引 
        self.downstream = [] # 下一层的节点信息
        self.upstream = [] # 上一层的节点信息
        self.output = 0 # 当前节点的输出
        self.delta = 0 # 当前节点的 delta

    def set_output(self, output):
        '''
        Desc:
            设置节点的输出值。如果节点属于输入层会用到这个函数
        Args:
            self
            output --- 节点的输出值
        Returns:
            None
        '''
        self.output = output

    def append_downstream_connection(self, conn):
        '''
        Desc:
            添加一个到下游节点的连接
        Args:
            self
            conn --- 到下游节点的连接
        Returns:
            None
        '''
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        '''
        Desc:
            添加一个到上游节点的连接
        Args:
            self
            conn --- 到上游节点的连接
        Returns:
            None
        '''
        self.upstream.append(conn)

    def calc_output(self):
        '''
        Desc:
            计算节点的输出值，根据 output = sigmoid(wTx)
        Args:
            self
        Returns:
            None
        '''
        # 理解 python 的 reduce 函数，请参考链接：https://www.cnblogs.com/XXCXY/p/5180245.html
        # 这里用的 lambda 匿名函数，实际上就是遍历 self.upstream （也就是上游的所有节点，因为我们是全连接）的所有节点，计算与权重相乘之后得到的 sum ，然后代入到 sigmoid 函数，得到output
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        '''
        Desc:
            节点属于隐藏层时，根据 式4 计算 delta
        Args:
            self
        Returns:
            None
        '''
        # 根据 BP 原理来计算隐藏层节点的 delta
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        '''
        Desc:
            节点属于输出层时，根据 式3 计算 delta
        Args:
            self
            label --- 对应数据的 label
        Returns:
            None
        '''
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        '''
        Desc:
            打印出节点信息
        Args:
            self
        Returns:
            node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str ---- 节点信息
        '''
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str 


class ConstNode(object):
    '''
    Desc:
        偏置节点类
    Args:
        object --- 节点类对象
    Returns:
        None
    '''
    def __init__(self, layer_index, node_index):
        '''
        Desc:
            构造节点对象
        Args:
            self
            layer_index --- 节点所属的层的编号 
            node_index --- 节点的编号
        Returns:
            None
        '''
        self.layer_index = layer_index # 节点所属层的索引
        self.node_index = node_index # 节点的索引
        self.downstream = [] # 下游节点的链接
        self.output = 1 # 输出，恒定为 1

    def append_downstream_connection(self, conn):
        '''
        Desc:
            添加一个到下游节点的连接
        Args:
            self
            conn --- 到下游节点的连接
        Returns:
            None
        '''
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        '''
        Desc:
            节点如果属于隐藏层，根据 式4 计算 delta
        Args:
            self
        Returns:
            None
        '''
        # 根据下游节点计算 delta
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        '''
        Desc:
            打印节点的信息
        Args:
            self
        Returns:
            node_str + '\n\tdownstream:' + downstream_str ---- 节点的信息
        '''
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str


class Layer(object):
    '''
    Desc:
        节点所属层的类
    Args:
        object --- 层对象
    Returns:
        None
    '''
    def __init__(self, layer_index, node_count):
        '''
        Desc:
            初始化一层，作为 Node 的集合对象，提供对 Node 集合的操作
        Args:
            layer_index --- 层的索引
            node_count --- 层所包含的节点个数
        Returns:
            None
        '''
        self.layer_index = layer_index # 层索引
        self.nodes = [] # 节点集合
        # 将节点添加到 nodes
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        # 向 nodes 中添加 偏置项
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        '''
        Desc:
            设置输出值
        Args:
            self
            data --- 要设置的输出值
        Returns:
            None
        '''
        # 为 nodes 中每个节点设置输出值
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        '''
        Desc:
            计算输出值
        Args:
            self
        Returns:
            None
        '''
        # 为 nodes 计算输出值
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        '''
        Desc:
            打印层的信息
        Args:
            self
        Returns:
            None
        '''
        # 将 nodes 中节点信息 print 出来
        for node in self.nodes:
            print node


class Connection(object):
    '''
    Desc:
        connection 对象，主要职责是记录连接的权重，以及这个连接所关联的上下游节点
    Args:
        object --- connection 对象
    Returns:
        None
    '''
    def __init__(self, upstream_node, downstream_node):
        '''
        Desc:
            初始化连接，权重初始化为一个很小的随机数
        Args:
            self
            upstream_node --- 上游节点
            downstream_node --- 下游节点
        Returns:
            None
        '''
        self.upstream_node = upstream_node # 上游节点
        self.downstream_node = downstream_node # 下游节点
        self.weight = random.uniform(-0.1, 0.1) # 设置 weights 权重
        self.gradient = 0.0 # 梯度

    def calc_gradient(self):
        '''
        Desc:
            计算梯度
        Args:
            self
        Returns:
            None
        '''
        # 梯度 = 下游的 delta * 上游的 output
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def update_weight(self, rate):
        '''
        Desc:
            根据梯度下降算法更新权重
        Args:
            self
            rate --- 学习率
        Returns:
            None
        '''
        self.calc_gradient()
        self.weight += rate * self.gradient

    def get_gradient(self):
        '''
        Desc:
            获取当前梯度
        Args:
            self
        Returns:
            gradient --- 梯度
        '''
        return self.gradient

    def __str__(self):
        '''
        Desc:
            打印连接信息
        Args:
            self
        Returns:
            None
        '''
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index, 
            self.upstream_node.node_index,
            self.downstream_node.layer_index, 
            self.downstream_node.node_index, 
            self.weight)


class Connections(object):
    '''
    Desc:
        Connections对象，提供Connection集合操作
    Args:
        object --- Connections 对象
    Returns:
        None
    '''
    def __init__(self):
        '''
        Desc:
            初始化 connections，初始化一层中的几个节点
        Args:
            self
        Returns:
            None
        '''
        self.connections = []

    def add_connection(self, connection):
        '''
        Desc:
            在层中的 connections list 中添加节点
        Args:
            self
            connection --- 需要在 connections 的 list 中添加的节点
        Returns:
            None
        '''
        self.connections.append(connection)

    def dump(self):
        '''
        Desc:
            打印 connections 信息
        Args:
            self
        Returns:
            None
        '''
        for conn in self.connections:
            print conn


class Network(object):
    '''
    Desc:
        Network 对象，对外提供 API
    Args:
        object --- network 对象
    Returns:
        None
    '''
    def __init__(self, layers):
        '''
        Desc:
            初始化一个全连接神经网络
        Args:
            self
            layers --- 二维数组，描述神经网络每层节点数
        Returns:
            None
        '''
        # 初始化 Connections 对象，节点待添加
        self.connections = Connections()
        # 初始化 layers 对象，表示第几层
        self.layers = []
        # layer_count 表示我们设计的神经网络有几层
        layer_count = len(layers)
        # node_count 表示有多少个节点，初始化为 0
        node_count = 0
        # 在 layers 中添加 0~layer_count-1 层
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        # 遍历现有层，每一层中添加上游节点和下游节点，并在上游节点和下游节点中
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node) 
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            # 遍历 connections 中的节点，即每一层的所有节点。在初始化的list 中添加节点。
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)


    def train(self, labels, data_set, rate, epoch):
        '''
        Desc:
            训练我们搭建的全连接神经网络
        Args:
            self
            labels --- 数组，训练样本的标签。每个元素是一个样本对应的标签
            data_set --- 二维数组，训练样本特征，每个元素是一个样本的特征
            rate --- 学习率
            epoch --- 迭代次数
        Returns:
            None
        '''
        # 迭代 epoch 次，训练神经网络
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)
                # print 'sample %d training finished' % d

    def train_one_sample(self, label, sample, rate):
        '''
        Desc:
            内部函数，用一个样本训练网络
        Args:
            label --- 样本的标签
            sample --- 样本特征
            rate --- 学习率
        Returns:
            None
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_delta(self, label):
        '''
        Desc:
            内部函数，计算每个节点的 delta
        Args:
            self
            label --- 样本的标签
        Returns:
            None
        '''
        # 输出层的节点 output_nodes
        output_nodes = self.layers[-1].nodes
        # 遍历所有的样本标签，计算输出层的 delta
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        # 计算隐藏层的 delta
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        '''
        Desc:
            内部函数，更新每个连接权重
        Args:
            rate --- 学习率
        Returns:
            None
        '''
        # 遍历除最后一层的每一层，根据下游节点进行更新权重
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        '''
        Desc:
            内部函数，计算每个连接的梯度
        Args:
            self
        Returns:
            None
        '''
        # 遍历除去最后一层的所有层，根据下游节点，计算梯度
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, label, sample):
        '''
        Desc:
            获得网络在一个样本下，每个连接上的梯度
        Args:
            self
            label --- 样本的标签
            sample --- 样本的特征
        Returns:
            None
        '''
        # 调用 predict() 函数进行预测
        self.predict(sample)
        # 调用 calc_delta 函数进行计算 delta
        self.calc_delta(label)
        # 调用 calc_gradient() 函数获取梯度
        self.calc_gradient()

    def predict(self, sample):
        '''
        Desc:
            根据输入的样本预测输出值
        Args:
            self 
            sample --- 输入的样本的特征
        Returns:
            map(lambda node: node.output, self.layers[-1].nodes[:-1]) --- 预测得到的样本输出
        '''
        # 对第一层（输入层）设置输出值，也就是输入样本的特征
        self.layers[0].set_output(sample)
        # 遍历除输入层的每一层，计算output
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        # 获得根据输入样本计算得到的样本预测输出
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

    def dump(self):
        '''
        Desc:
            打印层的信息
        Args:
            self
        Returns:
            None
        '''
        # 遍历每一层，打印出层的信息
        for layer in self.layers:
            layer.dump()


class Normalizer(object):
    '''
    Desc:
        归一化
    Args:
        object --- 要归一化的对象
    Returns:

    '''
    def __init__(self):
        self.mask = [
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
        ]

    def norm(self, number):
        return map(lambda m: 0.9 if number & m else 0.1, self.mask)

    def denorm(self, vec):
        binary = map(lambda i: 1 if i > 0.5 else 0, vec)
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x,y: x + y, binary)


def mean_square_error(vec1, vec2):
    '''
    Desc:
        计算 MES
    Args:
        vec1 --- 输入向量 1
        vec2 --- 输入向量 2
    Returns:
        计算得到的 MSE
    '''
    return 0.5 * reduce(lambda a, b: a + b, 
                        map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                            zip(vec1, vec2)
                        )
                 )


def gradient_check(network, sample_feature, sample_label):
    '''
    Desc:
        梯度检查
    Args:
        network --- 神经网络对象
        sample_feature --- 样本的特征
        sample_label --- 样本的标签
    Returns:
        None
    '''
    # 计算网络误差
    network_error = lambda vec1, vec2: \
            0.5 * reduce(lambda a, b: a + b, 
                      map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                          zip(vec1, vec2)))

    # 获取网络在当前样本下每个连接的梯度
    network.get_gradient(sample_feature, sample_label)

    # 对每个权重做梯度检查    
    for conn in network.connections.connections: 
        # 获取指定连接的梯度
        actual_gradient = conn.get_gradient()
    
        # 增加一个很小的值，计算网络的误差
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)
    
        # 减去一个很小的值，计算网络的误差
        conn.weight -= 2 * epsilon # 刚才加过了一次，因此这里需要减去2倍
        error2 = network_error(network.predict(sample_feature), sample_label)
    
        # 根据式6计算期望的梯度值
        expected_gradient = (error2 - error1) / (2 * epsilon)
    
        # 打印
        print 'expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_gradient, actual_gradient)


def train_data_set():
    '''
    Desc:
        获得训练数据集
    Args:
        None
    Returns:
        labels --- 数据集中样本对应的标签
        data_set --- 数据集的特征
    '''
    # 初始化一个 normalizer
    normalizer = Normalizer()
    # 初始化 data_set
    data_set = []
    # 初始化 labels
    labels = []
    for i in range(0, 256, 8):
        n = normalizer.norm(int(random.uniform(0, 256)))
        data_set.append(n)
        labels.append(n)
    return labels, data_set


def train(network):
    '''
    Desc:
        训练神经网络
    Args:
        network --- 神经网络
    Returns:
        None
    '''
    # 初始化数据集和对应的 labels
    labels, data_set = train_data_set()
    # 对 network 进行训练
    network.train(labels, data_set, 0.3, 50)


def test(network, data):
    '''
    Desc:
        测试神经网络
    Args:
        network --- 神经网络
        data --- 训练数据
    Returns:
        None
    '''
    # 初始化一个 normalizer
    normalizer = Normalizer()
    # 对数据进行归一化
    norm_data = normalizer.norm(data)
    # 用我们的神经网络进行对测试数据预测
    predict_data = network.predict(norm_data)
    # 打印
    print '\ttestdata(%u)\tpredict(%u)' % (
        data, normalizer.denorm(predict_data))


def correct_ratio(network):
    '''
    Desc:
        计算我们的神经网络的正确率
    Args:
        network --- 全连接神经网络
    Returns:
        None
    '''
    # 调用归一化函数
    normalizer = Normalizer()
    correct = 0.0
    # 256 个数据中进行计算正确率
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print 'correct_ratio: %.2f%%' % (correct / 256 * 100)


def gradient_check_test():
    '''
    Desc:
        梯度检查测试
    Args:
        None
    Returns:
        None
    '''
    # 初始化一个网络
    net = Network([2, 2, 2])
    # 设置 sample 的 feature
    sample_feature = [0.9, 0.1]
    # 设置 sample 的 label
    sample_label = [0.9, 0.1]
    # 梯度检查
    gradient_check(net, sample_feature, sample_label)


if __name__ == '__main__':
    '''
    Desc:
        主函数
    '''
    # 初始化一个全连接神经网络
    net = Network([8, 3, 8])
    # 训练我们的神经网络
    train(net)
    # 将信息打印出来
    net.dump()
    # 计算神经网络的正确率
    correct_ratio(net)