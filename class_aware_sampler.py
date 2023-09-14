import random
import numpy as np
from torch.utils.data.sampler import Sampler

class RandomCycleIter:
    
    def __init__ (self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode


    def __iter__ (self):
        return self
    

    def __next__ (self):
        self.i += 1
        
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)
            
        return self.data_list[self.i]


def class_aware_sample_generator (cls_iter, data_iter_list, n, num_samples_cls=1):
    i = 0
    j = 0

    while i < n:       
        if j >= num_samples_cls:
            j = 0
    
        if j == 0:
            # 这句话的意思是将当前迭代到的类的样本索引复制num_samples_cls倍，然后用next()做迭代访问，由于
            # 第n个元组当中的元素是当前类别样本索引的第n个，这里需要注意zip当中的迭代器是循环随机访问的
            # 所以每个元组的元素不会重复
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]]*num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]
        
        i += 1
        j += 1


class ClassAwareSampler(Sampler):
    """class aware sampler
        简单来说，就是每一次循环当中，打乱每一个类的访问顺序，然后按顺序采样每一个类的样本，
        每一个类的样本访问顺序在访问过一遍之后也打乱，通过这种打乱的操作来采样平衡数量的样本
    """

    def __init__(self, labels, num_samples_cls):
        """"
        :param labels: 标签
        :param num_samples_cls: 每个类别的样本重复采样几次
        """
        num_classes = len(np.unique(labels))  # 类别数
        
        self.class_iter = RandomCycleIter(range(num_classes))  # 能迭代访问的每一个类别的迭代器，
                                                            # 每访问一轮（即全部类别）就重新打乱

        cls_data_list = [list() for _ in range(num_classes)]  # 生成一个元素是列表类型的列表，
                                                            # 用于存储每一类别的所有样本的索引
        for i, label in enumerate(labels):
            # 给每一个样本索引存进对应的类别列表中
            cls_data_list[label].append(i)
        # 生成一个存储每一类别样本索引的列表，但是这里访问每个样本索引是通过RandomCycleIter来访问的（打乱循环访问）
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        # 样本数 = 最多样本的类别的样本数 x 总的类别数
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)

        self.num_samples_cls = num_samples_cls

    
    def __iter__(self,):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)

    
    def __len__ (self):
        return self.num_samples

if __name__ == '__main__':
    a = RandomCycleIter(range(5))
    b = [a]*3
    print(b)
