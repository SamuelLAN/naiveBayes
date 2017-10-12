#!/usr/bin/Python
# -*- coding: utf-8 -*-
import os
import sys

# 将运行路径切换到当前文件所在路径
cur_dir_path = os.path.split(__file__)[0]
if cur_dir_path:
    os.chdir(cur_dir_path)
    sys.path.append(cur_dir_path)

import re
import numpy as np
from collections import Counter
from six.moves import cPickle as pickle


'''
 数据模块
'''
class Data:
    TRAIN_PATH = r'data/train'
    TEST_PATH = r'data/test'

    DATA_PATH = r'data/data.pickle'

    DEBUG = False

    def __init__(self):
        self.__loadingTest = False
        self.__load()
        

    '''
     加载数据
    '''
    def __load(self):
        if os.path.exists(self.DATA_PATH):
            self.echo('already exist %s , loading data ...' % self.DATA_PATH)

            with open(self.DATA_PATH, 'rb') as f:
                (self.spamWordArray, self.hamWordArray, self.vocDict,
                 self.vocDictRev, self.spamTestData, self.hamTestData) = pickle.load(f)
        else:
            self.__loadFromOrigin()

        self.echo('finish loading')

        self.__process()


    '''
     加载原始数据并进行处理计数
    '''
    def __loadFromOrigin(self):
        # 加载 spam 数据并计数
        self.__spamWordCounter = self.__loadData(self.TRAIN_PATH, 'spam')

        # 加载 ham 数据并计数
        self.__hamWordCounter = self.__loadData(self.TRAIN_PATH, 'ham')

        # 处理上面两个 counter 数据成 array
        self.__processCounter()

        # 加载测试数据
        self.__spamTestData = self.__loadData(self.TEST_PATH, 'spam')
        self.__hamTestData = self.__loadData(self.TEST_PATH, 'ham')

        # 处理测试集的数据，将 counter 里的 word 根据词汇表转成对应的 index
        self.__processTestData()

        # 保存处理完的数据到 DATA_PATH 中，下次不需重新处理所有原始文本数据
        data = (self.spamWordArray, self.hamWordArray, self.vocDict,
                self.vocDictRev, self.spamTestData, self.hamTestData)
        with open(self.DATA_PATH, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


    '''
     处理训练集的两个 Counter
    '''
    def __processCounter(self):
        # 将所有 spam email 的计数合并
        self.echo('\nmerging counter of spamWordCounter ...')
        self.__spamWordCounter = reduce((lambda x, y: x + y), self.__spamWordCounter)
        self.echo('finish merging')

        # 将所有 ham email 的计数合并
        self.echo('\nmerging counter of hamWordCounter ...')
        self.__hamWordCounter = reduce((lambda x, y: x + y), self.__hamWordCounter)
        self.echo('finish merging')

        # 总的训练集的 counter
        total_train_counter = self.__spamWordCounter + self.__hamWordCounter

        # 将加载数据状态置为加载测试数据
        self.__loadingTest = True

        # 生成总的训练集的词汇表
        self.vocDict = {}           # 词汇表   word: id
        self.vocDictRev = {}        # 词汇表的 id: word
        for i, word in enumerate(total_train_counter.keys()):
            self.vocDict[word] = i
            self.vocDictRev[i] = word

        # 将 counter 转为 array
        spam_word_array = []
        ham_word_array = []
        for i in range(len(self.vocDictRev)):
            word = self.vocDictRev[i]
            spam_word_array.append(self.__spamWordCounter[word])
            ham_word_array.append(self.__hamWordCounter[word])

        # array 中的值为 词的计数，index 与 词汇表的一致
        self.spamWordArray = np.array(spam_word_array)
        self.hamWordArray = np.array(ham_word_array)


    '''
     处理测试集的数据
    '''
    def __processTestData(self):
        self.spamTestData = []
        for counter in self.__spamTestData:
            self.spamTestData.append([[self.vocDict[word], times] for word, times in counter.iteritems()])

        self.hamTestData = []
        for counter in self.__hamTestData:
            self.hamTestData.append([[self.vocDict[word], times] for word, times in counter.iteritems()])


    '''
     处理数据
    '''
    def __process(self):
        self.spamTrainWordNum = sum(self.spamWordArray)     # 训练集 spam email 中的单词数
        self.hamTrainWordNum = sum(self.hamWordArray)       # 训练集 ham email 中的单词数
        self.totalTrainWordNum = self.hamTrainWordNum + self.spamTrainWordNum  # 训练集中所有 email 的单词数

        self.spamTestNum = len(self.spamTestData)           # 测试集 spam email 的数量
        self.hamTestNum = len(self.hamTestData)             # 测试集 ham email 的数量


    '''
     加载数据，并计数，返回 list，list 中存储着多个计数器 counter
    '''
    def __loadData(self, folder_path, sub_folder_name):
        counter_list = []

        dir_path = os.path.join(folder_path, sub_folder_name)
        self.echo('loading %s data ...' % dir_path)

        file_list = os.listdir(dir_path)
        total_num = len(file_list)
        self.echo('\ttotal files num : %d' % total_num)

        for i, file_name in enumerate(file_list):
            if i % 3 == 0:
                progress = float(i) / total_num * 100
                self.echo('progress: %.4f    \r' % progress, False)

            file_path = os.path.join(dir_path, file_name)
            with open(file_path, 'rb') as f:
                content = f.read()
            counter_list.append( Counter(self.__filter(self.__tokenizer(content.lower()))) )

        return counter_list


    # 分词
    @staticmethod
    def __tokenizer(text):
        reg = re.compile(r'\s+')
        return reg.split(text)


    # 过滤 标点符号、数字、高频词
    def __filter(self, word_list):
        # 过滤高频词
        high_frequency_words = ['subject', 'to', 'a', 'the', 'of', 'is', 'are', 'am', 'we',
                                'he', 'her', 'his', 'her', 'cc', 'subject:', 'pm', 'etc']
        # 过滤非字母
        reg_alpha = re.compile(r'^[a-z]+$')
        return filter(( lambda x: len(x) > 1                            # 过滤标点符号
                                  and x not in high_frequency_words     # 过滤高频词
                                  and reg_alpha.search(x)               # 过滤非字母
                                  # 过滤不在训练集词典的词
                                  and (not self.__loadingTest or x in self.vocDict)
                        ), word_list)


    # 输出 msg 到 console ; crlf 为 False 表示不换行
    @staticmethod
    def echo(msg, crlf=True):
        if not Data.DEBUG:
            return

        if crlf:
            print msg
        else:
            sys.stdout.write(msg)
            sys.stdout.flush()
