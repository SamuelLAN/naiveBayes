#!/usr/bin/Python
# -*- coding: utf-8 -*-
import load
import numpy as np


class NaiveBayesClassifier:
    def __init__(self):
        self.__oData = load.Data()


    def run(self):
        print 'training: calculating bayes probability ...'
        self.__train()
        print 'finish training'

        print 'testing on test dataset'
        self.accuracy()
        print 'done'


    # 计算 training 数据的各种概率
    def __train(self):
        data = self.__oData

        # P(y = ham)  的概率
        self.__pHam = np.log10(float(data.hamTrainWordNum) / data.totalTrainWordNum)
        # P(y = spam) 的概率
        self.__pSpam = np.log10(float(data.spamTrainWordNum) / data.totalTrainWordNum)

        smooth = 1.0    # 拉普拉斯 平滑因子
        smooth_const = 10 * smooth  # 拉普拉斯平滑 分母所需的常量

        # P(x = word | y = ham)
        self.__pWordHam = np.log10(
            (data.hamWordArray + smooth) / (data.hamTrainWordNum + smooth_const) )

        # P(x = word | y = spam)
        self.__pWordSpam = np.log10(
            (data.spamWordArray + smooth) / (data.spamTrainWordNum + smooth_const) )


    def question1(self):
        self.__train()

        p_word_ham = np.power(10, self.__pWordHam)
        p_word_spam = np.power(10, self.__pWordSpam)

        ratio = list(p_word_spam / p_word_ham)
        ratio_list = []
        for i, val in enumerate(ratio):
            ratio_list.append([i, val])

        def __sort(a, b):
            if a[1] > b[1]:
                return -1
            elif a[1] == b[1]:
                return 0
            else:
                return 1

        ratio_list.sort(__sort)

        print 'top 10 ratio words:'
        string = ''
        for (voc_index, p) in ratio_list[0: 10]:
            string += self.__oData.vocDictRev[voc_index] + ' '
        print string


    # 计算测试集准确率
    def accuracy(self):
        data = self.__oData

        ham_error_num = 0
        for text_data in data.hamTestData:
            if self.classify(text_data):
                ham_error_num += 1

        spam_error_num = 0
        for text_data in data.spamTestData:
            if not self.classify(text_data):
                spam_error_num += 1

        total_test_num = data.hamTestNum + data.spamTestNum
        accuracy = float(total_test_num - ham_error_num - spam_error_num) / total_test_num

        print 'accuracy: %.6f' % accuracy

        tp = data.spamTestNum - spam_error_num
        fn = spam_error_num
        fp = ham_error_num

        precision = tp / float(tp + fp)
        recall = tp / float(tp + fn)

        print 'precision: %.6f' % precision
        print 'recall: %.6f' % recall


    def test(self):
        data = self.__oData

        textData = data.hamTestData[0]

        print 'ham text: '
        string = ''
        for word, times in textData.iteritems():
            string += word + ' '
        print string
        print ''

        print self.classify(textData)


    '''
     若为 spam 返回 true，若为 ham 返回 false
    '''
    def classify(self, text_data):
        p_text_ham = 1.0
        p_text_spam = 1.0

        for (word_id, times) in text_data:
            p_text_ham += self.__pWordHam[word_id] * times
            p_text_spam += self.__pWordSpam[word_id] * times

        p_ham_text = p_text_ham + self.__pHam
        p_spam_text = p_text_spam + self.__pSpam

        return p_ham_text <= p_spam_text


o_classifier = NaiveBayesClassifier()
o_classifier.run()
# o_classifier.question1()
# o_classifier.test()
