#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Lancer <lancer.dalu@hotmail.com>
#
# This program is used to calculate the similarity of a group of texts.
#
# Usage method:
#
#    python2.7 computing_similarity.py [file_name]
#
# Computing times:
#
#     n! / ((n - m)! * m!)
#     n = lines
#     m = 2
#
# e.g.:
#    n = 18142
#    m = 2
#    count = 164,557,011
#

from multiprocessing import Lock, Manager, Process
from scipy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from zhon.hanzi import punctuation

import datetime
import logging
import multiprocessing
import numpy as np
import os
import re
import string
import sys
import time
import xlrd
import xlwt

reload(sys)
sys.setdefaultencoding('utf8')

# 相似度，推荐值: 0.7
SIMILARITY_SCORE = 0.7

# 最相似的前TOP_K个结果
TOP_K = 5

# 并行进程数 (任务分片并不均匀).
# 如果用个人电脑独占CPU, 为了加快计算，推荐取CPU核数 * 4
# 此处采用CPU并行计算。考虑到计算为浮点数，若后续需要再加速，可以采用GPU并行计算方式，使用pycuda库
CONCURRENCY = 16

# 去除标点符号
def remove_punctuation(text):
    # 去除中文符号
    text = re.sub(ur"[%s]+" %punctuation, "", text)
    # 去除英文符号
    text = re.sub(ur"[%s]+" %string.punctuation, "", text)

    return text.strip().lower()

# 将字符串中间加入空格, 最简单分词
def add_space(s):
    return ' '.join(list(s))

# 计算相似度
def tfidf_similarity(s1, s2):
    if s1 == "" or s2 == "":
        return 0.0

    # 转化为Tf-Idf矩阵
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()

    # 计算范数
    n = norm(vectors[0]) * norm(vectors[1])
    if n == 0:
        return 0.0
    else:
        return np.dot(vectors[0], vectors[1]) / n

# 累计相似度
def computing_similarity(sheet_data, shard, region, similarity_score, lock):
    logger = logging.getLogger()

    sheet_data_len = len(sheet_data)
    upper_bound = shard * region
    lower_bound = (shard + 1) * region
    for i in range(upper_bound, lower_bound):
        for j in range(i+1, sheet_data_len):
            if (tfidf_similarity(sheet_data[i][0], sheet_data[j][0]) > similarity_score):
                # 加锁保证多进程间原子计数
                lock.acquire()
                # 相似问题+1, 并记录相似问题行号
                related_quest_list_i = sheet_data[i][3]
                related_quest_list_i.append(sheet_data[j][1])
                sheet_data[i] = [sheet_data[i][0], sheet_data[i][1], sheet_data[i][2] + 1, related_quest_list_i]

                related_quest_list_j = sheet_data[j][3]
                related_quest_list_j.append(sheet_data[i][1])
                sheet_data[j] = [sheet_data[j][0], sheet_data[j][1], sheet_data[j][2] + 1, related_quest_list_j]
                lock.release()

        logger.info('shard: %s, 范围：%s - %s, 计算至第: %s 行' % (shard, upper_bound, lower_bound, i))

def process_excel_data(source_file_path, similarity_score, top_k):
    logger = logging.getLogger()
    data = xlrd.open_workbook(source_file_path)
    table = data.sheet_by_name('Sheet1')
    row_num = table.nrows
    col_num = table.ncols
    logger.info('文件总行数: %s', row_num)

    # 该list可以在多进程间共享
    sheet_data = multiprocessing.Manager().list()
    for i in range(0, row_num):
        # 移除空格和换行
        c_cell = table.cell_value(i, 0).strip().replace(' ', '').replace("\n", "").replace("\r", "")
        # 移除标点符号
        c_cell = remove_punctuation(c_cell)
        if c_cell == '':
            logger.error('存在空行, 行号: %s', i)

        # 问题，行号，相似问题数量, 相似问题行号
        list_cell = [add_space(c_cell), i, 0 , []]
        sheet_data.append(list_cell)

    # concurrency 代表并行计算，值可以取CPU核数.
    concurrency = CONCURRENCY
    region = len(sheet_data) / concurrency
    lock = Lock()
    list_p = []
    for shard in range(concurrency):
        p = Process(target=computing_similarity, args=(sheet_data, shard, region, similarity_score, lock))
        p.start()
        list_p.append(p)

    # 阻塞等待所有子进程计算完毕
    for p in list_p:
        p.join()

    # 按照相似问题数排序
    sorted_list = sorted(sheet_data, key=lambda obj:obj[2], reverse=True)

    # 统计出前top_k个不重复的结果
    final_list = []
    for i in range(0, len(sorted_list)):
        if len(final_list) >= top_k:
            break

        exist_similar_question = False
        if len(final_list):
            for j in range(0, len(final_list)):
                if (tfidf_similarity(final_list[j][0], sorted_list[i][0]) > similarity_score):
                    exist_similar_question = True
                    break
            if not exist_similar_question and len(final_list) <= top_k:
                final_list.append(sorted_list[i])
        else:
            final_list.append(sorted_list[i])

    # 输出top_k个出现最多的问题
    for i in range(0, len(final_list)):
        logger.info('行号：%s, 问题：%s, 次数：%s, 相似问题号：%s' % (final_list[i][1], final_list[i][0].strip().replace(' ', ''), final_list[i][2], final_list[i][3]))

def main(argv):
    if len(argv) != 1:
        sys.stderr.write('Invalid arvgs : %s!', len(argv))
        print len(argv)
        sys.exit(1)

    # 数据源文件全路径, 此处为excle文件
    source_file_path = argv[0]

    # 相似度，推荐值: 0.7
    similarity_score = float(SIMILARITY_SCORE)

    # 最相似的前top_k个结果
    top_k = int(TOP_K)

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        filename='%s.log' % (source_file_path))

    process_excel_data(source_file_path, similarity_score, top_k)

if __name__ == "__main__":
    main(sys.argv[1:])
