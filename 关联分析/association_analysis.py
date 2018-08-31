# -*- coding: utf-8 -*-
import sys
from optparse import OptionParser
from 关联分析.input_tools import dataFromFile
from 关联分析.output_tools import printResults
from 关联分析 import apriori, fp_growth
from 关联分析.data_process_tools import getRetRulesbyRetItems

if __name__ == "__main__":

    optparser = OptionParser()
    optparser.add_option('-m', '--method',
                         dest='method',
                         help='use method apriori or fp_growth',
                         default=None)
    optparser.add_option('-f', '--inputFile',
                         dest='input',
                         help='filename containing csv',
                         default=None)
    optparser.add_option('-s', '--minSupport',
                         dest='minS',
                         help='minimum support value',
                         default=0.15,
                         type='float')
    optparser.add_option('-c', '--minConfidence',
                         dest='minC',
                         help='minimum confidence value',
                         default=0.6,
                         type='float')

    (options, args) = optparser.parse_args()

    inFile = None
    if options.input is None:
            inFile = sys.stdin
    elif options.input is not None:
            inFile = dataFromFile(options.input)
    else:
            print('No dataset filename specified, system with exit\n')
            sys.exit('System will exit')

    minSupport = options.minS
    minConfidence = options.minC

    if options.method == 'apriori':
        retItems = apriori.find_frequent_itemsets(inFile, minSupport)
        retRules = getRetRulesbyRetItems(retItems, minConfidence)
        printResults(retItems, retRules)
    if options.method == 'fp-growth':
        retItemsIter = fp_growth.find_frequent_itemsets(inFile, minSupport, True)
        retItems = []
        for item_and_support in retItemsIter:
            retItems.append(item_and_support)
        retRules = getRetRulesbyRetItems(retItems, minConfidence)
        printResults(retItems, retRules)

