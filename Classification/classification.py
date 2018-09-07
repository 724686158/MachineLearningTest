# -*- coding: utf-8 -*-
from optparse import OptionParser
import knn

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
    optparser.add_option('-t', '--testFile',
                         dest='test',
                         help='test file')
    optparser.add_option('-c', '--classFile',
                         dest='classfile',
                         help='class file')
    optparser.add_option('-k', '--Knumber',
                         dest='knumber',
                         help='number of k',
                         default=10)

    (options, args) = optparser.parse_args()

    if options.method == 'knn':
        # knn.showFileData(options.input)
        knn.autoTest(options.input, options.classfile, options.knumber, "OUTPUT.csv")
