# -*- coding: utf-8 -*-

def txt2csv(filename, out_filename):
    #打开文件
    fr = open(filename)
    #读取文件所有内容
    arrayOLines = fr.readlines()
    #得到文件行数
    numberOfLines = len(arrayOLines)
    index = 0
    fo = open(out_filename, "w")
    for line in arrayOLines:
        #s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        #使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        csvLine = line.replace('\t', ',')
        fo.write(csvLine + '\n')


if __name__ == '__main__':
    txt2csv('TestSet.txt', 'TestSet.csv')
