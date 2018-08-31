# -*- coding: utf-8 -*-

from collections import defaultdict
from association_analysis.data_process_tools import getItemSetTransactionList


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
    # calculates the support for items in the itemSet and returns a subset
    # of the itemSet each of whose elements satisfies the minimum support
    _itemSet = set()
    localSet = defaultdict(int)

    for item in itemSet:
        for transaction in transactionList:
            if item.issubset(transaction):
                freqSet[item] += 1
                localSet[item] += 1

    for item, count in localSet.items():
        support = float(count) / len(transactionList)

        if support >= minSupport:
            _itemSet.add(item)
    return _itemSet


def joinSet(itemSet, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


def find_frequent_itemsets(data_iter, minSupport):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    """
    # read from csv
    itemSet, transactionList = getItemSetTransactionList(data_iter)
    freqSet = defaultdict(int)
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport
    largeSet = dict()


    k = 1
    oneCSet = returnItemsWithMinSupport(itemSet,
                                        transactionList,
                                        minSupport,
                                        freqSet)

    currentLSet = oneCSet
    while(currentLSet != set([])):
        k = k + 1
        largeSet[k-1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        currentCSet = returnItemsWithMinSupport(currentLSet,
                                                transactionList,
                                                minSupport,
                                                freqSet)
        currentLSet = currentCSet

    def getSupport(item):
        """local function which Returns the support of an item"""
        return float(freqSet[item])/len(transactionList)

    RetItems = []
    for key, value in largeSet.items():
        RetItems.extend([(tuple(item), getSupport(item))
                           for item in value])
    return RetItems
