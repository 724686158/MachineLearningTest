# -*- coding: utf-8 -*-
from itertools import chain, combinations

def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))
    return itemSet, transactionList


def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def getRetRulesbyRetItems(RetItems, minConfidence):
    toRetRules = []
    item_supports = dict()

    for ret in RetItems:
        item = ret[0]
        support = ret[1]
        item_supports[frozenset(item)] = support

    for ret in RetItems:
        item = ret[0]
        _subsets = map(frozenset, [x for x in subsets(item)])

        for element in _subsets:
            remain = set(item).difference(element)
            if len(remain) > 0:
                confidence = item_supports[frozenset(item)] / item_supports[element]
                if confidence >= minConfidence:
                    toRetRules.append(((tuple(element), tuple(remain)), confidence))
    return toRetRules
