#!/usr/bin/python
# -*- coding: utf-8 -*-
#author: Tianming Lu

import sys
#import pdb
#pdb.set_trace()

PLACE_HOLDER = '_'


def read(filename):
    S = []
    with open(filename, 'r') as input:
        for line in input.readlines():
            elements = line.split(',')
            s = []
            for e in elements:
                s.append(e.split())
            S.append(s)
    return S


def loadTestData():
    S=[[['a'], ['a', 'b', 'c'], ['a', 'c'], ['d'], ['c', 'f']],
     [['a', 'd'], ['c'], ['b', 'c'], ['a', 'e']],
     [['e', 'f'], ['a', 'b'], ['d', 'f'], ['c'], ['b']],
     [['e'], ['g'], ['a', 'f'], ['c'], ['b'], ['c']]]
    return S


class sequencePattern:
    def __init__(self, sequence, support):
        self.sequence = []
        for s in sequence:
            self.sequence.append(list(s))
        self.support = support

    def append(self, p):
        if p.sequence[0][0] == PLACE_HOLDER:
            first_e = p.sequence[0]
            first_e.remove(PLACE_HOLDER)
            self.sequence[-1].extend(first_e)
            self.sequence.extend(p.sequence[1:])
        else:
            self.sequence.extend(p.sequence)
        self.support = min(self.support, p.support)


def print_patterns(patterns):
    for p in patterns:
        print("pattern:{0}, support:{1}".format(p.sequence, p.support))




# main function
def PrefixSpan(dataSet, threshold):

    def prefixSpan(S, threshold, pattern):
        patterns = []
        f_list = frequent_items(S, pattern, threshold)
        for i in f_list:
            p = sequencePattern(pattern.sequence, pattern.support)
            p.append(i)
            patterns.append(p)
            
            
            p_S = build_projected_database(S, p)
            p_patterns = prefixSpan(p_S, threshold, p)
            patterns.extend(p_patterns)
        return patterns
    
    
    def frequent_items(S, pattern, threshold):
        items = {}
        _items = {}
        f_list = []
        if S is None or len(S) == 0:
            return []
    
        if len(pattern.sequence) != 0:
            last_e = pattern.sequence[-1]
        else:
            last_e = []
        for s in S:
            #class 1
            is_prefix = True
            for item in last_e:
                if item not in s[0]:
                    is_prefix = False
                    break
            if is_prefix and len(last_e) > 0:
                index = s[0].index(last_e[-1])
                if index < len(s[0]) - 1:
                    for item in s[0][index + 1:]:
                        if item in _items:
                            _items[item] += 1
                        else:
                            _items[item] = 1
    
            #class 2
            if PLACE_HOLDER in s[0]:
                for item in s[0][1:]:
                    if item in _items:
                        _items[item] += 1
                    else:
                        _items[item] = 1
                s = s[1:]
    
            #class 3
            counted = []
            for element in s:
                for item in element:
                    if item not in counted:
                        counted.append(item)
                        if item in items:
                            items[item] += 1
                        else:
                            items[item] = 1
    
        #f_list.extend([sequencePattern([[PLACE_HOLDER, k]], v)
                       #for k, v in _items.iteritems()
                       #if v >= threshold])
        f_list.extend([sequencePattern([[k]], v)
                       for k, v in items.items()
                       if v >= threshold])
        sorted_list = sorted(f_list, key=lambda p: p.support)
        return sorted_list  
        
    
    
    def build_projected_database(S, pattern):
        """
        suppose S is projected database base on pattern's prefix,
        so we only need to use the last element in pattern to
        build projected database
        """
        p_S = []
        last_e = pattern.sequence[-1]
        last_item = last_e[-1]
        for s in S:
            p_s = []
            for element in s:
                is_prefix = False
                if PLACE_HOLDER in element:
                    if last_item in element and len(pattern.sequence[-1]) > 1:
                        is_prefix = True
                else:
                    is_prefix = True
                    for item in last_e:
                        if item not in element:
                            is_prefix = False
                            break
    
                if is_prefix:
                    e_index = s.index(element)
                    i_index = element.index(last_item)
                    if i_index == len(element) - 1:
                        p_s = s[e_index + 1:]
                    else:
                        p_s = s[e_index:]
                        index = element.index(last_item)
                        e = element[i_index:]
                        e[0] = PLACE_HOLDER
                        p_s[0] = e
                    break
            if len(p_s) != 0:
                p_S.append(p_s)
    
        return p_S
    
    
    
    if threshold<1:
        threshold = threshold * len(dataSet)
    
    patterns = prefixSpan(dataSet, threshold, pattern=sequencePattern([], sys.maxsize))
    #print_patterns(patterns)
    return [(p.sequence,p.support) for p in patterns]
    







if __name__ == "__main__":
    S = loadTestData()
    patterns = PrefixSpan(S, 2)
    print_patterns(patterns)
