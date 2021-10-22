# -*- coding: UTF-8 -*-
"""
Copyright (c) Yuwei Jin
Created at 2021-10-20 16:23 
Written by Yuwei Jin (642281525@qq.com)
"""

p = {'crop_pos':(1, 2)}

if __name__ == '__main__':
    print(p['crop_pos'])

    x = list(p['crop_pos'])

    x[0] *= 2
    x[1] *= 2

    p['crop_pos'] = x

    print(p['crop_pos'])
