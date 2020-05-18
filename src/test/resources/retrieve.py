#!/usr/bin/env python
# encoding: utf-8
from annoy import AnnoyIndex

test_dims = [64]


def do(indextype):
    for dim in test_dims:
        a = AnnoyIndex(dim, indextype)
        a.load('points.{}.annoy.{}'.format(indextype, dim))
        with open('points.{}.ann.{}.txt'.format(indextype, dim), 'w') as out:
            for q_index in range(a.get_n_items()):
                nns = a.get_nns_by_item(q_index, 21)
                out.write('{}\t{}\n'.format(q_index, ','.join([str(n) for n in nns])))


do('angular')
do('dot')
do('euclidean')
