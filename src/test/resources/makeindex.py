#!/usr/bin/env python
# encoding: utf-8
from annoy import AnnoyIndex

test_dims = [64]
for dim in test_dims:
    a = AnnoyIndex(dim, 'angular')
    d = AnnoyIndex(dim, 'dot')
    e = AnnoyIndex(dim, 'euclidean')
    a.set_seed(123)
    d.set_seed(123)
    e.set_seed(123)
    vectors = open('item_vector.txt').readlines()
    for index, vector in enumerate(vectors):
        v = [float(x) for x in vector.split(',')]
        a.add_item(index, v)
        d.add_item(index, v)
        e.add_item(index, v)
    a.build(3)
    a.save('points.angular.annoy.{}'.format(dim))
    d.build(3)
    d.save('points.dot.annoy.{}'.format(dim))
    e.build(3)
    e.save('points.euclidean.annoy.{}'.format(dim))
