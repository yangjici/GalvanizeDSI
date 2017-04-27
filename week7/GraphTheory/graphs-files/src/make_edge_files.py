'''
This script creates edge files for the actor only graph and movie only graph.

Created files:
    * data/actor_edges.tsv
    * data/movie_edges.tsv

Needs this file to run:
    * data/imdb_edges.tsv (actor, movie edges)
'''

from itertools import combinations
from load_imdb_data import load_imdb_data


def make_edge_file(filename, d):
    '''
    filename: name of file to write to
    d: dictionary of edge data

    Write edge list to the file.
    '''
    f = open(filename, 'w')
    edges = set()
    for key, values in d.iteritems():
        for edge in combinations(values, 2):
            edges.add(tuple(sorted(edge)))
    for one, two in edges:
        f.write("%s\t%s\n" % (one, two))
    f.close()


if __name__ == '__main__':
    actors, movies = load_imdb_data('data/imdb_edges.tsv')
    make_edge_file('data/actor_edges.tsv', movies)
    make_edge_file('data/movie_edges.tsv', actors)
