'''
This script creates three files that are sampled versions of the data:
    * data/small_imdb_edges.tsv
    * data/small_actor_edges.tsv
    * data/small_movie_edges.tsv

It uses two files:
    * data/actors.txt (list of actors to use in sampling)
    * data/imdb_edges.txt (full dataset)
'''


from load_imdb_data import load_imdb_data
from make_edge_files import make_edge_file


def load_actors(filename):
    '''
    filename: file where each line is an actor name

    Return a set of the actor names.
    '''
    with open(filename) as f:
        return set(line.strip() for line in f)


def make_small_imdb_file(actor_set, big_imdb_filename, small_imdb_filename):
    '''
    actor_set: set of actor names
    big_imdb_filename: file of actor, movie connections (to be read)
    small_imdb_filename: new file of actor, movie connections (to be written)

    Create a new file that contains only the edges from the big file that
    contain an actor from actors_set.
    '''
    bigf = open(big_imdb_filename)
    smallf = open(small_imdb_filename, 'w')
    for line in bigf:
        actor, movie = line.strip().split('\t')
        if actor in actor_set:
            smallf.write('%s\t%s\n' % (actor, movie))
    bigf.close()
    smallf.close()


if __name__ == '__main__':
    actor_set = load_actors('data/actors.txt')
    make_small_imdb_file(actor_set,
                         'data/imdb_edges.tsv',
                         'data/small_imdb_edges.tsv')
    actors, movies = load_imdb_data('data/small_imdb_edges.tsv')
    make_edge_file('data/small_actor_edges.tsv', movies)
    make_edge_file('data/small_movie_edges.tsv', actors)
