from collections import deque
from load_imdb_data import load_imdb_data
from sys import argv, exit


def shortest_path(actors, movies, actor1, actor2):
    '''
    INPUT:
        actors: dictionary of adjacency list of actors
        movies: dictionary of adjacency list of movies
        actor1: actor to start at
        actor2: actor to search for

    OUTPUT:
        path: list of actors and movies that starts at actor1 and ends at
              actor2

    Return the shortest path from actor1 to actor2. If there is more than one
    path, return any of them.
    '''
    pass


def print_path(path):
    '''
    INPUT:
        path: list of strings (node names)
    OUTPUT: None

    Print out the length of the path and all the nodes in the path.
    '''
    if path:
        print "length:", len(path) / 2
        for i, item in enumerate(path):
            if i % 2 == 0:
                print "    %s" % item
            else:
                print item
    else:
        print "No path!"


if __name__ == '__main__':
    if len(argv) != 4:
        print "Usage: python " + argv[0] + " <data_file> <actor1> <actor2>"
        exit(1)
    filename = argv[1]
    actor1 = argv[2]
    actor2 = argv[3]
    actors, movies = load_imdb_data(filename)
    print "Searching for shortest path from %s to %s" % (actor1, actor2)
    path = shortest_path(actors, movies, actor1, actor2)
    print_path(path)
