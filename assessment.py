'''
Fill each each function stub (replace the "pass") according to the docstring.
To run the unit tests: Make sure you are in the root dir:assessment-day1 Then run the tests with this command: "make test"
'''

import numpy as np
import pandas as pd
from collections import defaultdict
import re
import itertools

#Python

def count_characters(string):
    count = defaultdict()
    for char in string:
        if char not in count:
            count[char]=1
        else:
            count[char]+=1
    return count
    '''
    INPUT: STRING
    OUTPUT: DICT (STRING => INT)

    Return a dictionary which contains a count of the number of times each
    character appears in the string.
    Characters which would have a count of 0 should not need to be included in
    your dictionary.
    '''


def invert_dictionary(d):
    return {v: k for k, v in d.iteritems()}
    '''
    INPUT: DICT (STRING => INT)
    OUTPUT: DICT (INT => SET OF STRINGS)

    Given a dictionary d, return a new dictionary with d's values as keys and
    the value for a given key being the set of d's keys which have the same
    value.
    e.g. {'a': 2, 'b': 4, 'c': 2} => {2: {'a', 'c'}, 4: {'b'}}
    '''

def word_count(filename):
    f=open(filename,"r")
    all_string=f.read()
    lines= len(re.findall("\n", all_string))
    words= len(re.findall(" ", all_string))
    char=len(all_string)
    return tuple([lines, words, char])
    '''
    INPUT: STRING
    OUTPUT: (INT, INT, INT)

    filename refers to a text file.
    Return a tuple containing these stats for the file in this order:
      1. number of lines
      2. number of words (broken by whitespace)
      3. number of characters
    '''



def matrix_multiplication(A, B):
    result = [ [ [None] ]*len(A) ] * len(B[0])
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B[0])):
                result[i][j] += A[i][k] * B[k][j]
    return result
    '''
    INPUT: LIST OF LIST OF INTEGERS, LIST OF LIST OF INTEGERS
    OUTPUT: LIST OF LIST of INTEGERS

    A and B are matrices with integer values, encoded as lists of lists:
    e.g. A = [[2, 3, 4], [6, 4, 2], [-1, 2, 0]] corresponds to the matrix:
    | 2  3  4 |
    | 6  4  2 |
    |-1  2  0 |
    Return the matrix which is the product of matrix A and matrix B.
    You may assume that A and B are square matrices of the same size.
    You may not use numpy. Write your solution in straight python.
    '''


### Probability

def cookie_jar(a, b):
    c=0.5*a+0.5*b
    prob= a*0.5/c
    return prob


    '''
    INPUT: FLOAT, FLOAT
    OUTPUT: FLOAT

    There are two jars of cookies with chocolate and peanut butter cookies.
    a: fraction of Jar A which is chocolate
    b: fraction of Jar B which is chocolate
    A jar is chosen at random and a cookie is drawn.
    The cookie is chocolate.
    Return the probability that the cookie came from Jar A.
    '''

### NumPy

def array_work(rows, cols, scalar, matrixA):
    return np.dot(matrixA,np.array([scalar]*(rows*cols)).reshape(rows, cols))
    '''
    INPUT: INT, INT, INT, NUMPY ARRAY
    OUTPUT: NUMPY ARRAY

    Create matrix of size (rows, cols) with the elements initialized to the
    scalar value. Right multiply that matrix with the passed matrixA (i.e. AB,
    not BA).
    Return the result of the multiplication.
    You should be able to accomplish this in a single line.

    Ex: array_work(2, 3, 5, [[3, 4], [5, 6], [7, 8]])
           [[3, 4],      [[5, 5, 5],
            [5, 6],   *   [5, 5, 5]]
            [7, 8]]
    '''

def boolean_indexing(arr, minimum):
    return [x for x in itertools.chain(*arr) if x>= minimum]
    '''
    INPUT: NUMPY ARRAY, INT
    OUTPUT: NUMPY ARRAY

    Returns an array with all the elements of "arr" greater than
    or equal to "minimum"

    Ex:
    In [1]: boolean_indexing([[3, 4, 5], [6, 7, 8]], 7)
    Out[1]: array([7, 8])
    '''

### Pandas

def make_series(start, length, index):
    return pd.Series(np.arange(start,start+length),index=index,dtype="int64")
    '''
    INPUT: INT, INT, LIST
    OUTPUT: PANDAS SERIES

    Create a pandas Series of length "length"; its elements should be
    sequential integers starting from "start".
    The series' index should be "index".

    Ex:
    In [1]: make_series(5, 3, ['a', 'b', 'c'])
    Out[1]:
    a    5
    b    6
    c    7
    dtype: int64
    '''

def data_frame_work(df, colA, colB, colC):
    df[colC]=df[colA]+df[colB]
    '''
    INPUT: DATAFRAME, STR, STR, STR
    OUTPUT: None

    Insert a column (colC) into the dataframe that is the sum of colA and colB.
    '''


### SQL
# For each of these, your python function should return a string that is the
# SQL statement which answers the question.
# For example:
#    return "SELECT * FROM farmersmarkets;"
# You may want to run "sqlite3 markets.sql" in the command line to test out
# your queries.
#
# There are two tables in the database with these columns:
# statepopulations: state, pop2010, pop2000
# farmersmarkets: FMID, MarketName, Website, Street, City, County, State,
#    WIC, WICcash
#    (plus other columns we don't care about for this exercise)

def markets_per_state():
    '''
    INPUT: NONE
    OUTPUT: STRING

    Return a SQL query which gives the states and the number of markets
    that take WIC or WICcash for each state.
    The WIC and WICcash columns contain either 'Y' or 'N'
    '''
    pass


def state_population_gain():
    '''
    INPUT: NONE
    OUTPUT: STRING

    Return a SQL statement which gives the 10 states with the highest
    population gain from 2000 to 2010.
    '''
    pass


def market_density_per_state():
    '''
    INPUT: NONE
    OUTPUT: STRING

    Return a SQL statement which gives a table containing each state and the
    number of people per farmers market (use the population number from 2010).
    If a state does not appear in the farmersmarket table, it should still
    appear in your result with a value of 0.
    '''
    pass
