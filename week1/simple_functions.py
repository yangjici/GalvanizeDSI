from collections import Counter
from itertools import izip, count
from StringIO import StringIO


def write_to_file(lst, f):
    for i, line in enumerate(lst,start=1):
        f.write("{} {}\n".format(i,line))

    """
    INPUT: list, open file object
    OUTPUT: None

    Write the list to the file with line numbers, starting at 1.
    INPUT: ["a", "b", "c"]
    FILE CONTENTS:
    1 a
    2 b
    3 c

    Hint: Use enumerate for cleaner code
    """


def merge_files(f1, f2, out):

    file1 = f1.getvalue().strip("\n")
    file2 = f2.getvalue().strip("\n")

    for first, last in izip(file1.split("\n"), file2.split("\n")):
        out.write("{},{}\n".format(first, last))

    """
    INPUT: open file, open file, open file
    OUTPUT: None

    f1 and f2 are two files with the same number of lines. Merge the contents
    together, separated with a comma.

    INPUT FILES:
    cat
    dog

    mouse
    rabbit

    OUTPUT FILE:
    cat,mouse
    dog,rabbit

    Hint: Use izip
    """




def key_in_value(d):
    return[k for k,v in d.iteritems() if k in v]
    """
    INPUT: dict
    OUTPUT: list

    Return the keys from the dictionary where the key is a member in the
    associated value.

    example:
    INPUT: {"a": ["b", "c", "a"], "b": ["a", "c"], "c": ["c"]}
    OUTPUT: ["a", "c"]

    Hint: Use iteritems
    (Can be done on one line with a list comprehension)
    """



def most_common_letters(sentence):
    ls=[]
    for i in sentence.split(" "):
        dic={}
        count=Counter(i.lower())
        for k,v in count.iteritems():
            if v in dic:
                dic[v].append(k)
            else:
                dic[v]=k
        ls.append(dic[max(dic.keys())])

    return " ".join(ls)
    """
    INPUT: string
    OUTPUT: list of strings

    Given a sentence, give the most common letter for each word.
    You should lowercase the letters. If there's a tie, include any of them.

    example:
    INPUT: "Welcome to Zipfian Academy!"
    OUTPUT: 'e t i a'

    Hint: use Counter and the string join method
    (It is possible to do this in one line, but you might lose some
    readability)
    """




def merge_dictionaries(d1, d2):
    output = {}
    for key in d1:
        output[key] = d1[key]
    for key in d2:
        if key not in output:
            output[key] = d2[key]
        else:
            output[key] += d2[key]
    return output



    """
    INPUT: dict (string => int), dict (string => int)
    OUTPUT: dict (string => int)

    example:
    INPUT: {"a": 2, "b": 5}, {"a": 7, "c":10}
    OUTPUT: {"a": 9, "b": 5, "c": 10}

    Create a new dictionary that contains all the key, value pairs from d1 and
    d2. If a key is in both dictionaries, sum the values.
    """

    pass

out = StringIO()
write_to_file(["one", "two", "three", "four"], out)
