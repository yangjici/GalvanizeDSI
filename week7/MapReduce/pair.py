from mrjob.job import MRJob
from mrjob.step import MRStep

from itertools import combinations
import pandas as pd
import numpy as np



class Friecommender(MRJob):

    def mapper1(self, _, line):
        yield (int(line.split()[0]),int(line.split()[1]))
        yield (int(line.split()[1]),int(line.split()[0]))
    def reducer1(self, person, friends):
        friends_list=list(friends)
        yield (int(person), friends_list)


    # def mapper2(self,person, friends_list):
    #     for i in range(len(friends_list)):
    #         for j in range(i,len(friends_list)):
    #             yield([friends_list[i],friends_list[j]], 1)


    def mapper2(self, key, friends_list):

        for pair in combinations(friends_list,2):
            # if #already friend:
            #     yield (pair,0)
            # else:
            yield (pair,1)
        for friend in friends_list:
            known_friends=sorted([key, friend])
            yield (known_friends, False)
    def mutual_reducer(self, pair, count):
        for p,count in zip(pair,count):
            if count:
                yield(pair,count)

    def steps(self):
        return [
                MRStep(mapper=self.mapper1,
                       reducer=self.reducer1),
                MRStep(mapper=self.mapper2, reducer= self.mutual_reducer)
            ]


if __name__=='__main__':
    Friecommender.run()
