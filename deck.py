import random

class Card(object):
    value_dict = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                  '8': 8, '9': 9, '10': 10, 'J': 10, 'Q': 10,
                  'K': 10, 'A': 11}

    def __init__(self, number):
        self.number = number

    def __repr__(self):
        return "{}" .format(self.number)


    # def __cmp__(self, other):
    #     return cmp(self.value_dict[self.number], self.value_dict[other.number])


class Deck(object):
    def __init__(self):
        self.cards = []
        for num in Card.value_dict.values():
            for i in range(4):
                self.cards.append(num)

    def shuffle(self):
        random.shuffle(self.cards)

    def draw_card(self):
        if not self.isempty():
            return self.cards.pop()

    def __len__(self):
        return len(self.cards)

    def isempty(self):
        return self.cards == []
