import random


class Player(object):
    def __init__(self, name, money=100):
        self.name = name
        self.hand = []
        self.money = money

    def receive_card(self, card):
        self.hand.append(card)

    def sum_cards(self):
        return sum(map(int,self.hand))

    def restart(self):
        self.hand=[]

    def receive_money(self,amount):
        self.money += amount

    def show_hand(self):
        return self.hand

    def __repr__(self):
        return self.name

    def __len__(self):
        return len(self.hand) + len(self.discard)
