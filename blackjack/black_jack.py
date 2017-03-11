# -*- coding: utf-8 -*-

from deck import Deck
from jack_player import Player

class BlackJack(object):
    def __init__(self):
        self.player = self.create_player("Player")
        self.dealer = Player("Dealer")
        self.deck= Deck()


    def create_player(self, title):
        name = raw_input("Enter Player's name: ")
        return Player(name)

    def deal(self):
        self.deck.shuffle()
        for i in range(2):
            self.player.receive_card(self.deck.draw_card())
            self.dealer.receive_card(self.deck.draw_card())

    def player_decide(self):
        print "you have {} points ".format(self.player.sum_cards())
        print self.player.show_hand()
        decision = raw_input("hit or stay: ")
        while decision == "hit":
            self.player.receive_card(self.deck.draw_card())
            print self.player.show_hand()
            print "you have {} points ".format(self.player.sum_cards())
            if self.player.sum_cards()<21:
                decision = raw_input("hit or stay: ")
            else:
                decision= "stay"

    def dealer_decide(self):
        while self.dealer.sum_cards()<=16:
            self.dealer.receive_card(self.deck.draw_card())
        print self.dealer.show_hand()

    def bet(self):
        self.bet_amount = int(raw_input("How much do you want to bet? "))
        self.player.money -= self.bet_amount

    def pay_out(self):
        if self.player.sum_cards() > 21 and self.dealer.sum_cards() > 21:
            self.player.receive_money(self.bet_amount)
        elif self.player.sum_cards() <= 21 and (self.player.sum_cards() > self.dealer.sum_cards()):
            self.player.receive_money(self.bet_amount*1.5)
        elif self.player.sum_cards() <= 21 and self.dealer.sum_cards() > 21:
            self.player.receive_money(self.bet_amount*1.5)
        elif self.player.sum_cards() <= 21 and (self.player.sum_cards() == self.dealer.sum_cards()):
            self.player.receive_money(self.bet_amount)

    def play_game(self):
        continue_playing="yes"
        while self.player.money > 0 and continue_playing=="yes":
            self.deal()
            self.bet()
            self.player_decide()
            self.dealer_decide()
            self.pay_out()
            self.player.restart()
            self.dealer.restart()
            print "you have {} dollars left".format(self.player.money)
            continue_playing=raw_input("Do you want to keep playing? ")

        print "thank you come again sucker"






if __name__ == '__main__':
    game = BlackJack()
    game.play_game()
