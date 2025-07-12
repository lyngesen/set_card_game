from main import (
    Deck,
    Card,
    Draw,
    Triplet,
    AllTriplets,
    plot_graph,
    setup_MKP,
    add_table_readme,
    draw_solution,
    draw_triplets,
    draw_deck
)
import matplotlib.pyplot as plt
import numpy as np


def test_deck():

    deck = Deck()

    print(deck)


def test_card():

    card1 = Card((1, 0, 1, 2))
    card2 = Card((1, 0, 0, 2))
    card3 = Card((2, 2, 2, 2))

    T = Triplet(card1, card2, card3)

    print(T)
    print(f"{T.is_valid()=}")

    fig, ax = plt.subplots()

    card1.draw(ax, (0, 0), size=0.5)
    card2.draw(ax, (3, 0), size=0.5)
    card3.draw(ax, (6, 0), size=0.5)

    ax.set_aspect('equal')
    fig.savefig("figures/tests/draw_cards.pdf")

def test_triplet():

    deck = Deck()
    U = AllTriplets(deck)

    for u in U.triplets:
        assert u.is_valid(), "Triplet is not valid"
        for card in u.cards:
            assert card in deck.cards, "Card not in deck"


    t = U.triplets[0]
    # draw triplet

    fig, ax = plt.subplots()
    t.draw(ax, (0, 0), size=1)
    U.triplets[3].draw(ax, (10, 0), size=1)
    ax.set_aspect('equal')
    fig.savefig("figures/tests/draw_triplet.pdf")


def test_draw_triplets():
    draw_triplets()

def test_draw_deck():
    draw_deck()
        
def test_plot_graph():
    plot_graph()

def test_setup_MKP():
    setup_MKP()

def test_add_table_readme():
    add_table_readme()

def test_draw_solution():
    draw_solution()


