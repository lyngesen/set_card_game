from main import Deck, Card, Triplet, AllTriplets, plot_graph, setup_MKP, add_table_readme


def test_deck():

    deck = Deck()

    print(deck)


def test_card():

    card1 = Card((1, 0, 1, 2))
    card2 = Card((1, 0, 0, 2))
    card3 = Card((1, 0, 2, 2))

    T = Triplet(card1, card2, card3)

    print(T)
    print(f"{T.is_valid()=}")


def test_triplet():

    deck = Deck()
    U = AllTriplets(deck)

    for u in U.triplets:
        assert u.is_valid(), "Triplet is not valid"
        for card in u.cards:
            assert card in deck.cards, "Card not in deck"


def test_plot_graph():
    plot_graph()

def test_setup_MKP():
    setup_MKP()


def test_add_table_readme():
    add_table_readme()
