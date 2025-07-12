from typing import List, Literal
from itertools import product, combinations
import json, math, os, re
import pyomo.environ as pyomo
from pyomo.common.fileutils import Executable
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

PropertyValue = int  # each property has exactly 3 values
CardTuple = tuple[PropertyValue, ...]
N_PROPERTIES = 4  # number of properties, can be 2, 3 or 4
N_PROPERTY_VALUES = 3  # number of values for each property, can be 3 or 4

class Draw(object):
    ALPHA = 1
    X_LSPACE = np.linspace(-1, 1, 200)
    SHAPES = {
        0: {
            "name": "rectangle",
            "x": [-1, 1],
            "y1": [-0.5, -0.5],
            "y2": [0.5,  0.5],
        },
        1: {
            "name": "oval",
            "x": X_LSPACE,
            "y1" : np.sqrt(1 - (X_LSPACE) ** 2) * 0.5,
            "y2": -np.sqrt(1 - (X_LSPACE) ** 2) * 0.5
        },
        2: {
            "name": "squickly",
            "x": X_LSPACE,
            "y1": np.sqrt(1 - (X_LSPACE) ** 2) * 0.5 * np.cos(0.2 * np.pi + X_LSPACE),
            "y2": -np.sqrt(1 - (X_LSPACE) ** 2) * 0.5 * np.sin(0.2 * np.pi + X_LSPACE)
        },
        3: {
            "name": "paralellogram",
            "x": [-1,0, 1],
            "y1": [-0.0,-0.5 ,-0.0],
            "y2": [0,0.5,0],
        },
    }
    FILLS = {
        0: {'name': 'none', 'fill':'none'},
        1: {'name': 'hatched', 'fill':'////'},
        2: {'name': 'full', 'fill':'filled'},
        3: {'name': 'hatched', 'fill':'++'}
             }
    COLORS = {
        0: {'name': 'blue', 'color': '#2300FF'},
        1: {'name': 'purple', 'color': '#FB0072'},
        2: {'name': 'green', 'color': '#138D00'},
        3: {'name': 'red', 'color': 'red'},
    }
    NUMBERS = {
            0: {'name': 'one', 'number': 1},
            1: {'name': 'two', 'number': 2},
            2: {'name': 'three', 'number': 3},
            3: {'name': 'four', 'number': 4},
            }


    @staticmethod
    def draw_outline(ax, c, size=1, card_height = 3, card_width = 2, color ='black'):
        x = [-card_width/2, card_width/2, card_width/2, -card_width/2, -card_width/2]
        y = [-card_height/2, -card_height/2, card_height/2, card_height/2, -card_height/2]

        # scale to size but same position
        x = [xi * size for xi in x]
        y = [yi * size for yi in y]

        # translate to center c
        x = [xi + c[0] for xi in x]
        y = [yi + c[1] for yi in y]

        # set aspect ratio to be equal
        ax.plot(x, y, color=color, linewidth=size*0.1)

    @staticmethod
    def draw_shape(ax, shape, fill, color, pos=(0, 0), size=1):
        x = [xi * size + pos[0] for xi in shape["x"]]
        y1 = [yi * size + pos[1] for yi in shape["y1"]]
        y2 = [yi * size + pos[1] for yi in shape["y2"]]

        if fill['fill'] == 'filled':
            ax.fill_between(x, y1, y2, color=color['color'], alpha=Draw.ALPHA, linewidth = size)
        elif fill['fill'] == 'none':
            ax.fill_between(x, y1, y2, color='none', edgecolor=color['color'], linewidth = size)
        else:
            ax.fill_between(x, y1, y2, color='none', hatch=fill['fill'], edgecolor=color['color'], linewidth = size)

    @staticmethod
    def draw_card(ax, shape, fill, color,number, card_pos=(0, 0), size=1):
        """
        Draw a card with the given shape, fill, color, position, and size.
        The card is drawn centered at card_pos.
        """
        Draw.draw_outline(ax, card_pos, size)
        POS_DIR = {
            1: [(0, 0)],
            2: [(0, -0.5), (0, 0.5)],
            3: [(0, 0), (0, 1), (0, -1)],
            4: [(0, 1), (0, -1), (0, - 1/3), (0, 1/3)],
        }
        for pos in POS_DIR[number['number']]:
            pos = (pos[0] * size, pos[1] * size)  # scale position by size
            pos_draw = (card_pos[0] + pos[0], card_pos[1] + pos[1])
            Draw.draw_shape(ax, shape, fill, color, pos=pos_draw, size=0.8*size)

class Card(object):
    def __init__(self, values: CardTuple):
        self.values = values
        self.id: int = -1
        assert len(values) == N_PROPERTIES, "Card must have exactly 4 values"
        assert all(v in range(N_PROPERTY_VALUES) for v in values)

    def __str__(self):
        return f"({self.values})"

    def __getitem__(self, index: int):
        return self.values[index]

    def __repr__(self):
        return self.__str__()

    def is_in_triplet(self, triplet: "Triplet") -> bool:
        return self in triplet.cards

    def draw(self, ax, pos=(0, 0), size=1):
        """
        Draw the card on the given axes at position pos with size size.
        """
        shape = Draw.SHAPES[self[0]]
        fill = Draw.FILLS[self[1]]
        color = Draw.COLORS[self[2]]
        number = Draw.NUMBERS[self[3]]
        Draw.draw_card(ax, shape, fill, color, number, card_pos=pos, size=size)


class Triplet(object):
    def __init__(self, card1: Card, card2: Card, card3: Card):
        self.cards = [card1, card2, card3]
        self.id: int = -1
        assert len(self.cards) == 3, "Triplet must contain exactly 3 cards"

    def __str__(self):
        return f"Triplet: {', '.join(str(card) for card in self.cards)}"

    def __getitem__(self, index: int):
        return self.cards[index]

    def is_valid(self):
        if len(set(self.cards)) != 3:  # if there are not exactly 3 unique cards
            print("Not a valid triplet: not 3 unique cards", set(self.cards))
            return False

        for p in range(N_PROPERTIES):
            if self[0][p] == self[1][p] == self[2][p]:
                continue
            elif self[0][p] != self[1][p] != self[2][p] != self[0][p]:
                continue
            else:
                return False
        return True

    def contains_card(self, card: Card) -> bool:
        return card in self.cards

    def __repr__(self):
        return self.__str__()

    def draw(self, ax, pos=(0, 0), size=1):
        Draw.draw_outline(ax, pos, size=size, card_height=3*1.1, card_width=2.2*3, color = 'lightgray')
        for i in range(-1,2):
            self.cards[i].draw(ax, pos=(pos[0] + i * 2.2 * size, pos[1]), size=size)


class Deck(object):
    def __init__(self):

        self.cards: List[Card] = []
        for p_list in product(list(range(N_PROPERTY_VALUES)), repeat=N_PROPERTIES):
            self.cards.append(Card(tuple(p_list)))
        # make general

        self.cards = list(set(self.cards))

        if N_PROPERTIES == 4 and N_PROPERTY_VALUES == 3:
            assert len(self.cards) == 81, "Deck must contain exactly 81 cards"

        # set unique IDs for cards
        self.cards = list(sorted(self.cards, key=lambda card: card.values))
        for i, card in enumerate(self.cards):
            card.id = i

    def __str__(self):
        return f"Deck with {len(self.cards)} cards"

    def __getitem__(self, index: int):
        return self.cards[index]

    def __iter__(self):
        return iter(self.cards)

    def __repr__(self):
        return self.__str__()


class AllTriplets(object):
    def __init__(self, deck: Deck):
        self.triplets: List[Triplet] = []
        for card1, card2, card3 in combinations(deck.cards, 3):
            triplet = Triplet(card1, card2, card3)
            if triplet.is_valid():
                self.triplets.append(triplet)

        self.triplets = list(set(self.triplets))
        print(f"Found {len(self.triplets)} valid triplets")
        if N_PROPERTIES == 4 and N_PROPERTY_VALUES == 3:
            assert len(self.triplets) == 1080, len(self.triplets)
        # set unique ID for each triplet
        self.triplets = list(
            sorted(
                self.triplets,
                key=lambda triplet: tuple(card.values for card in triplet.cards),
            )
        )
        for i, triplet in enumerate(self.triplets):
            triplet.id = i

    def __iter__(self):
        return iter(self.triplets)

    def __str__(self):
        return f"AllTriplets with {len(self.triplets)} triplets"


def plot_graph():

    fig, ax = plt.subplots(figsize=(100, 800))

    V = Deck()
    U = AllTriplets(V)
    E = [(v, u) for u in U for v in V if u.contains_card(v)]

    G = nx.Graph()
    G.add_edges_from(E)

    # bipartite position of nodes split into V and U
    pos = nx.bipartite_layout(G, nodes=V.cards, align="vertical")
    pos = {v: (0, v.id * (len(U.triplets) / len(V.cards))) for v in V.cards} | {
        u: (100, u.id) for u in U.triplets
    }
    # pos = nx.spiral_layout(G)
    ax.set_title(
        "Bipartite graph of Cards (left) and Triplets (rights)"
        + f"\nProperties: {N_PROPERTIES}, Values: {N_PROPERTY_VALUES}",
        fontsize=30,
    )
    # add caption

    NODE_SIZE = 5

    ax.set_ylabel("Index for each card", fontsize=26)
    # read solution.json and color corresponding nodes in V
    with open("solution.json", "r") as f:
        solution = json.load(f)
        selected_cards = solution["cards"]
        selected_nodes = [card for card in V.cards if card.id in selected_cards]
    # draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=[v for v in V if v in selected_nodes],
        node_color="red",
        node_size=NODE_SIZE,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        nodelist=[v for v in V if v not in selected_nodes],
        node_color="lightgrey",
        node_size=NODE_SIZE,
    )
    # add edges
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.05, edge_color="lightgray")
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(v, u) for v, u in E if v in selected_nodes],
        ax=ax,
        width=0.05,
        edge_color="red",
    )

    # add node labels
    labels_V = {card: i for i, card in enumerate(V.cards)}
    labels_U = {triplet: i for i, triplet in enumerate(U.triplets)}
    # labels_U = {triplet:  for i, triplet in enumerate(U.triplets)}
    incident_edges_selected = lambda u: [
        (v, um) for v, um in E if v in selected_nodes and u == um
    ]
    labels_U = {
        triplet: len(incident_edges_selected(triplet)) for triplet in U.triplets
    }
    nx.draw_networkx_labels(
        G, pos, labels=labels_V, ax=ax, font_size=2, horizontalalignment="center"
    )
    nx.draw_networkx_labels(
        G, pos, labels=labels_U, ax=ax, font_size=1, horizontalalignment="center"
    )

    # ax.set_xlabel(
        # f"A total of ${len(selected_cards)}$ cards have been chosen (marked red). Each possible valid triplet has an edge\n for each card it consists of, this is colored red if the card is selected in the solution. \n The number for each of the 1080 triplets indicate the number of cards in the tripliet selected. \n\nNote: see file figures/graph_{N_PROPERTIES}_{N_PROPERTY_VALUES}_{len(selected_nodes)}.pdf in order to zoom — to read the values",
        # fontsize=26,
    # )


    if N_PROPERTIES == 4 and N_PROPERTY_VALUES == 3:
        if True: # draw cards
            for card in V.cards:
                pos_card = pos[card]
                card.draw(ax, np.array(pos_card) + np.array((-10,0)), size=3)
        if True:  # draw triplets
            for triplet in U.triplets:
                pos_triplet = pos[triplet]
                triplet.draw(ax, np.array(pos_triplet) + np.array((1,0)), size=0.2)

    ax.set_aspect("equal")
    # zoom
    fig.savefig(
        f"figures/graph_{N_PROPERTIES}_{N_PROPERTY_VALUES}_{len(selected_nodes)}.pdf"
    )
    fig.set_size_inches(80, 40)
    fig.savefig(
        f"figures/graph_{N_PROPERTIES}_{N_PROPERTY_VALUES}_{len(selected_nodes)}.png", dpi = 300, bbox_inches='tight', pad_inches=0
    )


def setup_MKP():
    # setup multi-dimensional knapsack (MKP) problem

    Executable("cplex").set_path(
        "/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex"
    )
    solver_str = "cplex"  # or "cplex_direct", "plpk", "glpk"
    assert solver_str in ["cbc", "cplex_direct", "plpk", "glpk", "cplex"]

    deck = Deck()
    all_triplets = AllTriplets(deck)
    # # setup Pyomo model
    model = pyomo.ConcreteModel()

    # Use Python sets or lists for initialization
    V_ids = [v.id for v in deck]
    U_ids = [u.id for u in all_triplets]

    # Assign to model sets
    model.V = pyomo.Set(initialize=V_ids)
    model.U = pyomo.Set(initialize=U_ids)

    # define binary variable for each card in the deck
    model.x = pyomo.Var(model.V, within=pyomo.Binary)

    # Set objective function — maximize number of cards selected
    model.objective = pyomo.Objective(
        expr=sum(model.x[v] for v in model.V), sense=pyomo.maximize
    )

    model.constraints = pyomo.ConstraintList()
    # set constraints
    for u in all_triplets:
        model.constraints.add(
            sum(model.x[v.id] for v in deck if u.contains_card(v)) <= 2
        )

    # solve model
    from pyomo.opt import SolverFactory

    solver = SolverFactory(solver_str)
    solver.solve(model, tee=True)
    # print results
    print("Objective value:", model.objective())
    print("Selected cards:")
    for v in model.V:
        if model.x[v].value > 0:
            print(f"Card {v} selected with value {model.x[v].value}")

    # save optimal solution to file
    solution_dict = {
        "cards": [v for v in model.V if not math.isclose(model.x[v].value, 0)]
    }

    with open("solution.json", "w") as f:
        json.dump(solution_dict, f, indent=4)


def add_table_readme():

    readme_file = "readme.md"
    with open(readme_file, "r") as f:
        content = f.readlines()
        # replace between lines
        # start = content.find("<!-- TABLE START -->")
        start_line = "<!--TABLE START-->\n"
        end_line = "<!--TABLE END-->\n"
        start = content.index(start_line)
        end = content.index(end_line)

    table_content = (
        start_line
        + """|Plot|$M^P$|$M^V$|Answer|
|-|-|-|-|"""
    )
    for image in os.listdir("figures"):
        if image.endswith(".png"):
            print(image)
            if not re.match(r"graph_\d+_\d+_\d+", image):
                continue
            n_props, n_vals, n_cards = re.findall(r"graph_(\d+)_(\d+)_(\d+)", image)[0]
            if n_cards == "1":
                continue
            table_content += (
                f"\n|![{image}](figures/{image})|{n_props}|{n_vals}|{n_cards}|"
            )

    table_content += "\n" + end_line

    # delete stuff between start and end (the old table)
    content = content[:start] + content[end + 1 :]
    content.insert(start + 1, table_content + "\n")

    with open(readme_file, "w") as f:
        f.writelines(content)


def draw_solution():
    V = Deck()

    fig, ax = plt.subplots(figsize=(20, 20))
    with open("solution.json", "r") as f:
        solution = json.load(f)
        selected_cards = solution["cards"]
        selected_nodes = [card for card in V.cards if card.id in selected_cards]

    for i, card in enumerate(selected_nodes):
        posx = (i % 5)*1.2
        posy = (i // 5)*1.8
        card.draw(ax, (posx, posy), size=0.5)
        ax.set_title("Solution Deck of Cards")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    fig.savefig("figures/solution_deck_of_cards.pdf")
    fig.savefig("figures/solution_deck_of_cards.png", dpi=300, bbox_inches='tight', pad_inches=0)

def draw_triplets():
    # draw all triplets
    U = AllTriplets(Deck())
    fig, ax = plt.subplots(figsize = (50,50))

    for i, t in enumerate(U.triplets):
        posx = (i % 32)*1.2*3
        posy = (i // 32)*1.8*1
        t.draw(ax, (posx, posy), size=0.5)
        ax.set_title("All Triplets", fontsize=50)
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_aspect('equal')
        ax.axis('off')

    fig.savefig("figures/all_triplets.pdf",bbox_inches='tight', transparent="True", pad_inches=0)
    fig.savefig("figures/all_triplets.png", dpi=100, bbox_inches='tight', pad_inches=0)

def draw_deck():
    # draw deck
    deck = Deck()
    fig, ax = plt.subplots(figsize=(20, 20))

    for i, card in enumerate(deck.cards):
        posx = (i % 9)*1.2
        posy = (i // 9)*1.8
        card.draw(ax, (posx, posy), size=0.5)
        ax.set_title("Deck of Cards")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    fig.savefig("figures/deck_of_cards.pdf", bbox_inches='tight', pad_inches=0
)
    fig.savefig("figures/deck_of_cards.png", dpi=300, bbox_inches='tight', pad_inches=0
)


if __name__ == "__main__":

    for N_PROPERTIES, N_PROPERTY_VALUES in product((2, 3, 4), (3,)):

        # N_PROPERTIES = 4 # default
        # N_PROPERTY_VALUES = 3 # default

        # formulate and solve the knapsack problem
        setup_MKP()

        # plot the graph of cards and triplets, showing the solution
        plot_graph()

        # update the readme with the new table
        add_table_readme()

        # draw if the cards ar the defaults
        if N_PROPERTIES == 4 and N_PROPERTY_VALUES == 3:
            # draw deck of cards
            draw_deck() # saves figures/deck_of_cards.pdf
            # draw all triplets
            draw_triplets() # saves figures/all_triplets.pdf
            # draw solution.json
            draw_solution() # saves figures/solution_deck_of_cards.pdf

