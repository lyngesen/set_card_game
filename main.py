from typing import List, Literal
from itertools import product, combinations
import json, math, os, re
import pyomo.environ as pyomo
from pyomo.common.fileutils import Executable
import networkx as nx
import matplotlib.pyplot as plt


PropertyValue = int  # each property has exactly 3 values
CardTuple = tuple[PropertyValue, ...]


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

    fig, ax = plt.subplots(figsize=(20, 20))

    V = Deck()
    U = AllTriplets(V)
    E = [(v, u) for u in U for v in V if u.contains_card(v)]

    G = nx.Graph()
    G.add_edges_from(E)

    # bipartite position of nodes split into V and U
    pos = nx.bipartite_layout(G, nodes=V.cards, align="vertical")
    pos = {v: (0, v.id * (len(U.triplets) / len(V.cards))) for v in V.cards} | {
        u: (1, u.id) for u in U.triplets
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

    ax.set_xlabel(
        f"A total of ${len(selected_cards)}$ cards have been chosen (marked red). Each possible valid triplet has an edge\n for each card it consists of, this is colored red if the card is selected in the solution. \n The number for each of the 1080 triplets indicate the number of cards in the tripliet selected. \n\nNote: see file figures/graph_{N_PROPERTIES}_{N_PROPERTY_VALUES}_{len(selected_nodes)}.pdf in order to zoom — to read the values",
        fontsize=26,
    )

    # zoom
    fig.savefig(
        f"figures/graph_{N_PROPERTIES}_{N_PROPERTY_VALUES}_{len(selected_nodes)}.pdf"
    )
    fig.savefig(
        f"figures/graph_{N_PROPERTIES}_{N_PROPERTY_VALUES}_{len(selected_nodes)}.png"
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
