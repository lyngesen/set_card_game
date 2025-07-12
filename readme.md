
# Finding a maximal number of cards without any valid sets in the game Set

[The game set](https://en.wikipedia.org/wiki/Set_(card_game)) is a card game where players compete in identifying *valid sets* from a pool of cards.

Each card consists of a *color, shape, number,* and *filling*. Each of the four properties have exactly three possibilities, that is, there are three colors, three shapes etc. One can represent a card using a vector $v \in \{1,2,3\}^4 := V$. Eg. the card $v= \{1,3,1,2\}$ contains the 1st *color*, 3rd *shape*, 1st *number* (the number 1) and the 2nd *filling*. There are a total of $|V|= 3^4 = 81$ cards.

<!--![All 81 cards](figures/deck_of_cards.png) -->
<img src="figures/deck_of_cards.png" alt="Deck of cards" width="300"/>

Three cards are said to be a valid *set* if for each of the four properties: all cards have the identical value of the property, or all cards have different values of the property. (See the rules [here](https://en.wikipedia.org/wiki/Set_(card_game)) for a better explanation).

Doing the game, the players are presented with $12$ cards. If no set can be made, an additional card is placed until a set is found. One can ask the question: **What is the smallest number of cards for which there must exist a valid set?**. This has been answered already, but I will find a solution using [integer programming](https://en.wikipedia.org/wiki/Integer_programming).

One can do this by creating a integer programming formulation of the decision problem **What is the largest number of cards possible, that does not contain a valid set?**.

Consider the set of all cards $V$, and use this to define the set of all valid triplets $U$. Given three cards $v^1, v^2$ and $v^3$ one can check if the triplet $(v^1,v^2,v^3)$ is valid using the following function *is_valid*:

```{python}
def is_valid(card1,card2,card3):
    for p in range(N_PROPERTIES):
        if card1[p] == card2[p] == card3[p]:
            continue
        elif card1[p] != card2[p] != card3[p] != card1[p]:
            continue
        else:
            return False
    return True
```

Doing this for all triplets gives us the set $U$ of all $1080$ valid triplets.


![All 81 cards](figures/all_triplets.png) 

$$x_v = \begin{cases}1, \quad \text{if card $v$ is selected}\\ 0,\quad \text{else}\end{cases}$$

One can formulate an integer program with the above decision variables, and constraints ensuring that no valid set exists among the selected cards:

$$
\begin{align}
\max&\sum_{v\in V} x_v\\
s.t. &\sum_{v \in N(u)} x_v &\le 2,\quad \forall u \in U\\
&x_v &\in \{0,1\}, \forall v \in V
\end{align}
$$

In the optimal solution — solved using CPLEX — $20$ cards where chosen. This implies that for a set of $21$ cards there must exist at least one valid set. 

<img src="figures/solution_deck_of_cards.png" alt="Solution" width="300"/>

For a bipartite graph showing the solution, where the left side is the set of cards and the right side is the set of valid triplets, see [this PDF](figures/graph_4_3_20.pdf).

Note: [The solution of 20 was already found in 1971](https://en.wikipedia.org/wiki/Set_(card_game)#Basic_combinatorics_of_Set)


# Usage 

- Create a virtual environment with `python3 -m venv .venv` — then enter the enviroment.
- (inside the virtual environment) Install the dependencies with `pip install -r requirements.txt`
    - The knapsack problem is modeled using the python package `pyomo` and solved using `CPLEX`. To configure `CPLEX` or any other solver: add the path to the executible as done on  `Executable(<name of solver eg. cplex/gurobi/glpk>).set_path(<path to solver exe>)`.
- Run the script with `python main.py`



# Alternative sets of cards

The deck of cards are determined a set of $M^P$ properties containing $M^V$ values. In the game set one considers $M^P=4$ properties and $M^V=3$ values. One can answer the question **What is the largest number of cards possible, that does not contain a valid set** for different values of $M^P$ and $M^V$.














<!--TABLE START-->
|Plot|$M^P$|$M^V$|Answer|
|-|-|-|-|
|![graph_2_3_4.png](figures/graph_2_3_4.png)|2|3|4|
|![graph_4_3_20.png](figures/graph_4_3_20.png)|4|3|20|
|![graph_3_3_9.png](figures/graph_3_3_9.png)|3|3|9|
<!--TABLE END-->

