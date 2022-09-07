import numpy as np
from collections import defaultdict

def test():

    a = Lattice("abcd")
    b = Lattice("bzde")
    c = combine(a,b)
    d = Lattice("bcdef")
    e = combine(c, d)

    print(e)

def ensemble(pred_lst):

    assert len(pred_lst) >= 1

    lattice_base = None

    for pred in pred_lst:
        if not isinstance(pred, list):
            pred = pred.split()

        if lattice_base is None:
            lattice_base = Lattice(pred)

        else:
            lattice_base = combine(lattice_base, Lattice(pred))

    return lattice_base.compute()


def combine(lattice_a, lattice_b, verbose=True, out=None):

    # output
    lattice = Lattice()

    cs_lst = []

    # length of each string
    len_a = len(lattice_a)
    len_b = len(lattice_b)

    # dp table
    dp = [[0 for x in range(len_a+1)] for y in range(len_b+1)]
    path = [[(0, 0) for x in range(len_a+1)] for y in range(len_b+1)]

    # initialize first row and first column
    for i in range(1, len_a+1):
        dp[0][i] = i
        path[0][i] = (0, i-1)

    for i in range(1, len_b+1):
        dp[i][0] = i
        path[i][0] = (i-1, 0)

    # dp update
    for i in range(1, len_b+1):
        for j in range(1, len_a+1):
            index_a = j-1
            index_b = i-1

            cs_a = lattice_a[index_a]
            cs_b = lattice_b[index_b]

            sub_cost = cs_a.substitute_cost(cs_b)
            del_cost = cs_a.delete_cost()
            add_cost = cs_b.delete_cost()

            dp[i][j] = dp[i-1][j-1]+sub_cost
            path[i][j] = (i-1,j-1)

            if dp[i][j] > dp[i-1][j]+del_cost:
                dp[i][j] = dp[i-1][j]+del_cost
                path[i][j] = (i-1,j)

            if dp[i][j] > dp[i][j-1]+add_cost:
                dp[i][j] = dp[i][j-1]+add_cost
                path[i][j] = (i, j-1)

    cur_node = (len_b, len_a)

    while(cur_node != (0,0)):
        prev_node = path[cur_node[0]][cur_node[1]]

        cs_a = lattice_a[cur_node[1]-1]
        cs_b = lattice_b[cur_node[0]-1]

        # substitution or match
        if prev_node[0]+1 == cur_node[0] and prev_node[1]+1 == cur_node[1]:

            cs_a.merge(cs_b)
            cs_lst.append(cs_a)


        # addition
        if prev_node[0] + 1 == cur_node[0] and prev_node[1] == cur_node[1]:

            cs_e = cs_a.create_empty_set()
            cs_e.merge(cs_b)

            cs_lst.append(cs_e)

        # deletion
        if prev_node[0] == cur_node[0] and prev_node[1]+1 == cur_node[1]:

            cs_e = cs_b.create_empty_set()
            cs_e.merge(cs_a)
            cs_lst.append(cs_e)

        cur_node = prev_node

    cs_lst.reverse()
    lattice.cs_lst = cs_lst

    return lattice

class CorrespondenceSet:

    def __init__(self, units=None, scores=None):

        if units is None:
            units = []

        if scores is None:
            scores = [1.0]*len(units)

        self.units = units
        self.unit_set = set(units)
        self.scores = scores


    def __repr__(self):
        return '{'+','.join(self.units)+'}'

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        for unit in self.units:
            yield unit

    def __len__(self):
        return len(self.units)

    def __getitem__(self, idx):
        return self.units[idx]

    def compute(self):
        unit2score = defaultdict(float)

        for i in range(len(self.units)):
            unit = self.units[i]
            score = self.scores[i]
            unit2score[unit] += score

        return sorted(unit2score.items(), key=lambda x:-x[1])[0][0]

    def has_epsilon(self):
        return '<blk>' in self.unit_set

    def contains(self, unit):
        return unit in self.unit_set

    def delete_cost(self):
        if self.has_epsilon():
            return 0
        else:
            return 1

    def substitute_cost(self, other):
        overlap = False
        for unit in self.units:
            if other.contains(unit):
                overlap = True

        if overlap:
            return 0
        else:
            return 1

    def create_empty_set(self):
        ave_score = np.mean(self.scores)
        cs = CorrespondenceSet(['<blk>'], [ave_score])
        return cs

    def merge(self, other):

        for unit, score in zip(other.units, other.scores):
            self.units.append(unit)
            self.scores.append(score)
            self.unit_set.add(unit)


class Lattice:

    def __init__(self, units=None, scores=None):

        if units is None:
            units = []
        else:
            units = list(units)

        if scores is None:
            scores = [1.0]*len(units)
        else:
            scores = list(scores)

        assert len(units) == len(scores)

        self.cs_lst = []

        for unit, score in zip(units, scores):
            self.cs_lst.append(CorrespondenceSet([unit], [score]))

    def __repr__(self):
        return '\n'.join([str(cs) for cs in self.cs_lst])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.cs_lst)

    def __getitem__(self, idx):
        return self.cs_lst[idx]

    def __iter__(self):
        for cs in self.cs_lst:
            yield cs

    def compute(self):
        cs_items = [cs.compute() for cs in self.cs_lst]

        return [cs_item for cs_item in cs_items if cs_item != "<blk>" and not cs_item.isspace()]