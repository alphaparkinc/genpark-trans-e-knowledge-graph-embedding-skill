import math

class TransEEmbedding:
    """
    TransE: Translating Embeddings for Modeling Multi-relational Data (Bordes et al.).
    Models relationship as translation: h + r approx t.
    Calculates L1/L2 distance d(h+r, t) and margin-based loss.
    """
    def __init__(self, dim=4):
        self.dim = dim
        self.entities = {}
        self.relations = {}

    def set_vector(self, name, vec, is_relation=False):
        norm = math.sqrt(sum(x*x for x in vec)) or 1.0
        normalized = [x / norm for x in vec]
        if is_relation:
            self.relations[name] = normalized
        else:
            self.entities[name] = normalized

    def distance(self, h_name, r_name, t_name):
        h = self.entities[h_name]
        r = self.relations[r_name]
        t = self.entities[t_name]
        diff = [h[i] + r[i] - t[i] for i in range(self.dim)]
        return math.sqrt(sum(x*x for x in diff))
