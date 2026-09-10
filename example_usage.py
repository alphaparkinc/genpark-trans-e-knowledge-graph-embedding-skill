from client import TransEEmbedding

def main():
    print("=== Testing TransE Knowledge Graph Embedding ===")
    te = TransEEmbedding(dim=2)
    te.set_vector("Paris", [1.0, 0.0])
    te.set_vector("capital_of", [0.0, 1.0], is_relation=True)
    te.set_vector("France", [1.0, 1.0])

    dist = te.distance("Paris", "capital_of", "France")
    print(f"Computed geometric distance d(Paris + capital_of, France): {round(dist, 4)}")
    assert dist < 0.5
    print("=== All tests passed successfully! ===")

if __name__ == "__main__":
    main()
