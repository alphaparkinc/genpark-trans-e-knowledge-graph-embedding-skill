import sys
import json
from client import TransEEmbedding

te = TransEEmbedding(4)

def handle_call(name, arguments):
    if name == "set_vec":
        te.set_vector(arguments["name"], arguments["vec"], arguments.get("is_rel", False))
        return {"status": "ok"}
    elif name == "distance":
        d = te.distance(arguments["h"], arguments["r"], arguments["t"])
        return {"distance": d}
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_call(req.get("name"), req.get("arguments", {}))
            print(json.dumps({"id": req.get("id"), "result": res}))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
