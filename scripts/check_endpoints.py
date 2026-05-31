import os, ast
data = []
for root, dirs, files in os.walk("api"):
    for fn in files:
        if not fn.endswith(".py"): continue
        p = os.path.join(root, fn)
        try:
            t = ast.parse(open(p, encoding="utf-8").read())
        except: continue
        for node in ast.walk(t):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    ds = ast.unparse(dec) if hasattr(ast, "unparse") else ""
                    if "router." in ds:
                        for a in dec.args if hasattr(dec, "args") else []:
                            if isinstance(a, ast.Constant):
                                m = "GET" if "get" in ds.lower() else "POST"
                                data.append({"file": p, "route": a.value, "method": m, "handler": node.name})
                                break
routes = {}
conflicts = 0
for ep in data:
    k = (ep["method"], ep["route"])
    if k in routes:
        print("CONFLICT: " + ep["method"] + " " + ep["route"] + " in " + ep["file"] + " and " + routes[k]["file"])
        conflicts += 1
    else:
        routes[k] = ep
print("Total: " + str(len(data)) + " endpoints, " + str(conflicts) + " conflicts")
for ep in sorted(data, key=lambda x: x["route"]):
    print("  " + ep["method"] + " " + ep["route"] + " -> " + ep["handler"] + " (" + ep["file"] + ")")