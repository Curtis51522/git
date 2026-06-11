import os, ast, builtins

data = []
for root, dirs, files in os.walk("api"):
    for fn in files:
        if not fn.endswith(".py"): continue
        p = os.path.join(root, fn)
        try: t = ast.parse(open(p, encoding="utf-8").read())
        except: continue
        for node in ast.walk(t):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    ds = ast.unparse(dec) if hasattr(ast, "unparse") else ""
                    if "router." in ds:
                        for a in (dec.args if hasattr(dec, "args") else []):
                            if isinstance(a, ast.Constant):
                                m = "GET" if "get" in ds.lower() else "POST"
                                data.append({"file": p, "route": a.value, "method": m, "handler": node.name, "tree": t, "node": node})
                                break

# Check route conflicts
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

# Check undefined names in handlers
for ep in data:
    names = set()
    for n in ast.walk(ep["node"]):
        if isinstance(n, ast.Name):
            names.add(n.id)
    imports = set(dir(builtins))
    for n in ast.walk(ep["tree"]):
        if isinstance(n, ast.Import):
            for alias in n.names:
                imports.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                imports.add(n.module.split(".")[0])
            for alias in n.names:
                imports.add(alias.asname or alias.name)
    undefined = names - imports - {"self", "True", "False", "None"}
    if undefined:
        print("UNDEFINED: " + ep["handler"] + " missing " + str(undefined))