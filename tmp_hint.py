path = r"C:\Users\Curtis\Desktop\learningmaterials\SEMESTER3\bakery-ai-system\api\module4_frontend\static\index.html"
with open(path, "rb") as f:
    content = f.read().decode("utf-8")
# Check bundle hint text
idx = content.find("Add items to cart then generate")
if idx >= 0:
    print(content[idx:idx+120])
idx = content.find("Generate Top-3 bundles first")
if idx >= 0:
    print(content[idx:idx+80])
