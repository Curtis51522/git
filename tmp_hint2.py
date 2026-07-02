path = r"C:\Users\Curtis\Desktop\learningmaterials\SEMESTER3\bakery-ai-system\api\module4_frontend\static\index.html"
with open(path, "rb") as f:
    content = f.read().decode("utf-8")

# Update hint text
old_hint = "Add items to cart then generate Top-3 bundles"
new_hint = "Click Generate Top-3 for bundle recommendations"
content = content.replace(old_hint, new_hint)

# Also update the ZH version
old_zh = "先将商品加入购物车，再生成前三捆绑"
if old_zh in content:
    content = content.replace(old_zh, "点击生成前三捆绑，获取套餐推荐")
    print("ZH hint updated")

print("Hint text updated")
with open(path, "wb") as f:
    f.write(content.encode("utf-8"))
print("Done")
