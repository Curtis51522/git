path = r"C:\Users\Curtis\Desktop\learningmaterials\SEMESTER3\bakery-ai-system\api\module4_frontend\static\index.html"
with open(path, "rb") as f:
    content = f.read().decode("utf-8")

old = "if(!cartItems.length){alert('Add items to cart first');return}"
if old in content:
    # Replace: just skip the check, always allow bundle generation
    # The cart items array is built from cartItems, if empty it will just send empty items
    content = content.replace(old, "")
    print("Removed cart-empty block")
else:
    print("NOT FOUND")

import re, time
content = re.sub(r'console\.log\("JS loaded v\d+"\)', f'console.log("JS loaded v{int(time.time())}")', content)

with open(path, "wb") as f:
    f.write(content.encode("utf-8"))
print("Done")
