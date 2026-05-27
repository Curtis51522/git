
import os

# Read the current working skeleton
filepath = r"C:\Users\Curtis\Desktop\learningmaterials\SEMESTER3\bakery-ai-system\api\module4_frontend\static\index.html"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Find the </script> tag and everything between <script> and </script>
script_start = content.find("<script>")
script_end = content.find("</script>") + len("</script>")

# Read the new JS from _full.js
js_path = r"C:\Users\Curtis\Desktop\learningmaterials\SEMESTER3\bakery-ai-system\api\module4_frontend\static\_full.js"
with open(js_path, "r", encoding="utf-8") as f:
    new_js = f.read()

# Replace the old script with the new one
content = content[:script_start] + new_js + content[script_end:]

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("File rebuilt successfully. Size:", len(content))
