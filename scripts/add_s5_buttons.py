import os, shutil

SRC = r"C:\Users\Curtis\Desktop\learningmaterials\SEMESTER3\bakery-ai-system\api\module4_frontend\static\index.html"
BAK = SRC + ".bak"

# Backup
shutil.copy(SRC, BAK)
print(f"Backup: {BAK}")

c = open(SRC, "r", encoding="utf-8").read()
orig_len = len(c)
print(f"Original: {orig_len} bytes")

# 1. Script tag before </body>
c = c.replace("</body>", '<script src="/s5_analysis.js"></script>\n</body>')

# 2. Forecast button
c = c.replace(
    "<h4>Demand Forecast</h4>",
    '<h4>Demand Forecast</h4><button class="btn btn-sm" style="margin-left:8px;background:linear-gradient(135deg,#6c5ce7,#a29bfe);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:11px;font-weight:600;cursor:pointer" onclick="runModuleS5Analysis(\'forecast\',\'fc-date\',\'fc-s5-result\')">\U0001f50d \'+t(\'AI Analysis\')+\'</button>'
)
c = c.replace(
    '<div id="forecast-result"',
    '<div id="fc-s5-result" style="display:none;background:linear-gradient(135deg,#f8f4ff,#ede4ff);border:2px solid #a29bfe;border-radius:10px;padding:14px;margin-bottom:12px"></div><div id="forecast-result"'
)

# 3. Schedule
c = c.replace(
    "renderScheduleTable(s,r);",
    'renderScheduleTable(s,r);\n  var scBtn=document.getElementById("sc-s5-btn-row");if(scBtn)scBtn.innerHTML="<button class=\\"btn btn-sm\\" style=\\"margin-left:8px;background:linear-gradient(135deg,#6c5ce7,#a29bfe);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:13px;font-weight:600;cursor:pointer\\" onclick=\\"runModuleS5Analysis(\'schedule\',\'sc-date\',\'sc-s5-result\')\\">\U0001f50d "+t(\"AI Analysis\")+"</button>";'
)
c = c.replace(
    "<h4>Weekly Schedule</h4>",
    '<div id="sc-s5-result" style="display:none;background:linear-gradient(135deg,#f8f4ff,#ede4ff);border:2px solid #a29bfe;border-radius:10px;padding:14px;margin-bottom:12px"></div><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px" id="sc-s5-btn-row"><h4>Weekly Schedule</h4>'
)

# 4. Inventory
c = c.replace(
    "renderInventoryTable(d);",
    'renderInventoryTable(d);\n  var invBtn=document.getElementById("inv-s5-btn");if(invBtn)invBtn.innerHTML="<button class=\\"btn btn-sm\\" style=\\"background:linear-gradient(135deg,#6c5ce7,#a29bfe);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:13px;font-weight:600;cursor:pointer;margin-left:8px\\" onclick=\\"runModuleS5Analysis(\'inventory\',\'inv-date\',\'inv-s5-result\')\\">\U0001f50d "+t(\"AI Analysis\")+"</button>";'
)
c = c.replace(
    "<h4>Batch Inventory</h4>",
    '<div id="inv-s5-result" style="display:none;background:linear-gradient(135deg,#f8f4ff,#ede4ff);border:2px solid #a29bfe;border-radius:10px;padding:14px;margin-bottom:12px"></div><div style="display:flex;align-items:center;gap:8px" id="inv-s5-btn"><h4>Batch Inventory</h4>'
)

# 5. Wastage
c = c.replace(
    "renderWastageLog(d);",
    'renderWastageLog(d);\n  var wsBtn=document.getElementById("ws-s5-btn");if(wsBtn)wsBtn.innerHTML="<button class=\\"btn btn-sm\\" style=\\"background:linear-gradient(135deg,#6c5ce7,#a29bfe);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:13px;font-weight:600;cursor:pointer;margin-left:8px\\" onclick=\\"runModuleS5Analysis(\'wastage\',\'ws-date\',\'ws-s5-result\')\\">\U0001f50d "+t(\"AI Analysis\")+"</button>";'
)
c = c.replace(
    "<h4>Wastage Log</h4>",
    '<div id="ws-s5-result" style="display:none;background:linear-gradient(135deg,#f8f4ff,#ede4ff);border:2px solid #a29bfe;border-radius:10px;padding:14px;margin-bottom:12px"></div><div style="display:flex;align-items:center;gap:8px" id="ws-s5-btn"><h4>Wastage Log</h4>'
)

# Verify content is valid before writing
new_len = len(c)
if new_len > orig_len - 1000:
    open(SRC, "w", encoding="utf-8").write(c)
    # Verify write
    verify = open(SRC, "r", encoding="utf-8").read()
    if len(verify) == new_len:
        print(f"OK: {new_len} bytes written and verified")
    else:
        print(f"FAIL: wrote {len(verify)} but expected {new_len}")
else:
    print(f"ABORT: content too short ({new_len} vs {orig_len}), file may be corrupted")
