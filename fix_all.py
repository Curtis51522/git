import sys, os, shutil

BASE = r"C:\Users\Curtis\Desktop\learningmaterials\SEMESTER3\bakery-ai-system"
path = os.path.join(BASE, "api", "module4_frontend", "static", "index.html")

# Backup
bak = path + ".backup"
shutil.copy2(path, bak)
print(f"Backup saved: {bak}")

with open(path, "r", encoding="utf-8") as f:
    c = f.read()

changes = 0

# ============================================
# CHANGE 1: Payment Modal JS + HTML
# ============================================
old_cp = """async function completePayment(){
if(cartItems.length===0){alert(t("No items in cart"));return;}
var total=0;for(var i=0;i<cartItems.length;i++){var it=cartItems[i];var pr=getItemPrice(it.product_name,it.freshness);total+=pr*it.quantity;}
total=Math.round(total*100)/100;
if(!confirm(t("Checkout")+" - "+t("Total")+": RM "+total.toFixed(2)))return;
try{
var hdrs={"Content-Type":"application/json","Authorization":"Bearer "+token};
var items=cartItems.map(function(ci){return{product_name:ci.product_name,quantity:ci.quantity,freshness:ci.freshness};});
var resp=await fetch(API_BASE+"/s4/checkout/complete",{method:"POST",headers:hdrs,body:JSON.stringify({items:items})});
if(!resp.ok)throw new Error("Checkout failed");
var d=await resp.json();
alert(t("Payment Complete")+"! Total: RM "+d.receipt.total.toFixed(2)+". "+d.receipt.items.length+" "+t("items deducted"));
hitlLog.push({action:"Checkout: "+d.receipt.total.toFixed(2)+" ("+d.receipt.items.length+" items)",time:new Date().toLocaleTimeString()});
cartItems=[];detections=[];lastScanResult=null;bundleRecs=[];
renderPOS(document.getElementById("content-area"));
}catch(e){alert(t("Payment Failed")+": "+e.message);}
}"""

new_cp = """var _payTotal=0,_payMethod="",_payReceipt=null,_payCashInput=0;
function completePayment(){
if(cartItems.length===0){alert(t("No items in cart"));return;}
var total=0;_payTotal=0;
for(var i=0;i<cartItems.length;i++){var it=cartItems[i];var pr=getItemPrice(it.product_name,it.freshness);total+=pr*it.quantity;}
_payTotal=Math.round(total*100)/100;
cartItemsRaw=cartItems.map(function(ci){return Object.assign({},ci);});
document.getElementById("pay-total").textContent="RM "+_payTotal.toFixed(2);
document.getElementById("pay-cash-row").style.display="none";
document.getElementById("pay-qr-msg").style.display="none";
document.getElementById("pay-cash").value="";
document.getElementById("pay-change").textContent="";
document.getElementById("pay-confirm-btn").disabled=true;
document.getElementById("payment-modal").classList.add("show");
}
function selectPaymentMethod(m){
_payMethod=m;
document.getElementById("pay-cash-row").style.display=m==="cash"?"block":"none";
document.getElementById("pay-qr-msg").style.display=m==="qr"?"block":"none";
document.getElementById("pay-confirm-btn").disabled=m==="cash";
if(m==="card"){confirmPayment();return;}
}
function calcChange(){
var rc=parseFloat(document.getElementById("pay-cash").value)||0;
var ch=rc-_payTotal;
document.getElementById("pay-change").textContent=ch>=0?"Change: RM "+ch.toFixed(2):"";
document.getElementById("pay-confirm-btn").disabled=ch<0;
_payCashInput=rc;
}
async function confirmPayment(){
if(_payMethod==="qr"){showReceipt(null);return;}
try{
var hdrs={"Content-Type":"application/json","Authorization":"Bearer "+token};
var items=cartItems.map(function(ci){return{product_name:ci.product_name,quantity:ci.quantity,freshness:ci.freshness};});
var resp=await fetch(API_BASE+"/s4/checkout/complete",{method:"POST",headers:hdrs,body:JSON.stringify({items:items})});
if(!resp.ok)throw new Error("Checkout failed");
var d=await resp.json();
if(d.status==="ok"&&d.receipt){_payReceipt=d.receipt;_payReceipt.payment_method=_payMethod;if(_payCashInput>0)_payReceipt.cash_received=_payCashInput;}
showReceipt(_payReceipt);
hitlLog.push({action:"Checkout: "+(_payReceipt?(_payReceipt.total||_payTotal).toFixed(2):_payTotal.toFixed(2))+" ("+cartItems.length+" items)",time:new Date().toLocaleTimeString()});
cartItems=[];detections=[];lastScanResult=null;bundleRecs=[];
}catch(e){alert("Payment Failed: "+e.message);}
}
function showReceipt(r){
document.getElementById("payment-modal").classList.remove("show");
if(!r){r={items:cartItemsRaw.map(function(ci){return{product_name:ci.product_name,quantity:ci.quantity,line_total:getItemPrice(ci.product_name,ci.freshness)*ci.quantity};}),total:_payTotal,payment_method:_payMethod};}
var h="<div class=\\"receipt\\">";
h+="<h3 style=\\"text-align:center;margin:0 0 12px 0\\">Receipt</h3>";
for(var i=0;i<r.items.length;i++){var it=r.items[i];h+="<div class=\\"receipt-line\\"><span class=\\"item-name\\">"+it.product_name+"</span><span class=\\"item-qty\\">x"+it.quantity+"</span><span>RM"+(it.line_total||0).toFixed(2)+"</span></div>";}
h+="<hr>";h+="<div class=\\"receipt-total\\"><span>Total</span><span>RM "+(r.total||_payTotal).toFixed(2)+"</span></div>";
if(r.payment_method){h+="<div style=\\"font-size:11px;color:#666;margin-top:4px\\">"+r.payment_method.toUpperCase()+"</div>";}
if(r.cash_received&&r.cash_received>0){h+="<div style=\\"font-size:11px;color:#666\\">Received: RM "+r.cash_received.toFixed(2)+"</div>";var ch2=r.cash_received-(r.total||_payTotal);h+="<div style=\\"font-size:11px;color:#666\\">Change: RM "+ch2.toFixed(2)+"</div>";}
h+="<div style=\\"text-align:center;margin-top:16px\\"><button class=\\"btn btn-primary btn-sm\\" onclick=\\"document.getElementById('receipt-overlay').classList.remove('show');renderPOS(document.getElementById('content-area'))\\">New Sale</button></div>";
h+="</div>";
document.getElementById("receipt-content").innerHTML=h;
document.getElementById("receipt-overlay").classList.add("show");
}
function closePaymentModal(){document.getElementById("payment-modal").classList.remove("show");}
function closeReceipt(){document.getElementById("receipt-overlay").classList.remove("show");renderPOS(document.getElementById("content-area"));}"""

if old_cp in c:
    c = c.replace(old_cp, new_cp)
    changes += 1
    print("1. Payment JS replaced")
else:
    print("1. FAIL: completePayment not found")

# Add payment HTML before </body> (ONLY the last occurrence)
if "</body>" in c:
    last_body = c.rindex("</body>")
    payment_html = """
<!-- Payment Modal -->
<div id="payment-modal" class="payment-overlay" onclick="if(event.target===this)closePaymentModal()">
<div class="payment-panel">
<div class="payment-header"><h3>Payment</h3><button class="close-btn" onclick="closePaymentModal()">&times;</button></div>
<div style="padding:12px 0">Total: <strong id="pay-total" style="font-size:18px">RM 0.00</strong></div>
<div style="display:flex;gap:8px;margin-bottom:12px">
<button class="btn btn-outline btn-sm" style="flex:1" onclick="selectPaymentMethod('cash')">Cash</button>
<button class="btn btn-outline btn-sm" style="flex:1" onclick="selectPaymentMethod('card')">Card</button>
<button class="btn btn-outline btn-sm" style="flex:1" onclick="selectPaymentMethod('qr')">QR Code</button>
</div>
<div id="pay-cash-row" style="display:none;margin-bottom:12px">
<label style="font-size:12px;color:#666">Received (RM)</label>
<input type="number" id="pay-cash" step="0.01" min="0" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:6px;font-size:16px" oninput="calcChange()">
<div id="pay-change" style="font-size:14px;color:#4caf50;margin-top:4px;font-weight:600"></div>
</div>
<div id="pay-qr-msg" style="display:none;text-align:center;padding:16px;background:#f5f5f5;border-radius:8px;margin-bottom:12px">
<div style="font-size:48px">&#x1F4F1;</div>
<div style="font-size:14px;color:#666;margin-top:8px">Scan QR code to pay<br>Click Confirm after payment</div>
</div>
<button id="pay-confirm-btn" class="btn btn-primary" style="width:100%;padding:12px;font-size:16px" onclick="confirmPayment()" disabled>Confirm Payment</button>
</div>
</div>

<!-- Receipt Overlay -->
<div id="receipt-overlay" class="payment-overlay">
<div class="receipt-container" id="receipt-content"></div>
</div>
"""
    c = c[:last_body] + payment_html + "\n</body>" + c[last_body+7:]
    changes += 1
    print("1b. Payment HTML added")
else:
    print("1b. FAIL: </body> not found")

# ============================================
# CHANGE 2: Coffee Customization
# ============================================
# 2a. Global vars
old_vars = "var COFFEE_DRINKS=["
new_vars = "var _ci='ice',_cs='sugar',_cn='normal',_cl='less',_ch='half',_cx='none',_ht='hot',_ic='iced';\nvar _coffeeKey='',_coffeeName='',_coffeePrice=0,_coffeeIce='',_coffeeSugar='',_coffeeTemp='hot';\n\nvar COFFEE_DRINKS=["
c = c.replace(old_vars, new_vars)
changes += 1
print("2a. Coffee vars added")

# 2b. Button onclick  
old_btn = "quickAddCoffee('+" + '"' + "'" + '"'
new_btn = "openCoffeeOptions('+" + '"' + "'" + '"' + "+cd.key+" + '"' + "'" + '"' + "+','+" + '"' + "'" + '"' + "+cd.name+" + '"' + "'" + '"' + "+',\"+cd.price+')"
c = c.replace(old_btn, new_btn)
changes += 1
print("2b. Coffee button onClick fixed")

# 2c. Replace quickAddCoffee function
old_func = "function quickAddCoffee(name){\nvar found=false;\nvar price=COFFEE_PRICES[name];\nfor(var i=0;i<cartItems.length;i++){\nif(cartItems[i].product_name===name){cartItems[i].quantity+=1;found=true;break;}}\nif(!found){cartItems.push({product_name:name,quantity:1,price:price,freshness:'N/A'});}\nrenderPOS(document.getElementById('content-area'));\n}"
new_func = """function openCoffeeOptions(key,name,price){
_coffeeKey=key;_coffeeName=name;_coffeePrice=price;_coffeeIce='normal';_coffeeSugar='normal';
document.getElementById('coffee-opt-name').textContent=name+' - RM '+price.toFixed(2);
var isIced=key==='cold_brew'||key==='iced_americano';
_coffeeTemp=isIced?'iced':'hot';
document.getElementById('coffee-hot-btn').classList.toggle('selected',_coffeeTemp==='hot');
document.getElementById('coffee-iced-btn').classList.toggle('selected',_coffeeTemp==='iced');
document.getElementById('coffee-ice-row').style.display=isIced?'flex':'none';
var iceRow=document.getElementById('coffee-ice-row');
for(var i=1;i<iceRow.children.length;i++)iceRow.children[i].classList.toggle('selected',i===1);
var sugRow=document.querySelectorAll('#coffee-options .coffee-pill-row')[2]||document.querySelectorAll('#coffee-options .coffee-pill-row')[1];
if(sugRow)for(var j=1;j<sugRow.children.length;j++)sugRow.children[j].classList.toggle('selected',j===1);
document.getElementById('coffee-options').style.display='block';
}
function confirmCoffeeAdd(){
var ice=_coffeeIce,sugar=_coffeeSugar,temp=_coffeeTemp;
var label=_coffeeName,sub=[];
if(temp==='iced'){if(ice&&ice!=='normal')sub.push({less:'Less Ice',none:'No Ice'}[ice]||ice);sub.push('Iced');}
else sub.push('Hot');
if(sugar&&sugar!='normal')sub.push({less:'Less Sugar',half:'Half Sugar',none:'No Sugar'}[sugar]||sugar);
if(sub.length)label+=' ('+sub.join(', ')+')';
var found=false;
for(var i=0;i<cartItems.length;i++){
if(cartItems[i].product_name===_coffeeKey&&(cartItems[i].ice||'')===ice&&(cartItems[i].sugar||'')===sugar&&(cartItems[i].temp||'hot')===temp){
cartItems[i].quantity+=1;found=true;break;
}}
if(!found){cartItems.push({product_name:_coffeeKey,quantity:1,price:_coffeePrice,freshness:'N/A',ice:ice,sugar:sugar,temp:temp,label:label});}
document.getElementById('coffee-options').style.display='none';
renderPOS(document.getElementById('content-area'));
}
function cancelCoffeeAdd(){document.getElementById('coffee-options').style.display='none';}
function selCoffeePill(type,val,el){
if(type==='ice')_coffeeIce=val;else if(type==='sugar')_coffeeSugar=val;
var row=el.parentElement;
for(var i=0;i<row.children.length;i++){if(row.children[i].tagName==='BUTTON')row.children[i].classList.toggle('selected',row.children[i]===el);}
}
function selCoffeeTemp(temp,el){
_coffeeTemp=temp;
document.getElementById('coffee-hot-btn').classList.toggle('selected',temp==='hot');
document.getElementById('coffee-iced-btn').classList.toggle('selected',temp==='iced');
document.getElementById('coffee-ice-row').style.display=temp==='iced'?'flex':'none';
}"""
c = c.replace(old_func, new_func)
changes += 1
print("2c. Coffee functions replaced")

# 2d. Coffee panel HTML
old_panel = "h+='<div class=\"drink-grid\">'+coffeeBtns+'</div></div>';"
new_panel = """h+='<div class=\"drink-grid\">'+coffeeBtns+'</div>';
h+='<div id=\"coffee-options\" style=\"display:none;background:#fdfaf5;border:1px solid #8b6914;border-radius:10px;padding:10px 12px;margin-top:6px\">';
h+='<div style=\"font-weight:700;margin-bottom:8px;font-size:14px;color:#8b6914\" id=\"coffee-opt-name\"></div>';
h+='<div class=\"coffee-pill-row\" id=\"coffee-hot-row\"><span class=\"pill-label\">Temp</span><button class=\"coffee-pill selected\" id=\"coffee-hot-btn\" onclick=\"selCoffeeTemp(_ht,this)\">Hot</button><button class=\"coffee-pill\" id=\"coffee-iced-btn\" onclick=\"selCoffeeTemp(_ic,this)\">Iced</button></div>';
h+='<div class=\"coffee-pill-row\" id=\"coffee-ice-row\"><span class=\"pill-label\">Ice</span><button class=\"coffee-pill selected\" onclick=\"selCoffeePill(_ci,_cn,this)\">Normal</button><button class=\"coffee-pill\" onclick=\"selCoffeePill(_ci,_cl,this)\">Less</button><button class=\"coffee-pill\" onclick=\"selCoffeePill(_ci,_cx,this)\">None</button></div>';
h+='<div class=\"coffee-pill-row\"><span class=\"pill-label\">Sugar</span><button class=\"coffee-pill selected\" onclick=\"selCoffeePill(_cs,_cn,this)\">Normal</button><button class=\"coffee-pill\" onclick=\"selCoffeePill(_cs,_cl,this)\">Less</button><button class=\"coffee-pill\" onclick=\"selCoffeePill(_cs,_ch,this)\">Half</button><button class=\"coffee-pill\" onclick=\"selCoffeePill(_cs,_cx,this)\">None</button></div>';
h+='<div style=\"display:flex;gap:6px;margin-top:8px\"><button class=\"btn btn-primary btn-sm\" style=\"flex:1\" onclick=\"confirmCoffeeAdd()\">Add to Cart</button><button class=\"btn btn-outline btn-sm\" style=\"flex:1\" onclick=\"cancelCoffeeAdd()\">Cancel</button></div>';
h+='</div></div>';"""
c = c.replace(old_panel, new_panel)
changes += 1
print("2d. Coffee panel HTML added")

# 2e. Coffee CSS
old_css = "@media print{"
new_css = """.coffee-pill-row{display:flex;align-items:center;gap:6px;margin:6px 0}
.coffee-pill-row .pill-label{font-size:12px;color:#888;min-width:36px;font-weight:600}
.coffee-pill{padding:4px 10px;border:1px solid #ccc;border-radius:14px;background:#fff;font-size:12px;cursor:pointer;color:#555;transition:all .15s}
.coffee-pill.selected{background:#4caf50;color:#fff;border-color:#4caf50}
.coffee-pill:hover:not(.selected){border-color:#8b6914;color:#8b6914}
@media print{"""
c = c.replace(old_css, new_css)
changes += 1
print("2e. Coffee CSS added")

# Write back
with open(path, "w", encoding="utf-8") as f:
    f.write(c)

print(f"\nTotal changes applied: {changes}/7")
print(f"Backup at: {bak}")
print("Restore with: git checkout -- api/module4_frontend/static/index.html")
