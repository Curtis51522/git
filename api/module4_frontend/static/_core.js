<script>
var token='', role='', username='', currentPanel='pos';
var API='';
var cartItems=[], detections=[], hitlLog=[], bundleRecs=[], lastScanResult=null;
var PRODUCT_PRICES={donut:4.5,croissant:5.5,bread_coconut:4.0,bread_roll:3.5,chiffon:6.0,croissant_chocolate:6.5};
var COFFEE_PRICES={latte:8.5,americano:6.5,cappuccino:9.0,cold_brew:10.0,espresso:5.5,flat_white:9.5,mocha:10.5,iced_americano:7.2};
var DISCOUNT_RATE=0.3;

async function doLogin(){
var u=document.getElementById('username').value.trim();
var p=document.getElementById('password').value;
if(!u||!p){document.getElementById('error-msg').textContent='Enter username and password';return}
document.getElementById('signin-btn').disabled=true;
document.getElementById('signin-btn').textContent='Signing in...';
try{
var r=await fetch(API+'/s4/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:u,password:p})});
if(!r.ok){var txt=await r.text();throw new Error(txt||'Invalid credentials')}
var d=await r.json();
token=d.access_token;role=d.role;username=d.username;
document.getElementById('login-page').classList.add('hidden');
document.getElementById('dashboard').classList.remove('hidden');
document.getElementById('user-display').textContent=username;
var badge=document.getElementById('role-badge');
badge.textContent=role==='manager'?'Manager':'Staff';
badge.className='role-badge '+(role==='manager'?'role-manager':'role-staff');
var mb=document.querySelectorAll('.manager-only');
for(var i=0;i<mb.length;i++)mb[i].style.display=role==='manager'?'':'none';
showPanel('pos');
}catch(e){
document.getElementById('error-msg').textContent='Error: '+e.message;
document.getElementById('signin-btn').disabled=false;
document.getElementById('signin-btn').textContent='Sign In';
}
}
function doLogout(){
token='';role='';username='';cartItems=[];detections=[];hitlLog=[];bundleRecs=[];
document.getElementById('dashboard').classList.add('hidden');
document.getElementById('login-page').classList.remove('hidden');
document.getElementById('signin-btn').disabled=false;
document.getElementById('signin-btn').textContent='Sign In';
}
function showPanel(panel){
currentPanel=panel;
var btns=document.querySelectorAll('.sidebar nav button');
for(var i=0;i<btns.length;i++)btns[i].classList.remove('active');
var btn=document.querySelector('[data-panel='+panel+']');
if(btn)btn.classList.add('active');
var titles={pos:'POS Checkout',forecast:'Sales Forecast',schedule:'Shift Schedule',inventory:'Inventory',agent:'Agent Query'};
document.getElementById('panel-title').textContent=titles[panel]||panel;
var ca=document.getElementById('content-area');
var panels={pos:renderPOS,forecast:renderForecast,schedule:renderSchedule,inventory:renderInventory,agent:renderAgent};
if(panels[panel])panels[panel](ca);
}
async function api(url,opts){
opts=opts||{};
var headers={'Authorization':'Bearer '+token};
if(opts.body&&!opts.bodyIsForm){headers['Content-Type']='application/json';opts.body=JSON.stringify(opts.body);}
var r=await fetch(API+url,{method:opts.method||'GET',headers:headers,body:opts.body});
if(!r.ok){var txt=await r.text();throw new Error(txt||'Request failed');}
return r.json();
}
function getItemPrice(item){
var base=PRODUCT_PRICES[item.product_name]||COFFEE_PRICES[item.product_name]||5.0;
return item.tray_color==='orange'?base*(1-DISCOUNT_RATE):base;
}
</script>
