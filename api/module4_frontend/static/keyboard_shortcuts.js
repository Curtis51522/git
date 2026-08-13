(function(global){
"use strict";

var ENTER_CONTEXTS=[
{name:"login",selector:'[data-keyboard-context="login"]'},
{name:"cart-edit",selector:'[data-keyboard-context="cart-edit"]',layer:true},
{name:"manager-attendance",selector:'[data-keyboard-context="manager-attendance"]',layer:true},
{name:"attendance-correction",selector:'[data-keyboard-context="attendance-correction"]',layer:true},
{name:"beverage-customization",selector:'[data-keyboard-context="beverage-customization"]',layer:true},
{name:"business-event",selector:'[data-keyboard-context="business-event"]',layer:true},
{name:"payment",selector:'[data-keyboard-context="payment"]',layer:true},
{name:"staff-attendance",selector:'[data-keyboard-context="staff-attendance"]'},
{name:"raw-material-restock",selector:'[data-keyboard-context="raw-material-restock"]'}
];

var ESCAPE_LAYERS=[
{id:"camera-modal",close:function(){if(global.POSCamera)global.POSCamera.close();}},
{id:"edit-modal",close:function(){callGlobal("closeEditModal");}},
{id:"attendance-pin-modal",close:function(){callGlobal("closeManagerAttendancePin");}},
{id:"attendance-correction-modal",close:function(){callGlobal("closeAttendanceCorrection");}},
{id:"coffee-modal",close:function(){callGlobal("closeCoffeeModal");}},
{id:"swap-modal",close:function(){removeShow("swap-modal");}},
{id:"business-event-modal",close:function(){callGlobal("closeBusinessEventModal");}},
{id:"payment-modal",close:function(){callGlobal("closePaymentModal");}},
{id:"receipt-overlay",close:function(){callGlobal("closeReceipt");}},
{id:"receipt-modal",present:true,close:function(){removeElement("receipt-modal");}}
];

function callGlobal(name){if(typeof global[name]==="function")global[name]();}
function removeShow(id){var element=document.getElementById(id);if(element)element.classList.remove("show");}
function removeElement(id){var element=document.getElementById(id);if(element&&typeof element.remove==="function")element.remove();}

function isVisible(element,present){
if(!element)return false;
if(present)return true;
if(element.hidden||element.classList.contains("hidden"))return false;
return element.classList.contains("show");
}

function zIndex(element){
if(!global.getComputedStyle)return 0;
var value=parseInt(global.getComputedStyle(element).zIndex,10);
return Number.isFinite(value)?value:0;
}

function laterInDocument(left,right){
if(!left||!right||typeof left.compareDocumentPosition!=="function"||!global.Node)return false;
return Boolean(left.compareDocumentPosition(right)&global.Node.DOCUMENT_POSITION_FOLLOWING);
}

function topVisible(items,resolve){
var top=null;
for(var index=0;index<items.length;index+=1){
var candidate=resolve(items[index]);
if(!candidate||!candidate.element)continue;
if(!top||candidate.z>top.z||(candidate.z===top.z&&laterInDocument(top.element,candidate.element)))top=candidate;
}
return top;
}

function topEscapeLayer(){
return topVisible(ESCAPE_LAYERS,function(layer){
var element=document.getElementById(layer.id);
if(!isVisible(element,layer.present))return null;
return {element:element,z:zIndex(element),close:layer.close};
});
}

function visibleEnterLayer(){
return topVisible(ENTER_CONTEXTS,function(context){
if(!context.layer)return null;
var element=document.querySelector(context.selector);
if(!isVisible(element,false))return null;
return {element:element,z:zIndex(element),context:context};
});
}

function closestEnterContext(target){
if(!target||typeof target.closest!=="function")return null;
var element=target.closest("[data-keyboard-context]");
if(!element)return null;
for(var index=0;index<ENTER_CONTEXTS.length;index+=1){
if(element.matches(ENTER_CONTEXTS[index].selector))return {element:element,context:ENTER_CONTEXTS[index]};
}
return null;
}

function hasModifier(event){return event.ctrlKey||event.altKey||event.metaKey||event.shiftKey;}

function isEditableException(target,contextElement){
if(!target)return true;
if(target.isContentEditable)return true;
var tag=String(target.tagName||"").toUpperCase();
if(tag==="TEXTAREA"||tag==="A")return true;
if(contextElement&&typeof contextElement.contains==="function"&&contextElement.contains(target)&&(tag==="BUTTON"||tag==="SELECT"))return true;
return false;
}

function primaryButton(contextElement){
if(!contextElement||typeof contextElement.querySelector!=="function")return null;
var button=contextElement.querySelector('[data-keyboard-primary="true"]');
if(!button||button.disabled||button.hidden||button.classList.contains("hidden"))return null;
return button;
}

function handleKeydown(event){
if(!event||event.isComposing||event.repeat||hasModifier(event))return;
if(event.key==="Escape"){
var layer=topEscapeLayer();
if(!layer)return;
event.preventDefault();
layer.close();
return;
}
if(event.key!=="Enter")return;
var active=visibleEnterLayer();
if(!active)active=closestEnterContext(event.target);
if(!active||isEditableException(event.target,active.element))return;
var button=primaryButton(active.element);
if(!button)return;
event.preventDefault();
button.click();
}

document.addEventListener("keydown",handleKeydown);
global.KeyboardShortcuts={handleKeydown:handleKeydown,topEscapeLayer:topEscapeLayer};
})(window);
