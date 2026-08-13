(function(global){
"use strict";

var stream=null;
var capturedBlob=null;
var capturedUrl="";
var deviceIds=[];
var deviceIndex=0;
var submitImage=null;
var translate=function(value){return value;};
var sessionId=0;
var streamRequestId=0;
var captureRequestId=0;
var permissionDenied=false;
var previousFocus=null;

function byId(id){return document.getElementById(id);}

function setError(message){
var element=byId("camera-error");
if(element)element.textContent=message||"";
}

function stopTracks(mediaStream){
if(!mediaStream)return;
mediaStream.getTracks().forEach(function(track){track.stop();});
}

function stopStream(){
if(!stream)return;
stream.getTracks().forEach(function(track){track.stop();});
stream=null;
var video=byId("camera-video");
if(video)video.srcObject=null;
}

function clearCapture(){
captureRequestId+=1;
capturedBlob=null;
if(capturedUrl){URL.revokeObjectURL(capturedUrl);capturedUrl="";}
var photo=byId("camera-photo");
if(photo){photo.removeAttribute("src");photo.hidden=true;}
var video=byId("camera-video");
if(video)video.hidden=false;
var canvas=byId("camera-canvas");
if(canvas){
var context=canvas.getContext("2d");
if(context&&canvas.width&&canvas.height)context.clearRect(0,0,canvas.width,canvas.height);
canvas.width=0;
canvas.height=0;
}
}

function showCaptureControls(captured){
var captureButton=byId("camera-capture");
var retakeButton=byId("camera-retake");
var recognizeButton=byId("camera-recognize");
if(captureButton)captureButton.hidden=captured;
if(retakeButton)retakeButton.hidden=!captured;
if(recognizeButton)recognizeButton.hidden=!captured;
}

function cameraMessage(error){
if(!error)return translate("Camera unavailable");
if(error.name==="NotAllowedError")return translate("Camera permission denied");
if(error.name==="NotFoundError")return translate("No camera found");
if(error.name==="NotReadableError")return translate("Camera is already in use");
return translate("Camera unavailable");
}

async function refreshDevices(){
var devices=await navigator.mediaDevices.enumerateDevices();
deviceIds=devices.filter(function(device){return device.kind==="videoinput";}).map(function(device){return device.deviceId;}).filter(Boolean);
var switchButton=byId("camera-switch");
if(switchButton)switchButton.hidden=deviceIds.length<2;
}

async function startStream(deviceId,currentSession){
var currentRequest=++streamRequestId;
stopStream();
var videoConstraints=deviceId?{
deviceId:{exact:deviceId},width:{ideal:1280},height:{ideal:720}
}:{
facingMode:{ideal:"environment"},width:{ideal:1280},height:{ideal:720}
};
var nextStream;
try{
nextStream=await navigator.mediaDevices.getUserMedia({video:videoConstraints,audio:false});
}catch(error){
if(currentSession!==sessionId||currentRequest!==streamRequestId)return false;
throw error;
}
if(currentSession!==sessionId||currentRequest!==streamRequestId){stopTracks(nextStream);return false;}
stream=nextStream;
var video=byId("camera-video");
if(!video){stopStream();return;}
video.srcObject=stream;
try{
await video.play();
}catch(error){
if(currentSession!==sessionId||currentRequest!==streamRequestId){
if(stream===nextStream){stopTracks(nextStream);stream=null;if(video.srcObject===nextStream)video.srcObject=null;}
return false;
}
throw error;
}
if(currentSession!==sessionId||currentRequest!==streamRequestId){
if(stream===nextStream){stopTracks(nextStream);stream=null;if(video.srcObject===nextStream)video.srcObject=null;}
return false;
}
try{
await refreshDevices();
}catch(error){
if(currentSession!==sessionId||currentRequest!==streamRequestId)return false;
throw error;
}
return true;
}

async function open(options){
submitImage=options&&options.submitImage;
translate=options&&options.translate?options.translate:function(value){return value;};
if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia){
alert(translate("Camera is not supported by this browser"));
return;
}
sessionId+=1;
var currentSession=sessionId;
deviceIndex=0;
clearCapture();
showCaptureControls(false);
setError("");
var modal=byId("camera-modal");
var wasOpen=modal&&modal.classList.contains("show");
if(!wasOpen)previousFocus=document.activeElement;
if(modal)modal.classList.add("show");
var closeButton=byId("camera-close");
if(closeButton)closeButton.focus();
if(permissionDenied){setError(translate("Camera permission denied"));return;}
try{
await startStream("",currentSession);
}catch(error){
if(currentSession!==sessionId)return;
stopStream();
if(error.name==="NotAllowedError")permissionDenied=true;
setError(cameraMessage(error));
}
}

function capture(){
var video=byId("camera-video");
if(!stream||!video||!video.videoWidth||!video.videoHeight){setError(translate("Camera frame is not ready"));return;}
var currentSession=sessionId;
var currentCapture=++captureRequestId;
var canvas=byId("camera-canvas");
canvas.width=video.videoWidth;
canvas.height=video.videoHeight;
canvas.getContext("2d").drawImage(video,0,0,canvas.width,canvas.height);
canvas.toBlob(function(blob){
if(currentSession!==sessionId||currentCapture!==captureRequestId)return;
if(!blob){setError(translate("Camera capture failed"));return;}
capturedBlob=blob;
if(capturedUrl)URL.revokeObjectURL(capturedUrl);
capturedUrl=URL.createObjectURL(blob);
var photo=byId("camera-photo");
photo.src=capturedUrl;
photo.hidden=false;
video.hidden=true;
showCaptureControls(true);
setError("");
},"image/jpeg",0.9);
}

function retake(){
clearCapture();
showCaptureControls(false);
setError("");
}

async function recognize(){
if(!capturedBlob||typeof submitImage!=="function")return;
var imageFile=new File([capturedBlob],"camera-capture.jpg",{type:"image/jpeg"});
close();
await submitImage(imageFile);
}

async function switchCamera(){
if(deviceIds.length<2)return;
deviceIndex=(deviceIndex+1)%deviceIds.length;
var currentSession=sessionId;
clearCapture();
showCaptureControls(false);
setError("");
try{
await startStream(deviceIds[deviceIndex],currentSession);
}catch(error){
if(currentSession!==sessionId)return;
stopStream();
if(error.name==="NotAllowedError")permissionDenied=true;
setError(cameraMessage(error));
}
}

function close(){
sessionId+=1;
streamRequestId+=1;
stopStream();
clearCapture();
showCaptureControls(false);
setError("");
var modal=byId("camera-modal");
if(modal)modal.classList.remove("show");
if(previousFocus&&typeof previousFocus.focus==="function")previousFocus.focus();
previousFocus=null;
}

document.addEventListener("keydown",function(event){
var modal=byId("camera-modal");
if(!modal||!modal.classList.contains("show")||event.key!=="Tab")return;
var controls=Array.prototype.filter.call(modal.querySelectorAll("button:not([hidden])"),function(button){return !button.disabled;});
if(!controls.length)return;
var first=controls[0];
var last=controls[controls.length-1];
if(event.shiftKey&&document.activeElement===first){event.preventDefault();last.focus();}
else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus();}
});
global.addEventListener("beforeunload",close);

global.POSCamera={open:open,close:close,capture:capture,retake:retake,recognize:recognize,switchCamera:switchCamera};
})(window);
