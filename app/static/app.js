"use strict";
const $ = id => document.getElementById(id);
const canvas = $("reality"), ctx = canvas.getContext("2d");
const state = {chunk:1,nfe:1,playing:false,busy:false,pending:false,frame:0,images:[],data:null,
  goal:null,session:crypto.randomUUID(),episodes:[],lastTick:0,revision:0};
const cameraNames=["FRONT LEFT","FRONT","FRONT RIGHT","REAR LEFT","REAR","REAR RIGHT"];
const cameraImages=cameraNames.map((name,index)=>{
  const card=document.createElement("div");card.className="camera-card";
  const img=document.createElement("img");img.alt=name.toLowerCase()+" surround camera";
  const label=document.createElement("span");label.className="camera-label";label.textContent=name;
  const num=document.createElement("span");num.className="camera-index";num.textContent=`CAM ${String(index+1).padStart(2,"0")}`;
  card.append(img,label,num);$("camera-grid").append(card);return img;
});
function notice(message,error=false){$("status").textContent=message;$("status").classList.toggle("error",error);}
function setPlaying(value){state.playing=value;$("play").textContent=value?"Ⅱ":"▶";$("play").setAttribute("aria-label",value?"Pause":"Play");state.lastTick=0;}
function maxChunks(){const ep=state.episodes.find(e=>e.id===$("episode").value);return Math.floor(((ep?.frames??180)-8)/8);}
function reset(){setPlaying(false);state.chunk=1;state.frame=0;state.session=crypto.randomUUID();state.revision++;renderChunk();}
function changeConfig(){reset();}
function loadImage(src){return new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(new Error("A frame could not be loaded."));img.src=src;});}
function format(value,digits=1){return value==null||!Number.isFinite(value)?"—":value.toFixed(digits);}
function updateTelemetry(data){
  const t=data.telemetry;
  for(const [id,key] of [["dit-ms","dit_ms"],["decode-ms","decode_ms"],["policy-ms","policy_ms"],["effective-fps","effective_fps"],["psnr","psnr_db"],["sram","sram_mib"]])$(id).textContent=format(t[key]);
  const stages=t.uve_stage_frames.slice(0,4);$("stage-frames").textContent=stages.length?stages.join(" / "):"— / — / — / —";
  $("stream-mode").textContent=t.stream_mode;
  $("telemetry-note").textContent=data.trained?"Local inference timings · model compute only":"Measured values appear with trained inference";
  $("view-label").textContent=data.source.toUpperCase();
  $("runtime-state").textContent=data.trained?`EMA CHECKPOINT / STEP ${data.checkpoint_step}`:data.source.toUpperCase();
  $("episode-label").textContent=$("episode").value==="demo"?"PREVIEW / SCENE 01":$("episode").value.toUpperCase().replaceAll("_"," ");
  if(data.goal_xy){const [x,y]=data.goal_xy;$("goal-readout").textContent=`${x.toFixed(1)} m forward · ${Math.abs(y).toFixed(1)} m ${y>=0?"left":"right"}`;$("goal-label").textContent="Destination set · ego ground plane";}
  else{$("goal-readout").textContent="No destination selected";$("goal-label").textContent="Tap the road to set a goal";}
}
function draw(){
  const image=state.images[state.frame];
  if(!image)return;
  const w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);ctx.drawImage(image,0,0,w,h);
  const shade=ctx.createLinearGradient(0,0,0,h);shade.addColorStop(0,"#08131e66");shade.addColorStop(.25,"#08131e00");shade.addColorStop(.7,"#08131e00");shade.addColorStop(1,"#08131e88");ctx.fillStyle=shade;ctx.fillRect(0,0,w,h);
  const points=state.data?.projected_waypoints??[];
  ctx.save();ctx.scale(w/320,h/192);ctx.lineWidth=1.3;ctx.strokeStyle="#8acbff";ctx.fillStyle="#8acbff";ctx.shadowColor="#5b9be6";ctx.shadowBlur=6;
  for(let i=0;i<points.length;i++){
    const [x,y]=points[i];if(x<0||x>320||y<0||y>192)continue;
    const scale=Math.max(.7,2.8-i*.16);
    ctx.globalAlpha=1-i*.055;ctx.beginPath();ctx.moveTo(x-scale,y+scale*.55);ctx.lineTo(x,y-scale*.35);ctx.lineTo(x+scale,y+scale*.55);ctx.stroke();
  }
  if(state.goal&&state.data?.goal_xy){ctx.globalAlpha=1;ctx.strokeStyle="#c5e7ff";ctx.lineWidth=.7;ctx.beginPath();ctx.arc(state.goal.u,state.goal.v,4.5,0,Math.PI*2);ctx.stroke();ctx.beginPath();ctx.arc(state.goal.u,state.goal.v,1,0,Math.PI*2);ctx.fill();}
  ctx.restore();
  const frame=(state.data?.frame_start??state.chunk*8)+state.frame;
  $("frame-label").textContent=`FRAME ${String(frame).padStart(5,"0")}`;
  $("chunk-label").textContent=`CHUNK ${String(state.chunk).padStart(2,"0")}`;
  const ms=Math.round(frame/36*1000);$("time-label").textContent=`00:${String(Math.floor(ms/1000)).padStart(2,"0")}.${String(ms%1000).padStart(3,"0")}`;
  $("progress").style.width=`${Math.min(100,(state.chunk-1+state.frame/8)/maxChunks()*100)}%`;
}
async function renderChunk(){
  if(state.busy){state.pending=true;return;}
  state.busy=true;state.pending=false;const revision=state.revision;
  canvas.setAttribute("aria-busy","true");
  try{
    const source=$("source").value;
    const response=await fetch("/api/render",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
      episode:$("episode").value==="demo"?"val/episode_0050":$("episode").value,chunk:state.chunk,nfe:state.nfe,
      source,rollout:$("rollout").value,interleaving:$("interleaving").checked,tiled:$("tiled").checked,
      goal:state.goal,session_id:state.session,seed:42
    })});
    const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==="string"?data.detail:"The render request was rejected.");
    const images=await Promise.all(data.frames.map(loadImage));
    if(revision!==state.revision)return;
    state.images=images;state.data=data;state.frame=0;state.lastTick=0;
    data.cameras.forEach((src,i)=>cameraImages[i].src=src);updateTelemetry(data);draw();
    notice(data.trained?"Trained Mean-Flow inference. Cached camera arrivals emulate streaming; tap goals condition the independent planner.":source==="demo"?"Procedural preview — illustrative scenery and route, no learned inference. Select a cached episode for ground truth or trained predictions.":"Ground-truth simulator replay. Waypoints come from recorded ego poses; tap goals affect only model predictions.");
  }catch(error){if(revision===state.revision){setPlaying(false);notice(error.message,true);}}
  finally{state.busy=false;canvas.setAttribute("aria-busy","false");if(state.pending){state.pending=false;renderChunk();}}
}
function nextChunk(){if(state.busy)return;if(state.chunk>=maxChunks()){setPlaying(false);notice("End of episode. Reset playback or select the next episode.");return;}state.chunk++;state.frame=0;renderChunk();}
function nextEpisode(){const options=Array.from($("episode").options);const index=options.findIndex(o=>o.value===$("episode").value);$("episode").value=options[(index+1)%options.length].value;selectEpisode();}
function selectEpisode(){const demo=$("episode").value==="demo";$("source").value=demo?"demo":"ground_truth";Array.from($("source").options).forEach(o=>{o.disabled=demo?o.value!=="demo":o.value==="demo"||(o.value==="model"&&!state.checkpoint);});state.goal=null;reset();}
function animate(timestamp){if(state.playing&&!state.busy&&state.images.length){if(!state.lastTick)state.lastTick=timestamp;const elapsed=timestamp-state.lastTick;if(elapsed>=1000/36){const advance=Math.floor(elapsed/(1000/36));state.lastTick+=advance*1000/36;state.frame+=advance;if(state.frame>=state.images.length){state.frame=state.images.length-1;draw();nextChunk();}else draw();}}requestAnimationFrame(animate);}
$("play").addEventListener("click",()=>setPlaying(!state.playing));$("next-chunk").addEventListener("click",nextChunk);$("next-episode").addEventListener("click",nextEpisode);$("reset").addEventListener("click",reset);
$("episode").addEventListener("change",selectEpisode);
for(const id of ["source","rollout","interleaving","tiled"])$(id).addEventListener("change",changeConfig);
for(const button of document.querySelectorAll("[data-nfe]"))button.addEventListener("click",()=>{state.nfe=Number(button.dataset.nfe);for(const other of document.querySelectorAll("[data-nfe]")){const selected=other===button;other.classList.toggle("selected",selected);other.setAttribute("aria-pressed",String(selected));}changeConfig();});
canvas.addEventListener("click",event=>{if($("source").value==="ground_truth"){notice("Select Mean-Flow prediction to condition the planner, or procedural preview to illustrate a route.");return;}const box=canvas.getBoundingClientRect();state.goal={u:(event.clientX-box.left)/box.width*320,v:(event.clientY-box.top)/box.height*192};reset();});
$("clear-goal").addEventListener("click",()=>{state.goal=null;reset();});
$("fullscreen").addEventListener("click",async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await $("canvas-wrap").requestFullscreen();}catch{notice("Fullscreen is unavailable in this browser.");}});
$("help").addEventListener("click",()=>$("help-dialog").showModal());$("close-help").addEventListener("click",()=>$("help-dialog").close());
document.addEventListener("keydown",event=>{if(["INPUT","SELECT","BUTTON","TEXTAREA"].includes(document.activeElement?.tagName)||$("help-dialog").open)return;if(event.code==="Space"){event.preventDefault();setPlaying(!state.playing);}if(event.key.toLowerCase()==="n")nextChunk();if(event.key.toLowerCase()==="r")reset();});
async function boot(){try{const [healthResponse,episodesResponse]=await Promise.all([fetch("/api/health"),fetch("/api/episodes")]);if(!healthResponse.ok||!episodesResponse.ok)throw new Error("Local runtime unavailable.");const health=await healthResponse.json(),data=await episodesResponse.json();state.checkpoint=health.checkpoint_available;state.episodes=data.episodes;$("device").textContent=health.device.toUpperCase()+" RUNTIME";for(const ep of state.episodes){const option=document.createElement("option");option.value=ep.id;option.textContent=ep.id.replaceAll("_"," ")+` · ${ep.frames} frames`;$("episode").append(option);}selectEpisode();requestAnimationFrame(animate);}catch(error){notice(error.message,true);}}
boot();
