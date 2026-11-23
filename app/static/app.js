"use strict";
const $ = id => document.getElementById(id);
const canvas = $("reality"), ctx = canvas.getContext("2d");
const state = {chunk:1,nfe:1,playing:false,busy:false,pending:false,frame:0,images:[],data:null,
  goal:null,session:crypto.randomUUID(),episodes:[],lastTick:0,revision:0,cameraFrames:[],fps:10,styles:["reality"]};
const cameraNames=["FRONT LEFT","FRONT","FRONT RIGHT","REAR LEFT","REAR","REAR RIGHT"];
const cameraImages=cameraNames.map((name,index)=>{
  const card=document.createElement("div");card.className="camera-card";
  const img=document.createElement("img");img.hidden=true;img.alt=name.toLowerCase()+" surround camera";
  const label=document.createElement("span");label.className="camera-label";label.textContent=name;
  const num=document.createElement("span");num.className="camera-index";num.textContent=`CAM ${String(index+1).padStart(2,"0")}`;
  card.append(img,label,num);$("camera-grid").append(card);return img;
});
function notice(message,error=false){$("status").textContent=message;$("status").classList.toggle("error",error);}
function setPlaying(value){if(value&&!state.images.length)return;state.playing=value;$("play").textContent=value?"Ⅱ":"▶";$("play").setAttribute("aria-label",value?"Pause":"Play");state.lastTick=0;}
function maxChunks(){const ep=state.episodes.find(e=>e.id===$("episode").value);return Math.floor(((ep?.frames??180)-8)/8);}
function reset(){setPlaying(false);state.chunk=1;state.frame=0;state.session=crypto.randomUUID();state.revision++;renderChunk();}
function changeConfig(){reset();}
function loadImage(src){return new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(new Error("A frame could not be loaded."));img.src=src;});}
function format(value,digits=1){return value==null||!Number.isFinite(value)?"—":value.toFixed(digits);}
function updateTelemetry(data){
  const t=data.telemetry;
  state.fps=data.fps??36;
  $("cadence-target").textContent=`${state.fps.toFixed(1)} fps recorded cadence`;
  const front=data.target_view==="native_front";
  document.querySelector(".view-caption").textContent=front?"NATIVE FRONT CAMERA TARGET":"ELEVATED CHASE CAMERA";
  document.querySelector(".camera-symbol").textContent=front?"CAMERA REFERENCE":"75° FOV";
  $("clear-goal").disabled=front;canvas.setAttribute("aria-label",front?"Native front-camera reconstruction; tap-to-drive unavailable.":"Elevated chase output; click a road point to set a destination.");
  cameraImages.forEach((img,i)=>{const label=img.parentElement.querySelector(".camera-label");label.textContent=(data.camera_names?.[i]??cameraNames[i]).toUpperCase()+(data.camera_available?.[i]===false?" / UNAVAILABLE":"");});
  for(const [id,key] of [["dit-ms","dit_ms"],["decode-ms","decode_ms"],["policy-ms","policy_ms"],["effective-fps","effective_fps"],["psnr","psnr_db"],["sram","sram_mib"]])$(id).textContent=format(t[key]);
  const stages=t.uve_stage_frames.slice(0,4);$("stage-frames").textContent=stages.length?stages.join(" / "):"— / — / — / —";
  $("stream-mode").textContent=t.stream_mode;
  $("telemetry-note").textContent=data.trained?"Local inference timings · model compute only":"Measured values appear with trained inference";
  $("view-label").textContent=data.source.toUpperCase();
  $("runtime-state").textContent=data.trained?`EMA CHECKPOINT / STEP ${data.checkpoint_step}`:data.source.toUpperCase();
  $("episode-label").textContent=$("episode").value.toUpperCase().replaceAll("_"," ");
  $("lora-size").textContent=format(t.lora_size_mb,2);$("lora-merge").textContent=format(t.lora_merge_ms,2);
  if(data.goal_xy){const [x,y]=data.goal_xy;$("goal-readout").textContent=`${x.toFixed(1)} m forward · ${Math.abs(y).toFixed(1)} m ${y>=0?"left":"right"}`;$("goal-label").textContent="Destination set · ego ground plane";}
  else{$("goal-readout").textContent=front?"Tap-to-drive unavailable for front-view baseline":"No destination selected";$("goal-label").textContent=front?"Front-camera baseline · not a novel chase view":"Tap the road to set a goal";}
}
function draw(){
  const image=state.images[state.frame];
  if(!image){ctx.fillStyle="#111b25";ctx.fillRect(0,0,canvas.width,canvas.height);ctx.fillStyle="#a1b5c8";ctx.font="24px system-ui";ctx.textAlign="center";ctx.fillText("Load real PandaSet recordings to begin",canvas.width/2,canvas.height/2);ctx.textAlign="left";return;}
  const w=canvas.width,h=canvas.height;ctx.clearRect(0,0,w,h);ctx.drawImage(image,0,0,w,h);
  if(state.cameraFrames[state.frame])state.cameraFrames[state.frame].forEach((img,i)=>{cameraImages[i].hidden=false;if(cameraImages[i].src!==img.src)cameraImages[i].src=img.src;});
  const shade=ctx.createLinearGradient(0,0,0,h);shade.addColorStop(0,"#08131e66");shade.addColorStop(.25,"#08131e00");shade.addColorStop(.7,"#08131e00");shade.addColorStop(1,"#08131e88");ctx.fillStyle=shade;ctx.fillRect(0,0,w,h);
  const points=state.data?.projected_waypoints??[];
  ctx.save();ctx.scale(w/320,h/192);ctx.lineWidth=1.3;const routeColor={reality:"#8acbff",comic:"#ffcf72",cyberpunk:"#ff78da",hybrid:"#7ce9d0"}[state.data?.telemetry?.active_style??"reality"];ctx.strokeStyle=routeColor;ctx.fillStyle=routeColor;ctx.shadowColor=routeColor;ctx.shadowBlur=6;
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
  const ms=Math.round(frame/state.fps*1000);$("time-label").textContent=`00:${String(Math.floor(ms/1000)).padStart(2,"0")}.${String(ms%1000).padStart(3,"0")}`;
  $("progress").style.width=`${Math.min(100,(state.chunk-1+state.frame/8)/maxChunks()*100)}%`;
}
async function renderChunk(){
  if(!$("episode").value){notice("No recordings loaded. Import a PandaSet cache to begin.");return;}
  if(state.busy){state.pending=true;return;}
  state.busy=true;state.pending=false;const revision=state.revision;
  canvas.setAttribute("aria-busy","true");
  try{
    const source=$("source").value;
    const response=await fetch("/api/render",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
      episode:$("episode").value,chunk:state.chunk,nfe:state.nfe,
      source,rollout:$("rollout").value,interleaving:$("interleaving").checked,tiled:$("tiled").checked,
      style:$("style").value,style_strength:Number($("style-strength").value),closed_loop:$("style-closed-loop").checked,
      goal:state.goal,session_id:state.session,seed:42
    })});
    const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==="string"?data.detail:"The render request was rejected.");
    const images=await Promise.all(data.frames.map(loadImage));
    const cameraFrames=await Promise.all((data.camera_frames??[data.cameras]).map(row=>Promise.all(row.map(loadImage))));
    if(revision!==state.revision)return;
    state.images=images;state.cameraFrames=cameraFrames;state.data=data;state.frame=0;state.lastTick=0;
    data.cameras.forEach((src,i)=>cameraImages[i].src=src);updateTelemetry(data);draw();
    notice(data.trained?`UnrealityView · ${data.telemetry.active_style} · checkpoint step ${data.checkpoint_step} · ${data.training_data_source}. Style affects video; the planner is isolated.`:`Reference cache: ${data.dataset_source}. This is recorded data, not a generated result.`);
  }catch(error){if(revision===state.revision){setPlaying(false);notice(error.message,true);}}
  finally{state.busy=false;canvas.setAttribute("aria-busy","false");if(state.pending){state.pending=false;renderChunk();}}
}
function nextChunk(){if(!state.episodes.length)return;if(state.busy)return;if(state.chunk>=maxChunks()){setPlaying(false);notice("End of episode. Reset playback or select the next episode.");return;}state.chunk++;state.frame=0;renderChunk();}
function nextEpisode(){if(!state.episodes.length)return;const options=Array.from($("episode").options);const index=options.findIndex(o=>o.value===$("episode").value);$("episode").value=options[(index+1)%options.length].value;selectEpisode();}
function selectEpisode(){$("source").value="ground_truth";Array.from($("source").options).forEach(o=>{o.disabled=o.value==="model"&&!state.checkpoint;});$("style").value="reality";updateStyleControls();state.goal=null;reset();}
function animate(timestamp){if(state.playing&&!state.busy&&state.images.length){if(!state.lastTick)state.lastTick=timestamp;const elapsed=timestamp-state.lastTick;if(elapsed>=1000/state.fps){const advance=Math.floor(elapsed/(1000/state.fps));state.lastTick+=advance*1000/state.fps;state.frame+=advance;if(state.frame>=state.images.length){state.frame=state.images.length-1;draw();nextChunk();}else draw();}}requestAnimationFrame(animate);}
$("play").addEventListener("click",()=>{if(state.images.length)setPlaying(!state.playing);});$("next-chunk").addEventListener("click",nextChunk);$("next-episode").addEventListener("click",nextEpisode);$("reset").addEventListener("click",reset);
$("episode").addEventListener("change",selectEpisode);
for(const id of ["source","rollout","interleaving","tiled"])$(id).addEventListener("change",()=>{if(id==="rollout")$("style-closed-loop").checked=$("rollout").value==="closed_loop";if(id==="source"){$("style").value="reality";updateStyleControls();}changeConfig();});
for(const button of document.querySelectorAll("[data-nfe]"))button.addEventListener("click",()=>{state.nfe=Number(button.dataset.nfe);for(const other of document.querySelectorAll("[data-nfe]")){const selected=other===button;other.classList.toggle("selected",selected);other.setAttribute("aria-pressed",String(selected));}changeConfig();});
canvas.addEventListener("click",event=>{if(!state.data)return;if(state.data?.target_view==="native_front"){notice("Tap-to-drive is unavailable for the native-front baseline; the chase-camera raycast does not apply to this view.");return;}if($("source").value==="ground_truth"){notice("Select Mean-Flow prediction to condition the planner.");return;}const box=canvas.getBoundingClientRect();state.goal={u:(event.clientX-box.left)/box.width*320,v:(event.clientY-box.top)/box.height*192};reset();});
$("clear-goal").addEventListener("click",()=>{state.goal=null;reset();});
$("fullscreen").addEventListener("click",async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await $("canvas-wrap").requestFullscreen();}catch{notice("Fullscreen is unavailable in this browser.");}});
$("help").addEventListener("click",()=>$("help-dialog").showModal());$("close-help").addEventListener("click",()=>$("help-dialog").close());
document.addEventListener("keydown",event=>{if(["INPUT","SELECT","BUTTON","TEXTAREA"].includes(document.activeElement?.tagName)||$("help-dialog").open)return;if(event.code==="Space"){event.preventDefault();setPlaying(!state.playing);}if(event.key.toLowerCase()==="n")nextChunk();if(event.key.toLowerCase()==="r")reset();});
async function boot(){try{const [healthResponse,episodesResponse]=await Promise.all([fetch("/api/health"),fetch("/api/episodes")]);if(!healthResponse.ok||!episodesResponse.ok)throw new Error("Local runtime unavailable.");const health=await healthResponse.json(),data=await episodesResponse.json();state.styles=health.styles_available??["reality"];state.checkpoint=health.checkpoint_available;state.episodes=data.episodes;$("device").textContent=health.device.toUpperCase()+" RUNTIME";for(const ep of state.episodes){const option=document.createElement("option");option.value=ep.id;option.textContent=ep.id.replaceAll("_"," ")+` · ${ep.frames} frames`;$("episode").append(option);}if(state.episodes.length)selectEpisode();else{notice("No recordings loaded. Import a PandaSet cache to begin.");draw();for(const id of ["play","next-chunk","next-episode","reset","source","episode"])$(id).disabled=true;updateStyleControls();}requestAnimationFrame(animate);}catch(error){notice(error.message,true);}}
function updateStyleControls(){const enabled=$("source").value==="model"&&state.checkpoint&&state.episodes.length;$("style").disabled=!enabled;for(const option of $("style").options)option.disabled=!state.styles.includes(option.value);$("style-strength").disabled=!enabled||$("style").value==="reality";$("style-closed-loop").disabled=!state.episodes.length;}
let strengthTimer;
function refreshStyle(){updateStyleControls();$("strength-value").textContent=Number($("style-strength").value).toFixed(2);state.revision++;renderChunk();}
$("style").addEventListener("change",refreshStyle);
$("style-strength").addEventListener("input",()=>{$("strength-value").textContent=Number($("style-strength").value).toFixed(2);clearTimeout(strengthTimer);strengthTimer=setTimeout(refreshStyle,150);});
$("style-closed-loop").addEventListener("change",()=>{$("rollout").value=$("style-closed-loop").checked?"closed_loop":"teacher_forced";reset();});
boot();
