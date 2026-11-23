"""Check real loopback HTTP endpoints; run while the cockpit server is active."""
import argparse
import json
from uuid import uuid4
import httpx

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--episode",required=True)
p.add_argument("--model",action="store_true")
a=p.parse_args()
with httpx.Client(base_url="http://127.0.0.1:8088",timeout=120,trust_env=False) as client:
    assert client.get("/").status_code==200
    health=client.get("/api/health");health.raise_for_status()
    payload={"session_id":str(uuid4()),"source":"model" if a.model else "ground_truth","episode":a.episode}
    response=client.post("/api/render",json=payload);response.raise_for_status()
    result=response.json()
    assert len(result["frames"])==8 and len(result["cameras"])==6
    assert len(result["camera_frames"])==8
    assert client.post("/api/render",json={**payload,"nfe":3}).status_code==422
    if a.model:
        assert result["trained"] and result["telemetry"]["dit_ms"]>0
        tapped=client.post("/api/render",json={**payload,"goal":{"u":180.,"v":110.}})
        if result["target_view"]=="native_front":assert tapped.status_code==422
        else:
            tapped.raise_for_status()
            assert tapped.json()["goal_xy"] is not None
            change=max(abs(x-y) for p,q in zip(result["waypoints"],tapped.json()["waypoints"]) for x,y in zip(p,q))
            assert change>0,"Tap goal must change learned planner output"
    print(json.dumps({"status":"LIVE_HTTP_VERIFIED","health":health.json(),"source":result["source"],
                      "training_data_source":result.get("training_data_source"),"telemetry":result["telemetry"]},indent=2))
