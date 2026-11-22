"""Labelled procedural preview, available without datasets or pretrained weights."""
import math
from PIL import Image,ImageDraw
from fsd_reality.geometry import project_waypoints


def demo_frame(frame=0, camera=6):
    w,h=320,192
    image=Image.new("RGB",(w,h),"#8ba8ba")
    draw=ImageDraw.Draw(image)
    horizon=55 if camera==6 else 80
    draw.rectangle((0,horizon,w,h),fill="#566452")
    bend=math.sin(frame/160)*14
    road=[(w/2-11+bend,horizon),(w/2+11+bend,horizon),(310,h),(10,h)]
    draw.polygon(road,fill="#323b41")
    draw.line((w/2-11+bend,horizon,10,h),fill="#d0d6cc",width=2)
    draw.line((w/2+11+bend,horizon,310,h),fill="#d0d6cc",width=2)
    for lane in (-1,1):
        for j in range(10):
            z=((j/10+frame/160)%1)**2
            y=horizon+z*(h-horizon)
            x=w/2+lane*(4+z*48)+bend*(1-z)
            length=2+z*12
            draw.line((x,y,x+lane*length*.35,y+length),fill="#b6bcbb",width=max(1,int(1+z*2)))
    for i in range(8):
        z=((i/8+frame/500)%1)**2
        y=horizon+z*(h-horizon)
        radius=3+z*17
        for side in (-1,1):
            x=w/2+side*(22+z*160)
            draw.rectangle((x-2,y-radius/2,x+2,y+radius),fill="#53604e")
            draw.ellipse((x-radius,y-radius*2,x+radius,y),fill="#3b5548")
    for offset,color in ((-35,"#9bafba"),(30,"#738896")):
        y=95+math.sin(frame/70+offset)*5
        x=w/2+offset+math.sin(frame/100)*8
        draw.rounded_rectangle((x-9,y-16,x+9,y+12),radius=4,fill=color,outline="#24313a",width=1)
        draw.rectangle((x-6,y-11,x+6,y-4),fill="#334650")
    if camera==6:
        draw.ellipse((140,141,180,178),fill="#28323a")
        draw.rounded_rectangle((143,137,177,178),radius=7,fill="#b8c1c8",outline="#71808c",width=2)
        draw.polygon([(148,143),(172,143),(174,156),(146,156)],fill="#30404d")
        draw.rectangle((148,172,153,174),fill="#e37e74")
        draw.rectangle((167,172,172,174),fill="#e37e74")
    else:
        if camera in (0,2,3,5):
            image=image.transform((w,h),Image.Transform.AFFINE,(1,0,35 if camera in (0,3) else -35,0,1,0))
        if camera in (3,4,5):
            image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return image


def demo_waypoints(goal=None):
    goal=(35.,0.) if goal is None else goal
    points=[]
    for i in range(1,13):
        t=i/12
        x=max(1.,goal[0])*t
        y=goal[1]*(3*t*t-2*t*t*t)
        yaw=math.atan2(goal[1]*6*t*(1-t),max(1.,goal[0]))
        points.append([x,y,math.sin(yaw),math.cos(yaw)])
    return points
