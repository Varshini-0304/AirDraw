import cv2
import numpy as np
import cvzone
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)
canvas = None
prev_x, prev_y = 0, 0
colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
draw_color = colors[0]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if canvas is None:
        canvas = np.zeros_like(frame)

    for i, color in enumerate(colors):
        cv2.rectangle(frame,(i*80+10,10),(i*80+70,60),color,-1)
    cv2.rectangle(frame,(330,10),(420,60),(200,200,200),-1)
    cv2.putText(frame,"CLEAR",(335,42),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

    hands, frame = detector.findHands(frame)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        ix, iy = lmList[8][0], lmList[8][1]
        cv2.circle(frame,(ix,iy),10,draw_color,-1)

        if iy < 60:
            for i, color in enumerate(colors):
                if i*80+10 < ix < i*80+70:
                    draw_color = colors[i]
            if 330 < ix < 420:
                canvas = np.zeros_like(frame)
            prev_x, prev_y = 0, 0
        else:
            if prev_x and prev_y:
                cv2.line(canvas,(prev_x,prev_y),(ix,iy),draw_color,5)
            prev_x, prev_y = ix, iy
    else:
        prev_x, prev_y = 0, 0

    output = cv2.addWeighted(frame,0.7,canvas,0.9,0)
    cv2.imshow("Air Canvas Gesture", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()