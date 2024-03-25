import cv2
import utils as ht
import mediapipe as mp
import pygame

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
#handTracking = ht.handDetector()

pygame.mixer.init()

correct_sound = pygame.mixer.Sound('Audio/correct.mp3')
wrong_sound = pygame.mixer.Sound('Audio/wrong.mp3')

drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands
hand_obj = hands.Hands(max_num_hands=1)

bg = cv2.imread("Resources/bg_resize.png")
monster1 = cv2.imread("Resources/monster_test.png", cv2.IMREAD_UNCHANGED)
monster2 = cv2.imread("Resources/green_monster.png", cv2.IMREAD_UNCHANGED)
deadmonster1 = cv2.imread("Resources/monster_test.png", cv2.IMREAD_UNCHANGED)
deadmonster2 = cv2.imread("Resources/green_dead.png", cv2.IMREAD_UNCHANGED)

def count_fingers(lst):
    cnt = 0

    thresh = (lst.landmark[0].y*100 - lst.landmark[9].y*100)/2

    if (lst.landmark[5].y*100 - lst.landmark[8].y*100) > thresh:
        cnt += 1

    if (lst.landmark[9].y*100 - lst.landmark[12].y*100) > thresh:
        cnt += 1

    if (lst.landmark[13].y*100 - lst.landmark[16].y*100) > thresh:
        cnt += 1

    if (lst.landmark[17].y*100 - lst.landmark[20].y*100) > thresh:
        cnt += 1

    return cnt

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.flip(frame, 1)

        res = hand_obj.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        monster_visible = True  # Assume the monster is visible by default
        monster_dead = False
        text_wrong = False
        text_correct = False

        if res.multi_hand_landmarks:
            hand_keyPoints = res.multi_hand_landmarks[0]

            count = count_fingers(hand_keyPoints)

            # If one finger is up, set monster_visible to False
            if count == 2:
                monster_visible = False
                monster_dead = True
                text_correct = True
                #correct_sound.play()

            elif count == 1 or count == 3 or count:
                # If not one finger, consider it wrong and display "Wrong"
                monster_visible = True
                text_wrong = True
                #wrong_sound.play()

            drawing.draw_landmarks(frame, hand_keyPoints, hands.HAND_CONNECTIONS)

        # Apply background blending
        frame = cv2.addWeighted(frame, 0.3, bg, 0.7, 0)

        # Only overlay the monster image if monster_visible is True
        if monster_visible:
            frame = ht.overlayPNG(frame, monster2, (640, 720 - 350))
        if monster_dead:
            frame = ht.overlayPNG(frame, deadmonster2, (640, 720 - 400))
        if text_wrong:
            cv2.putText(frame, "Wrong", (500, 380), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA)
        if text_correct:
            cv2.putText(frame, "Correct", (500, 380), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA)


        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

