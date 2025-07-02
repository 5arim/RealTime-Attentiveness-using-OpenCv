import cv2 as cv
import mediapipe as mp
import csv
from datetime import datetime
import time
import math

# Initialize MediaPipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
my_drawing_spec = mp_drawing.DrawingSpec(color=(0, 55, 0), thickness=1, circle_radius=1)

# Initialize CSV file
csv_file = 'attentiveness_scores.csv'
csv_header = ['Name', 'Timestamp', 'Attentiveness_Score_Percent']
try:
    with open(csv_file, 'x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
except FileExistsError:
    pass

# Get user's name
name = input("Please enter your name: ")

# Initialize video capture
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Helper functions
def calculate_ear(eye_landmarks):
    """Calculate Eye Aspect Ratio (EAR) for an eye."""
    top = eye_landmarks[1]    # Top eyelid
    bottom = eye_landmarks[3] # Bottom eyelid
    left = eye_landmarks[0]   # Left corner
    right = eye_landmarks[2]  # Right corner
    
    vertical_dist = abs(top.y - bottom.y)
    horizontal_dist = abs(left.x - right.x)
    
    if horizontal_dist == 0:
        return 0
    return vertical_dist / horizontal_dist

def perpendicular_distance(P1_x, P1_y, P2_x, P2_y, I_x, I_y):
    """Calculate perpendicular distance from point (I_x, I_y) to line (P1_x, P1_y)-(P2_x, P2_y)."""
    vec_P1P2 = (P2_x - P1_x, P2_y - P1_y)
    vec_P1I = (I_x - P1_x, I_y - P1_y)
    cross_product = vec_P1I[0] * vec_P1P2[1] - vec_P1I[1] * vec_P1P2[0]
    distance = abs(cross_product) / math.sqrt(vec_P1P2[0]**2 + vec_P1P2[1]**2 + 1e-6)
    return distance

def is_looking_at_screen(iris_landmarks, eye_landmarks, ear):
    """Check if gaze is toward the screen, considering EAR."""
    if ear < 0.2:  # Eye is closed
        return False, 0, 0
    
    # Calculate iris center
    iris_center_x = sum(lm.x for lm in iris_landmarks) / len(iris_landmarks)
    iris_center_y = sum(lm.y for lm in iris_landmarks) / len(iris_landmarks)
    
    # Get eye corners
    left_corner = eye_landmarks[0]
    right_corner = eye_landmarks[2]
    
    # Calculate eye width
    eye_width = math.sqrt((right_corner.x - left_corner.x)**2 + (right_corner.y - left_corner.y)**2)
    if eye_width == 0:
        return False, 0, 0
    
    # Calculate perpendicular distance
    perp_distance = perpendicular_distance(left_corner.x, left_corner.y, right_corner.x, right_corner.y, iris_center_x, iris_center_y)
    normalized_perp_distance = perp_distance / eye_width
    
    # Calculate horizontal position
    horizontal_pos = (iris_center_x - left_corner.x) / eye_width
    is_horizontal_centered = 0.3 < horizontal_pos < 0.7  # Stricter range
    
    # Stricter threshold
    perp_threshold = 0.1
    is_looking = (normalized_perp_distance < perp_threshold) and is_horizontal_centered
    
    return is_looking, normalized_perp_distance, horizontal_pos

def is_facing_forward(face_landmarks):
    """Check if face is facing forward."""
    nose_tip = face_landmarks.landmark[1]
    left_eye_outer = face_landmarks.landmark[263]
    right_eye_outer = face_landmarks.landmark[33]
    face_width = right_eye_outer.x - left_eye_outer.x
    nose_rel_pos = (nose_tip.x - left_eye_outer.x) / face_width
    return 0.3 < nose_rel_pos < 0.7

# Initialize Face Mesh
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    
    frame_count = 0
    attentive_frames = 0
    total_scored_frames = 0
    score_interval = 15
    max_note_taking_frames = 5400
    note_taking_frames_used = 0
    note_taking_mode = False
    
    start_time = time.time()
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty frame")
            continue
            
        frame_count += 1
        
        # Handle note-taking mode
        key = cv.waitKey(1) & 0xFF
        if key == ord('n'):
            if not note_taking_mode and note_taking_frames_used < max_note_taking_frames:
                note_taking_mode = True
                print("Note-taking mode ON")
            elif note_taking_mode:
                note_taking_mode = False
                print("Note-taking mode OFF")
        
        if note_taking_mode and note_taking_frames_used < max_note_taking_frames:
            note_taking_frames_used += 1
        
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        # Initialize attentiveness variables
        is_attentive = False
        left_looking_at_screen = False
        right_looking_at_screen = False
        facing_forward = False
        depth_attentive = False
        left_distance = 0
        right_distance = 0
        left_horizontal = 0
        right_horizontal = 0
        
        if not note_taking_mode or note_taking_frames_used >= max_note_taking_frames:
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    nose_tip = face_landmarks.landmark[1]
                    z_value = nose_tip.z
                    depth_attentive = abs(z_value) < 0.15
                    
                    # Define eye landmarks
                    left_iris = face_landmarks.landmark[473:478]
                    left_eye_landmarks = [face_landmarks.landmark[362], face_landmarks.landmark[159], 
                                          face_landmarks.landmark[263], face_landmarks.landmark[145]]
                    right_iris = face_landmarks.landmark[468:473]
                    right_eye_landmarks = [face_landmarks.landmark[33], face_landmarks.landmark[386], 
                                           face_landmarks.landmark[133], face_landmarks.landmark[374]]
                    
                    # Calculate EAR
                    left_ear = calculate_ear(left_eye_landmarks)
                    right_ear = calculate_ear(right_eye_landmarks)
                    
                    # Check gaze with EAR
                    left_looking_at_screen, left_distance, left_horizontal = is_looking_at_screen(left_iris, left_eye_landmarks, left_ear)
                    right_looking_at_screen, right_distance, right_horizontal = is_looking_at_screen(right_iris, right_eye_landmarks, right_ear)
                    
                    facing_forward = is_facing_forward(face_landmarks)
                    
                    # Calculate attention score
                    attention_score = 0
                    if left_looking_at_screen: attention_score += 0.25
                    if right_looking_at_screen: attention_score += 0.25
                    if facing_forward: attention_score += 0.3
                    if depth_attentive: attention_score += 0.2
                    is_attentive = attention_score >= 0.7
                    
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=my_drawing_spec
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
        
        if frame_count % score_interval == 0:
            total_scored_frames += 1
            if note_taking_mode and note_taking_frames_used < max_note_taking_frames:
                attentive_frames += 1
            elif is_attentive:
                attentive_frames += 1
        
        # Flip image and draw text
        image = cv.flip(image, 1)
        
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        timer_text = f"Time: {minutes:02d}:{seconds:02d}"
        
        if total_scored_frames > 0:
            current_score = int((attentive_frames / total_scored_frames) * 100)
            cv.putText(image, f'Score: {current_score}%', (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv.putText(image, timer_text, (10, 60), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(image, f'Right Looking at Screen: {left_looking_at_screen}', (10, 90), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(image, f'Left Looking at Screen: {right_looking_at_screen}', (10, 120), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(image, f'Facing Forward: {facing_forward}', (10, 150), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(image, f'Depth Attentive: {depth_attentive}', (10, 180), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(image, f'Attentive: {is_attentive}', (10, 210), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(image, f'Left Perp Distance: {left_distance:.3f}', (10, 300), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(image, f'Right Perp Distance: {right_distance:.3f}', (10, 330), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(image, f'Left Horizontal: {left_horizontal:.3f}', (10, 360), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(image, f'Right Horizontal: {right_horizontal:.3f}', (10, 390), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        remaining_frames = max_note_taking_frames - note_taking_frames_used
        if remaining_frames < 0:
            remaining_frames = 0
        note_text = f"Note-taking frames left: {remaining_frames}"
        if remaining_frames == 0 and note_taking_mode:
            note_text = "Note-taking limit reached"
        cv.putText(image, note_text, (10, 240), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if note_taking_mode and note_taking_frames_used < max_note_taking_frames:
            cv.putText(image, "Note-taking Mode Active", (10, 270), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv.imshow('Capturing Video', image)
        
        if key == ord('q'):
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(int(elapsed_time), 60)
            if total_scored_frames > 0:
                final_score = int((attentive_frames / total_scored_frames) * 100)
            else:
                final_score = 0
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, timestamp, final_score])
            
            print(f"Session ended. Final attentiveness score for {name}: {final_score}%")
            print(f"Session duration: {minutes:02d}:{seconds:02d}")
            break

cap.release()
cv.destroyAllWindows()