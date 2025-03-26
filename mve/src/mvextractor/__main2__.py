import sys
import os
import time
from datetime import datetime
import argparse

#import GPUtil
#import psutil

import numpy as np
import cv2

from sklearn.cluster import DBSCAN
from mvextractor.videocap import VideoCap
from sklearn.neighbors import NearestNeighbors





#Afficher les vecteurs sur un fond noir
def draw_motion_vectors_black(frame, motion_vectors):
    black_frame = np.zeros_like(frame)
    if len(motion_vectors) > 0:
        num_mvs = np.shape(motion_vectors)[0]
        for mv in np.split(motion_vectors, num_mvs):
            start_pt = (mv[0, 3], mv[0, 4])
            end_pt = (mv[0, 5], mv[0, 6])
            #if (np.sqrt(np.abs((end_pt[0]-start_pt[0])^2+(end_pt[1]-start_pt[1])^2))>5):
            cv2.arrowedLine(black_frame, start_pt, end_pt, (70, 255, 255), 1, cv2.LINE_AA, 0, 0.1)
    return black_frame

def draw_motion_vectors(frame, motion_vectors):

    if len(motion_vectors) > 0:
        num_mvs = np.shape(motion_vectors)[0]
        for mv in np.split(motion_vectors, num_mvs):
            start_pt = (mv[0, 3], mv[0, 4])
            end_pt = (mv[0, 5], mv[0, 6])
            #if (np.sqrt(np.abs((end_pt[0]-start_pt[0])^2+(end_pt[1]-start_pt[1])^2))>5):
            cv2.arrowedLine(frame, start_pt, end_pt, (70, 255, 255), 1, cv2.LINE_AA, 0, 0.1)
    return frame

def select_vectors_norm(motion_vectors, value):
    start_pt = motion_vectors[:, [3, 4]]
    end_pt = motion_vectors[:, [5, 6]]
    norm = np.linalg.norm(end_pt - start_pt, axis=1)
    return motion_vectors[norm>value]

def select_vectors_zone(motion_vectors):
    if motion_vectors.shape[0] == 0:
        return motion_vectors
    
    start_pt = motion_vectors[:, [3, 4]]
    end_pt = motion_vectors[:, [5, 6]]

    min_neighbors = 3
    distance_threshold = 32

    #Utiliser NearestNeighbors pour trouver les voisins proches des points finaux
    nbrs = NearestNeighbors(radius=distance_threshold).fit(end_pt)
    distances, indices = nbrs.radius_neighbors(end_pt)

    # Filtrer les vecteurs qui ont au moins `min_neighbors` voisins proches pour leurs points finaux
    mask = [len(neighbors) > min_neighbors for neighbors in indices]

    return motion_vectors[mask]

#Crop to zoom on vector zone
def crop_frame(frame,motion_vectors):
    if motion_vectors.shape[0] == 0 :
        return frame
    start_x = motion_vectors[:, 3]
    start_y = motion_vectors[:, 4]
    end_x = motion_vectors[:, 5]
    end_y = motion_vectors[:, 6]
    
    # Trouver les coordonnées minimales et maximales
    min_x = int(min(np.min(start_x), np.min(end_x)))
    max_x = int(max(np.max(start_x), np.max(end_x)))
    min_y = int(min(np.min(start_y), np.min(end_y)))
    max_y = int(max(np.max(start_y), np.max(end_y)))
    
    
    # Rogner la frame selon ces limites
    cropped_frame = frame[(min_y):(max_y), (min_x):(max_x)]
    
    return cropped_frame

def angle_between_vectors(v1, v2):
    """Calcule l'angle en radians entre deux vecteurs en évitant les divisions par zéro."""
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return np.pi  # Si un vecteur est nul, retour d'un grand angle (180°)

    dot_product = np.dot(v1, v2)
    cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)  # Évite les erreurs numériques
    return np.arccos(cos_angle)

# Fonction pour filtrer les champs vectoriels avec au moins 5 vecteurs alignés
def filter_vector_fields(data, max_angle_deg=1, min_count=5):
    if data.shape[0] == 0:
        return data
    max_angle_rad = np.deg2rad(max_angle_deg)
    filtered_vectors = []
    positions = data[:, 5:6]
    # Calcul des vecteurs à partir des points initiaux et finaux
    vectors = data[:, 5:6] - data[:, 3:4]  # (x_f - x_0, y_f - y_0)
    
    # Parcourir chaque vecteur pour comparer avec les autres
    # Parcourir chaque vecteur pour comparer avec les autres
    for i, (pos1, vec1) in enumerate(zip(positions, vectors)):
        count_similar = 0
        for j, (pos2, vec2) in enumerate(zip(positions, vectors)):
            if i != j:
                distance = np.linalg.norm(pos1 - pos2)
                angle = angle_between_vectors(vec1, vec2)
                
                if angle <= max_angle_rad and distance <= 32:
                    count_similar += 1
        
        # Si au moins min_count vecteurs sont proches et alignés avec vec1, on le conserve
        if count_similar >= min_count - 1:
            filtered_vectors.append(data[i])
    
    return np.array(filtered_vectors)

def main(args=None):
    value=5
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Extract motion vectors from video.')
    parser.add_argument('video_url', type=str, nargs='?', help='File path or url of the video stream')
    parser.add_argument('-p', '--preview', action='store_true', help='Show a preview video with overlaid motion vectors')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailled text output')
    parser.add_argument('-d', '--dump', action='store_true', help='Dump frames, motion vectors, frame types, and timestamps to output directory')
    args = parser.parse_args()

    if args.dump:
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        #for child in ["frames", "motion_vectors"]:
         #   os.makedirs(os.path.join(f"out-{now}", child), exist_ok=True)


    cap = VideoCap()

    # open the video file
    ret = cap.open(args.video_url)
    if not ret:
        raise RuntimeError(f"Could not open {args.video_url}")
    
    if args.verbose:
        print("Sucessfully opened video file")

    step = 0
    times = []

    # continuously read and display video frames and motion vectors
    while True:
        if args.verbose:
            print("Frame: ", step, end=" ")

        tstart = time.perf_counter()

        # read next video frame and corresponding motion vectors
        ret, frame, motion_vectors, frame_type, timestamp = cap.read()

        # select motion vectors
    
        print (step)
        print("1 : ",np.shape(motion_vectors))
        
        motion_vectors=select_vectors_norm(motion_vectors,value)
        print("2 : ",np.shape(motion_vectors))
        
        motion_vectors=select_vectors_zone(motion_vectors)
        print("3 : ",np.shape(motion_vectors))

        #motion_vectors=filter_vector_fields(motion_vectors)
        #print("4 : ",np.shape(motion_vectors))

    

        # if there is an error reading the frame
        if not ret:
            if args.verbose:
                print("No frame read. Stopping.")
            break

        # print results
        if args.verbose:
            print("timestamp: {} | ".format(timestamp), end=" ")
            print("frame type: {} | ".format(frame_type), end=" ")

            print("frame size: {} | ".format(np.shape(frame)), end=" ")
            print("motion vectors: {} | ".format(np.shape(motion_vectors)), end=" ")
            #print("elapsed time: {} s".format(telapsed))

        frame = draw_motion_vectors(frame, motion_vectors)
        #frame = draw_motion_vectors_black(frame, motion_vectors)
    
        #frame=crop_frame(frame, motion_vectors)
        # store motion vectors, frames, etc. in output directory
        if args.dump:
            # Définir un répertoire parent pour tous les dossiers de sortie
            parent_dir = "outputs_frames"
            os.makedirs(parent_dir, exist_ok=True)  # Crée le répertoire parent s'il n'existe pas

            # Chemin du dossier spécifique à cette session
            session_dir = os.path.join(parent_dir, f"out-{now}")
            os.makedirs(session_dir, exist_ok=True)  # Crée le dossier de la session s'il n'existe pas

         # Crée les sous-dossiers pour frames et motion_vectors
            frames_dir = os.path.join(session_dir, "frames")
            motion_vectors_dir = os.path.join(session_dir, "motion_vectors")
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(motion_vectors_dir, exist_ok=True)

            # Sauvegarde les fichiers dans les sous-dossiers
            if frame.size != 0:
                cv2.imwrite(os.path.join(frames_dir, f"frame-{step}.jpg"), frame)
                np.save(os.path.join(motion_vectors_dir, f"mvs-{step}.npy"), motion_vectors)

        # Ajoute les informations dans les fichiers texte
            with open(os.path.join(session_dir, "timestamps.txt"), "a") as f:
                f.write(str(timestamp) + "\n")
            with open(os.path.join(session_dir, "frame_types.txt"), "a") as f:
                f.write(frame_type + "\n")


        step += 1

        if args.preview:
            cv2.imshow("Frame", frame)

            # if user presses "q" key stop program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        tend = time.perf_counter()
        telapsed = tend - tstart
        times.append(telapsed)
    
    if args.verbose:
        print("average dt: ", np.mean(times))

    cap.release()

    # close the GUI window
    if args.preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
