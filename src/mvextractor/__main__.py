import sys
import os
import time
from datetime import datetime
import argparse

import numpy as np
import cv2

from sklearn.cluster import DBSCAN
from mvextractor.videocap import VideoCap
from sklearn.neighbors import NearestNeighbors

def calculate_angle(mv):
    start_pt = np.array([mv[3], mv[4]])
    end_pt = np.array([mv[5], mv[6]])
    direction = end_pt - start_pt
    angle = np.arctan2(direction[1], direction[0])  # En radians
    return angle


def filter_by_magnitude(motion_vectors, min_magnitude):
    filtered_vectors = []
    for mv in motion_vectors:
        start_pt = np.array([mv[0,3], mv[0,4]])  # source_x, source_y
        end_pt = np.array([mv[0,5], mv[0,6]])    # dst_x, dst_y
        magnitude = np.linalg.norm(end_pt - start_pt)  # Calculer la magnitude du vecteur
        if magnitude > min_magnitude:
            filtered_vectors.append(mv)
    return np.array(filtered_vectors)


def cluster_motion_vectors(motion_vectors, eps=15, min_samples=3):
    """Cluster motion vectors based on spatial proximity."""
    if len(motion_vectors) == 0:
        return []

    # Créer un tableau de features [x, y] pour clusterisation
    feature_vectors = []
    for mv in motion_vectors:
        start_pt = np.array(mv[3], mv[4])  # source_x, source_y
        feature_vectors.append([start_pt[0], start_pt[1]])
    
    # Cluster en utilisant DBSCAN sur les coordonnées spatiales
    feature_vectors = np.array(feature_vectors)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(feature_vectors)
    
    clusters = []
    for cluster_id in np.unique(clustering.labels_):
        if cluster_id != -1:  # Ignorer les points de bruit (label = -1)
            cluster = np.array(motion_vectors)[clustering.labels_ == cluster_id]
            clusters.append(cluster)
    
    return clusters


def is_cluster_rectilinear(cluster, angle_tolerance=0.1):
    """Vérifie si un cluster de vecteurs suit une direction rectiligne."""
    angles = np.array([calculate_angle(mv) for mv in cluster])
    mean_angle = np.mean(angles)
    
    # Vérifier si tous les angles sont proches du même angle moyen
    deviations = np.abs(angles - mean_angle)
    
    return np.all(deviations < angle_tolerance)  # Si toutes les déviations sont inférieures à une tolérance


def draw_clusters(frame, clusters, motion_vectors):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Quelques couleurs pour les clusters
    valid_clusters = []
    
    for i, cluster in enumerate(clusters):
        if is_cluster_rectilinear(cluster):  # Ne garder que les clusters rectilignes
            valid_clusters.append(cluster)
            for mv in cluster:
                start_pt = (int(mv[3]), int(mv[4]))  # source_x, source_y
                end_pt = (int(mv[5]), int(mv[6]))    # dst_x, dst_y
                # Assigner une couleur en fonction du cluster
                color = colors[i % len(colors)]
                cv2.arrowedLine(frame, start_pt, end_pt, color, 1, cv2.LINE_AA, 0, 0.1)
    
    return frame, valid_clusters
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

def select_vectors_norm(motion_vectors):
    start_pt = motion_vectors[:, [3, 4]]
    end_pt = motion_vectors[:, [5, 6]]
    norm = np.linalg.norm(end_pt - start_pt, axis=1)
    return motion_vectors[norm>10]

def select_vectors_zone(motion_vectors):
    if motion_vectors.shape[0] == 0:
        return motion_vectors
    
    start_pt = motion_vectors[:, [3, 4]]
    end_pt = motion_vectors[:, [5, 6]]

    min_neighbors = 2
    distance_threshold = 16

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
    cropped_frame = frame[(min_y-16):(max_y+16), (min_x-16):(max_x+16)]
    
    return cropped_frame

def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angle = dot_product / norm_product
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle)

# Fonction pour filtrer les champs vectoriels avec au moins 5 vecteurs alignés
def filter_vector_fields(data, max_angle_deg=5, min_count=5):
    if data.shape[0] == 0:
        return data
    max_angle_rad = np.deg2rad(max_angle_deg)
    filtered_vectors = []
    
    # Calcul des vecteurs à partir des points initiaux et finaux
    vectors = data[:, 5:6] - data[:, 3:4]  # (x_f - x_0, y_f - y_0)
    
    # Parcourir chaque vecteur pour comparer avec les autres
    for i, vec1 in enumerate(vectors):
        count_similar = 0
        for j, vec2 in enumerate(vectors):
            if i != j:
                angle = angle_between_vectors(vec1, vec2)
                if angle <= max_angle_rad:
                    count_similar += 1
        
        # Si au moins min_count vecteurs sont alignés avec vec1, on le conserve
        if count_similar >= min_count - 1:
            filtered_vectors.append(data[i])  # Conserver la ligne correspondante
    
    return np.array(filtered_vectors)

def main(args=None):
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
        for child in ["frames", "motion_vectors"]:
            os.makedirs(os.path.join(f"out-{now}", child), exist_ok=True)

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
        
        motion_vectors=select_vectors_norm(motion_vectors)
        print("2 : ",np.shape(motion_vectors))
        
        motion_vectors=select_vectors_zone(motion_vectors)
        print("3 : ",np.shape(motion_vectors))

        #motion_vectors=filter_vector_fields(motion_vectors)
        #print("4 : ",np.shape(motion_vectors))

        #frame=crop_frame(frame, motion_vectors)

        # Filtrer les vecteurs de mouvement par magnitude
        #min_magnitude = 2.0  # Ajuster cette valeur selon le seuil de magnitude désiré
        #filtered_mvs = filter_by_magnitude(motion_vectors, min_magnitude)

        # Cluster les vecteurs de mouvement
        #clusters = cluster_motion_vectors(filtered_mvs, eps=15, min_samples=3)

        # Dessiner les clusters rectilignes sur la frame
        #frame, valid_clusters = draw_clusters(frame, clusters, filtered_mvs)

        tend = time.perf_counter()
        telapsed = tend - tstart
        times.append(telapsed)

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
            print("elapsed time: {} s".format(telapsed))

        #frame = draw_motion_vectors(frame, motion_vectors)
        frame = draw_motion_vectors_black(frame, motion_vectors)
        # store motion vectors, frames, etc. in output directory
        if args.dump:
            cv2.imwrite(os.path.join(f"out-{now}", "frames", f"frame-{step}.jpg"), frame)
            np.save(os.path.join(f"out-{now}", "motion_vectors", f"mvs-{step}.npy"), motion_vectors)
            with open(os.path.join(f"out-{now}", "timestamps.txt"), "a") as f:
                f.write(str(timestamp)+"\n")
            with open(os.path.join(f"out-{now}", "frame_types.txt"), "a") as f:
                f.write(frame_type+"\n")


        step += 1

        if args.preview:
            cv2.imshow("Frame", frame)

            # if user presses "q" key stop program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    if args.verbose:
        print("average dt: ", np.mean(times))

    cap.release()

    # close the GUI window
    if args.preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
