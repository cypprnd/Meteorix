# Annexes

### Motion Vector Extractor

#### Calcul de l’angle d’un vecteur


Calcul de l'angle d'un vecteur
``` py
def calculate_angle(mv):
    start_pt = np.array([mv[3], mv[4]])
    end_pt = np.array([mv[5], mv[6]])
    direction = end_pt - start_pt
    angle = np.arctan2(direction[1], direction[0])  # En radians
    return angle
```

Filtre les vecteurs en fonction de la distance entre le point de départ et le point d'arrivé
``` py
def filter_by_magnitude(motion_vectors, min_magnitude):
    filtered_vectors = []
    for mv in motion_vectors:
        start_pt = np.array([mv[0,3], mv[0,4]])  # source_x, source_y
        end_pt = np.array([mv[0,5], mv[0,6]])    # dst_x, dst_y
        magnitude = np.linalg.norm(end_pt - start_pt)  # Calculer la magnitude du vecteur
        if magnitude > min_magnitude:
            filtered_vectors.append(mv)
    return np.array(filtered_vectors)
```

Clusterise les vecteurs de mouvement
``` py
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
```

Vérifie si tous les vecteurs d'un cluster vont dans la même direction
``` py
def is_cluster_rectilinear(cluster, angle_tolerance=0.1):
    """Vérifie si un cluster de vecteurs suit une direction rectiligne."""
    angles = np.array([calculate_angle(mv) for mv in cluster])
    mean_angle = np.mean(angles)
    
    # Vérifier si tous les angles sont proches du même angle moyen
    deviations = np.abs(angles - mean_angle)
    
    return np.all(deviations < angle_tolerance)  # Si toutes les déviations sont inférieures à une tolérance
```

Dessine les vecteurs du cluster
``` py
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
```

Dessine les vecteurs sur fond noir
``` py
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
```

Dessine les vecteurs sur les frames originaux
``` py
def draw_motion_vectors(frame, motion_vectors):

    if len(motion_vectors) > 0:
        num_mvs = np.shape(motion_vectors)[0]
        for mv in np.split(motion_vectors, num_mvs):
            start_pt = (mv[0, 3], mv[0, 4])
            end_pt = (mv[0, 5], mv[0, 6])
            #if (np.sqrt(np.abs((end_pt[0]-start_pt[0])^2+(end_pt[1]-start_pt[1])^2))>5):
            cv2.arrowedLine(frame, start_pt, end_pt, (70, 255, 255), 1, cv2.LINE_AA, 0, 0.1)
    return frame
```

Sélectionne les vecteurs vérifiant une condition sur la norme
``` py
def select_vectors_norm(motion_vectors):
    start_pt = motion_vectors[:, [3, 4]]
    end_pt = motion_vectors[:, [5, 6]]
    norm = np.linalg.norm(end_pt - start_pt, axis=1)
    return motion_vectors[norm>10]
```

Sélectionne les vecteurs ayant un certain nombre de voisins
``` py
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
```

Crop pour zoomer sur une frame où se trouve un météore
``` py
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
```

Calcule l'angle entre deux vecteurs donnés
``` py
def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angle = dot_product / norm_product
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle)
```

Filtre les champs vectoriels avec au moins 5 vecteurs alignés
``` py
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
```