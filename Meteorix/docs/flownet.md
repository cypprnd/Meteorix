# Flownet

### Architecture de Flownet

**FlowNet** est un réseau de neurones qui n’a pas eu besoin d’un dataset réaliste pour avoir des résultats intéressants. Le réseau est capable de généraliser et il est même meilleur que certaines méthodes à la pointe (à ce moment-là) telles que **Deepflow** et **Epicflow** sur ses données d’entraînement. Nous avons trouvé le code source [ici](#).

Pour calculer le flux optique, la taille des entrées et sorties étant très grande, les couches de **pooling** sont nécessaires pour la praticité du réseau de neurones. De plus, nous en avons besoin pour pouvoir agréger des parties de l’image en flux "globaux" optique. Faire du pooling réduit la résolution de l’image, nous avons donc une partie du réseau qui "pool", la **partie contractive**. Puis, la deuxième partie du réseau réaugmente la taille de l’"image" (pour pouvoir fournir le flux optique demandé), la **partie expansive** dite de "refinement". Ce réseau est entraîné en tant qu’une seule entité (pas d’entraînement différents entre les deux parties), par rétropropagation du gradient.

### Partie contractive

- **FlowNetSimple** : Les deux images sont données dans la même entrée au réseau, et il cherche "seul" la correspondance entre les deux images et le flot optique. Plus précisément, on "superpose" les deux images dans 6 canaux.
- **FlowNetCorr** : Les deux images ont un prétraitement où leurs caractéristiques sont extraites en **coarse features** dans deux "branches" indépendantes du réseau en parallèle puis sont regroupées.

#### Détails sur la corrélation

Les deux **feature maps** sont associées par partie (appelée **patch**) par une somme de produit scalaire appelée **multiplicative patch comparison**.

Pour comprendre comment ça fonctionne, regardons la formule d’une simple comparaison de deux patchs centrés sur la position \( x_1 \) dans la première map et \( x_2 \) dans la deuxième :

\[
c : (w \times h) \times (w \times h) \rightarrow \mathbb{R}
\]
\[
c(x_1, x_2) = \sum_{o \in [-k, k] \times [-k, k]} \langle f_1(x_1 + o), f_2(x_2 + o) \rangle
\]

Ici, les deux patchs sont de taille \( (2k + 1) \times (2k + 1) \), et \( f_1 \) (respectivement \( f_2 \)) représentent les données d’un pixel sur toutes les **coarse features** d’entrée de l’image 1 (respectivement de l’image 2), autrement dit, sur toute la profondeur de l’"image" en position \( x_1 + o \) ou \( x_2 + o \).

Si on cherche les correspondances pour tous les points de la première feature map avec la deuxième, le réseau ne sera pas entraînable. En effet, nous aurons trop de données à calculer. Aussi, deux pixels très loin l’un de l’autre avec la même intensité lumineuse pourraient correspondre alors que ce n’est pas ce qu’on veut. Pour pallier à ce problème, une limitation sur le décalage entre \( x_1 \) et \( x_2 \) a été mise en place (recherche de correspondance entre \( x_1 \) et \( x_2 \) seulement dans leur voisinage respectif). Le déplacement entre les deux feature maps a été mis en place en fonction de la limitation. Le voisinage autorisé est une fenêtre de diamètre \( D = 2d + 1 \).

### Partie expansive

**Upconvolutional layer** : Le principe est d’étendre (en largeur et hauteur) les feature maps à la fin de la partie contractive. Pour cela, on fait un **bed of nails** et une convolution avec un masque de notre entrée (la fin de notre partie contractive), puis une concaténation avec des feature maps de la partie contractive gardées en mémoire.

Les données sauvegardées des précédentes convolutions sont utilisées comme informations à ajouter sur les **coarse feature maps** de la fin de la première partie. Cette méthode permet de garder les informations des **coarse feature maps** (le "flux optique global") et des informations précises locales avec les niveaux précédents.

Ils se sont arrêtés à une résolution de l’image (de sortie) de 96x128 pour un input de 384x512 (la taille est divisée par 4). Ce format donne déjà les informations nécessaires sur le flux optique, et plus de couches dans la partie expansive n’améliorent pas le résultat. Ils réalisent une interpolation bilinéaire avec la sortie pour avoir la bonne taille de l’image d’entrée.

### Entraînement

FlowNet étant novateur (à ce moment-là) et le flux optique étant compliqué à calculer avec des images réelles, trouver des données d’entraînement adaptées est difficile.

#### Jeux de données existants

- **Middlebury** : Très petit nombre de données et de très petits mouvements.
- **KITTI** : Mouvements spécifiques avec un pourcentage faible (environ 50 %) de données vérifiées sur le flux optique "GT" (les données sont réelles : elles ne sont pas générées artificiellement).
- **MPISintel** : Nombre de données considérable, GT avec un pourcentage de 100 % (de certitude) mais pas assez pour entraîner un réseau de neurones (base de données de synthèse).

#### Flying Chairs Dataset

Pour pouvoir entraîner le réseau, ils ont créé un jeu de données, le **Flying Chairs Dataset**. Ils ont choisi aléatoirement :
- Des transformations affines en 2D sur le fond et sur les chaises pour les déplacements.
- La taille, l’endroit où les chaises se placent.

La distribution utilisée pour générer les données aléatoires est ajustée pour que les données soient similaires à celle de **MPISintel**. Les créateurs de **Flying Chairs** auraient pu apparemment générer plus de données pour le dataset mais se sont arrêtés à 22 872 paires d’images.

Pour éviter le problème de l’**overfitting**, la **data augmentation** a été utilisée sur :
- La translation,
- La rotation,
- La taille,
- Le bruit gaussien,
- La luminosité,
- Le gamma,
- Le contraste,
- La couleur.

### Détails pratiques sur le réseau

FlowNet est composé de 9 couches de convolution. Parmi ces couches de convolution, 6 d’entre elles effectuent aussi du pooling avec la méthode du **stride of 2**, où la taille des feature maps est divisée par 2 et le nombre de feature maps est multiplié par 2. Un **ReLU** est présent après chaque couche de convolution.

De plus, on peut avoir en entrée du réseau une taille variable entre les paires d’images car il n’y a pas de couches complètement connectées (FC).

### Résultats dans l’article

FlowNet est testé sur **Sintel**, **Kitti**, **Middlebury** et **Flying chairs**. Pour pouvoir comparer les données, ils utilisent le **average end-to-end point error (EPE)**, ce qui correspond à la distance euclidienne entre le flux trouvé et le véritable flux, moyennée sur tous les pixels. Plus le chiffre (EPE) est faible, plus le résultat de FlowNet est proche de la réalité et plus on peut considérer que FlowNet est performant.

En général, on peut observer que FlowNet n’est pas le plus performant sur les données de test par rapport à ses adversaires **EpicFlow**, **DeepFlow**, etc. Néanmoins, les concepteurs de FlowNet considèrent que leur réseau est le plus efficace en termes de performances relativement au temps d’exécution. On peut noter que FlowNet est le meilleur réseau sur le dataset **Chairs** (en considérant les données de test). Ce résultat peut s’expliquer par le fait que FlowNet a été entraîné sur le même dataset (mais sur d’autres données que celles du test). Concernant le dataset **Chairs**, la version **corr** de FlowNet (FlowNetC) est un peu plus performante que **FlowNetS**. Mais, sur d’autres datasets (notamment avec **KITTI**), la tendance peut s’inverser à cause du fait que les mouvements sur ces datasets peuvent être plus grands/amples. De même, **FlowNetS** semble être meilleur que **FlowNetC** lorsqu’il y a des mouvements constants et larges dans l’image comme du brouillard ou des nuages (voir dataset **SintelFinal** qui en contient). Finalement, FlowNet a une certaine capacité de généralisation entre l’entraînement et de nouvelles données utilisées pour les tests.