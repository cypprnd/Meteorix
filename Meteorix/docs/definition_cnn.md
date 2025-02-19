# Définition CNN

Un **CNN (Convolutional Neural Network)** est un cas particulier de réseau de neurones. En effet, il présente comme les autres réseaux une couche d’entrée (qui correspondra à l’image d’entrée) et une couche de sortie (qui correspondra en général à une couche de classification). Les CNN présentent 3 types de couches : les couches de convolution, les couches de pooling et les couches entièrement connectées (FC).

### Couches de convolution

Les couches de convolution ont une architecture de sorte que la propagation du signal au travers de cette couche effectue un produit de convolution avec celui-ci. La sortie de cette couche correspondra au produit de convolution entre la sortie de la couche précédente et un **masque de convolution** correspondant aux poids neuronaux (associés aux connexions neuronales entre la couche précédente et la couche de convolution).

### Couches de Pooling

Les couches de **Pooling** servent essentiellement à réduire la taille de l’image entre 2 couches de convolution. Cela permet de réduire la complexité du réseau et de le rendre ainsi utilisable en pratique. On parle souvent de **max-pooling** car on utilise en général une fonction qui prend le maximum de l’intensité du voisinage d’un neurone/pixel.

### Couches entièrement connectées (FC)

Les couches **FC (Fully Connected)** sont en général à la fin du réseau pour "interpréter" les **feature maps** (ou cartes caractéristiques) de l’image obtenues après la succession de couches de convolutions et de pooling. À la suite de ces couches, on obtient une sortie qui en général (ce n’est pas le cas de FlowNet) classe quelque chose dans l’image de départ (comme des chats par exemple).

### Remarques

1. **FlowNet**, lui, n’est pas un classifieur. Sa sortie correspond bien à une image, ce qui est une différence significative par rapport aux CNN classiques.
2. **FlowNet** semble remplacer les couches FC par l’étape de "refinement" (voir la figure 1 plus bas) pour obtenir une image en sortie plutôt qu’une classification.

### Force des CNN

La force des CNN pour le traitement d’image (par rapport aux réseaux classiques) semble être cette astuce de limiter le nombre de paramètres entre chaque couche de convolutions. En effet, si on souhaite faire du traitement d’image avec un réseau naïf de neurones (avec que des couches FC), le nombre d’entrées étant très grand pour une image, le nombre de paramètres entre chaque couche exploserait. Tandis qu’avec un CNN, si on choisit des noyaux de convolution 3x3, on se limite à 9 paramètres pour chaque couche de convolution, ce qui réduit drastiquement la complexité du réseau.