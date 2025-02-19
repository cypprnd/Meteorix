# Apprentissage

### Descente de gradient

Dans le cas de **FlowNet**, l’apprentissage par descente de gradient se fait (sur une couche) sur une matrice \( w \) des poids en fonction de "vecteurs" de pixels.

Avec \( \alpha \) un réel (le **learning rate**) et pour l’itération \( n \), on a la formule :

\[
w_n = w_{n-1} - \alpha \nabla J(w_{n-1})
\]

Où \( J \) est une fonction de mesure de l’erreur avec comme données d’entraînement \( \{(x^{(i)}, y^{(i)})\}_{i \in \{1, \dots, m\}} \), \( m \) étant le nombre de paires de données, où \( x^{(i)} \) est la donnée d’entrée et \( y^{(i)} \) le flux optique réel ("GT") :

\[
J(w_{n-1}) = \frac{1}{2m} \times \sum_{i=1}^m (h(x^{(i)}) - y^{(i)})^2
\]

\( h \) étant la fonction qui associe à \( x^{(i)} \) le flux optique prédit par le réseau :

\[
h(x^{(i)}) = w_0 + \sum_{j=1}^n w_j x_j^{(i)}
\]

Ainsi, avec \( x^{*(i)} = [1, x^{(i)}] \) pour prendre en compte \( w_0 \) :

\[
\nabla J(w_{n-1}) = \frac{1}{2m} \times \sum_{i=1}^m (h(x^{(i)}) - y^{(i)}) \times x^{*(i)}
\]

La descente de gradient peut être utilisée de 3 manières différentes :

1. **Batch** : On utilise toutes les données de la data d’entraînement dans une matrice géante, et on calcule le gradient sur toutes les données, mais cela ralentit énormément l’apprentissage.
2. **Stochastique** : On sélectionne une donnée aléatoire (une paire d’images) à la \( i \)-ème itération en pensant que cette donnée fiable ressemble à toutes les autres.
3. **Mini-batch** : On sélectionne de manière aléatoire une partie des données d’entraînement. Pour FlowNet, ils ont pris 8 paires d’images comme taille pour une itération.

### Problème de l’underfitting et de l’overfitting

À partir de l’apprentissage par descente de gradients stochastique ou en mini-batch, le problème de l’**underfitting** et de l’**overfitting** est lié au nombre d’itérations effectué pour entraîner le réseau. En effet, l’underfitting est le fait de ne pas assez entraîner le réseau, alors que l’overfitting entraîne "trop" le réseau, et il perd sa capacité à généraliser.

Pour faire face à ce problème, il existe différentes méthodes. Dans le cas de FlowNet, ils ont utilisé un **validation set** qui teste à chaque itération quels sont les meilleurs poids à garder.

### Backpropagation

Le principe de **backpropagation** permet d’entraîner un réseau de neurones avec plusieurs couches. Cela consiste à, au début, calculer la sortie avec les poids existants au temps \( t \), puis changer les poids en partant de la fin et en se basant sur le principe (avec \( z(x, y) \), \( x(s) \) et \( y(s) \)) :

\[
\frac{\partial z}{\partial s} = \frac{\partial z}{\partial x} \times \frac{\partial x}{\partial s} + \frac{\partial z}{\partial y} \times \frac{\partial y}{\partial s}
\]

### Algorithme d’Adam

L’algorithme d’**Adam** est un algorithme qui modifie le **learning rate** pour optimiser l’apprentissage dans la méthode de descente de gradient. Pour essayer de comprendre l’algorithme, nous allons expliciter les optimisations faites de manière temporelle :

1. **AdaGrad** : La première optimisation a été de diminuer le learning rate en fonction du temps et du gradient (et ainsi de prendre en compte le fait que l’algorithme converge vers la solution). Nous avons donc, avec \( \delta \) une valeur pour ne pas diviser par 0, \( \epsilon \) le learning rate de départ :

\[
r \leftarrow r + \nabla J(w_{t-1})^2
\]
\[
w_t \leftarrow w_{t-1} - \frac{\epsilon}{\delta + \sqrt{r}} \times \nabla J(w_{t-1})
\]

2. **RMSProp** : Pour ralentir la progression, une restriction a été mise sur \( r \). Une moyenne logarithmique est faite entre le terme précédent et le terme rajouté :

\[
r_t \leftarrow \gamma \times r_{t-1} + (1 - \gamma) \times \nabla J(w_{t-1})^2
\]

3. **Adam** : La dernière optimisation est de prendre en compte le **moment (s)**, qui est une variable qui permet de ne pas tomber dans un minimum local et de chercher le minimum global. Mathématiquement, c’est une variable qui change en fonction du gradient au premier ordre (plus le gradient est grand, plus le moment est grand et plus le learning rate est élevé).