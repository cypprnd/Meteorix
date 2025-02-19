# Flux optique

À partir des images avec la zone d’intérêt extraites grâce à **Motion Vector Extractor (MVE)**, nous allons maintenant chercher à calculer le flux optique.

On considère l’application \( I \) qui quantifie l’intensité lumineuse d’un pixel dans une des images (en faisant la simplification que l’image est en nuances de gris par exemple). L’intensité lumineuse d’un pixel de coordonnées \( (x, y) \) à l’instant \( t \) est donc donnée par \( I(x, y, t) \).

Dans la suite de cette partie, on cherche à déterminer le flux optique entre 2 images successives séparées d’un délai de \( dt \).

## Hypothèse d’illumination constante

Premièrement, il est nécessaire de faire l’hypothèse d’illumination constante, qui consiste à dire que :

\[
I(x + dx, y + dy, t + dt) = I(x, y, t)
\]

## Développement de Taylor

Par ailleurs, on peut appliquer la formule de Taylor à l’ordre 1 sur \( I \) pour \( dx, dy, dt \) proches de 0 :

\[
I(x + dx, y + dy, t + dt) = I(x, y, t) + \frac{\partial I}{\partial x}(x, y, t) \, dx + \frac{\partial I}{\partial y}(x, y, t) \, dy + \frac{\partial I}{\partial t}(x, y, t) \, dt
\]

Or, d’après l’hypothèse d’illumination constante, on obtient :

\[
\frac{\partial I}{\partial x}(x, y, t) \, dx + \frac{\partial I}{\partial y}(x, y, t) \, dy + \frac{\partial I}{\partial t}(x, y, t) \, dt = 0
\]

En divisant par \( dt \) (non nul), on a :

\[
\frac{\partial I}{\partial x}(x, y, t) \frac{dx}{dt} + \frac{\partial I}{\partial y}(x, y, t) \frac{dy}{dt} + \frac{\partial I}{\partial t}(x, y, t) = 0
\]

## Vecteur de flux optique

On peut définir le vecteur de flux optique au point \( (x, y) \) à l’instant \( t \) qui correspond au vecteur \( \mathbf{u} = \left( \frac{dx}{dt}, \frac{dy}{dt} \right) \). Ce dernier peut être vu comme le vecteur vitesse du point image \( (x, y) \) à l’instant \( t \).

## Conclusion

On peut interpréter le flux optique entre 2 images successives comme étant l’ensemble des vecteurs vitesses à l’instant \( t \) associés à chaque point image. Plus généralement, dans une vidéo, on peut considérer que le flux optique est la vitesse de chaque point image (fonction à 2 composantes) en faisant l’hypothèse de l’illumination constante.