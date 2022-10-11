# Implémentation d'un modèle de scoring (prédiction du remboursement d'un emprunt bancaire)


## 1. Le Contexte
"Prêt à dépenser" est une société financière qui propose des crédits à la consommation pour des personnes
ayant peu ou pas du tout d'historique de prêt.
- L’entreprise souhaite mettre en oeuvre un outil de scoring de crédit pour déterminer si elle peut accorder un
emprunt en minimisant le risque financier.
- Cette étude consiste à :
    - Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon
automatique.
    - Construire un dashboard interactif à destination des gestionnaires de la relation client permettant
d'expliquer de façon la plus transparente possible les décisions d’octroi de crédit.


## 2. Les Données
Les données proviennent de Home Credit (https://www.kaggle.com/c/home-credit-default-risk/data)

Elles sont séparées en 7 fichiers qui seront fusionnés pour obtenir une base de données unique concernant 307k
emprunts caractérisés par 218 variables (informations bancaires et personnelles des emprunteurs), ainsi que 47k
emprunts sans variable cible. Ce dernier set ne sera utilisé que pour illustrer le dashboard.

![image](https://user-images.githubusercontent.com/108366684/195057349-fc71a3c6-3411-424a-b81b-3de807940ce3.png)

La taille des différents jeux de données est :
- Training + Validation set : 80% * 307k = 246k (+ validation croisée à 5 plis)
- Test set indépendant : 20% * 307k = 61k
- Kaggle Test set : 1k (exemples utilisés dans le dashboard)

La Variable cible à prédire peut prendre 2 valeurs :
- 1 (classe positive) : clients avec des difficultés de paiement (retard de paiement pendant plus de X jours
pour au moins un des Y premiers versements)
- 0 (classe négative) : tous les autres cas

La base de données est “déséquilibrée” avec seulement 8 % des emprunts classés en classe 1.


## 3. Fetures engineering (extraction de nouvelles caractéristiques)
Actuellement le features engineering utilise principalement le kernel suivant :
https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script, ce qui conduit à plus de 800
variables.

TO DO: C'est l'unique partie de ce projet dont je ne suis pas l'auteur. Cette décision avait été prise à l'époque pour gagner du temps. 
Le features engineering sera prochainement entièrement revu pour incorporer uniquement mon analyse.


## 4. La sélection des variables
Pour contrebalancer le fléau de la dimension, améliorer le score, accélérer les calculs et gagner en
interprétabilité, l’étape suivante consiste à diminuer le nombre de variables en particulier en recherchant les
variables les plus importantes.

On retient les 50 variables les plus importantes pour le modèle suivant l’algorithme de SHAP (méthode itérative). 
On examine ensuite le tableau de corrélation des variables. Pour les couples de variables qui ont un coefficient de corrélation supérieur
à 0.85, on retire une des deux variables à chaque fois → 46 variables.

TO DO: A la réflexion, il semble nécessaire d'inclure la sélection des variables à l'intérieur de chacun des plis de la validation croisée. Sera mis à jour en utilisant un pipeline.






