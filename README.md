# Prédiction du remboursement d'un emprunt bancaire


## 1. Le Contexte

Une société financière qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt souhaite mettre en oeuvre un outil de scoring de crédit pour déterminer si elle peut accorder un emprunt en minimisant le risque financier.

Cette étude consiste à :
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


## 5. Traitement des variables

En fonction de l’algorithme d’apprentissage automatique utilisé, un ou plusieurs des opérations ci-dessous ont été utilisés :
- Remplacement des valeurs manquantes :
on utilise la médiane pour les variables numériques et le mode pour les variables catégorielles.
- Transformation des variables de type catégoriel en valeurs numériques :
on remplace par le nombre de clients qui ont la modalité en question (« count encoding »).
- Mise à l’échelle (« standard scaling ») :
on soustrait la moyenne et on divise par l’écart-type.


## 6. Score “métier”

Après prédiction par l’algorithme de classification binaire (0/1), les résultats peuvent être interprétés en les classant dans 4 catégories
: Vrai Positif (TP pour True Positive), Vrai Négatif (TN), Faux Positif (FP pour False Positive)) et Faux Négatif (FN).
![image](https://user-images.githubusercontent.com/108366684/195102441-b16212eb-d092-4ac0-adfa-69ad1eaefbc2.png)

On appelle Faux Négatifs les clients que l’algorithme de Machine Learning classe comme bons payeurs (classe négative) alors qu’en
réalité ils n’ont pas payé l’intégralité de leur emprunt (classe positive). A l’inverse les Faux Positifs sont les clients qui sont prédits
comme conduisant à un défaut de paiement alors qu’ils ont payé l’intégralité de leur emprunt.

L’institution financière cherche à maximiser ses profits. Sur la base d’hypothèses qui seraient à vérifier par un expert métier, nous
attribuons un coût(-)/profit(+) aux quatre catégories :
FN : -10 | FP : -1 | TP : 0 | TN : +1

Afin de pouvoir comparer le score entre des jeux de données de taille différente (par exemple le training set et le test set), on normalise le score en divisant
par le nombre de clients TP + TN + FP + FN :

![image](https://user-images.githubusercontent.com/108366684/195102899-099fa3ec-aad0-4928-9b1b-9ed5372026c2.png)

Fonction coût:
Le score défini précédemment n’étant pas une fonction différentiable, la fonction coût/objectif qui est minimisée par l’algorithme de Machine Learning est la fonction d’entropie croisée (log-loss).


## 7. Modélisation

Plusieurs algorithmes de Machine Learning ont été testés :
- Un « dummy » classifieur qui prédit toujours la classe minoritaire
- Un modèle de régression logistique
- Trois algorithmes de boosters de gradient : CatBoost, LightGBM et XGBoost

Pour chaque algorithme, on recherche les meilleurs hyper-paramètres via la fonction
RandomSearch en fonction du score ‘métier’.

20% du jeu de données avec la variable cible est mis de côté (test set) afin d’avoir une
généralisation de l’erreur du modèle sélectionné. 80% des données sont utilisées afin
d’estimer l’erreur de chaque algorithme via une validation croisée à 5 plis.

Vu le déséquilibre relativement important entre les classes, des techniques
d’under/over-sampling ont été testées mais donnent de moins bons résultats comparé au
cas avec les données d’origine (8% de classe 1) dans la mesure où le meilleur seuil de
décision est recherché.

La fonction score « métier » est tracée en fonction du seuil de décision (qui sépare les 2 classes à partir du score provenant de predict_proba()) afin de déterminer la valeur du seuil qui maximise le profit de Prêt à dépenser.

![image](https://user-images.githubusercontent.com/108366684/195104002-4a0b18fa-c9fb-4283-ba98-84a400789f1e.png)


Résultats obtenus pour les différents algorithmes:
![image](https://user-images.githubusercontent.com/108366684/195104423-1eccaec0-e36c-4010-9a15-fc1727f3b1b3.png)

Le modèle CatBoost est retenu pour créer le Dashboard interactif.
![image](https://user-images.githubusercontent.com/108366684/195104685-c520cde0-e43b-4a4d-aeb7-28e81738c992.png)


## 8. Interprétabilité du modèle

L’interprétation des résultats au niveau global (modèle) et local (client) est effectuée par la méthode SHAP
(SHapley Additive exPlanations) basée sur la théorie des jeux coopératifs. Les valeurs de Shapley donnent une
répartition équitable des gains aux joueurs. Cet algorithme permet de déterminer les variables qui sont les plus
importantes dans les prédictions d’un modèle d’apprentissage automatique.
![image](https://user-images.githubusercontent.com/108366684/195105640-26cb968a-9b9a-4abb-ab90-02a354ab4662.png)


## 9. Déploiement de l’application

L’interface graphique du Dashboard intéractif a été créée avec Streamlit et a été déployée sur le cloud via
Streamlit Share. Le Dashboard calcule le score pour le client sélectionné en passant par une API mise en place
avec Flask et déployé via Heroku.
![image](https://user-images.githubusercontent.com/108366684/195106209-1088d739-4098-4550-b7ed-edffdafc08aa.png)

URL du Dashboard : https://rehaqds-oc-p7-app-streamlit-m672oa.streamlitapp.com/


## 10. Conclusion


