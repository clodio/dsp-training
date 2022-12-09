# utilisation d'une image docker déjà existante qui a python3.7 préinstallée
# la liste complète des images peut être retrouvée sur https://hub.docker.com/
FROM python:3.7-slim-buster

# installation de git (mlflow tracking demande à avoir accès à la commande git)
RUN apt update
RUN apt install -y git

# notre dossier de travail au sein de l'image docker (il va être créé automatiquement)
WORKDIR /dsp-training

# copie du projet de la machine hôte (notre laptop) vers l'image docker
COPY . .

# installation des dépendances du projet

# j'ai du remplacer sklearn par scikit-learn dans les requirements pour qu eca fonctionne sur mon poste
# je n'ai pas de poste avec docker je n'ai pas pu tester
RUN pip3 install -r requirements

RUN python -m pytest tests

# notre script qui va être exécuté quand l'image sera lancée via "docker run"
# TODO
CMD ["python3", "main.py"]
