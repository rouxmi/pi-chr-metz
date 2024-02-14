# Application Prédiction de lésion péri-apicale.

## Description

L'application est conçue pour convertir des fichiers DICOM au format PNG et appliquer un modèle YOLO fine-tune pour la détection de ses lésions sur des photos panoramiques dentaires. Les prédictions de boîtes englobantes sont ensuite utilisées pour créer des fichiers GSPS contenant uniquement un layer d'annotations.

## Fonctionnalités

- Conversion DICOM vers PNG
- Détection d'objets en utilisant un modèle YOLO custom
- Création de fichiers GSPS basés sur les prédictions de boîtes englobantes
- Choix entre plusieurs modèle de détection (DETR et yolo)

## Configuration requise

- Python 3.7 ou version ultérieure (testé avec 3.11.7)
- Bibliothèques principales : ultralytics, cv2, numpy, pydicom
- Détails des librairies et des versions dans `requirements.txt`

## Installation

1. Clonez le dépôt sur votre machine locale.
2. Installez les bibliothèques Python requises en utilisant la commande `pip install -r requirements.txt`.

## Utilisation

1. Lancer le script `run.sh` (besoin des droirs d'éxécution dessus)
2. Placez vos fichiers DICOM dans le répertoire `input`.
3. Le script principal `python main.py` sera executé après 30 sec si le dossier input est non vide.
4. Les fichiers GSPS seront envoyés sur le PACS donnée dans le `config.ini`

## Configuration

Tout les paramètres de l'application peuvent être modifier dans le fichier `config.ini` comme le PACS de destination, le modèle utiliser ou l'utilisation du mode debug. Il existe 2 modèles possibles pour prédire les lésions Yolo et DETR. Yolo est le plus performant mais il est conseillé de l'utiliser avec width=1024 et height=512.

## Mode debug

Le mode debug permet de visualiser les boudings boxes trouvés avant l'envoie sur le pacs et la visualisation des logs directement dans le terminal.

## Logs 

Des logs sont disponibles dans le dossier log à chaque éxécution du script `main.py` un fichier est crée pour suivre l'éxécution du script et les éventuels problèmes.

## Contribution

Les contributions sont les bienvenues. Veuillez soumettre une pull request avec vos modifications.

## Support

Si vous avez des questions ou des problèmes, veuillez ouvrir une issue sur le dépôt GitHub.

