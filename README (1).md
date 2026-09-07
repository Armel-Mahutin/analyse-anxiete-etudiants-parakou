# Prévalence et facteurs associés à l'anxiété chez les étudiants en sciences médicales et paramédicales — Université de Parakou, 2025

Script Python d'analyse statistique (descriptive, bivariée et multivariée) réalisé dans le cadre d'un mémoire de recherche en santé publique.

## Objectifs de l'analyse

1. Estimer la prévalence de l'anxiété (score **GAD-7**)
2. Identifier les facteurs associés à l'anxiété (tests bivariés + régression logistique multivariée, OR ajustés)
3. Décrire le profil-type de l'étudiant anxieux

L'anxiété est mesurée par le **GAD-7** ; le stress perçu (facteur associé potentiel) par la **PSS-10** (échelle de Cohen).

## ⚠️ Confidentialité des données (RGPD)

**Ce dépôt ne contient et ne doit jamais contenir de données individuelles.** Seul le code d'analyse est public.

- Le fichier de données (`.xlsx`) et le dossier de résultats (`resultats_anxiete/`, qui inclut un export des données individuelles nettoyées) sont exclus via `.gitignore`.
- Le script supprime dès le chargement les identifiants techniques bruts (`_uuid`, `root_uuid`).
- Cela ne suffit pas à anonymiser complètement le jeu de données : la combinaison de plusieurs variables (âge, sexe, filière, année d'étude...) peut suffire à réidentifier une personne sur un petit échantillon. Ne diffuse **jamais** le fichier de données lui-même, y compris par un canal non sécurisé (mail, drive public, etc.).
- Conserve le fichier de données uniquement en local, dans le dossier `data/` (non versionné), et assure-toi d'avoir respecté les procédures d'éthique/consentement de ton étude avant toute collecte ou traitement.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

1. Place ton fichier de données dans un dossier local `data/` (non versionné) :
   ```
   data/donnees_anxiete.xlsx
   ```
2. Lance le script en indiquant le chemin du fichier :
   ```bash
   python analyse_anxiete_parakou.py --data data/donnees_anxiete.xlsx
   ```

## Résultats produits

Dans le dossier `resultats_anxiete/` (non versionné) :

- `tableaux_resultats.xlsx` — tous les tableaux de résultats (un onglet par tableau)
- `figures/*.png` — tous les graphiques (distribution du score GAD-7, prévalence, facteurs associés...)
- `log_analyse.txt` — récapitulatif texte complet de l'analyse
- `base_nettoyee_avec_scores.xlsx` — données individuelles nettoyées avec scores calculés (**usage local uniquement**)

Les graphiques s'affichent aussi à l'écran à la fin de l'exécution.

## Méthodes statistiques

- Prévalence présentée à deux seuils GAD-7 en parallèle : ≥5 (légère à sévère) et ≥10 (modérée à sévère), avec IC95% (méthode de Wilson)
- Analyse bivariée : test du Khi2 (ou Fisher exact si effectifs attendus <5) pour les variables qualitatives, t-test de Welch ou Mann-Whitney (selon normalité, testée par Shapiro-Wilk) pour les variables quantitatives
- Régression logistique multivariée : variables présélectionnées à p<0,20 en bivarié (règle de Hosmer-Lemeshow), OR ajustés avec IC95%

## Licence

À toi de choisir une licence adaptée si tu rends ce dépôt public (ex. MIT pour le code). Le jeu de données n'est, en tout état de cause, pas concerné par cette licence : il reste soumis aux règles de confidentialité rappelées ci-dessus.
