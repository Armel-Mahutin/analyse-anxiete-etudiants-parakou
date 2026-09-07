# -*- coding: utf-8 -*-
"""
================================================================================
 PREVALENCE ET FACTEURS ASSOCIES A L'ANXIETE CHEZ LES ETUDIANTS EN SCIENCES
 MEDICALES ET PARAMEDICALES A L'UNIVERSITE DE PARAKOU, 2025
================================================================================

Ce script réalise l'analyse complète correspondant aux 3 objectifs spécifiques :

  Objectif 1 : Estimer la prévalence de l'anxiété (score GAD-7)
  Objectif 2 : Identifier les facteurs associés à l'anxiété
               (analyse bivariée : Khi2/Fisher/Mann-Whitney,
                puis régression logistique multivariée -> OR ajustés)
  Objectif 3 : Décrire le profil-type de l'étudiant anxieux

Outils utilisés :
  - GAD-7 (Generalized Anxiety Disorder-7)   -> variable de résultat (anxiété)
  - PSS-10 (Perceived Stress Scale de Cohen) -> facteur associé potentiel

La prévalence est présentée aux DEUX seuils usuels du GAD-7 :
  - seuil >= 5  (anxiété légère à sévère)
  - seuil >= 10 (anxiété modérée à sévère, seuil le plus utilisé en clinique)
Les objectifs 2 et 3 sont également dupliqués pour les deux seuils, afin que
tu puisses comparer et choisir celui que tu retiens pour ton mémoire.

--------------------------------------------------------------------------------
COMMENT UTILISER CE SCRIPT DANS VS CODE
--------------------------------------------------------------------------------
1. Crée un dossier local "data/" (non versionné, voir .gitignore) et place ton
   fichier .xlsx dedans, par exemple : data/donnees_anxiete.xlsx
2. Installe les dépendances si nécessaire (terminal VS Code) :
       pip install -r requirements.txt
3. Lance le script en indiquant le chemin de tes données :
       python analyse_anxiete_parakou.py --data data/donnees_anxiete.xlsx
   (ou modifie simplement DATA_PATH_PAR_DEFAUT ci-dessous si tu préfères ne
   pas taper l'option à chaque fois -- mais ne commite JAMAIS cette valeur si
   elle contient un chemin ou un nom de fichier propre à ton ordinateur/étude)
4. Résultats produits dans le dossier ./resultats_anxiete/ :
       - tableaux_resultats.xlsx  (un onglet par tableau)
       - figures/*.png            (tous les graphiques)
       - log_analyse.txt          (récapitulatif texte de l'analyse)

--------------------------------------------------------------------------------
RGPD / CONFIDENTIALITE -- A LIRE AVANT DE PUBLIER SUR GITHUB
--------------------------------------------------------------------------------
Ce dépôt doit contenir UNIQUEMENT du code. Ne jamais committer :
  - le fichier de données brutes ou nettoyées (dossier data/)
  - le dossier de résultats (resultats_anxiete/), qui contient notamment
    "base_nettoyee_avec_scores.xlsx" = données individuelles des étudiants
  Ces éléments sont déjà exclus via le fichier .gitignore fourni.
Le script supprime dès le chargement les identifiants techniques bruts
(_uuid, root_uuid) qui ne servent à rien pour l'analyse. Cela reste
insuffisant à soi seul pour anonymiser complètement un jeu de données
(l'âge, le sexe, la filière, etc. combinés peuvent réidentifier une
personne dans un petit échantillon) : ne diffuse jamais les fichiers de
données eux-mêmes, publiquement ou par un canal non sécurisé.
================================================================================
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # backend interactif par défaut -> les figures s'affichent à l'écran
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# --- Chemin par défaut vers le fichier de données ---------------------------
# Ne mets ici qu'un chemin générique (ex: dossier "data/"), jamais un chemin
# personnel type "C:/Users/prenom/...". Le chemin réel est de toute façon
# surchargeable via l'option --data au lancement du script (voir ci-dessus),
# ce qui évite d'avoir à modifier/committer ce fichier pour changer de PC.
DATA_PATH_PAR_DEFAUT = "data/donnees_anxiete.xlsx"

# --- Dossier de sortie -----------------------------------------------------
OUTPUT_DIR = "resultats_anxiete"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

# --- Seuils GAD-7 étudiés en parallèle --------------------------------------
SEUILS_GAD7 = [5, 10]

# --- Seuil de significativité pour la présélection des variables
#     avant régression logistique multivariée (règle de Hosmer-Lemeshow) -----
SEUIL_SELECTION_BIVARIEE = 0.20
SEUIL_SIGNIFICATIVITE = 0.05

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["font.size"] = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

LOG_LINES = []


def log(msg=""):
    """Affiche et conserve chaque message (pour le fichier log_analyse.txt)."""
    print(msg)
    LOG_LINES.append(str(msg))


def titre(txt):
    log("\n" + "=" * 90)
    log(txt)
    log("=" * 90)


# ==============================================================================
# 2. CHARGEMENT ET NETTOYAGE DES DONNEES
# ==============================================================================

def charger_donnees(path):
    titre("1. CHARGEMENT DES DONNEES")
    df = pd.read_excel(path)
    log(f"Fichier chargé : {path}")
    log(f"Dimensions brutes : {df.shape[0]} lignes x {df.shape[1]} colonnes")

    # --- Renommage des colonnes (par position, dans l'ordre du questionnaire) ---
    nouveaux_noms = [
        "consentement", "age", "sexe", "statut_matrimonial", "autre_statut_precision",
        "acces_sante_mentale", "milieu_residence", "argent_poche", "note_gad_intro",
        "gad1", "gad2", "gad3", "gad4", "gad5", "gad6", "gad7",
        "entite_formation", "annee_etude", "expo_stress_clinique", "note_pss_intro",
        "pss1", "pss2", "pss3", "pss4", "pss5", "pss6", "pss7", "pss8", "pss9", "pss10",
        "soutien_social", "qualite_sommeil", "heures_sommeil", "difficulte_endormissement",
        "reveils_nocturnes", "consommation_substances", "activite_physique",
        "uuid", "root_uuid",
    ]
    assert len(nouveaux_noms) == df.shape[1], (
        "Le nombre de colonnes du fichier ne correspond plus à celui attendu par le "
        "script : vérifie que le questionnaire n'a pas changé, ou adapte la liste "
        "'nouveaux_noms' ci-dessus."
    )
    df.columns = nouveaux_noms

    # Colonnes inutiles pour l'analyse (notes de section ODK, identifiants techniques)
    df = df.drop(columns=["note_gad_intro", "note_pss_intro", "uuid", "root_uuid",
                           "autre_statut_precision"])

    # Nettoyage des espaces superflus dans les variables textuelles
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})

    return df


def nettoyer_donnees(df):
    titre("2. NETTOYAGE ET RECODAGE DES VARIABLES")

    # --- Consentement : ne garder que les répondants ayant accepté -------------
    n_avant = len(df)
    df = df[df["consentement"] == "Oui"].copy()
    log(f"Consentement 'Non' exclus : {n_avant - len(df)} -> échantillon analysé : {len(df)}")

    # --- Age : correction des valeurs aberrantes (années de naissance saisies
    #     par erreur à la place de l'âge, ex : 2005 au lieu de 21 ans) ----------
    age_num = pd.to_numeric(df["age"], errors="coerce")
    n_aberrant = ((age_num < 14) | (age_num > 45)).sum()
    if n_aberrant > 0:
        log(f"Ages implausibles mis en donnée manquante (n={n_aberrant}) : "
            f"{sorted(age_num[(age_num < 14) | (age_num > 45)].dropna().unique().tolist())}")
    age_num = age_num.where((age_num >= 14) & (age_num <= 45))
    df["age"] = age_num

    # --- Argent de poche : conversion numérique + info sur les données manquantes
    df["argent_poche"] = pd.to_numeric(df["argent_poche"], errors="coerce")
    pct_manquant = df["argent_poche"].isna().mean() * 100
    log(f"Variable 'argent de poche' : {pct_manquant:.1f}% de valeurs manquantes "
        f"(question probablement sensible / sautée) -> conservée en analyse en cas disponibles.")

    # --- Regroupement de modalités trop rares pour des tests fiables -----------
    df["statut_matrimonial"] = df["statut_matrimonial"].replace(
        {"Marié": "Marié/Autre", "Autre": "Marié/Autre"}
    )
    df["consommation_substances"] = df["consommation_substances"].apply(
        lambda x: "Aucune" if x == "Aucune" else ("Au moins une substance" if pd.notna(x) else np.nan)
    )

    return df


# ==============================================================================
# 3. CALCUL DES SCORES : GAD-7 (anxiété) ET PSS-10 (stress perçu)
# ==============================================================================

GAD_ITEMS = [f"gad{i}" for i in range(1, 8)]
PSS_ITEMS = [f"pss{i}" for i in range(1, 11)]
PSS_ITEMS_INVERSES = ["pss4", "pss5", "pss7", "pss8"]  # items formulés positivement

GAD_MAP = {
    "Jamais": 0,
    "Plusieurs jours": 1,
    "Plus de la moitié des jours": 2,
    "Presque tous les jours": 3,
}
PSS_MAP = {
    "Jamais": 0,
    "Presque jamais": 1,
    "Parfois": 2,
    "Assez souvent": 3,
    "Souvent": 4,
}


def calculer_scores(df):
    titre("3. CALCUL DES SCORES GAD-7 (ANXIETE) ET PSS-10 (STRESS PERCU)")

    # ---------------------------- GAD-7 -----------------------------------
    for item in GAD_ITEMS:
        df[item + "_score"] = df[item].map(GAD_MAP)
    gad_score_cols = [c + "_score" for c in GAD_ITEMS]
    df["gad7_total"] = df[gad_score_cols].sum(axis=1, min_count=7)  # NaN si un item manque

    def cat_gad7(score):
        if pd.isna(score):
            return np.nan
        if score <= 4:
            return "Minimale (0-4)"
        elif score <= 9:
            return "Légère (5-9)"
        elif score <= 14:
            return "Modérée (10-14)"
        else:
            return "Sévère (15-21)"

    df["gad7_categorie"] = df["gad7_total"].apply(cat_gad7)

    for seuil in SEUILS_GAD7:
        df[f"anxiete_seuil{seuil}"] = np.where(
            df["gad7_total"].isna(), np.nan, (df["gad7_total"] >= seuil).astype(float)
        )

    n_valide = df["gad7_total"].notna().sum()
    log(f"Score GAD-7 calculé pour {n_valide}/{len(df)} étudiants "
        f"({len(df) - n_valide} non calculables : item(s) manquant(s)).")

    # ---------------------------- PSS-10 -----------------------------------
    for item in PSS_ITEMS:
        score = df[item].map(PSS_MAP)
        if item in PSS_ITEMS_INVERSES:
            score = 4 - score  # inversion des items formulés positivement
        df[item + "_score"] = score
    pss_score_cols = [c + "_score" for c in PSS_ITEMS]
    df["pss10_total"] = df[pss_score_cols].sum(axis=1, min_count=10)

    def cat_pss10(score):
        if pd.isna(score):
            return np.nan
        if score <= 13:
            return "Faible (0-13)"
        elif score <= 26:
            return "Modéré (14-26)"
        else:
            return "Élevé (27-40)"

    df["pss10_categorie"] = df["pss10_total"].apply(cat_pss10)

    n_valide_pss = df["pss10_total"].notna().sum()
    log(f"Score PSS-10 calculé pour {n_valide_pss}/{len(df)} étudiants.")

    return df


# ==============================================================================
# 4. OBJECTIF 1 : PREVALENCE DE L'ANXIETE
# ==============================================================================

def ic95_proportion(n_succes, n_total):
    """IC95% d'une proportion (méthode de Wilson, recommandée sur petits effectifs)."""
    if n_total == 0:
        return (np.nan, np.nan)
    borne_inf, borne_sup = proportion_confint(n_succes, n_total, alpha=0.05, method="wilson")
    return borne_inf * 100, borne_sup * 100


def tableau_caracteristiques(df):
    """Tableau 1 : description de l'échantillon (variables sociodémographiques)."""
    lignes = []

    def ajouter_quanti(nom, serie):
        s = serie.dropna()
        lignes.append({"Variable": nom, "Modalité": "Moyenne ± ET",
                        "n": len(s), "%": f"{s.mean():.1f} ± {s.std():.1f}"})

    def ajouter_quali(nom, serie):
        eff = serie.value_counts(dropna=True)
        total = eff.sum()
        for modalite, n in eff.items():
            lignes.append({"Variable": nom, "Modalité": modalite,
                            "n": int(n), "%": f"{100 * n / total:.1f}"})

    ajouter_quanti("Âge (années)", df["age"])
    ajouter_quali("Sexe", df["sexe"])
    ajouter_quali("Statut matrimonial", df["statut_matrimonial"])
    ajouter_quali("Milieu de résidence", df["milieu_residence"])
    ajouter_quali("Entité de formation", df["entite_formation"])
    ajouter_quali("Année d'étude", df["annee_etude"])
    ajouter_quali("Accès aux services de santé mentale", df["acces_sante_mentale"])
    ajouter_quali("Exposition au stress clinique", df["expo_stress_clinique"])
    ajouter_quali("Niveau de soutien social", df["soutien_social"])
    ajouter_quali("Qualité du sommeil", df["qualite_sommeil"])
    ajouter_quali("Activité physique", df["activite_physique"])
    ajouter_quali("Consommation de substances", df["consommation_substances"])
    ajouter_quali("Catégorie de stress perçu (PSS-10)", df["pss10_categorie"])

    return pd.DataFrame(lignes)


def objectif1_prevalence(df):
    titre("OBJECTIF 1 : PREVALENCE DE L'ANXIETE (SCORE GAD-7)")
    resultats = {}

    resultats["tableau_caracteristiques"] = tableau_caracteristiques(df)

    # --- Statistiques descriptives du score GAD-7 -------------------------
    desc = df["gad7_total"].describe()
    log(f"\nScore GAD-7 (n={int(desc['count'])}) : "
        f"moyenne={desc['mean']:.2f} ± {desc['std']:.2f} | "
        f"médiane={df['gad7_total'].median():.1f} | min={desc['min']:.0f} | max={desc['max']:.0f}")

    # --- Répartition par catégorie de sévérité -----------------------------
    cat_order = ["Minimale (0-4)", "Légère (5-9)", "Modérée (10-14)", "Sévère (15-21)"]
    tab_cat = df["gad7_categorie"].value_counts(dropna=True).reindex(cat_order).fillna(0).astype(int)
    tab_cat_pct = (100 * tab_cat / tab_cat.sum()).round(1)
    resultats["repartition_categories_gad7"] = pd.DataFrame(
        {"Catégorie GAD-7": tab_cat.index, "n": tab_cat.values, "%": tab_cat_pct.values}
    )
    log("\nRépartition par catégorie de sévérité (GAD-7) :")
    log(resultats["repartition_categories_gad7"].to_string(index=False))

    # --- Prévalence aux deux seuils, avec IC95% -----------------------------
    lignes_prev = []
    for seuil in SEUILS_GAD7:
        col = f"anxiete_seuil{seuil}"
        n_total = df[col].notna().sum()
        n_anxieux = df[col].sum()
        prev = 100 * n_anxieux / n_total
        ic_inf, ic_sup = ic95_proportion(n_anxieux, n_total)
        lignes_prev.append({
            "Seuil GAD-7": f">= {seuil}",
            "n anxieux": int(n_anxieux),
            "n total": int(n_total),
            "Prévalence (%)": round(prev, 1),
            "IC95% inf": round(ic_inf, 1),
            "IC95% sup": round(ic_sup, 1),
        })
        log(f"\nPrévalence de l'anxiété (seuil >= {seuil}) : "
            f"{n_anxieux:.0f}/{n_total} = {prev:.1f}% [IC95% {ic_inf:.1f}-{ic_sup:.1f}]")
    resultats["prevalence_anxiete"] = pd.DataFrame(lignes_prev)

    # ---------------------------- GRAPHIQUES ----------------------------
    # 1) Histogramme du score GAD-7
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(df["gad7_total"].dropna(), bins=range(0, 23), color="#4C72B0", ax=ax)
    for seuil, style in zip(SEUILS_GAD7, ["--", "-"]):
        ax.axvline(seuil, color="red", linestyle=style, linewidth=1.5, label=f"Seuil = {seuil}")
    ax.set_xlabel("Score total GAD-7")
    ax.set_ylabel("Nombre d'étudiants")
    ax.set_title("Distribution du score GAD-7")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "01_distribution_score_gad7.png"))

    # 2) Barres : répartition par catégorie de sévérité
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=tab_cat_pct.index, y=tab_cat_pct.values, ax=ax, color="#55A868")
    ax.set_ylabel("Pourcentage (%)")
    ax.set_xlabel("")
    ax.set_title("Répartition des étudiants par sévérité de l'anxiété (GAD-7)")
    for i, v in enumerate(tab_cat_pct.values):
        ax.text(i, v + 0.5, f"{v}%", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_categories_severite_gad7.png"))

    # 3) Camemberts : proportion anxieux / non-anxieux, pour chaque seuil
    fig, axes = plt.subplots(1, len(SEUILS_GAD7), figsize=(5 * len(SEUILS_GAD7), 4.5))
    if len(SEUILS_GAD7) == 1:
        axes = [axes]
    for ax, seuil in zip(axes, SEUILS_GAD7):
        col = f"anxiete_seuil{seuil}"
        vc = df[col].value_counts(dropna=True).sort_index()
        labels = ["Non anxieux", "Anxieux"]
        ax.pie(vc.values, labels=labels, autopct="%1.1f%%", startangle=90,
               colors=["#8DA0CB", "#FC8D62"])
        ax.set_title(f"Seuil GAD-7 >= {seuil}")
    fig.suptitle("Proportion d'étudiants anxieux selon le seuil retenu")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "03_prevalence_anxiete_camembert.png"))

    log(f"\nFigures enregistrées dans : {FIG_DIR}")
    return resultats


# ==============================================================================
# 5. OBJECTIF 2 : FACTEURS ASSOCIES A L'ANXIETE
# ==============================================================================

# Variables qualitatives candidates (facteurs potentiels)
VARIABLES_QUALI = [
    "sexe", "statut_matrimonial", "milieu_residence", "entite_formation",
    "annee_etude", "acces_sante_mentale", "expo_stress_clinique",
    "soutien_social", "qualite_sommeil", "heures_sommeil",
    "difficulte_endormissement", "reveils_nocturnes",
    "consommation_substances", "activite_physique",
    # NB : pss10_categorie n'est PAS incluse ici : elle est dérivée du même score
    # que pss10_total (variable quantitative ci-dessous). Les inclure toutes les
    # deux dans le même modèle multivarié crée une redondance qui fait "exploser"
    # les OR (séparation quasi-parfaite, IC absurdes type [0 - inf]).
]
# Variables quantitatives candidates
VARIABLES_QUANTI = ["age", "argent_poche", "pss10_total"]


def test_association_quali(df, var, outcome):
    """Khi2 (ou Fisher exact si tableau 2x2 avec effectif attendu < 5)."""
    sous = df[[var, outcome]].dropna()
    table = pd.crosstab(sous[var], sous[outcome])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, "Test impossible (modalité manquante)", table
    chi2, p, dof, attendu = stats.chi2_contingency(table)
    if table.shape == (2, 2) and (attendu < 5).any():
        _, p = stats.fisher_exact(table)
        test_nom = "Fisher exact"
    else:
        test_nom = "Khi2"
    return p, test_nom, table


def or_crude_2x2(df, var, outcome, modalite_reference=None):
    """OR brut + IC95% (méthode de Woolf, correction de Haldane si case=0)."""
    sous = df[[var, outcome]].dropna()
    modalites = sous[var].unique().tolist()
    if len(modalites) != 2:
        return None
    if modalite_reference is None:
        modalite_reference = sorted(modalites)[0]
    modalite_exposee = [m for m in modalites if m != modalite_reference][0]

    a = ((sous[var] == modalite_exposee) & (sous[outcome] == 1)).sum()
    b = ((sous[var] == modalite_exposee) & (sous[outcome] == 0)).sum()
    c = ((sous[var] == modalite_reference) & (sous[outcome] == 1)).sum()
    d = ((sous[var] == modalite_reference) & (sous[outcome] == 0)).sum()

    if 0 in (a, b, c, d):
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5  # correction de Haldane-Anscombe

    orv = (a * d) / (b * c)
    se_log_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ic_inf = np.exp(np.log(orv) - 1.96 * se_log_or)
    ic_sup = np.exp(np.log(orv) + 1.96 * se_log_or)
    return {
        "Modalité exposée": modalite_exposee, "Référence": modalite_reference,
        "OR brut": round(orv, 2), "IC95% inf": round(ic_inf, 2), "IC95% sup": round(ic_sup, 2),
    }


def test_association_quanti(df, var, outcome):
    """Compare les scores entre anxieux/non-anxieux : t-test si normalité, sinon Mann-Whitney."""
    sous = df[[var, outcome]].dropna()
    g0 = sous.loc[sous[outcome] == 0, var]
    g1 = sous.loc[sous[outcome] == 1, var]
    if len(g0) < 3 or len(g1) < 3:
        return np.nan, "Test impossible (effectif insuffisant)", g0, g1
    normal0 = stats.shapiro(g0.sample(min(len(g0), 500), random_state=0)).pvalue > 0.05
    normal1 = stats.shapiro(g1.sample(min(len(g1), 500), random_state=0)).pvalue > 0.05
    if normal0 and normal1:
        _, p = stats.ttest_ind(g0, g1, equal_var=False)
        test_nom = "t-test (Welch)"
    else:
        _, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        test_nom = "Mann-Whitney U"
    return p, test_nom, g0, g1


def analyse_bivariee(df, outcome):
    """Boucle sur toutes les variables candidates -> tableau récapitulatif."""
    lignes = []

    for var in VARIABLES_QUALI:
        p, test_nom, table = test_association_quali(df, var, outcome)
        info_or = ""
        if isinstance(p, float) and table.shape == (2, 2):
            orr = or_crude_2x2(df, var, outcome)
            if orr:
                info_or = f"OR={orr['OR brut']} [{orr['IC95% inf']}-{orr['IC95% sup']}] " \
                          f"({orr['Modalité exposée']} vs {orr['Référence']})"
        lignes.append({
            "Variable": var, "Type": "Qualitative", "Test": test_nom,
            "p-value": round(p, 4) if isinstance(p, float) else p,
            "OR brut (IC95%)": info_or,
        })

    for var in VARIABLES_QUANTI:
        p, test_nom, g0, g1 = test_association_quanti(df, var, outcome)
        moyennes = ""
        if isinstance(p, float):
            moyennes = f"Non anxieux: {g0.mean():.1f}±{g0.std():.1f} | " \
                       f"Anxieux: {g1.mean():.1f}±{g1.std():.1f}"
        lignes.append({
            "Variable": var, "Type": "Quantitative", "Test": test_nom,
            "p-value": round(p, 4) if isinstance(p, float) else p,
            "OR brut (IC95%)": moyennes,
        })

    tab = pd.DataFrame(lignes).sort_values("p-value", na_position="last")
    tab["Significatif (p<0.05)"] = tab["p-value"].apply(
        lambda p: "Oui" if isinstance(p, float) and p < SEUIL_SIGNIFICATIVITE else ""
    )
    return tab


def regression_logistique_multivariee(df, outcome, tab_bivariee):
    """Régression logistique multivariée sur les variables p<0.20 en bivarié."""
    variables_retenues = tab_bivariee.loc[
        (tab_bivariee["p-value"].apply(lambda p: isinstance(p, float) and p < SEUIL_SELECTION_BIVARIEE)),
        "Variable",
    ].tolist()

    if len(variables_retenues) == 0:
        log("Aucune variable avec p<0.20 en bivarié : régression multivariée non réalisée.")
        return None, []

    log(f"\nVariables retenues pour le modèle multivarié (p<0.20 en bivarié) : "
        f"{variables_retenues}")

    colonnes = variables_retenues + [outcome]
    sous = df[colonnes].dropna().copy()

    # Encodage des variables qualitatives (référence = modalité la plus fréquente)
    quali_retenues = [v for v in variables_retenues if v in VARIABLES_QUALI]
    for v in quali_retenues:
        modalite_ref = sous[v].value_counts().idxmax()
        cats = [c for c in sous[v].unique() if c != modalite_ref]
        sous[v] = pd.Categorical(sous[v], categories=[modalite_ref] + cats)

    X = pd.get_dummies(sous[variables_retenues], drop_first=True, dtype=float)
    X = sm.add_constant(X)
    y = sous[outcome].astype(float)

    try:
        modele = sm.Logit(y, X).fit(disp=0, maxiter=200)
    except Exception as e:
        log(f"Échec de la régression logistique multivariée : {e}")
        return None, variables_retenues

    resultats = pd.DataFrame({
        "OR ajusté": np.exp(modele.params),
        "IC95% inf": np.exp(modele.conf_int()[0]),
        "IC95% sup": np.exp(modele.conf_int()[1]),
        "p-value": modele.pvalues,
    }).round(3)
    resultats = resultats.drop(index="const", errors="ignore").reset_index()
    resultats = resultats.rename(columns={"index": "Variable (modalité vs référence)"})
    resultats["Significatif (p<0.05)"] = resultats["p-value"].apply(
        lambda p: "Oui" if p < SEUIL_SIGNIFICATIVITE else ""
    )

    log(f"\nModèle logistique multivarié -- n={int(modele.nobs)} | "
        f"Pseudo R² (McFadden)={modele.prsquared:.3f}")
    log(resultats.to_string(index=False))

    return resultats, variables_retenues


def figure_facteurs_significatifs(tab_bivariee, seuil, suffixe):
    """Diagramme en barres des p-values des variables testées (repère visuel rapide)."""
    tab = tab_bivariee.dropna(subset=["p-value"]).copy()
    tab = tab[tab["p-value"].apply(lambda x: isinstance(x, float))]
    tab = tab.sort_values("p-value")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(tab))))
    couleurs = ["#C44E52" if p < SEUIL_SIGNIFICATIVITE else "#8C8C8C" for p in tab["p-value"]]
    ax.barh(tab["Variable"], tab["p-value"], color=couleurs)
    ax.axvline(SEUIL_SIGNIFICATIVITE, color="black", linestyle="--", linewidth=1,
               label="seuil p=0.05")
    ax.invert_yaxis()
    ax.set_xlabel("p-value (test bivarié)")
    ax.set_title(f"Facteurs associés à l'anxiété (seuil GAD-7 >= {seuil})\n"
                 f"rouge = significatif (p<0.05)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"04_pvalues_facteurs_associes_{suffixe}.png"))


def objectif2_facteurs_associes(df):
    titre("OBJECTIF 2 : FACTEURS ASSOCIES A L'ANXIETE")
    resultats = {}

    for seuil in SEUILS_GAD7:
        outcome = f"anxiete_seuil{seuil}"
        log(f"\n--- Analyse bivariée (seuil GAD-7 >= {seuil}) ---")
        tab_biv = analyse_bivariee(df, outcome)
        log(tab_biv.to_string(index=False))
        resultats[f"bivarie_seuil{seuil}"] = tab_biv
        figure_facteurs_significatifs(tab_biv, seuil, f"seuil{seuil}")

        log(f"\n--- Régression logistique multivariée (seuil GAD-7 >= {seuil}) ---")
        tab_multi, vars_retenues = regression_logistique_multivariee(df, outcome, tab_biv)
        if tab_multi is not None:
            resultats[f"multivarie_seuil{seuil}"] = tab_multi

    return resultats


# ==============================================================================
# 6. OBJECTIF 3 : PROFIL-TYPE DE L'ETUDIANT ANXIEUX
# ==============================================================================

def profil_type(df, outcome, seuil):
    """Compare, pour chaque variable, la modalité dominante chez les anxieux
    vs les non-anxieux, et construit une phrase de synthèse du profil-type."""
    lignes = []
    for var in VARIABLES_QUALI:
        sous = df[[var, outcome]].dropna()
        if sous.empty:
            continue
        tab_pct = pd.crosstab(sous[var], sous[outcome], normalize="columns") * 100
        if 1.0 not in tab_pct.columns or 0.0 not in tab_pct.columns:
            continue
        modalite_dominante_anx = tab_pct[1.0].idxmax()
        pct_anx = tab_pct.loc[modalite_dominante_anx, 1.0]
        pct_non_anx = tab_pct.loc[modalite_dominante_anx, 0.0]
        lignes.append({
            "Variable": var,
            "Modalité la plus fréquente chez les anxieux": modalite_dominante_anx,
            "% chez anxieux": round(pct_anx, 1),
            "% chez non-anxieux": round(pct_non_anx, 1),
        })
    tab = pd.DataFrame(lignes)

    # Age moyen comparé
    age_anx = df.loc[df[outcome] == 1, "age"].mean()
    age_non_anx = df.loc[df[outcome] == 0, "age"].mean()

    log(f"\nÂge moyen -- anxieux : {age_anx:.1f} ans | non-anxieux : {age_non_anx:.1f} ans")
    log(tab.to_string(index=False))

    return tab


def synthese_texte_profil(tab_profil, df, outcome, seuil):
    """Génère une phrase de synthèse en français à partir du tableau de profil-type."""
    elements = []
    for _, row in tab_profil.iterrows():
        ecart = row["% chez anxieux"] - row["% chez non-anxieux"]
        if abs(ecart) >= 5:  # ne retenir que les écarts les plus marqués
            elements.append(f"{row['Variable']} = « {row['Modalité la plus fréquente chez les anxieux']} » "
                             f"({row['% chez anxieux']}% vs {row['% chez non-anxieux']}% chez les non-anxieux)")
    phrase = (
        f"Portrait-type de l'étudiant anxieux (seuil GAD-7 >= {seuil}) : "
        + "; ".join(elements) + "."
        if elements else
        f"Aucune caractéristique ne se démarque nettement (seuil GAD-7 >= {seuil})."
    )
    log("\n" + phrase)
    return phrase


def objectif3_profil_type(df):
    titre("OBJECTIF 3 : PROFIL-TYPE DE L'ETUDIANT ANXIEUX")
    resultats = {}
    for seuil in SEUILS_GAD7:
        outcome = f"anxiete_seuil{seuil}"
        log(f"\n--- Profil-type (seuil GAD-7 >= {seuil}) ---")
        tab = profil_type(df, outcome, seuil)
        resultats[f"profil_seuil{seuil}"] = tab
        synthese_texte_profil(tab, df, outcome, seuil)
    return resultats


# ==============================================================================
# 7. EXPORT DES RESULTATS
# ==============================================================================

def exporter_excel(dictionnaire_tableaux, chemin):
    with pd.ExcelWriter(chemin, engine="openpyxl") as writer:
        for nom_feuille, tab in dictionnaire_tableaux.items():
            nom_feuille_court = nom_feuille[:31]  # limite Excel = 31 caractères
            tab.to_excel(writer, sheet_name=nom_feuille_court, index=False)
    log(f"\nTous les tableaux ont été exportés dans : {chemin}")


# ==============================================================================
# PROGRAMME PRINCIPAL
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyse de prévalence et des facteurs associés à l'anxiété (GAD-7 / PSS-10)."
    )
    parser.add_argument(
        "--data", type=str, default=DATA_PATH_PAR_DEFAUT,
        help="Chemin vers le fichier .xlsx de données (par défaut : %(default)s)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Fichier de données introuvable : '{args.data}'.\n"
            f"-> Vérifie le chemin, ou précise-le avec : "
            f"python {os.path.basename(__file__)} --data \"chemin/vers/ton_fichier.xlsx\""
        )

    df = charger_donnees(args.data)
    df = nettoyer_donnees(df)
    df = calculer_scores(df)

    tous_les_tableaux = {}
    tous_les_tableaux.update(objectif1_prevalence(df))
    tous_les_tableaux.update(objectif2_facteurs_associes(df))
    tous_les_tableaux.update(objectif3_profil_type(df))

    exporter_excel(tous_les_tableaux, os.path.join(OUTPUT_DIR, "tableaux_resultats.xlsx"))

    with open(os.path.join(OUTPUT_DIR, "log_analyse.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))

    # Sauvegarde de la base nettoyée + scores calculés (utile pour vérifications)
    df.to_excel(os.path.join(OUTPUT_DIR, "base_nettoyee_avec_scores.xlsx"), index=False)

    titre("ANALYSE TERMINEE")
    log(f"Tous les résultats sont dans le dossier : {os.path.abspath(OUTPUT_DIR)}")

    # --- Affichage à l'écran de tous les graphiques générés ---------------
    log("\nAffichage des graphiques à l'écran : ferme toutes les fenêtres "
        "de graphiques pour terminer le script.")
    plt.show()


if __name__ == "__main__":
    main()
