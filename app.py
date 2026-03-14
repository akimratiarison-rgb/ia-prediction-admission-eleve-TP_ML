import streamlit as st
import numpy as np
import pickle
import pandas as pd
import time

# css pour le fond sombre, j'ai trouvé comment faire ca sur la doc streamlit
st.markdown(
    """
    <style>
    .stApp {
        background-color: #2D3748!important;
    }
    h1, h2, h3, p, span, div, label, .stMarkdown {
        color: #E0E0E0 !important;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }
    input {
        background-color: #2D2D2D !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="Admission Etudiant", page_icon="👽")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# je charge le modele, si le fichier est pas la ca affiche une erreur
try:
    with open("model_etudiant.pkl", "rb") as f:
        data = pickle.load(f)
except FileNotFoundError:
    st.error("le fichier model_etudiant.pkl est introuvable")
    st.stop()

w = data["w"]
b = data["b"]

st.title("Assistant d'Admission")
st.write("Prédit si un étudiant sera admis ou non.")

# j'ai utilisé les tabs pour séparer les deux modes
onglet1, onglet2 = st.tabs(["Saisie manuelle", "Importer un fichier"])

#  premiere partie : saisie a la main 
with onglet1:

    note1 = st.number_input("Note de l'examen n°1 (0-100)", min_value=0, max_value=100, value=0)
    note2 = st.number_input("Note de l'examen n°2 (0-100)", min_value=0, max_value=100, value=0)

    if st.button("Lancer la prédiction"):
        with st.spinner("Analyse en cours..."):
            time.sleep(2.5)
            x = np.array([note1, note2])
            z = np.dot(x, w) + b
            prob = float(sigmoid(z))

        st.write(f"### Probabilité : {prob*100:.1f}%")
        st.progress(prob)

        # j'ai mis 0.4 apres plusieurs tests, 0.5 refusait trop de monde
        if prob >= 0.4:
            st.snow()
            st.success("ADMIS, félicitations !")
        else:
            st.error("REFUSÉ, les notes ne sont pas suffisantes.")
            st.warning("Une préparation supplémentaire est nécessaire.")

#  deuxieme partie : import fichier 
with onglet2:

    st.write("Le fichier doit avoir deux colonnes : note1 et note2")

    fichier = st.file_uploader("Choisis un fichier", type=["txt", "csv", "xlsx"])

    if fichier is not None:
        # je lis le fichier selon son extension
        try:
            if fichier.name.endswith(".xlsx"):
                df = pd.read_excel(fichier)
            else:
                # pour txt et csv j'essaie d'abord avec virgule
                try:
                    df = pd.read_csv(fichier)
                except:
                    df = pd.read_csv(fichier, sep=";")

            st.write("Aperçu :")
            st.dataframe(df.head())

            if st.button("Prédire pour tout le fichier"):
                with st.spinner("Calcul en cours..."):
                    time.sleep(1.5)

                    resultats = []
                    for _, row in df.iterrows():
                        x = np.array([row.iloc[0], row.iloc[1]])
                        z = np.dot(x, w) + b
                        prob = float(sigmoid(z))
                        # meme seuil que la saisie manuelle
                        decision = "ADMIS" if prob >= 0.4 else "REFUSÉ"
                        resultats.append({
                            "Note 1": row.iloc[0],
                            "Note 2": row.iloc[1],
                            "Probabilité (%)": round(prob * 100, 1),
                            "Décision": decision
                        })

                df_resultats = pd.DataFrame(resultats)
                st.write("Résultats :")
                st.dataframe(df_resultats)

                # petit résumé a la fin
                nb_admis = sum(1 for r in resultats if r["Décision"] == "ADMIS")
                st.write(f"**{nb_admis} admis** sur {len(resultats)} étudiants")

        except Exception as e:
            st.error(f"problème avec le fichier : {e}")
            st.info("vérifie que ton fichier a bien deux colonnes de notes")

st.divider()

st.caption("Projet ML - Régression Logistique - © 2026 by Kiki")