import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from PIL import Image
import os

# --- CONFIGURATION INITIALE & THÈME GLOBAL ---
st.set_page_config(
    page_title="Asaad Saadi | Portefeuille Data & CV",
    page_icon="⚡", # Icône plus percutante
    layout="wide"
)

# --- CSS GLOBAL STYLÉ (Dark Mode Overlays) ---
# Ce CSS ajoute un style Dark Mode avec des polices modernes et des cartes élégantes
st.markdown("""
    <style>
        /* Couleurs du thème */
        :root {
            --primary-color: #00FFFF; /* Cyan Électrique */
            --background-color: #0E1117; /* Fond sombre */
            --secondary-background-color: #1F2430; /* Cartes sombres */
            --text-color: #FAFAFA; /* Texte blanc */
            --accent-color: #FF8C00; /* Orange Vif */
            --danger-color: #FF6347; /* Rouge Tomate */
        }
        
        /* Styles pour le CV */
        .cv-header-card {
            background-color: var(--secondary-background-color);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
            margin-bottom: 30px;
        }

        /* Cartes pour les projets */
        .project-card {
            background-color: var(--secondary-background-color);
            border-left: 5px solid #FFFFFF; /* MODIFIÉ EN BLANC PUR */
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .project-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5);
        }
        
        /* Styles pour les KPIs du Dashboard */
        .kpi-card {
            background-color: var(--secondary-background-color);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
            transition: all 0.3s ease;
        }
        .kpi-value {
            font-size: 2.5em;
            font-weight: bold;
            color: var(--accent-color); 
        }
        .kpi-label {
            font-size: 0.9em;
            color: #AAAAAA;
            margin-top: 5px;
        }
        
        /* Style des barres de progression */
        .stProgress > div > div > div > div {
            background-color: var(--primary-color); 
        }
        
    </style>
""", unsafe_allow_html=True)

# --- FONCTIONS DE CHARGEMENT ET DE PRÉPARATION DES DONNÉES (Pour le Dashboard) ---

@st.cache_data
def load_simulation_data():
    """Génère des données simulées pour la régularité du métro (pour les graphiques temporels)."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq='D')
    lignes = [str(i) for i in range(1, 15)]
    data = []
    for ligne in lignes:
        base_reg = 98.0 if ligne in ['1', '14'] else (92.0 if ligne == '13' else 95.0)
        for date in dates:
            variation = np.random.normal(0, 1.5)
            is_weekend = date.weekday() >= 5
            factor = 0.5 if is_weekend else 0.0
            taux = min(100, max(0, base_reg + variation - factor))
            data.append({
                'Date': date, 'Ligne': ligne, 'Taux_Regularite': round(taux, 2),
                'Trafic': int(np.random.normal(500000, 50000))
            })
    df = pd.DataFrame(data)
    df['Mois'] = df['Date'].dt.month_name()
    df['Jour_Semaine'] = df['Date'].dt.day_name()
    ordre_jours = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['Jour_Semaine'] = pd.Categorical(df['Jour_Semaine'], categories=ordre_jours, ordered=True)
    return df

@st.cache_data
def load_real_csv_data(): 
    """Charge le fichier CSV des fontaines à eau (avec correction du chemin Windows)."""
    # VEUILLEZ VÉRIFIER QUE CE CHEMIN EST CORRECT SUR VOTRE MACHINE
    file_path = "fontaines-a-eau-dans-le-reseau-ratp.csv" 
    
    try:
        df = pd.read_csv(file_path, sep=';') 
        df['Ligne'] = df['Ligne'].astype(str)
        df = df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Erreur lors de la lecture du CSV ({file_path}): {e}. Vérifiez le chemin et le format du fichier.")
        return None

# Chargement des données au lancement de l'application
df_sim = load_simulation_data()
df_real = load_real_csv_data()

# --- BLOCS DE RENDU DES PAGES ---

def render_cv_page():
    """Rend la page du CV Interactif."""
    
    # Chargement de l'image
    img = None
    try:
        image_path = "profile.jpg" 
        img = Image.open(image_path)
    except FileNotFoundError:
        st.warning("L'image 'profile.jpg' est introuvable. Veuillez la placer dans le même répertoire que app.py.")
        
    # --- EN-TÊTE STYLISÉE ---
    st.markdown('<div class="cv-header-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 4], gap="medium")
    with col1:
        if img:
            st.image(img, width=200, use_column_width=False)
    with col2:
        st.title("Asaad Saadi")
        st.subheader("Étudiant en BUT Sciences des Données | Data Analyst Junior")
        st.markdown(f'<p style="color:#AAAAAA;">📍 Le Bourget (France) | 📧 saadi_asaad@outlook.fr | 📞 07 52 07 70 35</p>', unsafe_allow_html=True)
        st.markdown("[LinkedIn Professionnel](https://www.linkedin.com/in/Asaad%20Saadi)")

    st.markdown('</div>', unsafe_allow_html=True)

    # --- 1. À PROPOS ---
    st.subheader("🔥 À propos de moi")
    st.write(
        """
        Étudiant en BUT Sciences des Données à l'IUT de Paris Rives de Seine (2023-2026), 
        je recherche une alternance en chargé d'étude. 
        Passionné par la modélisation et la visualisation, j’ai développé des compétences solides en 
        statistiques, en programmation (Python/R) et en gestion de bases de données, cherchant toujours à transformer la donnée brute en information stratégique.
        """
    )

    # --- 2. COMPÉTENCES ---
    st.subheader("💡 Compétences Techniques")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Langages & Outils Data :**")
        st.write("`Python` (Pandas, Numpy, Plotly, Streamlit) / `R` / `SAS` / `SQL`")
        st.write("`Excel` (VBA) / `Access`")

    with col2:
        st.markdown("**Analyse & Visualisation :**")
        st.write("`Power BI` / Data Storytelling / Statistiques Descriptives")
        st.write("Modélisation de données / Analyse d’enquêtes")

    # --- 3. PROJETS UNIVERSITAIRES (AVEC LIEN INTERNE) ---
    st.subheader("💻 Projets Data & Développement")
    
    st.markdown("""
    <div class="project-card">
    <b>Tableau de Bord RATP (Streamlit Data Viz)</b><br>
    - Objectif : Créer une preuve de concept pour suivre la ponctualité du métro.<br>
    - Réalisation : Application Streamlit intégrant données simulées de régularité et données réelles de services (fontaines). Visualisation de KPIs, courbes temporelles, et cartographie interactive (Plotly).<br>
    - Compétences :Python, Streamlit, Pandas, Plotly.
    </div>
    """, unsafe_allow_html=True)

    # Bouton pour basculer vers le Dashboard
    if st.button("🚀 Accéder au Tableau de Bord RATP (Voir la Data Viz)", type="primary"):
        st.session_state.page = "Dashboard RATP"
        st.rerun() 

    st.markdown("---")
    
    st.markdown("""
    <div class="project-card">
    <b>Étude sur les Jeux Olympiques (2023-2024)</b><br>
    - Collecte, nettoyage et création de bases de données volumineuses.<br>
    - Réalisation de statistiques et graphiques complexes.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="project-card">
    <b>Traitement de fichiers CSV (2023)</b><br>
    - Nettoyage et transformation des données avec Python pour préparer l'analyse.
    </div>
    """, unsafe_allow_html=True)

    # --- 4. FORMATION ---
    st.subheader("🎓 Formation")
    st.write("---")
    st.markdown("**BUT Sciences des Données** - IUT Rives de Seine Paris (2023 - 2026)")
    st.markdown("**Baccalauréat Général** - Lycée Germain Tillon, Le Bourget (2022)")

    # --- 5. LANGUES ---
    st.subheader("🌐 Langues")
    st.write("Français : Langue maternelle")
    st.progress(1.0)
    st.write("Arabe : Niveau C1")
    st.progress(0.85)
    st.write("Anglais : Niveau B1")
    st.progress(0.6)

    st.caption("Fait avec Streamlit pour un affichage dynamique et moderne.")

def render_dashboard_page(df_sim, df_real):
    """Rend la page du Tableau de Bord RATP (Look stylé)."""
    
    st.title("🚇 Tableau de Bord RATP : Qualité de Service (POC)")
    st.markdown("Analyse combinée de la **Régularité (Simulée)** et des **Services (Réels)**. Thème sombre pour un impact maximal.")
    
    # --- FILTRES (RÉPÉTÉS DANS LA SIDEBAR POUR LE DASHBOARD) ---
    liste_lignes = sorted(df_sim['Ligne'].unique())
    st.sidebar.header("🔍 Filtres d'Analyse")

    choix_lignes = st.sidebar.multiselect(
        "Choisir les lignes :",
        options=liste_lignes,
        default=['1', '14', '13'] # Mettre 14 en défaut pour montrer la différence simulée
    )

    min_date = df_sim['Date'].min()
    max_date = df_sim['Date'].max()
    date_range = st.sidebar.date_input(
        "Période d'analyse (Ponctualité) :",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if not choix_lignes:
        st.warning("Veuillez sélectionner au moins une ligne.")
        return

    # --- FILTRAGE DES DONNÉES ---
    mask_sim = (df_sim['Date'].dt.date >= date_range[0]) & \
               (df_sim['Date'].dt.date <= date_range[1]) & \
               (df_sim['Ligne'].isin(choix_lignes))
    df_sim_filtered = df_sim[mask_sim]

    if df_real is not None:
        df_real_filtered = df_real[df_real['Ligne'].isin(choix_lignes)]
    else:
        df_real_filtered = None

    # --- KPIs STYLÉS (Cartes CSS personnalisées) ---
    col1, col2, col3 = st.columns(3)
    avg_reg = df_sim_filtered['Taux_Regularite'].mean()
    min_reg = df_sim_filtered['Taux_Regularite'].min()
    nb_fontaines = len(df_real_filtered) if df_real_filtered is not None else 0

    # KPI 1 : Régularité Moyenne
    col1.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">RÉGULARITÉ MOYENNE (Sim.)</div>
            <div class="kpi-value" style="color:#00C080;">{avg_reg:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)
    
    # KPI 2 : Pire Performance
    col2.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">PIRE PERFORMANCE JOURNALIÈRE (Sim.)</div>
            <div class="kpi-value" style="color:#FF4B4B;">{min_reg:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

    # KPI 3 : Services Réels
    col3.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">SERVICES RÉELS (Fontaines)</div>
            <div class="kpi-value">{nb_fontaines} 🚰</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True) # Espace

    # --- Onglets ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Évolution Temporelle", 
        "🏆 Comparaison Lignes", 
        "📅 Analyse Hebdomadaire", 
        "🗺️ Carte des Services (CSV)"
    ])

    with tab1:
        st.subheader("Suivi de la performance jour après jour")
        fig_line = px.line(
            df_sim_filtered, x='Date', y='Taux_Regularite', color='Ligne',
            title="Taux de régularité journalier par ligne (Thème Sombre)",
            labels={'Taux_Regularite': 'Régularité (%)'},
            template="plotly_dark", # Force le thème sombre
            color_discrete_sequence=px.colors.sequential.Plasma # Utilisation d'une palette vibrante
        )
        fig_line.add_hline(y=95, line_dash="dash", line_color="#00C080", annotation_text="Objectif 95%", annotation_position="bottom right")
        st.plotly_chart(fig_line, use_container_width=True)

    with tab2:
        st.subheader("Classement des lignes sur la période sélectionnée")
        df_grouped = df_sim_filtered.groupby('Ligne')['Taux_Regularite'].mean().reset_index()
        df_grouped = df_grouped.sort_values(by='Taux_Regularite', ascending=False)
        fig_bar = px.bar(
            df_grouped, x='Taux_Regularite', y='Ligne', orientation='h',
            color='Taux_Regularite', 
            color_continuous_scale='Reds', # Palette pour le contraste sur fond sombre
            range_color=[90, 100], text_auto='.1f',
            labels={'Taux_Regularite': 'Régularité Moyenne (%)'},
            template="plotly_dark"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader("Visualisation des plages horaires et jours critiques")
        heatmap_data = df_sim_filtered.pivot_table(
            index='Ligne', columns='Jour_Semaine', values='Taux_Regularite', aggfunc='mean'
        )
        fig_heat = px.imshow(
            heatmap_data, 
            color_continuous_scale='Viridis', # Palette qui fonctionne bien en dark mode
            aspect="auto", text_auto='.1f',
            title="Régularité moyenne par Ligne et Jour de la semaine (Plus le chiffre est élevé, mieux c'est)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab4:
        st.subheader("🗺️ Localisation des Fontaines à eau (Services Réels)")
        
        if df_real is None:
            # Récupère le chemin d'accès dans la fonction pour l'affichage de l'erreur
            file_path_display = r"C:\Users\rayan.rami\Desktop\data viz web\fontaines-a-eau-dans-le-reseau-ratp.csv" 
            st.error(f"⚠️ Fichier CSV introuvable au chemin : {file_path_display}. Impossible d'afficher la carte.")
        
        elif df_real_filtered.empty:
            st.info("Aucune fontaine n'est répertoriée pour les lignes sélectionnées.")
        
        else:
            st.markdown(f"Affichage des **{len(df_real_filtered)}** fontaines disponibles pour ces lignes.")
            map_data = df_real_filtered.dropna(subset=['latitude', 'longitude'])
            
            # Utilisation de st.map est simple et s'intègre bien au dark mode de Streamlit
            st.map(map_data, zoom=11, size=20, color='#00C080')
            
            with st.expander("Voir le détail des stations (Tableau)"):
                st.dataframe(df_real_filtered[['Ligne', 'Station ou Gare', 'Adresse', 'Commune', 'En zone contrôlée ou non']])

    st.divider()
    st.caption("Projet Streamlit (POC Data Viz).")

# --- FONCTION PRINCIPALE DE L'APPLICATION ---
def main():
    # Gestion de l'état pour la navigation
    if 'page' not in st.session_state:
        st.session_state.page = "Mon CV"

    # Sélecteur de page dans la sidebar
    page_selection = st.sidebar.radio(
        "Navigation Principale",
        ["Mon CV", "Dashboard RATP"],
        index=0 if st.session_state.page == "Mon CV" else 1
    )
    
    st.session_state.page = page_selection

    # Affichage de la page sélectionnée
    if st.session_state.page == "Mon CV":
        render_cv_page()
    elif st.session_state.page == "Dashboard RATP":
        render_dashboard_page(df_sim, df_real)

if __name__ == "__main__":
    main()


