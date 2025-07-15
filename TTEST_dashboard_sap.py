import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import io
import re
import plotly.figure_factory as ff
import scipy # Ajouté pour résoudre ImportError avec create_distplot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --- Chemins vers vos fichiers de données ---
# ATTENTION : Ces chemins ont été mis à jour pour être RELATIFS.
# Cela signifie que les fichiers Excel/CSV doivent se trouver dans le MÊME dossier
# que ce script Python lorsque vous le déployez (par exemple, sur GitHub pour Streamlit Community Cloud).
DATA_PATHS = {
    "memory": "memory_final_cleaned_clean.xlsx",
    "hitlist_db": "HITLIST_DATABASE_final_cleaned_clean.xlsx",
    "times": "Times_final_cleaned_clean.xlsx",
    "tasktimes": "TASKTIMES_final_cleaned_clean.xlsx",
    "usertcode": "USERTCODE_cleaned.xlsx",
    "performance": "AL_GET_PERFORMANCE_final_cleaned_clean.xlsx",
    "sql_trace_summary": "performance_trace_summary_final_cleaned_clean.xlsx",
    "usr02": "usr02_data.xlsx",
}

# --- Configuration de la page Streamlit ---
st.set_page_config(layout="wide", page_title="Dashboard SAP Complet Multi-Sources")

# --- Fonctions de Nettoyage et Chargement des Données (avec cache) ---

def clean_string_column(series, default_value="Non défini"):
    """
    Nettoyage d'une série de type string : supprime espaces, remplace NaN/vides/caractères non imprimables.
    """
    cleaned_series = series.astype(str).str.strip()
    cleaned_series = cleaned_series.apply(lambda x: re.sub(r'[^\x20-\x7E\s]+', ' ', x).strip())
    cleaned_series = cleaned_series.replace({'nan': default_value, '': default_value, ' ': default_value})
    return cleaned_series

def clean_column_names(df):
    """
    Nettoyage des noms de colonnes : supprime les espaces, les caractères invisibles,
    et s'assure qu'ils sont valides pour l'accès.
    """
    new_columns = []
    for col in df.columns:
        cleaned_col = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', str(col)).strip()
        cleaned_col = re.sub(r'[^a-zA-Z0-9_]', '_', cleaned_col)
        cleaned_col = re.sub(r'_+', '_', cleaned_col)
        cleaned_col = cleaned_col.strip('_')
        new_columns.append(cleaned_col)
    df.columns = new_columns
    return df

def convert_mm_ss_to_seconds(time_str):
    """
    Convertit une chaîne de caractères au format MM:SS en secondes.
    Gère les cas où les minutes ou secondes sont manquantes ou invalides.
    """
    if pd.isna(time_str) or not isinstance(time_str, str):
        return 0
    try:
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return int(minutes * 60 + seconds)
        elif len(parts) == 1:
            return int(float(parts[0]))
        else:
            return 0
    except ValueError:
        return 0

def clean_numeric_with_comma(series):
    """
    Nettoyage d'une série de chaînes numériques qui peuvent contenir des virgules
    comme séparateurs de milliers ou décimaux, et conversion en float.
    """
    cleaned_series = series.astype(str).str.replace(' ', '').str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    return pd.to_numeric(cleaned_series, errors='coerce').fillna(0)


@st.cache_data
def load_and_process_data(file_key, path):
    """Charge et nettoie un fichier Excel/CSV."""
    df = pd.DataFrame()
    try:
        if path.lower().endswith('.xlsx'):
            df = pd.read_excel(path)
        elif path.lower().endswith('.csv'):
            df = pd.read_csv(path)
        else:
            st.error(f"Format de fichier non supporté pour {file_key}: {path}")
            return pd.DataFrame()

        df = clean_column_names(df.copy())

        # --- Gestion spécifique des types de données et valeurs manquantes ---
        if file_key == "memory":
            numeric_cols = ['MEMSUM', 'PRIVSUM', 'USEDBYTES', 'MAXBYTES', 'MAXBYTESDI', 'PRIVCOUNT', 'RESTCOUNT', 'COUNTER']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
            
            if 'ACCOUNT' in df.columns:
                df['ACCOUNT'] = clean_string_column(df['ACCOUNT'], 'Compte Inconnu')
            if 'MANDT' in df.columns:
                df['MANDT'] = clean_string_column(df['MANDT'], 'MANDT Inconnu')
            # TASKTYPE removal: Removed df['TASKTYPE'] = clean_string_column(df['TASKTYPE'], 'Type de Tâche Inconnu')

            if 'ENDDATE' in df.columns and 'ENDTIME' in df.columns:
                df['ENDTIME_STR'] = df['ENDTIME'].astype(str).str.zfill(6)
                df['FULL_DATETIME'] = pd.to_datetime(df['ENDDATE'].astype(str) + df['ENDTIME_STR'], format='%Y%m%d%H%M%S', errors='coerce')
                df.drop(columns=['ENDTIME_STR'], inplace=True, errors='ignore')
            elif 'FULL_DATETIME' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['FULL_DATETIME']):
                df['FULL_DATETIME'] = pd.to_datetime(df['FULL_DATETIME'], errors='coerce')
            
            subset_cols_memory = []
            if 'USEDBYTES' in df.columns:
                subset_cols_memory.append('USEDBYTES')
            if 'ACCOUNT' in df.columns:
                subset_cols_memory.append('ACCOUNT')
            if subset_cols_memory:
                df.dropna(subset=subset_cols_memory, inplace=True)


        elif file_key == "hitlist_db":
            numeric_cols = [
                'GENERATETI', 'REPLOADTI', 'CUALOADTI', 'DYNPLOADTI', 'QUETI', 'DDICTI', 'CPICTI',
                'LOCKCNT', 'LOCKTI', 'BTCSTEPNR', 'RESPTI', 'PROCTI', 'CPUTI', 'QUEUETI', 'ROLLWAITTI',
                'GUITIME', 'GUICNT', 'GUINETTIME', 'DBP_COUNT', 'DBP_TIME', 'DSQLCNT', 'QUECNT',
                'CPICCNT', 'SLI_CNT', 'TAB1DIRCNT', 'TAB1SEQCNT', 'TAB1UPDCNT', 'TAB2DIRCNT',
                'TAB2SEQCNT', 'TAB2UPDCNT', 'TAB3DIRCNT', 'TAB3SEQCNT', 'TAB3UPDCNT', 'TAB4DIRCNT',
                'TAB4SEQCNT', 'TAB4UPDCNT', 'TAB5DIRCNT', 'TAB5SEQCNT', 'TAB5UPDCNT',
                'READDIRCNT', 'READDIRTI', 'READDIRBUF', 'READDIRREC', 'READSEQCNT', 'READSEQTI',
                'READSEQBUF', 'READSEQREC', 'PHYREADCNT', 'INSCNT', 'INSTI', 'INSREC', 'PHYINSCNT',
                'UPDCNT', 'UPDTI', 'UPDREC', 'PHYUPDCNT', 'DELCNT', 'DELTI', 'DELREC', 'PHYDELCNT',
                'DBCALLS', 'COMMITTI', 'INPUTLEN', 'OUTPUTLEN', 'MAXROLL', 'MAXPAGE',
                'ROLLINCNT', 'ROLLINTI', 'ROLLOUTCNT', 'ROLLOUTTI', 'ROLLED_OUT', 'PRIVSUM',
                'USEDBYTES', 'MAXBYTES', 'MAXBYTESDI', 'RFCRECEIVE', 'RFCSEND',
                'RFCEXETIME', 'RFCCALLTIM', 'RFCCALLS', 'VMC_CALL_COUNT', 'VMC_CPU_TIME', 'VMC_ELAP_TIME'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
            
            if 'ENDDATE' in df.columns and 'ENDTIME' in df.columns:
                df['ENDTIME_STR'] = df['ENDTIME'].astype(str).str.zfill(6)
                df['FULL_DATETIME'] = pd.to_datetime(df['ENDDATE'].astype(str) + df['ENDTIME_STR'], format='%Y%m%d%H%M%S', errors='coerce')
                df.drop(columns=['ENDTIME_STR'], inplace=True, errors='ignore')
            elif 'FULL_DATETIME' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['FULL_DATETIME']):
                df['FULL_DATETIME'] = pd.to_datetime(df['FULL_DATETIME'], errors='coerce')

            subset_cols_hitlist = []
            if 'RESPTI' in df.columns: subset_cols_hitlist.append('RESPTI')
            if 'PROCTI' in df.columns: subset_cols_hitlist.append('PROCTI')
            if 'CPUTI' in df.columns: subset_cols_hitlist.append('CPUTI')
            if 'DBCALLS' in df.columns: subset_cols_hitlist.append('DBCALLS')
            if subset_cols_hitlist:
                df.dropna(subset=subset_cols_hitlist, inplace=True)
            if 'FULL_DATETIME' in df.columns:
                df.dropna(subset=['FULL_DATETIME'], inplace=True)

            for col in ['WPID', 'ACCOUNT', 'REPORT', 'ROLLKEY', 'PRIVMODE', 'WPRESTART']: # Removed 'TASKTYPE'
                if col in df.columns:
                    df[col] = clean_string_column(df[col])


        elif file_key == "times":
            numeric_cols = [
                'COUNT', 'LUW_COUNT', 'RESPTI', 'PROCTI', 'CPUTI', 'QUEUETI', 'ROLLWAITTI',
                'GUITIME', 'GUICNT', 'GUINETTIME', 'DBP_COUNT', 'DBP_TIME', 'READDIRCNT',
                'READDIRTI', 'READDIRBUF', 'READDIRREC', 'READSEQCNT', 'READSEQTI',
                'READSEQBUF', 'READSEQREC', 'CHNGCNT', 'CHNGTI', 'CHNGREC', 'PHYREADCNT',
                'PHYCHNGREC', 'PHYCALLS', 'VMC_CALL_COUNT', 'VMC_CPU_TIME', 'VMC_ELAP_TIME'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
            
            subset_cols_times = []
            if 'RESPTI' in df.columns: subset_cols_times.append('RESPTI')
            if 'PHYCALLS' in df.columns: subset_cols_times.append('PHYCALLS')
            if 'COUNT' in df.columns: subset_cols_times.append('COUNT')
            if subset_cols_times:
                df.dropna(subset=subset_cols_times, inplace=True)
            
            if 'TIME' in df.columns:
                df['TIME'] = clean_string_column(df['TIME'])
            # TASKTYPE removal: Removed df['TASKTYPE'] = clean_string_column(df['TASKTYPE'])
            if 'ENTRY_ID' in df.columns:
                df['ENTRY_ID'] = clean_string_column(df[col])

        elif file_key == "tasktimes":
            numeric_cols = [
                'COUNT', 'RESPTI', 'PROCTI', 'CPUTI', 'QUEUETI', 'ROLLWAITTI', 'GUITIME',
                'GUICNT', 'GUINETTIME', 'DBP_COUNT', 'DBP_TIME', 'READDIRCNT', 'READDIRTI',
                'READDIRBUF', 'READDIRREC', 'READSEQCNT', 'READSEQTI',
                'READSEQBUF', 'READSEQREC', 'CHNGCNT', 'CHNGTI', 'CHNGREC', 'PHYREADCNT',
                'PHYCHNGREC', 'PHYCALLS', 'CNT001', 'CNT002', 'CNT003', 'CNT004', 'CNT005', 'CNT006', 'CNT007', 'CNT008', 'CNT009'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
            
            subset_cols_tasktimes = []
            if 'COUNT' in df.columns: subset_cols_tasktimes.append('COUNT')
            if 'RESPTI' in df.columns: subset_cols_tasktimes.append('RESPTI')
            if 'CPUTI' in df.columns: subset_cols_tasktimes.append('CPUTI')
            if subset_cols_tasktimes:
                df.dropna(subset=subset_cols_tasktimes, inplace=True)
            
            # TASKTYPE removal: Removed df['TASKTYPE'] = clean_string_column(df['TASKTYPE'], 'Type de tâche non spécifié')
            if 'TIME' in df.columns:
                df['TIME'] = clean_string_column(df['TIME'])


        elif file_key == "usertcode":
            numeric_cols = [
                'COUNT', 'DCOUNT', 'UCOUNT', 'BCOUNT', 'ECOUNT', 'SCOUNT', 'LUW_COUNT',
                'TMBYTESIN', 'TMBYTESOUT', 'RESPTI', 'PROCTI', 'CPUTI', 'QUEUETI',
                'ROLLWAITTI', 'GUITIME', 'GUICNT', 'GUINETTIME', 'DBP_COUNT', 'DBP_TIME',
                'READDIRCNT', 'READDIRTI', 'READDIRBUF', 'READDIRREC', 'READSEQCNT',
                'READSEQTI', 'READSEQBUF', 'READSEQREC', 'CHNGCNT', 'CHNGTI', 'CHNGREC',
                'PHYREADCNT', 'PHYCHNGREC', 'PHYCALLS', 'DSQLCNT', 'QUECNT', 'CPICCNT',
                'SLI_CNT', 'VMC_CALL_COUNT', 'VMC_CPU_TIME', 'VMC_ELAP_TIME'
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
            
            # Add FULL_DATETIME creation for usertcode
            if 'ENDDATE' in df.columns and 'ENDTIME' in df.columns:
                df['ENDTIME_STR'] = df['ENDTIME'].astype(str).str.zfill(6)
                df['FULL_DATETIME'] = pd.to_datetime(df['ENDDATE'].astype(str) + df['ENDTIME_STR'], format='%Y%m%d%H%M%S', errors='coerce')
                df.drop(columns=['ENDTIME_STR'], inplace=True, errors='ignore')
            elif 'FULL_DATETIME' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['FULL_DATETIME']):
                df['FULL_DATETIME'] = pd.to_datetime(df['FULL_DATETIME'], errors='coerce')

            critical_usertcode_cols = []
            if 'RESPTI' in df.columns: critical_usertcode_cols.append('RESPTI')
            if 'ACCOUNT' in df.columns: critical_usertcode_cols.append('ACCOUNT')
            if 'COUNT' in df.columns: critical_usertcode_cols.append('COUNT')
            
            if critical_usertcode_cols:
                df.dropna(subset=critical_usertcode_cols, inplace=True)
            
            for col in ['ENTRY_ID', 'ACCOUNT']: # Removed 'TASKTYPE'
                if col in df.columns:
                    df[col] = clean_string_column(df[col])

        elif file_key == "performance": # Nouveau bloc pour AL_GET_PERFORMANCE
            # Convertir WP_CPU de MM:SS en secondes
            if 'WP_CPU' in df.columns:
                df['WP_CPU_SECONDS'] = df['WP_CPU'].apply(convert_mm_ss_to_seconds).astype(float)
            
            # Convertir WP_IWAIT en secondes (s'il est en ms, diviser par 1000)
            if 'WP_IWAIT' in df.columns:
                df['WP_IWAIT'] = pd.to_numeric(df['WP_IWAIT'], errors='coerce').fillna(0)
                # Keep WP_IWAIT as is, we will use it for plotting.
                # df['WP_IWAIT_SECONDS'] = df['WP_IWAIT'] / 1000.0 # This conversion might not be universally needed
            else:
                df['WP_IWAIT'] = 0 # Ensure column exists even if empty

            # Nettoyage des colonnes string
            for col in ['WP_SEMSTAT', 'WP_IACTION', 'WP_ITYPE', 'WP_RESTART', 'WP_ISTATUS', 'WP_TYP', 'WP_STATUS']:
                if col in df.columns:
                    df[col] = clean_string_column(df[col])
            
            # Nettoyage des colonnes numériques
            numeric_cols_perf = ['WP_NO', 'WP_IRESTRT', 'WP_PID', 'WP_INDEX']
            for col in numeric_cols_perf:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
            
            # Supprimer les lignes avec des valeurs critiques manquantes si nécessaire
            subset_cols_perf = []
            if 'WP_CPU_SECONDS' in df.columns: subset_cols_perf.append('WP_CPU_SECONDS')
            if 'WP_STATUS' in df.columns: subset_cols_perf.append('WP_STATUS')
            if subset_cols_perf:
                df.dropna(subset=subset_cols_perf, inplace=True)
        
        elif file_key == "sql_trace_summary": # Nouveau bloc pour performance_trace_summary
            # Nettoyage des colonnes numériques avec virgule/espace
            numeric_cols_sql = ['TOTALEXEC', 'IDENTSEL', 'EXECTIME', 'RECPROCNUM', 'TIMEPEREXE', 'RECPEREXE', 'AVGTPERREC', 'MINTPERREC']
            for col in numeric_cols_sql:
                if col in df.columns:
                    df[col] = clean_numeric_with_comma(df[col]).astype(float)
            
            # Nettoyage des colonnes string
            for col in ['SQLSTATEM', 'SERVERNAME', 'TRANS_ID']:
                if col in df.columns:
                    df[col] = clean_string_column(df[col])
            
            # Supprimer les lignes avec des valeurs critiques manquantes si nécessaire
            subset_cols_sql = []
            if 'EXECTIME' in df.columns: subset_cols_sql.append('EXECTIME')
            if 'TOTALEXEC' in df.columns: subset_cols_sql.append('TOTALEXEC')
            if 'SQLSTATEM' in df.columns: subset_cols_sql.append('SQLSTATEM')
            if subset_cols_sql:
                df.dropna(subset=subset_cols_sql, inplace=True)

        elif file_key == "usr02": # Nouveau bloc pour usr02_data.xlsx
            # Nettoyage des colonnes string
            for col in ['BNAME', 'USTYP']:
                if col in df.columns:
                    df[col] = clean_string_column(df[col])
            
            # Conversion de GLTGB en datetime
            if 'GLTGB' in df.columns:
                df['GLTGB'] = df['GLTGB'].astype(str).replace('00000000', np.nan)
                df['GLTGB_DATE'] = pd.to_datetime(df['GLTGB'], format='%Y%m%d', errors='coerce')
            else:
                df['GLTGB_DATE'] = pd.NaT

        return df

    except FileNotFoundError:
        st.error(f"Erreur: Le fichier '{path}' pour '{file_key}' est introuvable. Veuillez vérifier le chemin.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement du fichier '{file_key}' : {e}. Détails : {e}")
        return pd.DataFrame()

# --- Chargement de TOUTES les données ---
dfs = {}
for key, path in DATA_PATHS.items():
    dfs[key] = load_and_process_data(key, path)

# --- Contenu principal du Dashboard ---
st.title("📊 Tableau de Bord SAP Complet Multi-Sources")
st.markdown("Explorez les performances, l'utilisation mémoire, les transactions utilisateurs et la santé du système à travers différentes sources de données.")

# --- Affichage des KPIs ---
st.markdown("---")
kpi_cols = st.columns(5)

# KPI 1: Temps de Réponse Moyen Global (Hitlist DB)
avg_resp_time = 0
if not dfs['hitlist_db'].empty and 'RESPTI' in dfs['hitlist_db'].columns:
    # Ensure RESPTI is numeric before mean calculation
    if pd.api.types.is_numeric_dtype(dfs['hitlist_db']['RESPTI']):
        avg_resp_time = dfs['hitlist_db']['RESPTI'].mean() / 1000
kpi_cols[0].metric("Temps de Réponse Moyen (s)", f"{avg_resp_time:.2f}")

# KPI 2: Utilisation Mémoire Moyenne (USEDBYTES)
avg_memory_usage = 0
if not dfs['memory'].empty and 'USEDBYTES' in dfs['memory'].columns:
    # Ensure USEDBYTES is numeric before mean calculation
    if pd.api.types.is_numeric_dtype(dfs['memory']['USEDBYTES']):
        avg_memory_usage = dfs['memory']['USEDBYTES'].mean() / (1024 * 1024)
kpi_cols[1].metric("Mémoire Moyenne (Mo)", f"{avg_memory_usage:.2f}")

# KPI 3: Total des Appels Base de Données (Hitlist DB)
total_db_calls = 0
if not dfs['hitlist_db'].empty and 'DBCALLS' in dfs['hitlist_db'].columns:
    # Ensure DBCALLS is numeric before sum calculation
    if pd.api.types.is_numeric_dtype(dfs['hitlist_db']['DBCALLS']):
        total_db_calls = dfs['hitlist_db']['DBCALLS'].sum()
kpi_cols[2].metric("Total Appels DB", f"{int(total_db_calls):,}".replace(",", " "))

# KPI 4: Total des Exécutions SQL (performance_trace_summary) - NOUVEAU KPI
total_sql_executions = 0
if not dfs['sql_trace_summary'].empty and 'TOTALEXEC' in dfs['sql_trace_summary'].columns:
    # Ensure TOTALEXEC is numeric before sum calculation
    if pd.api.types.is_numeric_dtype(dfs['sql_trace_summary']['TOTALEXEC']):
        total_sql_executions = dfs['sql_trace_summary']['TOTALEXEC'].sum()
kpi_cols[3].metric("Total Exécutions SQL", f"{int(total_sql_executions):,}".replace(",", " "))

# KPI 5: Temps CPU Moyen Global (Hitlist DB)
avg_cpu_time = 0
if not dfs['hitlist_db'].empty and 'CPUTI' in dfs['hitlist_db'].columns:
    # Ensure CPUTI is numeric before mean calculation
    if pd.api.types.is_numeric_dtype(dfs['hitlist_db']['CPUTI']):
        avg_cpu_time = dfs['hitlist_db']['CPUTI'].mean() / 1000
kpi_cols[4].metric("Temps CPU Moyen (s)", f"{avg_cpu_time:.2f}")

st.markdown("---")

# --- Barre de navigation flexible ---
tab_titles = [
    "Analyse Mémoire",
    "Transactions Utilisateurs",
    "Statistiques Horaires",
    "Insights Hitlist DB",
    "Performance des Processus de Travail",
    "Résumé des Traces de Performance SQL",
    "Analyse des Utilisateurs",
    "Détection d'Anomalies",
    "Prédiction de Performance (ML)"
]

if 'current_section' not in st.session_state:
    st.session_state.current_section = tab_titles[0]

st.sidebar.header("Navigation Rapide")
selected_section = st.sidebar.radio(
    "Accéder à la section :",
    tab_titles,
    index=tab_titles.index(st.session_state.current_section)
)

st.session_state.current_section = selected_section

if all(df.empty for df in dfs.values()):
    st.error("Aucune source de données n'a pu être chargée. Le dashboard ne peut pas s'afficher. Veuillez vérifier les chemins et les fichiers.")
else:
    # --- Sidebar pour les filtres globaux ---
    st.sidebar.header("Filtres")

    all_accounts = pd.Index([])
    if not dfs['memory'].empty and 'ACCOUNT' in dfs['memory'].columns:
        all_accounts = all_accounts.union(dfs['memory']['ACCOUNT'].dropna().unique())
    if not dfs['usertcode'].empty and 'ACCOUNT' in dfs['usertcode'].columns:
        all_accounts = all_accounts.union(dfs['usertcode']['ACCOUNT'].dropna().unique())
    if not dfs['hitlist_db'].empty and 'ACCOUNT' in dfs['hitlist_db'].columns:
        all_accounts = all_accounts.union(dfs['hitlist_db']['ACCOUNT'].dropna().unique())
    
    selected_accounts = []
    if not all_accounts.empty:
        selected_accounts = st.sidebar.multiselect(
            "Sélectionner des Comptes",
            options=sorted(all_accounts.tolist()),
            default=[]
        )
        if selected_accounts:
            for key in ['memory', 'usertcode', 'hitlist_db']:
                if not dfs[key].empty and 'ACCOUNT' in dfs[key].columns:
                    dfs[key] = dfs[key][dfs[key]['ACCOUNT'].isin(selected_accounts)]

    selected_reports = []
    if not dfs['hitlist_db'].empty and 'REPORT' in dfs['hitlist_db'].columns:
        all_reports = dfs['hitlist_db']['REPORT'].dropna().unique().tolist()
        selected_reports = st.sidebar.multiselect(
            "Sélectionner des Rapports (Hitlist DB)",
            options=sorted(all_reports),
            default=[]
        )
        if selected_reports:
            dfs['hitlist_db'] = dfs['hitlist_db'][dfs['hitlist_db']['REPORT'].isin(selected_reports)]
    
    # TASKTYPE removal: Removed global tasktype filter from sidebar
    # all_tasktypes = pd.Index([])
    # if not dfs['usertcode'].empty and 'TASKTYPE' in dfs['usertcode'].columns:
    #     all_tasktypes = all_tasktypes.union(dfs['usertcode']['TASKTYPE'].dropna().unique())
    # if not dfs['times'].empty and 'TASKTYPE' in dfs['times'].columns:
    #     all_tasktypes = all_tasktypes.union(dfs['times']['TASKTYPE'].dropna().unique())
    # if not dfs['tasktimes'].empty and 'TASKTYPE' in dfs['tasktimes'].columns:
    #     all_tasktypes = all_tasktypes.union(dfs['tasktimes']['TASKTYPE'].dropna().unique())
    # if not dfs['hitlist_db'].empty and 'TASKTYPE' in dfs['hitlist_db'].columns:
    #     all_tasktypes = all_tasktypes.union(dfs['hitlist_db']['TASKTYPE'].dropna().unique())
    # if not dfs['memory'].empty and 'TASKTYPE' in dfs['memory'].columns:
    #     all_tasktypes = all_tasktypes.union(dfs['memory']['TASKTYPE'].dropna().unique())

    # selected_tasktypes = []
    # if not all_tasktypes.empty:
    #     selected_tasktypes = st.sidebar.multiselect(
    #         "Sélectionner des Types de Tâches",
    #         options=sorted(all_tasktypes.tolist()),
    #         default=[]
    #     )
    #     if selected_tasktypes:
    #         for key in ['usertcode', 'times', 'tasktimes', 'hitlist_db', 'memory']:
    #             if not dfs[key].empty and 'TASKTYPE' in dfs[key].columns:
    #                 dfs[key] = dfs[key][dfs[key]['TASKTYPE'].isin(selected_tasktypes)]
    
    selected_wp_types = []
    if not dfs['performance'].empty and 'WP_TYP' in dfs['performance'].columns:
        all_wp_types = dfs['performance']['WP_TYP'].dropna().unique().tolist()
        selected_wp_types = st.sidebar.multiselect(
            "Sélectionner des Types de Work Process (Performance)",
            options=sorted(all_wp_types),
            default=[]
        )
        if selected_wp_types:
            dfs['performance'] = dfs['performance'][dfs['performance']['WP_TYP'].isin(selected_wp_types)]

    df_hitlist_filtered = dfs['hitlist_db'].copy()


    # --- Contenu des sections basé sur la sélection de la barre latérale ---
    if st.session_state.current_section == "Analyse Mémoire":
        # --- Onglet 1: Analyse Mémoire (memory_final_cleaned_clean.xlsx) ---
        st.header("🧠 Analyse de l'Utilisation Mémoire")
        df_mem = dfs['memory'].copy()
        if selected_accounts:
            df_mem = df_mem[df_mem['ACCOUNT'].isin(selected_accounts)]
        # TASKTYPE removal: Removed if selected_tasktypes filter for df_mem
        # if selected_tasktypes:
        #     if 'TASKTYPE' in df_mem.columns:
        #         df_mem = df_mem[df_mem['TASKTYPE'].isin(selected_tasktypes)]
        #     else:
        #         st.warning("La colonne 'TASKTYPE' est manquante dans les données mémoire pour le filtrage.")

        if not df_mem.empty:
            st.subheader("Top 10 Utilisateurs par Utilisation Mémoire (USEDBYTES)")
            st.markdown("""
                Ce graphique identifie les 10 principaux utilisateurs ou comptes consommant le plus de mémoire (USEDBYTES). Il est essentiel pour détecter les "gros consommateurs" de ressources, comme JBELIM qui domine l'utilisation et ainsi cibler les optimisations de performance système. 
                Il aide à comprendre qui ou quel processus ABAP (comme WF-BATCH) impacte le plus les ressources mémoire.


                """)
            if all(col in df_mem.columns for col in ['ACCOUNT', 'USEDBYTES', 'MAXBYTES', 'PRIVSUM']) and df_mem['USEDBYTES'].sum() > 0:
                # Ensure numeric types before aggregation
                df_mem['USEDBYTES'] = pd.to_numeric(df_mem['USEDBYTES'], errors='coerce').fillna(0).astype(float)
                df_mem['MAXBYTES'] = pd.to_numeric(df_mem['MAXBYTES'], errors='coerce').fillna(0).astype(float)
                df_mem['PRIVSUM'] = pd.to_numeric(df_mem['PRIVSUM'], errors='coerce').fillna(0).astype(float)

                top_users_mem = df_mem.groupby('ACCOUNT', as_index=False)[['USEDBYTES', 'MAXBYTES', 'PRIVSUM']].sum().nlargest(10, 'USEDBYTES')
                if not top_users_mem.empty and top_users_mem['USEDBYTES'].sum() > 0:
                    fig_top_users_mem = px.bar(top_users_mem,
                                                x='ACCOUNT', y='USEDBYTES',
                                                title="Top 10 Comptes par USEDBYTES Total",
                                                labels={'USEDBYTES': 'Utilisation Mémoire (Octets)', 'ACCOUNT': 'Compte Utilisateur'},
                                                hover_data=['MAXBYTES', 'PRIVSUM'],
                                                color='USEDBYTES', color_continuous_scale=px.colors.sequential.Plasma)
                    st.plotly_chart(fig_top_users_mem, use_container_width=True)
                else:
                    st.info("Pas de données valides pour les Top 10 Utilisateurs par Utilisation Mémoire après filtrage.")
            else:
                st.info("Colonnes nécessaires (ACCOUNT, USEDBYTES, MAXBYTES, PRIVSUM) manquantes ou USEDBYTES total est zéro/vide après filtrage.")

            st.subheader("Moyenne de USEDBYTES par Client (ACCOUNT)")
            st.markdown("""
                Cette visualisation présente la consommation moyenne de mémoire (USEDBYTES) par client ou compte SAP. 
                Elle offre une perspective agrégée, utile pour la planification des capacités et la compréhension des tendances d'utilisation par groupe fonctionnel ou type d'activité. 
                Cela permet d'allouer les ressources mémoire plus efficacement en fonction des besoins moyens des différents clients.
                """)
            if 'ACCOUNT' in df_mem.columns and 'USEDBYTES' in df_mem.columns and df_mem['USEDBYTES'].sum() > 0:
                df_mem_account_clean = df_mem[df_mem['ACCOUNT'] != 'Compte Inconnu'].copy()
                
                if not df_mem_account_clean.empty:
                    # Ensure USEDBYTES is numeric here
                    df_mem_account_clean['USEDBYTES'] = pd.to_numeric(df_mem_account_clean['USEDBYTES'], errors='coerce').fillna(0).astype(float)
                    df_mem_account_clean['ACCOUNT_DISPLAY'] = df_mem_account_clean['ACCOUNT'].astype(str)

                    account_counts = df_mem_account_clean['ACCOUNT_DISPLAY'].nunique()
                    if account_counts > 6:
                        top_accounts = df_mem_account_clean['ACCOUNT_DISPLAY'].value_counts().nlargest(6).index
                        df_mem_account_filtered_for_plot = df_mem_account_clean.loc[df_mem_account_clean['ACCOUNT_DISPLAY'].isin(top_accounts)].copy()
                    else:
                        df_mem_account_filtered_for_plot = df_mem_account_clean.copy()

                    avg_mem_account = df_mem_account_filtered_for_plot.groupby('ACCOUNT_DISPLAY', as_index=False)['USEDBYTES'].mean().sort_values(by='USEDBYTES', ascending=False)
                    if not avg_mem_account.empty and not avg_mem_account['USEDBYTES'].sum() == 0:
                        fig_avg_mem_account = px.bar(avg_mem_account,
                                                     x='ACCOUNT_DISPLAY', y='USEDBYTES',
                                                     title="Moyenne de USEDBYTES par Client SAP (Top 6 ou tous)",
                                                     labels={'USEDBYTES': 'Moyenne USEDBYTES (Octets)', 'ACCOUNT_DISPLAY': 'Client SAP'},
                                                     color='USEDBYTES', color_continuous_scale=px.colors.sequential.Viridis)
                        fig_avg_mem_account.update_xaxes(type='category')
                        st.plotly_chart(fig_avg_mem_account, use_container_width=True)
                    else:
                        st.info("Pas de données valides pour la moyenne de USEDBYTES par Client SAP après filtrage (peut-être tous 'Compte Inconnu' ou USEDBYTES est zéro).")
                else:
                    st.info("Aucune donnée valide pour les clients (ACCOUNT) après filtrage.")
            else:
                st.info("Colonnes 'ACCOUNT' ou 'USEDBYTES' manquantes ou USEDBYTES total est zéro/vide après filtrage.")

            st.subheader("Distribution de l'Utilisation Mémoire (USEDBYTES) - Courbe de Densité")
            st.markdown("""
                 "Densité" indique la fréquence relative à laquelle une valeur d'utilisation mémoire apparaît. 
                 Un pic élevé sur la courbe signifie que les valeurs de USEDBYTES autour de ce point sont très fréquentes. 
                 À l'inverse, une faible densité (la courbe se rapprochant de zéro) indique que ces niveaux de consommation mémoire sont rares.
                """)
            if 'USEDBYTES' in df_mem.columns and df_mem['USEDBYTES'].sum() > 0:
                # Ensure USEDBYTES is numeric here
                df_mem['USEDBYTES'] = pd.to_numeric(df_mem['USEDBYTES'], errors='coerce').fillna(0).astype(float)
                if df_mem['USEDBYTES'].nunique() > 1:
                    fig_dist_mem = ff.create_distplot([df_mem['USEDBYTES'].dropna()], ['USEDBYTES'], bin_size=df_mem['USEDBYTES'].std()/5 if df_mem['USEDBYTES'].std() > 0 else 1, show_rug=False, show_hist=False)
                    fig_dist_mem.update_layout(title_text="Distribution de l'Utilisation Mémoire (USEDBYTES) - Courbe de Densité", xaxis_title='Utilisation Mémoire (Octets)', yaxis_title='Densité')
                    fig_dist_mem.data[0].line.color = 'lightcoral'
                    st.plotly_chart(fig_dist_mem, use_container_width=True)
                else:
                    st.info("La colonne 'USEDBYTES' contient des valeurs uniques ou est vide après filtrage, impossible de créer une courbe de densité.")
            else:
                st.info("Colonne 'USEDBYTES' manquante ou total est zéro/vide après filtrage.")

            if 'FULL_DATETIME' in df_mem.columns and pd.api.types.is_datetime64_any_dtype(df_mem['FULL_DATETIME']) and not df_mem['FULL_DATETIME'].isnull().all() and 'USEDBYTES' in df_mem.columns and df_mem['USEDBYTES'].sum() > 0:
                # Ensure USEDBYTES is numeric here
                df_mem['USEDBYTES'] = pd.to_numeric(df_mem['USEDBYTES'], errors='coerce').fillna(0).astype(float)
                hourly_mem_usage = df_mem.set_index('FULL_DATETIME')['USEDBYTES'].resample('H').mean().dropna()
                if not hourly_mem_usage.empty:
                    fig_hourly_mem = px.line(hourly_mem_usage.reset_index(), x='FULL_DATETIME', y='USEDBYTES', title="Tendance Moyenne USEDBYTES par Heure", labels={'FULL_DATETIME': 'Heure', 'USEDBYTES': 'Moyenne USEDBYTES'}, color_discrete_sequence=['purple'])
                    fig_hourly_mem.update_xaxes(dtick="H1", tickformat="%H:%M")
                    st.plotly_chart(fig_hourly_mem, use_container_width=True)
                else:
                    pass # Original code had pass here, maintaining
            else:
                pass # Original code had pass here, maintaining

            st.markdown("""
                Cette courbe de densité illustre la distribution de l'utilisation mémoire (USEDBYTES). Le pic prononcé près de zéro indique que la majorité des opérations consomment très peu de mémoire. La "longue queue" vers la droite révèle la présence, bien que rare, de processus gourmands en mémoire. Il pourrait s'agir de requêtes complexes, de traitements de lot volumineux (comme WF-BATCH qui apparaît dans les autres graphiques), ou d'éventuels problèmes nécessitant une optimisation.
                """)
            st.subheader("Comparaison des Métriques Mémoire (USEDBYTES, MAXBYTES, PRIVSUM) par Compte Utilisateur")
            st.markdown("""
                *USEDBYTES: Mémoire actuellement utilisée par un processus ou une session. *MAXBYTES: Quantité maximale de mémoire atteinte par un processus pendant son exécution. *PRIVSUM: Quantité de mémoire privée, non partagée, consommée par un processus.
                """)
            mem_metrics_cols = ['USEDBYTES', 'MAXBYTES', 'PRIVSUM']
            if all(col in df_mem.columns for col in mem_metrics_cols) and 'ACCOUNT' in df_mem.columns and df_mem[mem_metrics_cols].sum().sum() > 0:
                # Ensure numeric types before aggregation
                for col in mem_metrics_cols:
                    df_mem[col] = pd.to_numeric(df_mem[col], errors='coerce').fillna(0).astype(float)
                account_mem_summary = df_mem.groupby('ACCOUNT', as_index=False)[mem_metrics_cols].sum().nlargest(10, 'USEDBYTES')
                if not account_mem_summary.empty and account_mem_summary[mem_metrics_cols].sum().sum() > 0:
                    fig_mem_comparison = px.bar(account_mem_summary, x='ACCOUNT', y=mem_metrics_cols, title="Comparaison des Métriques Mémoire par Compte Utilisateur (Top 10 USEDBYTES)", labels={'value': 'Quantité (Octets)', 'variable': 'Métrique Mémoire', 'ACCOUNT': 'Compte Utilisateur'}, barmode='group', color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig_mem_comparison, use_container_width=True)
                else:
                    st.info("Pas de données valides pour la comparaison des métriques mémoire par compte utilisateur après filtrage.")
            else:
                st.info("Colonnes nécessaires (ACCOUNT, USEDBYTES, MAXBYTES, PRIVSUM) manquantes ou leurs totaux sont zéro/vides après filtrage pour la comparaison des métriques mémoire.")
            # TASKTYPE removal: Removed section "Top Types de Tâches par Utilisation Mémoire"
            # st.subheader("Top Types de Tâches (TASKTYPE) par Utilisation Mémoire (USEDBYTES)")
            # if 'TASKTYPE' in df_mem.columns and 'USEDBYTES' in df_mem.columns and df_mem['USEDBYTES'].sum() > 0:
            # df_mem['USEDBYTES'] = pd.to_numeric(df_mem['USEDBYTES'], errors='coerce').fillna(0).astype(float)
            # top_tasktype_mem = df_mem.groupby('TASKTYPE', as_index=False)['USEDBYTES'].sum().nlargest(3, 'USEDBYTES')
            # if not top_tasktype_mem.empty and top_tasktype_mem['USEDBYTES'].sum() > 0:
            # fig_top_tasktype_mem = px.bar(top_tasktype_mem, x='TASKTYPE', y='USEDBYTES', title="Top 3 Types de Tâches par Utilisation Mémoire (USEDBYTES)", labels={'USEDBYTES': 'Utilisation Mémoire Totale (Octets)', 'TASKTYPE': 'Type de Tâche'}, color='USEDBYTES', color_continuous_scale=px.colors.sequential.Greys)
            # st.plotly_chart(fig_top_tasktype_mem, use_container_width=True)
            # else:
            # st.info("Pas de données valides pour les Top Types de Tâches par Utilisation Mémoire après filtrage.")
            # else:
            # st.info("Colonnes 'TASKTYPE' ou 'USEDBYTES' manquantes ou USEDBYTES total est zéro/vide après filtrage pour les types de tâches mémoire.")
            st.subheader("Aperçu des Données Mémoire Filtrées")
            # Displaying only relevant columns for an overview, excluding TASKTYPE
            # TASKTYPE removal: Ensure TASKTYPE is not in this list
            columns_to_display = [col for col in df_mem.columns if col not in ['TASKTYPE', 'FULL_DATETIME']]
            st.dataframe(df_mem[columns_to_display].head())
        # FIX START: Replaced the problematic 'else' block with a new 'if' for clarity
        if df_mem.empty:
            st.warning("Données mémoire non disponibles ou filtrées à vide.")
        # FIX END
    elif st.session_state.current_section == "Transactions Utilisateurs":
        # --- Onglet 2: Transactions Utilisateurs (USERTCODE_cleaned.xlsx) ---
        st.header("👤 Analyse des Transactions Utilisateurs")
        df_user = dfs['usertcode'].copy()
        if selected_accounts:
            if 'ACCOUNT' in df_user.columns:
                df_user = df_user[df_user['ACCOUNT'].isin(selected_accounts)]
            else:
                st.warning("La colonne 'ACCOUNT' est manquante dans les données utilisateurs pour le filtrage.")
        # TASKTYPE removal: Removed if selected_tasktypes filter for df_user
        # if selected_tasktypes:
        #     if 'TASKTYPE' in df_user.columns:
        #         df_user = df_user[df_user['TASKTYPE'].isin(selected_tasktypes)]
        #     else:
        #         st.warning("La colonne 'TASKTYPE' est manquante dans les données utilisateurs pour le filtrage.")
        if not df_user.empty:
            # --- Section modifiée: Top 10 des Comptes (ACCOUNT) par Temps de Réponse Moyen (en cercle) ---
            st.subheader("Top 10 des Comptes (ACCOUNT) par Temps de Réponse Moyen")
            st.markdown("""
                Ce graphique montre la distribution des appels de base de données par serveur, ce qui est crucial pour identifier 
                les serveurs surchargés ou sous-utilisés et optimiser la répartition de la charge.
                """)
            
            if all(col in df_user.columns for col in ['ACCOUNT', 'RESPTI']) and df_user['RESPTI'].sum() > 0:
                df_user['RESPTI'] = pd.to_numeric(df_user['RESPTI'], errors='coerce').fillna(0).astype(float)
                top_accounts_resp = df_user.groupby('ACCOUNT', as_index=False)['RESPTI'].mean().nlargest(10, 'RESPTI')
                
                if not top_accounts_resp.empty and top_accounts_resp['RESPTI'].sum() > 0:
                    fig_top_accounts_resp = px.pie(top_accounts_resp, values='RESPTI', names='ACCOUNT',
                                                 title="Top 10 Comptes par Temps de Réponse Moyen (ms)",
                                                 hole=0.3)
                    fig_top_accounts_resp.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_top_accounts_resp, use_container_width=True)
                else:
                    st.info("Pas de données valides pour les Top 10 Comptes par Temps de Réponse Moyen après filtrage.")
            else:
                st.info("Colonnes 'ACCOUNT' ou 'RESPTI' manquantes ou RESPTI total est zéro/vide après filtrage.")

            st.subheader("Nombre de Transactions par TCode (Top 10)")
            st.markdown("""
                Ce graphique en barres présente les 10 codes de transaction (TCode) les plus fréquemment utilisés. Il permet d'identifier 
                les fonctionnalités les plus sollicitées dans SAP, ce qui est utile pour la planification des ressources et l'identification des processus métier clés.
                """)
            if 'ENTRY_ID' in df_user.columns and 'COUNT' in df_user.columns and df_user['COUNT'].sum() > 0:
                df_user['COUNT'] = pd.to_numeric(df_user['COUNT'], errors='coerce').fillna(0).astype(float)
                top_tcodes = df_user.groupby('ENTRY_ID', as_index=False)['COUNT'].sum().nlargest(10, 'COUNT')
                if not top_tcodes.empty and top_tcodes['COUNT'].sum() > 0:
                    fig_top_tcodes = px.bar(top_tcodes, x='ENTRY_ID', y='COUNT',
                                            title="Top 10 TCodes par Nombre de Transactions",
                                            labels={'ENTRY_ID': 'Code de Transaction (TCode)', 'COUNT': 'Nombre de Transactions'},
                                            color='COUNT', color_continuous_scale=px.colors.sequential.Sunset)
                    st.plotly_chart(fig_top_tcodes, use_container_width=True)
                else:
                    st.info("Pas de données valides pour les Top 10 TCodes après filtrage.")
            else:
                st.info("Colonnes 'ENTRY_ID' ou 'COUNT' manquantes ou COUNT total est zéro/vide après filtrage.")

            st.subheader("Distribution du Temps de Réponse (RESPTI) par Type d'Utilisateur (USTYP) - USR02")
            st.markdown("""
                Cette analyse explore comment le temps de réponse des transactions varie en fonction des types d'utilisateurs (par exemple, dialogue, batch). 
                Elle aide à comprendre les impacts des différents profils d'utilisation sur la performance globale du système SAP et à identifier les goulots d'étranglement 
                spécifiques à certains types d'activités.
                """)
            # Merge with usr02 to get USTYP
            if not dfs['usr02'].empty and not df_user.empty and 'ACCOUNT' in df_user.columns and 'BNAME' in dfs['usr02'].columns and 'RESPTI' in df_user.columns:
                df_merged_user_type = pd.merge(df_user, dfs['usr02'][['BNAME', 'USTYP']], left_on='ACCOUNT', right_on='BNAME', how='left')
                if 'USTYP' in df_merged_user_type.columns and not df_merged_user_type['USTYP'].isnull().all() and df_merged_user_type['RESPTI'].sum() > 0:
                    df_merged_user_type['RESPTI'] = pd.to_numeric(df_merged_user_type['RESPTI'], errors='coerce').fillna(0).astype(float)
                    
                    # Filter out NaN/None USTYP
                    df_merged_user_type_clean = df_merged_user_type.dropna(subset=['USTYP']).copy()
                    
                    if not df_merged_user_type_clean.empty and df_merged_user_type_clean['USTYP'].nunique() > 1:
                        # Ensure 'RESPTI' column has variation for meaningful box plot
                        if df_merged_user_type_clean.groupby('USTYP')['RESPTI'].nunique().min() > 1:
                            fig_resp_by_user_type = px.box(df_merged_user_type_clean, x='USTYP', y='RESPTI',
                                                        title="Distribution du Temps de Réponse (RESPTI) par Type d'Utilisateur",
                                                        labels={'USTYP': "Type d'Utilisateur", 'RESPTI': 'Temps de Réponse (ms)'},
                                                        color='USTYP')
                            st.plotly_chart(fig_resp_by_user_type, use_container_width=True)
                        else:
                            st.info("La colonne 'RESPTI' ne contient pas suffisamment de variation pour créer un Box Plot significatif pour chaque type d'utilisateur.")
                    else:
                        st.info("Pas de données valides pour la distribution du Temps de Réponse par Type d'Utilisateur après filtrage.")
                else:
                    st.info("Colonnes 'USTYP' ou 'RESPTI' manquantes/vides après fusion, ou RESPTI total est zéro/vide.")
            else:
                st.info("Données 'usr02' ou 'usertcode' non disponibles ou colonnes requises manquantes pour l'analyse par type d'utilisateur.")

            st.subheader("Aperçu des Données Utilisateurs Filtrées")
            # TASKTYPE removal: Ensure TASKTYPE is not in this list
            columns_to_display_user = [col for col in df_user.columns if col not in ['TASKTYPE', 'FULL_DATETIME']]
            st.dataframe(df_user[columns_to_display_user].head())
        else:
            st.warning("Données utilisateurs non disponibles ou filtrées à vide.")

    elif st.session_state.current_section == "Statistiques Horaires":
        # --- Onglet 3: Statistiques Horaires (Times_final_cleaned_clean.xlsx & TASKTIMES_final_cleaned_clean.xlsx) ---
        st.header("⏰ Statistiques Horaires et Types de Tâches")

        st.subheader("Temps de Réponse Moyen (RESPTI) par Heure (Times)")
        st.markdown("""
            Ce graphique en aires représente l'évolution du temps de réponse moyen par heure. 
            Il aide à identifier les périodes de pointe où le système est le plus sollicité, 
            ce qui est crucial pour la planification de la capacité et l'optimisation des performances en fonction des cycles d'activité.
            """)
        df_times = dfs['times'].copy()
        # TASKTYPE removal: Removed if selected_tasktypes filter for df_times
        # if selected_tasktypes:
        #     if 'TASKTYPE' in df_times.columns:
        #         df_times = df_times[df_times['TASKTYPE'].isin(selected_tasktypes)]
        #     else:
        #         st.warning("La colonne 'TASKTYPE' est manquante dans les données 'times' pour le filtrage.")

        if not df_times.empty and 'TIME' in df_times.columns and 'RESPTI' in df_times.columns and df_times['RESPTI'].sum() > 0:
            df_times['RESPTI'] = pd.to_numeric(df_times['RESPTI'], errors='coerce').fillna(0).astype(float)
            
            # Convert TIME column to proper time format for sorting
            df_times['TIME_HOUR_MIN'] = pd.to_datetime(df_times['TIME'], format='%H:%M:%S', errors='coerce').dt.time
            df_times_clean = df_times.dropna(subset=['TIME_HOUR_MIN']).copy()

            if not df_times_clean.empty:
                hourly_resp_time = df_times_clean.groupby('TIME_HOUR_MIN')['RESPTI'].mean().reset_index()
                hourly_resp_time['TIME_STR'] = hourly_resp_time['TIME_HOUR_MIN'].astype(str) # For display
                hourly_resp_time = hourly_resp_time.sort_values(by='TIME_HOUR_MIN')

                if not hourly_resp_time.empty and hourly_resp_time['RESPTI'].sum() > 0:
                    fig_hourly_resp = px.area(hourly_resp_time, x='TIME_STR', y='RESPTI',
                                            title="Temps de Réponse Moyen (RESPTI) par Heure (Times)",
                                            labels={'TIME_STR': 'Heure de la Journée', 'RESPTI': 'Temps de Réponse Moyen (ms)'},
                                            color_discrete_sequence=['deepskyblue'])
                    fig_hourly_resp.update_xaxes(type='category', categoryorder='array', categoryarray=hourly_resp_time['TIME_STR'].tolist())
                    st.plotly_chart(fig_hourly_resp, use_container_width=True)
                else:
                    st.info("Pas de données valides pour le Temps de Réponse Moyen par Heure après filtrage.")
            else:
                st.info("La colonne 'TIME' est vide ou contient des formats non valides après filtrage.")
        else:
            st.info("Colonnes 'TIME' ou 'RESPTI' manquantes dans 'Times' ou RESPTI total est zéro/vide.")

        st.subheader("Temps CPU Moyen par Heure (Tasktimes)")
        st.markdown("""
            Ce graphique illustre la consommation moyenne de temps CPU par heure, offrant une vue détaillée des périodes où les ressources de traitement 
            sont le plus sollicitées. Cette information est essentielle pour le réglage fin des performances du système et la répartition des charges de travail.
            """)
        df_tasktimes = dfs['tasktimes'].copy()
        # TASKTYPE removal: Removed if selected_tasktypes filter for df_tasktimes
        # if selected_tasktypes:
        #     if 'TASKTYPE' in df_tasktimes.columns:
        #         df_tasktimes = df_tasktimes[df_tasktimes['TASKTYPE'].isin(selected_tasktypes)]
        #     else:
        #         st.warning("La colonne 'TASKTYPE' est manquante dans les données 'tasktimes' pour le filtrage.")

        if not df_tasktimes.empty and 'TIME' in df_tasktimes.columns and 'CPUTI' in df_tasktimes.columns and df_tasktimes['CPUTI'].sum() > 0:
            df_tasktimes['CPUTI'] = pd.to_numeric(df_tasktimes['CPUTI'], errors='coerce').fillna(0).astype(float)
            
            # Convert TIME column to proper time format for sorting
            df_tasktimes['TIME_HOUR_MIN'] = pd.to_datetime(df_tasktimes['TIME'], format='%H:%M:%S', errors='coerce').dt.time
            df_tasktimes_clean = df_tasktimes.dropna(subset=['TIME_HOUR_MIN']).copy()

            if not df_tasktimes_clean.empty:
                hourly_cpu_time = df_tasktimes_clean.groupby('TIME_HOUR_MIN')['CPUTI'].mean().reset_index()
                hourly_cpu_time['TIME_STR'] = hourly_cpu_time['TIME_HOUR_MIN'].astype(str) # For display
                hourly_cpu_time = hourly_cpu_time.sort_values(by='TIME_HOUR_MIN')
                
                if not hourly_cpu_time.empty and hourly_cpu_time['CPUTI'].sum() > 0:
                    fig_hourly_cpu = px.line(hourly_cpu_time, x='TIME_STR', y='CPUTI',
                                            title="Temps CPU Moyen (CPUTI) par Heure (Tasktimes)",
                                            labels={'TIME_STR': 'Heure de la Journée', 'CPUTI': 'Temps CPU Moyen (ms)'},
                                            color_discrete_sequence=['orange'])
                    fig_hourly_cpu.update_xaxes(type='category', categoryorder='array', categoryarray=hourly_cpu_time['TIME_STR'].tolist())
                    st.plotly_chart(fig_hourly_cpu, use_container_width=True)
                else:
                    st.info("Pas de données valides pour le Temps CPU Moyen par Heure après filtrage.")
            else:
                st.info("La colonne 'TIME' est vide ou contient des formats non valides après filtrage.")
        else:
            st.info("Colonnes 'TIME' ou 'CPUTI' manquantes dans 'Tasktimes' ou CPUTI total est zéro/vide.")

        st.subheader("Nombre de Traitements par Tâche (COUNT) (Tasktimes)")
        st.markdown("""
            Ce graphique met en évidence les types de tâches les plus fréquemment exécutées dans le système. Il offre un aperçu de la charge de travail 
            et des patterns d'utilisation, aidant à identifier les tâches dominantes qui pourraient nécessiter une attention particulière pour l'optimisation.
            """)
        if not df_tasktimes.empty and 'COUNT' in df_tasktimes.columns and df_tasktimes['COUNT'].sum() > 0:
            df_tasktimes['COUNT'] = pd.to_numeric(df_tasktimes['COUNT'], errors='coerce').fillna(0).astype(float)
            
            # Assuming 'ENTRY_ID' or similar column for task type, if not, use a generic count
            # Based on file_key "tasktimes", the data might not have a direct 'TASKTYPE' column.
            # If there's another column that represents task type, use it. Otherwise, count generally.
            # For this example, let's just show total count or a distribution if a suitable column exists.
            
            # If there's a column like 'ENTRY_ID' or 'TASKTYPE' to group by
            # For tasktimes, if a column like 'ENTRY_ID' or 'TASKTYPE' is present, use it.
            # If not, a simple sum of COUNT might be more appropriate.
            # Given the previous TASKTYPE removal, we'll just sum counts.
            total_task_count = df_tasktimes['COUNT'].sum()
            st.metric("Total des Traitements (Tasktimes)", f"{int(total_task_count):,}".replace(",", " "))
        else:
            st.info("Colonne 'COUNT' manquante dans 'Tasktimes' ou total est zéro/vide.")
        
        st.subheader("Aperçu des Données Statistiques Horaires Filtrées (Times)")
        columns_to_display_times = [col for col in df_times.columns if col not in ['TASKTYPE', 'TIME_HOUR_MIN']]
        st.dataframe(df_times[columns_to_display_times].head())

        st.subheader("Aperçu des Données Statistiques Horaires Filtrées (Tasktimes)")
        columns_to_display_tasktimes = [col for col in df_tasktimes.columns if col not in ['TASKTYPE', 'TIME_HOUR_MIN']]
        st.dataframe(df_tasktimes[columns_to_display_tasktimes].head())
        
        if df_times.empty and df_tasktimes.empty:
            st.warning("Données horaires non disponibles ou filtrées à vide.")

    elif st.session_state.current_section == "Insights Hitlist DB":
        # --- Onglet 4: Insights Hitlist DB (HITLIST_DATABASE_final_cleaned_clean.xlsx) ---
        st.header("🔍 Insights sur la Base de Données (Hitlist DB)")
        df_hitlist = df_hitlist_filtered.copy() # Use the already filtered df

        if not df_hitlist.empty:
            st.subheader("Distribution du Temps de Réponse (RESPTI) par Rapport (REPORT) - Top 10")
            st.markdown("""
                Ce graphique en barres présente les temps de réponse moyens pour les 10 rapports SAP les plus utilisés. 
                Il est essentiel pour identifier les rapports lents qui pourraient nécessiter une optimisation, 
                améliorant ainsi l'expérience utilisateur et l'efficacité des processus métier.
                """)
            if 'REPORT' in df_hitlist.columns and 'RESPTI' in df_hitlist.columns and df_hitlist['RESPTI'].sum() > 0:
                df_hitlist['RESPTI'] = pd.to_numeric(df_hitlist['RESPTI'], errors='coerce').fillna(0).astype(float)
                top_reports_resp = df_hitlist.groupby('REPORT', as_index=False)['RESPTI'].mean().nlargest(10, 'RESPTI')
                
                if not top_reports_resp.empty and top_reports_resp['RESPTI'].sum() > 0:
                    fig_reports_resp = px.bar(top_reports_resp, x='REPORT', y='RESPTI',
                                            title="Temps de Réponse Moyen (ms) par Rapport (Top 10)",
                                            labels={'REPORT': 'Rapport SAP', 'RESPTI': 'Temps de Réponse Moyen (ms)'},
                                            color='RESPTI', color_continuous_scale=px.colors.sequential.Bluyl)
                    st.plotly_chart(fig_reports_resp, use_container_width=True)
                else:
                    st.info("Pas de données valides pour le Temps de Réponse par Rapport après filtrage.")
            else:
                st.info("Colonnes 'REPORT' ou 'RESPTI' manquantes ou RESPTI total est zéro/vide après filtrage.")
            
            st.subheader("Corrélation Temps de Réponse (RESPTI) et Temps CPU (CPUTI) - Scatter Plot")
            st.markdown("""
                Ce nuage de points visualise la relation entre le temps de réponse global d'une transaction et le temps CPU qu'elle consomme. 
                Une forte corrélation indique que le CPU est un facteur limitant la performance. 
                Les points aberrants (haut temps de réponse, faible CPU ou vice-versa) peuvent signaler des goulots d'étranglement 
                liés à la base de données, au réseau, ou à des problèmes d'attente.
                """)
            if 'RESPTI' in df_hitlist.columns and 'CPUTI' in df_hitlist.columns:
                df_hitlist_scatter = df_hitlist[['RESPTI', 'CPUTI']].dropna()
                if not df_hitlist_scatter.empty:
                    df_hitlist_scatter['RESPTI'] = pd.to_numeric(df_hitlist_scatter['RESPTI'], errors='coerce').fillna(0).astype(float)
                    df_hitlist_scatter['CPUTI'] = pd.to_numeric(df_hitlist_scatter['CPUTI'], errors='coerce').fillna(0).astype(float)

                    if df_hitlist_scatter['RESPTI'].nunique() > 1 and df_hitlist_scatter['CPUTI'].nunique() > 1:
                        fig_resp_cpu_scatter = px.scatter(df_hitlist_scatter.sample(min(5000, len(df_hitlist_scatter)), random_state=42), # Sample for performance
                                                        x='CPUTI', y='RESPTI',
                                                        title="Corrélation Temps de Réponse vs. Temps CPU",
                                                        labels={'CPUTI': 'Temps CPU (ms)', 'RESPTI': 'Temps de Réponse (ms)'},
                                                        opacity=0.6,
                                                        hover_data={'CPUTI': True, 'RESPTI': True})
                        st.plotly_chart(fig_resp_cpu_scatter, use_container_width=True)
                    else:
                        st.info("Les colonnes 'RESPTI' ou 'CPUTI' ne contiennent pas suffisamment de variation pour créer un Scatter Plot significatif après filtrage.")
                else:
                    st.info("Pas de données valides pour la corrélation Temps de Réponse et Temps CPU après filtrage.")
            else:
                st.info("Colonnes 'RESPTI' ou 'CPUTI' manquantes dans 'Hitlist DB'.")

            st.subheader("Temps d'Attente en File (QUEUETI) par Rapport - Top 10")
            st.markdown("""
                Ce graphique identifie les rapports associés aux temps d'attente en file (QUEUETI) les plus élevés. 
                Des temps d'attente élevés indiquent des goulots d'étranglement dans le système, souvent liés à des ressources insuffisantes 
                (processus de travail, base de données) ou à un grand nombre de requêtes simultanées.
                """)
            if 'REPORT' in df_hitlist.columns and 'QUEUETI' in df_hitlist.columns and df_hitlist['QUEUETI'].sum() > 0:
                df_hitlist['QUEUETI'] = pd.to_numeric(df_hitlist['QUEUETI'], errors='coerce').fillna(0).astype(float)
                top_reports_queue = df_hitlist.groupby('REPORT', as_index=False)['QUEUETI'].mean().nlargest(10, 'QUEUETI')
                if not top_reports_queue.empty and top_reports_queue['QUEUETI'].sum() > 0:
                    fig_reports_queue = px.bar(top_reports_queue, x='REPORT', y='QUEUETI',
                                            title="Temps d'Attente en File (QUEUETI) Moyen par Rapport (Top 10)",
                                            labels={'REPORT': 'Rapport SAP', 'QUEUETI': "Temps d'Attente en File (ms)"},
                                            color='QUEUETI', color_continuous_scale=px.colors.sequential.Greens)
                    st.plotly_chart(fig_reports_queue, use_container_width=True)
                else:
                    st.info("Pas de données valides pour les Temps d'Attente en File par Rapport après filtrage.")
            else:
                st.info("Colonnes 'REPORT' ou 'QUEUETI' manquantes ou QUEUETI total est zéro/vide après filtrage.")
            
            st.subheader("Distribution des Appels Base de Données (DBCALLS)")
            st.markdown("""
                Ce graphique montre la distribution des appels de base de données (DBCALLS), ce qui est un indicateur clé de l'activité du système. 
                Des pics anormaux ou une distribution non uniforme peuvent signaler des requêtes inefficaces ou des problèmes de performance de la base de données.
                """)
            if 'DBCALLS' in df_hitlist.columns and df_hitlist['DBCALLS'].sum() > 0:
                df_hitlist['DBCALLS'] = pd.to_numeric(df_hitlist['DBCALLS'], errors='coerce').fillna(0).astype(float)
                if df_hitlist['DBCALLS'].nunique() > 1:
                    fig_dbcalls_dist = ff.create_distplot([df_hitlist['DBCALLS'].dropna()], ['DBCALLS'], show_rug=False, show_hist=False)
                    fig_dbcalls_dist.update_layout(title_text="Distribution des Appels Base de Données (DBCALLS)", xaxis_title='Nombre d\'Appels DB', yaxis_title='Densité')
                    fig_dbcalls_dist.data[0].line.color = 'darkblue'
                    st.plotly_chart(fig_dbcalls_dist, use_container_width=True)
                else:
                    st.info("La colonne 'DBCALLS' contient des valeurs uniques ou est vide après filtrage, impossible de créer une courbe de densité.")
            else:
                st.info("Colonne 'DBCALLS' manquante ou total est zéro/vide après filtrage.")

            st.subheader("Aperçu des Données Hitlist DB Filtrées")
            # TASKTYPE removal: Ensure TASKTYPE is not in this list
            columns_to_display_hitlist = [col for col in df_hitlist.columns if col not in ['TASKTYPE', 'FULL_DATETIME']]
            st.dataframe(df_hitlist[columns_to_display_hitlist].head())
        else:
            st.warning("Données Hitlist DB non disponibles ou filtrées à vide.")

    elif st.session_state.current_section == "Performance des Processus de Travail":
        # --- Onglet 5: Performance des Processus de Travail (AL_GET_PERFORMANCE_final_cleaned_clean.xlsx) ---
        st.header("🚀 Performance des Processus de Travail (Work Processes)")
        df_perf = dfs['performance'].copy()
        if selected_wp_types:
            df_perf = df_perf[df_perf['WP_TYP'].isin(selected_wp_types)]

        if not df_perf.empty:
            st.subheader("Temps CPU Moyen par Type de Processus de Travail (WP_TYP)")
            st.markdown("""
                Ce graphique compare la consommation moyenne de temps CPU par différents types de processus de travail (par exemple, dialogue, batch, spool). 
                Il permet de cibler l'optimisation des processus les plus gourmands en CPU et d'assurer une répartition équilibrée des ressources.
                """)
            if 'WP_TYP' in df_perf.columns and 'WP_CPU_SECONDS' in df_perf.columns and df_perf['WP_CPU_SECONDS'].sum() > 0:
                df_perf['WP_CPU_SECONDS'] = pd.to_numeric(df_perf['WP_CPU_SECONDS'], errors='coerce').fillna(0).astype(float)
                avg_cpu_by_wp_type = df_perf.groupby('WP_TYP', as_index=False)['WP_CPU_SECONDS'].mean().sort_values(by='WP_CPU_SECONDS', ascending=False)
                
                if not avg_cpu_by_wp_type.empty and avg_cpu_by_wp_type['WP_CPU_SECONDS'].sum() > 0:
                    fig_cpu_wp_type = px.bar(avg_cpu_by_wp_type, x='WP_TYP', y='WP_CPU_SECONDS',
                                            title="Temps CPU Moyen (secondes) par Type de Processus de Travail",
                                            labels={'WP_TYP': 'Type de Processus de Travail', 'WP_CPU_SECONDS': 'Temps CPU Moyen (s)'},
                                            color='WP_CPU_SECONDS', color_continuous_scale=px.colors.sequential.Cividis)
                    st.plotly_chart(fig_cpu_wp_type, use_container_width=True)
                else:
                    st.info("Pas de données valides pour le Temps CPU Moyen par Type de Processus de Travail après filtrage.")
            else:
                st.info("Colonnes 'WP_TYP' ou 'WP_CPU_SECONDS' manquantes ou WP_CPU_SECONDS total est zéro/vide.")

            st.subheader("Distribution des Temps d'Attente (WP_IWAIT)")
            st.markdown("""
                Ce graphique présente la distribution des temps d'attente (WP_IWAIT) pour les processus de travail. 
                Des temps d'attente élevés peuvent indiquer des goulots d'étranglement tels que la contention des ressources, 
                les blocages de base de données ou la lenteur des opérations d'E/S, nécessitant une investigation approfondie.
                """)
            if 'WP_IWAIT' in df_perf.columns and df_perf['WP_IWAIT'].sum() > 0:
                df_perf['WP_IWAIT'] = pd.to_numeric(df_perf['WP_IWAIT'], errors='coerce').fillna(0).astype(float)
                if df_perf['WP_IWAIT'].nunique() > 1:
                    fig_iwait_dist = ff.create_distplot([df_perf['WP_IWAIT'].dropna()], ['WP_IWAIT'], show_rug=False, show_hist=False)
                    fig_iwait_dist.update_layout(title_text="Distribution des Temps d'Attente (WP_IWAIT)", xaxis_title="Temps d'Attente (ms)", yaxis_title='Densité')
                    fig_iwait_dist.data[0].line.color = 'red'
                    st.plotly_chart(fig_iwait_dist, use_container_width=True)
                else:
                    st.info("La colonne 'WP_IWAIT' contient des valeurs uniques ou est vide après filtrage, impossible de créer une courbe de densité.")
            else:
                st.info("Colonne 'WP_IWAIT' manquante ou total est zéro/vide après filtrage.")

            st.subheader("Statut des Processus de Travail (WP_STATUS)")
            st.markdown("""
                Ce graphique à barres montre la répartition des processus de travail par leur statut (par exemple, en cours, en attente, stoppé). 
                Une grande proportion de processus en attente ou stoppés peut indiquer des problèmes de configuration, 
                des blocages ou des insuffisances de ressources qui affectent la disponibilité et la performance.
                """)
            if 'WP_STATUS' in df_perf.columns and not df_perf['WP_STATUS'].isnull().all():
                status_counts = df_perf['WP_STATUS'].value_counts().reset_index()
                status_counts.columns = ['WP_STATUS', 'Count']
                if not status_counts.empty and status_counts['Count'].sum() > 0:
                    fig_wp_status = px.bar(status_counts, x='WP_STATUS', y='Count',
                                        title="Répartition par Statut des Processus de Travail",
                                        labels={'WP_STATUS': 'Statut du Processus', 'Count': 'Nombre'},
                                        color='Count', color_continuous_scale=px.colors.sequential.Aggrnyl)
                    st.plotly_chart(fig_wp_status, use_container_width=True)
                else:
                    st.info("Pas de données valides pour les Statuts des Processus de Travail après filtrage.")
            else:
                st.info("Colonne 'WP_STATUS' manquante ou vide après filtrage.")

            st.subheader("Aperçu des Données Performance Filtrées")
            st.dataframe(df_perf.head())
        else:
            st.warning("Données de performance des processus de travail non disponibles ou filtrées à vide.")

    elif st.session_state.current_section == "Résumé des Traces de Performance SQL":
        # --- Onglet 6: Résumé des Traces de Performance SQL (performance_trace_summary_final_cleaned_clean.xlsx) ---
        st.header("⚡ Résumé des Traces de Performance SQL")
        df_sql = dfs['sql_trace_summary'].copy()

        if not df_sql.empty:
            st.subheader("Top 10 des Statements SQL par Temps d'Exécution (EXECTIME)")
            st.markdown("""
                Ce graphique identifie les 10 requêtes SQL les plus longues en termes de temps d'exécution. 
                L'optimisation de ces requêtes est souvent le moyen le plus efficace d'améliorer radicalement la performance globale de la base de données.
                """)
            if 'SQLSTATEM' in df_sql.columns and 'EXECTIME' in df_sql.columns and df_sql['EXECTIME'].sum() > 0:
                df_sql['EXECTIME'] = pd.to_numeric(df_sql['EXECTIME'], errors='coerce').fillna(0).astype(float)
                top_sql_exec_time = df_sql.groupby('SQLSTATEM', as_index=False)['EXECTIME'].sum().nlargest(10, 'EXECTIME')
                
                if not top_sql_exec_time.empty and top_sql_exec_time['EXECTIME'].sum() > 0:
                    fig_sql_exec_time = px.bar(top_sql_exec_time, x='SQLSTATEM', y='EXECTIME',
                                            title="Top 10 Statements SQL par Temps d'Exécution Total (ms)",
                                            labels={'SQLSTATEM': 'Statement SQL', 'EXECTIME': "Temps d'Exécution (ms)"},
                                            color='EXECTIME', color_continuous_scale=px.colors.sequential.Oranges)
                    st.plotly_chart(fig_sql_exec_time, use_container_width=True)
                else:
                    st.info("Pas de données valides pour les Top 10 Statements SQL par Temps d'Exécution après filtrage.")
            else:
                st.info("Colonnes 'SQLSTATEM' ou 'EXECTIME' manquantes ou EXECTIME total est zéro/vide.")

            st.subheader("Top 10 des Statements SQL par Nombre Total d'Exécutions (TOTALEXEC)")
            st.markdown("""
                Ce graphique met en lumière les 10 requêtes SQL les plus fréquemment exécutées. 
                Même si une requête n'est pas la plus lente individuellement, son exécution fréquente peut en faire un goulot d'étranglement majeur. 
                L'optimisation de ces requêtes peut avoir un impact significatif sur la performance globale.
                """)
            if 'SQLSTATEM' in df_sql.columns and 'TOTALEXEC' in df_sql.columns and df_sql['TOTALEXEC'].sum() > 0:
                df_sql['TOTALEXEC'] = pd.to_numeric(df_sql['TOTALEXEC'], errors='coerce').fillna(0).astype(float)
                top_sql_total_exec = df_sql.groupby('SQLSTATEM', as_index=False)['TOTALEXEC'].sum().nlargest(10, 'TOTALEXEC')
                
                if not top_sql_total_exec.empty and top_sql_total_exec['TOTALEXEC'].sum() > 0:
                    fig_sql_total_exec = px.bar(top_sql_total_exec, x='SQLSTATEM', y='TOTALEXEC',
                                            title="Top 10 Statements SQL par Nombre Total d'Exécutions",
                                            labels={'SQLSTATEM': 'Statement SQL', 'TOTALEXEC': "Nombre Total d'Exécutions"},
                                            color='TOTALEXEC', color_continuous_scale=px.colors.sequential.Blues)
                    st.plotly_chart(fig_sql_total_exec, use_container_width=True)
                else:
                    st.info("Pas de données valides pour les Top 10 Statements SQL par Nombre Total d'Exécutions après filtrage.")
            else:
                st.info("Colonnes 'SQLSTATEM' ou 'TOTALEXEC' manquantes ou TOTALEXEC total est zéro/vide.")

            st.subheader("Distribution du Temps par Exécution (TIMEPEREXE)")
            st.markdown("""
                Cette courbe de densité visualise la distribution du temps moyen passé par chaque exécution de requête SQL. 
                Elle aide à identifier si les requêtes sont généralement rapides avec quelques exceptions, 
                ou si elles sont globalement lentes, nécessitant une approche différente d'optimisation.
                """)
            if 'TIMEPEREXE' in df_sql.columns and df_sql['TIMEPEREXE'].sum() > 0:
                df_sql['TIMEPEREXE'] = pd.to_numeric(df_sql['TIMEPEREXE'], errors='coerce').fillna(0).astype(float)
                if df_sql['TIMEPEREXE'].nunique() > 1:
                    fig_timeperexe_dist = ff.create_distplot([df_sql['TIMEPEREXE'].dropna()], ['TIMEPEREXE'], show_rug=False, show_hist=False)
                    fig_timeperexe_dist.update_layout(title_text="Distribution du Temps par Exécution (TIMEPEREXE)", xaxis_title="Temps par Exécution (ms)", yaxis_title='Densité')
                    fig_timeperexe_dist.data[0].line.color = 'darkgreen'
                    st.plotly_chart(fig_timeperexe_dist, use_container_width=True)
                else:
                    st.info("La colonne 'TIMEPEREXE' contient des valeurs uniques ou est vide après filtrage, impossible de créer une courbe de densité.")
            else:
                st.info("Colonne 'TIMEPEREXE' manquante ou total est zéro/vide après filtrage.")

            st.subheader("Aperçu des Données SQL Trace Summary Filtrées")
            st.dataframe(df_sql.head())
        else:
            st.warning("Données de résumé des traces SQL non disponibles ou filtrées à vide.")

    elif st.session_state.current_section == "Analyse des Utilisateurs":
        # --- Onglet 7: Analyse des Utilisateurs (usr02_data.xlsx) ---
        st.header("👥 Analyse des Utilisateurs SAP (USR02)")
        df_usr02 = dfs['usr02'].copy()

        if not df_usr02.empty:
            st.subheader("Nombre d'Utilisateurs par Type (USTYP)")
            st.markdown("""
                Ce graphique en camembert présente la répartition des utilisateurs par leur type (par exemple, dialogue, service, système). 
                Il est utile pour la conformité et la sécurité, permettant d'identifier les proportions de différents types de comptes et de détecter les anomalies.
                """)
            if 'USTYP' in df_usr02.columns and not df_usr02['USTYP'].isnull().all():
                user_type_counts = df_usr02['USTYP'].value_counts().reset_index()
                user_type_counts.columns = ['USTYP', 'Count']
                if not user_type_counts.empty and user_type_counts['Count'].sum() > 0:
                    fig_user_type = px.pie(user_type_counts, values='Count', names='USTYP',
                                        title="Répartition des Utilisateurs par Type (USTYP)",
                                        hole=0.3)
                    fig_user_type.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_user_type, use_container_width=True)
                else:
                    st.info("Pas de données valides pour le Nombre d'Utilisateurs par Type après filtrage.")
            else:
                st.info("Colonne 'USTYP' manquante ou vide après filtrage.")

            st.subheader("Distribution des Dernières Connexions (GLTGB_DATE)")
            st.markdown("""
                Cet histogramme visualise la fréquence des dernières connexions des utilisateurs au fil du temps. 
                Il peut aider à identifier les comptes inactifs qui devraient être examinés pour des raisons de sécurité ou de conformité, 
                et à comprendre les patterns de connexion.
                """)
            if 'GLTGB_DATE' in df_usr02.columns and not df_usr02['GLTGB_DATE'].isnull().all():
                # Filter out dates before a reasonable start, e.g., 2000-01-01 if there are many '00000000'
                df_usr02_dates = df_usr02.dropna(subset=['GLTGB_DATE'])
                if not df_usr02_dates.empty and len(df_usr02_dates['GLTGB_DATE'].unique()) > 1:
                    fig_last_logon = px.histogram(df_usr02_dates, x='GLTGB_DATE',
                                                title="Distribution des Dernières Connexions des Utilisateurs",
                                                labels={'GLTGB_DATE': 'Date de Dernière Connexion'},
                                                nbins=30, # Adjust number of bins as needed
                                                color_discrete_sequence=['cadetblue'])
                    st.plotly_chart(fig_last_logon, use_container_width=True)
                else:
                    st.info("La colonne 'GLTGB_DATE' contient des valeurs uniques ou est vide après filtrage, impossible de créer un histogramme.")
            else:
                st.info("Colonne 'GLTGB_DATE' manquante ou vide après filtrage.")

            st.subheader("Aperçu des Données Utilisateurs Filtrées")
            st.dataframe(df_usr02.head())
        else:
            st.warning("Données utilisateurs (USR02) non disponibles ou filtrées à vide.")

    elif st.session_state.current_section == "Détection d'Anomalies":
        # --- Onglet 8: Détection d'Anomalies (sur Hitlist DB) ---
        st.header("🚨 Détection d'Anomalies sur le Temps de Réponse (Hitlist DB)")
        st.markdown("""
            Cette section utilise une méthode simple de détection d'anomalies basée sur l'écart type pour identifier les temps de réponse exceptionnellement élevés.
            Les points marqués comme 'Anomalie' sont ceux qui dépassent un seuil défini (moyenne + 3 * écart-type), indiquant des performances potentiellement problématiques.
            """)
        
        df_anomalies = dfs['hitlist_db'].copy()
        if not df_anomalies.empty and 'RESPTI' in df_anomalies.columns and 'FULL_DATETIME' in df_anomalies.columns:
            # Ensure columns are numeric and datetime
            df_anomalies['RESPTI'] = pd.to_numeric(df_anomalies['RESPTI'], errors='coerce').fillna(0).astype(float)
            df_anomalies['FULL_DATETIME'] = pd.to_datetime(df_anomalies['FULL_DATETIME'], errors='coerce')
            
            df_anomalies_clean = df_anomalies.dropna(subset=['RESPTI', 'FULL_DATETIME']).copy()

            if not df_anomalies_clean.empty and df_anomalies_clean['RESPTI'].nunique() > 1:
                # Calculer la moyenne mobile et l'écart type mobile
                window = 50 # Fenêtre pour la moyenne mobile et l'écart type
                df_anomalies_clean['Mean_RESPTI'] = df_anomalies_clean['RESPTI'].rolling(window=window, min_periods=1).mean()
                df_anomalies_clean['Std_RESPTI'] = df_anomalies_clean['RESPTI'].rolling(window=window, min_periods=1).std()

                # Définir le seuil pour l'anomalie (ex: 3 écarts types au-dessus de la moyenne)
                n_std = st.slider("Facteur d'Écart Type pour les Anomalies", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
                df_anomalies_clean['Upper_Bound'] = df_anomalies_clean['Mean_RESPTI'] + n_std * df_anomalies_clean['Std_RESPTI']
                
                # Identifier les anomalies
                df_anomalies_clean['Anomaly'] = df_anomalies_clean['RESPTI'] > df_anomalies_clean['Upper_Bound']

                # Visualisation
                fig_anomalies = px.line(df_anomalies_clean, x='FULL_DATETIME', y='RESPTI',
                                        title="Détection d'Anomalies dans le Temps de Réponse (RESPTI)",
                                        labels={'FULL_DATETIME': 'Date et Heure', 'RESPTI': 'Temps de Réponse (ms)'})
                
                # Ajouter la moyenne mobile et les bornes supérieures
                fig_anomalies.add_scatter(x=df_anomalies_clean['FULL_DATETIME'], y=df_anomalies_clean['Mean_RESPTI'], mode='lines', name=f'Moyenne Mobile ({window} pts)', line=dict(color='orange', dash='dash'))
                fig_anomalies.add_scatter(x=df_anomalies_clean['FULL_DATETIME'], y=df_anomalies_clean['Upper_Bound'], mode='lines', name=f'Seuil Anomalie (+{n_std} StdDev)', line=dict(color='red', dash='dot'))

                # Mettre en évidence les anomalies
                anomalies_points = df_anomalies_clean[df_anomalies_clean['Anomaly']]
                if not anomalies_points.empty:
                    fig_anomalies.add_scatter(x=anomalies_points['FULL_DATETIME'], y=anomalies_points['RESPTI'], mode='markers', name='Anomalie',
                                            marker=dict(color='red', size=8, symbol='x'))
                
                st.plotly_chart(fig_anomalies, use_container_width=True)

                st.subheader("Détails des Anomalies Détectées")
                if not anomalies_points.empty:
                    st.dataframe(anomalies_points[['FULL_DATETIME', 'RESPTI', 'Mean_RESPTI', 'Upper_Bound']].sort_values(by='RESPTI', ascending=False))
                else:
                    st.info("Aucune anomalie détectée avec les paramètres actuels.")
            else:
                st.info("La colonne 'RESPTI' ne contient pas suffisamment de variation pour la détection d'anomalies, ou les données sont vides après nettoyage.")
        else:
            st.warning("Données Hitlist DB non disponibles ou colonnes requises (RESPTI, FULL_DATETIME) manquantes pour la détection d'anomalies.")

    elif st.session_state.current_section == "Prédiction de Performance (ML)":
        # --- Onglet 9: Prédiction de Performance (ML) ---
        st.header("🤖 Prédiction de Performance (ML)")
        st.markdown("""
            Cette section utilise un modèle de Machine Learning (Forêt Aléatoire) pour prédire le temps de réponse (RESPTI) 
            basé sur d'autres métriques. Le modèle est entraîné sur les données 'Hitlist DB'.
            """)

        df_ml = dfs['hitlist_db'].copy()

        if not df_ml.empty:
            st.subheader("Configuration et Entraînement du Modèle")
            
            # Sélection de la colonne cible pour la prédiction
            available_target_cols = ['RESPTI', 'PROCTI', 'CPUTI', 'DBCALLS'] # Add other relevant numeric columns if applicable
            target_ml_column = st.selectbox("Sélectionner la colonne cible pour la prédiction (Y)", available_target_cols, index=0)

            # Sélection des colonnes de features
            # Exclure la cible, les identifiants et les colonnes de date/heure complexes ou non numériques
            all_ml_cols = [col for col in df_ml.columns if pd.api.types.is_numeric_dtype(df_ml[col]) or pd.api.types.is_string_dtype(df_ml[col])]
            
            # Removed irrelevant features and potentially problematic ones like 'ENDTIME', 'FULL_DATETIME'
            exclude_features = [target_ml_column, 'FULL_DATETIME', 'ENDDATE', 'ENDTIME', 'ENDTIME_STR', 'WPID', 'ACCOUNT', 'REPORT', 'ROLLKEY', 'PRIVMODE', 'WPRESTART']
            # Also exclude any columns that might be entirely NaN or have very few unique values
            final_features = [col for col in all_ml_cols if col not in exclude_features and df_ml[col].nunique() > 1]
            
            if not final_features:
                st.warning("Aucune colonne numérique valide n'est disponible pour les features après exclusion. Le modèle ne peut pas être entraîné.")
            else:
                selected_features = st.multiselect("Sélectionner les features (X) pour la prédiction", final_features, default=final_features[:min(len(final_features), 5)]) # Select first 5 by default

                if target_ml_column not in df_ml.columns or df_ml[target_ml_column].isnull().all():
                    st.error(f"La colonne cible '{target_ml_column}' est manquante ou vide dans les données.")
                elif not selected_features:
                    st.error("Veuillez sélectionner au moins une feature pour l'entraînement du modèle.")
                else:
                    # Préparation des données
                    X = df_ml[selected_features]
                    y = df_ml[target_ml_column]

                    # Gérer les valeurs non numériques (ex: strings) pour l'encodage One-Hot
                    # Gérer les valeurs manquantes avant l'entraînement
                    numeric_features = X.select_dtypes(include=np.number).columns
                    categorical_features = X.select_dtypes(include='object').columns

                    numeric_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='mean'))
                    ])

                    categorical_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                    ])

                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', numeric_transformer, numeric_features),
                            ('cat', categorical_transformer, categorical_features)
                        ])

                    # Création du pipeline du modèle
                    model = Pipeline(steps=[('preprocessor', preprocessor),
                                            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

                    # Division des données
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    st.write("Entraînement du modèle en cours...")
                    try:
                        model.fit(X_train, y_train)
                        st.success("Modèle entraîné avec succès!")

                        # Évaluation du modèle
                        y_pred = model.predict(X_test)
                        r2 = r2_score(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)

                        st.subheader("Évaluation du Modèle")
                        st.write(f"**Coefficient de Détermination (R²):** {r2:.2f}")
                        st.write(f"**Erreur Absolue Moyenne (MAE):** {mae:.2f}")

                        st.markdown("""
                            * **R² (Coefficient de Détermination)**: Mesure la proportion de la variance de la variable dépendante qui est prévisible à partir des variables indépendantes. Un R² de 1.0 indique que le modèle explique 100% de la variance, 0.0 indique qu'il n'explique rien.
                            * **MAE (Erreur Absolue Moyenne)**: Représente la moyenne des erreurs absolues entre les prédictions et les valeurs réelles. C'est une mesure de l'erreur moyenne en valeur absolue.
                            """)

                        # Importance des features
                        st.subheader("Importance des Features")
                        if hasattr(model.named_steps['regressor'], 'feature_importances_'):
                            # Get feature names after one-hot encoding
                            ohe_feature_names = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
                            all_processed_features = list(numeric_features) + list(ohe_feature_names)
                            
                            feature_importances = pd.Series(model.named_steps['regressor'].feature_importances_, index=all_processed_features)
                            
                            # Filter out features with 0 importance if they are from one-hot encoding of categories not present in train
                            feature_importances = feature_importances[feature_importances > 0].sort_values(ascending=False)
                            
                            if not feature_importances.empty:
                                fig_importance = px.bar(feature_importances.head(10), # Top 10 features
                                                        x=feature_importances.head(10).index, y=feature_importances.head(10).values,
                                                        title="Top 10 Importances des Features",
                                                        labels={'x': 'Feature', 'y': 'Importance'},
                                                        color=feature_importances.head(10).values, color_continuous_scale=px.colors.sequential.Viridis)
                                st.plotly_chart(fig_importance, use_container_width=True)
                            else:
                                st.info("Aucune feature n'a d'importance significative (toutes sont nulles ou très faibles).")
                        else:
                            st.info("Le modèle ne supporte pas l'extraction de l'importance des features.")

                        # Visualisation des prédictions vs. valeurs réelles
                        st.subheader("Prédictions vs. Valeurs Réelles")
                        df_results = pd.DataFrame({'Valeurs Réelles': y_test, 'Prédictions': y_pred})
                        fig_pred = px.scatter(df_results.sample(min(500, len(df_results)), random_state=42), # Échantillon pour la visibilité
                                            x='Valeurs Réelles', y='Prédictions',
                                            title=f"Prédictions vs. Valeurs Réelles pour {target_ml_column}",
                                            labels={'Valeurs Réelles': f'Valeurs Réelles de {target_ml_column}', 'Prédictions': f'Prédictions de {target_ml_column}'},
                                            trendline='ols', # Ligne de régression linéaire
                                            opacity=0.6)
                        fig_pred.update_traces(marker_size=5)
                        st.plotly_chart(fig_pred, use_container_width=True)

                        st.write("""
                            Un bon modèle aura des points groupés près de la ligne diagonale (où les prédictions égalent les valeurs réelles).
                            """)
                    except Exception as e:
                        st.error(f"Erreur lors de l'entraînement ou de l'évaluation du modèle : {e}")

# --- Téléchargement du script ---
st.sidebar.markdown("---")
st.sidebar.subheader("Télécharger le Script")
script_code = io.StringIO()
with open(__file__, "r", encoding="utf-8") as f:
    script_code.write(f.read())
st.sidebar.download_button(
    label="Télécharger le Script",
    data=script_code.getvalue(),
    file_name="TTEST_dashboard_sap_corrected.py",
    mime="text/x-python"
)
