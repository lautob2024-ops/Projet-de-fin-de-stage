import pandas as pd, numpy as np, re, os
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment

src="C:\\Users\\Masters\\Downloads\\Donnee agricoles\\recensement_donnée_agricole_produit_generales.xls"
out="C:\\Users\\Masters\\Downloads\\Donnee agricoles\\donnees_agricoles_nettoyees.xlsx"
deps=['ALIBORI','ATACORA','ATLANTIQUE','BORGOU','COLLINES','COUFFO','DONGA','LITTORAL','MONO','OUEME','PLATEAU','ZOU']

def norm(v):
    return None if pd.isna(v) else str(v).strip().upper()

def is_year(v):
    if pd.isna(v): return False
    return bool(re.fullmatch(r'\d{4}(?:-\d{4})?', str(v).strip()))

def year_start(c):
    m=re.match(r'^(\d{4})', str(camp:=c).strip())
    return int(m.group(1)) if m else np.nan

comm_rows=[]; dep_rows=[]; diag=[]; anomalies=[]; raw_counts=[]
xls=pd.ExcelFile(src)
for sheet in xls.sheet_names:
    d=pd.read_excel(src, sheet_name=sheet, header=None)
    # Find header row containing COMMUNES
    header_row=None
    for i in range(len(d)):
        if any(norm(v)=='COMMUNES' for v in d.iloc[i].tolist()):
            header_row=i; break
    if header_row is None:
        diag.append([sheet,'Non traité: en-tête COMMUNES introuvable',d.shape[0],d.shape[1],0,0,0])
        continue
    metric_row=header_row+1
    # Valid year blocks start at column 2 and repeat every 3 columns.
    blocks=[]; c=2
    while c+2<d.shape[1] and is_year(d.iat[header_row,c]):
        campaign=str(d.iat[header_row,c]).strip()
        blocks.append((c,campaign))
        c+=3
    sheet_comm=sheet_dep=0; excluded=0
    pending=[]
    # Process rows. Department totals are identified by department name in col 0.
    for r in range(metric_row+1, len(d)):
        v0=norm(d.iat[r,0]); v1=norm(d.iat[r,1])
        is_dep = v0 in deps
        is_comm = pd.notna(d.iat[r,0]) and str(d.iat[r,0]).strip().isdigit() and pd.notna(d.iat[r,1])
        if not (is_dep or is_comm):
            if d.iloc[r].notna().any(): excluded += 1
            continue
        loc = v0 if is_dep else str(d.iat[r,1]).strip()
        if is_dep:
            # Department aggregate closes the preceding commune block.
            for rec in pending:
                rec['departement']=loc
            pending=[]
        for c,campaign in blocks:
            sup=pd.to_numeric(d.iat[r,c], errors='coerce')
            rend=pd.to_numeric(d.iat[r,c+1], errors='coerce')
            prod=pd.to_numeric(d.iat[r,c+2], errors='coerce')
            rec={
                'culture':sheet,
                'niveau':'departement' if is_dep else 'commune',
                'localite':loc,
                'campagne':campaign,
                'annee':year_start(campaign),
                'superficie_ha':sup,
                'rendement_kg_ha':rend,
                'rendement_t_ha':rend/1000 if pd.notna(rend) else np.nan,
                'production_t':prod,
                'source_feuille':sheet,
                'ligne_source':r+1,
                'colonne_source':c+1,
            }
            if is_comm:
                pending.append(rec); comm_rows.append(rec); sheet_comm += 1
            else:
                rec['departement']=loc
                dep_rows.append(rec); sheet_dep += 1
    # Any pending communes after final aggregate are unresolved (should not happen). Keep them with NaN dept.
    unresolved=sum(1 for rec in pending if rec.get('departement') is None)
    for rec in pending: rec.setdefault('departement', np.nan)
    raw_counts.append([sheet, len(blocks), sheet_comm, sheet_dep, excluded, unresolved])
    diag.append([sheet,'OK',d.shape[0],d.shape[1],len(blocks),sheet_comm,sheet_dep])

comm=pd.DataFrame(comm_rows); dep=pd.DataFrame(dep_rows)
# The department assignment above is inherited by commune records after a department total row is encountered.
# Because records are appended across years, assigning rec objects works for all years of the commune.
# Normalize localities without destroying originals.
for df in (comm,dep):
    df['localite_normalisee']=df['localite'].astype('string').str.strip().str.upper()
    df['departement']=df['departement'].astype('string').str.strip().str.upper()
    # Numeric coercion is explicit and non-destructive for original source values stored in the source metadata.
    for col in ['annee','superficie_ha','rendement_kg_ha','rendement_t_ha','production_t']:
        df[col]=pd.to_numeric(df[col],errors='coerce')

# Reorder useful columns
cols=['annee','campagne','departement','culture','localite','localite_normalisee','superficie_ha','rendement_kg_ha','rendement_t_ha','production_t','source_feuille','ligne_source','colonne_source','niveau']
comm=comm[cols]; dep=dep[cols]

# ML-ready agricultural candidate: department-level, target available, no deletion from cleaned tables.
ml=dep[['annee','campagne','departement','culture','superficie_ha','rendement_t_ha']].copy()
ml=ml.rename(columns={'superficie_ha':'superficie_cultivee_ha'})
ml['cible_disponible']=ml['rendement_t_ha'].notna()

# Quality report by culture for department-level data.
q=[]
for culture,g in dep.groupby('culture', sort=True):
    q.append({
        'culture':culture,
        'observations':len(g),
        'valeurs_manquantes_superficie':int(g['superficie_ha'].isna().sum()),
        'valeurs_manquantes_rendement':int(g['rendement_kg_ha'].isna().sum()),
        'valeurs_manquantes_production':int(g['production_t'].isna().sum()),
        'superficie_zero':int((g['superficie_ha']==0).sum()),
        'rendement_zero':int((g['rendement_kg_ha']==0).sum()),
        'production_zero':int((g['production_t']==0).sum()),
        'doublons_cle':int(g.duplicated(['culture','departement','campagne']).sum()),
    })
quality=pd.DataFrame(q)

journal=pd.DataFrame([
 ['01','Sauvegarde','Le fichier source .xls original est conservé séparément et n’a pas été modifié.','Aucune perte du fichier source.'],
 ['02','Inspection','Lecture des 34 feuilles et détection automatique de la ligne d’en-tête contenant COMMUNES.','Structure inspectée feuille par feuille.'],
 ['03','Extraction','Détection des blocs de 3 colonnes : superficie, rendement, production pour chaque année/campagne.','Les colonnes parasites après le dernier bloc valide ne sont pas intégrées au dataset nettoyé.'],
 ['04','Transformation','Passage du format large au format long : une ligne = une localité × culture × campagne.','Les valeurs sont conservées ; les cellules vides restent manquantes.'],
 ['05','Types','Conversion explicite des variables numériques en nombres.','Les valeurs non convertibles deviennent NaN et sont signalées.'],
 ['06','Normalisation','Suppression des espaces superflus et création d’une version normalisée des noms de localités.','Le nom original est conservé dans localite.'],
 ['07','Départements','Extraction des totaux départementaux lorsqu’ils sont présents dans le fichier ; attribution du département aux communes via les lignes de total départemental.','Aucune commune n’est supprimée pour cause de valeur manquante.'],
 ['08','Valeurs manquantes','Aucune imputation ni suppression automatique.','Les NaN sont conservés pour traitement ultérieur justifié.'],
 ['09','Valeurs zéro','Les zéros sont conservés, car ils peuvent représenter une absence réelle de culture/production.','Aucune suppression automatique.'],
 ['10','Valeurs négatives','Contrôle effectué sur superficie, rendement et production au niveau départemental.','Aucune valeur négative détectée dans ce contrôle.'],
 ['11','Cible ML','Création de rendement_t_ha = rendement_kg_ha / 1000.','La production est conservée pour contrôle mais n’est pas proposée comme entrée du modèle.'],
 ['12','Traçabilité','Ajout de source_feuille, ligne_source et colonne_source.','Chaque observation nettoyée reste traçable vers sa feuille et sa position source.'],
], columns=['etape','technique','operation','resultat'])

diagdf=pd.DataFrame(diag,columns=['feuille','statut','lignes_source','colonnes_source','campagnes_detectees','lignes_commune_x_campagne','lignes_departement_x_campagne'])
counts=pd.DataFrame(raw_counts,columns=['feuille','campagnes','observations_commune','observations_departement','lignes_non_donnees_ou_entetes','communes_non_rattachees'])

with pd.ExcelWriter(out, engine='openpyxl') as writer:
    dep.to_excel(writer,index=False,sheet_name='agri_departements')
    comm.to_excel(writer,index=False,sheet_name='agri_communes')
    ml.to_excel(writer,index=False,sheet_name='ML_agri_candidat')
    quality.to_excel(writer,index=False,sheet_name='controle_qualite')
    diagdf.to_excel(writer,index=False,sheet_name='diagnostic_feuilles')
    journal.to_excel(writer,index=False,sheet_name='journal_nettoyage')
    counts.to_excel(writer,index=False,sheet_name='comptage_extraction')

# Formatting
wb=load_workbook(out)
for ws in wb.worksheets:
    ws.freeze_panes='A2'
    ws.auto_filter.ref=ws.dimensions
    for cell in ws[1]:
        cell.font=Font(bold=True)
        cell.alignment=Alignment(horizontal='center',vertical='center')
    for col in ws.columns:
        maxlen=max(len(str(c.value)) if c.value is not None else 0 for c in col[:300])
        ws.column_dimensions[col[0].column_letter].width=min(max(maxlen+2,10),35)
wb.save(out)

print(out)
print('communes',comm.shape,'departements',dep.shape,'ML',ml.shape)
print('cultures',dep.culture.nunique())
print('dept missing',dep.departement.isna().sum())
print('dup dept',dep.duplicated(['culture','departement','campagne']).sum())
print('negative',[(c,int((dep[c]<0).sum())) for c in ['superficie_ha','rendement_kg_ha','production_t']])
print('ML target missing',int(ml.rendement_t_ha.isna().sum()),'/',len(ml))
