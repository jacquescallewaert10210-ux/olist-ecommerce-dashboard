OLIST DASHBOARD FINAL — INSTALLATION
====================================

1. Mets ces éléments dans le même dossier :
   - app_olist_final.py
   - requirements_olist_final.txt
   - dossier data/

2. Dans data/, place les 9 fichiers CSV Olist :
   - olist_customers_dataset.csv
   - olist_geolocation_dataset.csv
   - olist_order_items_dataset.csv
   - olist_order_payments_dataset.csv
   - olist_order_reviews_dataset.csv
   - olist_orders_dataset.csv
   - olist_products_dataset.csv
   - olist_sellers_dataset.csv
   - product_category_name_translation.csv

3. Dans PowerShell :
   python -m pip install -r requirements_olist_final.txt

4. Lance le dashboard :
   python -m streamlit run app_olist_final.py

CHANGEMENTS DE CETTE VERSION
============================
- Le filtre "État client" est maintenant un MULTISELECT.
- Tu peux sélectionner plusieurs États en même temps.
- Si aucun État n'est sélectionné, tous les États sont conservés.
- Les États sélectionnés sont appliqués à tous les KPI et à tous les graphiques.
- La comparaison avec la période précédente conserve également la sélection multi-États.
- Le graphique Sankey a été remplacé par des BARRES EMPILÉES.
- Les barres empilées montrent les effectifs de satisfaction :
    1–2 étoiles = insatisfaction
    3 étoiles   = satisfaction intermédiaire
    4–5 étoiles = satisfaction élevée
  pour :
    - En retard
    - À l'heure / en avance
- La carte conserve son fond géographique Plotly visible.
- Aucun GeoJSON, GeoPandas ou service cartographique externe n'est nécessaire.

POUR LA SOUTENANCE
==================
Exemple de démonstration des filtres :
1. Sélectionne SP, RJ et MG dans "État client — sélection multiple".
2. Clique "Appliquer les filtres".
3. Montre que les KPI, les ventes, la satisfaction, la géographie et les corrélations changent.
4. Clique "Réinitialiser" pour revenir à l'analyse complète.
