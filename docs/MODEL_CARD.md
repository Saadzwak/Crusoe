# Model card — MH-PINN « presse de cuisson » (v2)

**Checkpoint courant** : [pinn/models/mh_pinn_v2.pt](../pinn/models/mh_pinn_v2.pt)
(447 Ko, commité exprès pour que la démo tourne sur clone frais).
Architecture : adaptateurs d'entrée par capteur → corps LSTM partagé
(hidden 128) → 7 sorties entraînées. Chargement :
`MHPinn(); model.load_state_dict(torch.load("pinn/models/mh_pinn_v2.pt", weights_only=True))`.
Entraînement reproductible : `python -m pinn.train --epochs 10` (CPU, ~15 min).

## Données et provenance (étiquetée jusque dans la télémétrie signée)

| Source | Provenance | Rôle |
|---|---|---|
| CWRU (161 fichiers, grille complète) | **RÉEL** (mesures accéléromètres) | classes de défaut roulement ; vérité du capteur retenu fan-end |
| NASA IMS (3 tests run-to-failure) | **RÉEL** (casses physiques réelles) | santé/fatigue ; calibration Basquin sur vraies durées de vie |
| NASA C-MAPSS FD001 | SIMULÉ (NASA haute-fidélité) | tête RUL |
| AI4I 2020 | SIMULÉ (règles génératives documentées, vérifiées 115/115) | tête thermique ; capteur retenu température |
| Cycles de presse | **SYNTHÉTIQUE** (générés in-repo, Kamal-Sourour + ODE, aucun dataset public n'existe) | tête pression/cuisson — jamais présenté comme mesuré |

## Capacités validées (val/test, runs/v2/metrics.json)

- **Reconstruction de capteur retenu (la démo phare)** : température process
  AI4I reconstruite à **0,56 K MAE (7,4 % de la plage)** sans son capteur ;
  accéléromètre fan-end CWRU (RÉEL, jamais en entrée) reconstruit depuis le
  drive-end à **12,8 % d'erreur RMS médiane**.
- Vibration : **97,8 %** (4 classes) ; santé IMS corr 0,992.
- Fatigue : D corr **0,992** avec la position de vie réelle ; sur la dernière
  fenêtre avant la casse réelle → « remplacer maintenant » (D=0,935).
- RUL : RMSE test **13,7 cycles** ; pente **−0,99**/cycle là où la cible
  décroît (contrainte physique appliquée sous le genou observable par unité).
- Pression/cuisson (SYNTHÉTIQUE) : détection de fuite 0,87 ; α_end MAE 0,011,
  calibration quasi-identité (1,11).
- Contrat de sortie : `SignedReading` HMAC-SHA256 — **testé en conditions
  réelles** contre les modules réels de la couche information :
  [scripts/test_cross_layer_contract.py](../scripts/test_cross_layer_contract.py) (6 checks).

## Limites connues et vérifiées (à dire en démo, pas à cacher)

1. **Tête pression = 100 % synthétique.** Constantes de temps τ plausibles,
   non mesurées (catégorie c) ; les plages P/T/durée sont, elles, sourcées.
2. **Détection de sous-cuisson au seuil α<0,90 : 3/6 attrapées, 1 fausse
   alarme** sur le val — n=6 cas, les ratés sont à ±MAE du seuil (limite de
   mesure, pas de tendance) ; la fausse alarme (apparue avec la calibration
   quasi-identité) n'est **pas élucidée individuellement** — v3. La métrique
   robuste est la corrélation de profondeur (0,89) ; pour alerter en démo,
   utiliser α̂<0,93 (marge).
3. Le seuil « cuit/sous-cuit » 0,90 est un choix démo (c), pas une norme.
4. Vibration : variance inter-runs ±1 pt (97,8-99,0 % selon seed/run) — non
   caractérisée par multi-seeds.
5. RUL : dans la région sous-genou la pente moyenne overshoot (−1,28) ;
   la pente globale −0,99 est la bonne lecture.
6. Zones vibratoires ISO 10816-3 : norme industrielle **générique**, pas
   spécifique presse (rappel obligatoire à chaque citation).
7. IMS test 1 : mapping canal→palier pris du readme, non re-vérifié contre le
   PDF (les têtes n'en dépendent pas ; le test 2 utilisé est vérifié).
8. Basquin : IMS = un seul niveau de charge → famille (b, σf′) calibrée, pas
   un fit 2 paramètres (dit tel quel dans [physics_heads.md](physics_heads.md) §4).

Investigations complètes (méthodes, chiffres bruts) :
[v1_physics_diagnosis.md](v1_physics_diagnosis.md) ·
[v2_equation_stress_test.md](v2_equation_stress_test.md) ·
`runs/v2/{metrics,diagnosis,stress_test}.json`.
