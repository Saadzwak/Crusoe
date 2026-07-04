# Step-7 — stress-test des équations : ce qui a changé, ce qui reste, pourquoi

Règle du brief : *une citation prouve qu'une équation est réelle, pas qu'elle
est le bon modèle pour CE système.* Chaque forme physique a donc été
confrontée aux données. Outillage : [pinn/stress_test.py](../pinn/stress_test.py),
données brutes : `runs/v2/stress_test.json`. Verdicts par question :

---

## 1. RUL — la pente dure à −1 sous le cap est-elle la bonne forme ? **NON, pas partout — corrigée**

**Mesure (A1)** : fit deux segments (plat → linéaire) sur l'indice de santé
composite de chacun des 100 moteurs train FD001. RUL au genou observable :
**médiane 92 cycles, IQR [79, 108], plage [63, 179]**. Le cap conventionnel de
125 (Heimes 2008) se situe vers le 90ᵉ percentile des genoux : pour le moteur
médian, la cible décroît pendant ~33 cycles alors que **les capteurs sont
encore plats** — la contrainte −1 y exigeait de prédire un déclin inobservable.

**Changement** : la contrainte de pente −1 n'est plus appliquée sous le cap
mais **sous le genou estimé par unité** (`estimate_knee_rul` dans
[pinn/physics/rul.py](../pinn/physics/rul.py), masque calculé par le loader) ;
ailleurs, seule la non-remontée (relu²) est exigée. La *cible* reste la
convention 125 (comparabilité d'évaluation) ; c'est la **portée de la
contrainte** qui devient conforme aux données. La métrique d'éval rapporte
désormais la pente dans la région du genou (là où −1 a un sens).

## 2. Interférence multi-tâches RUL — conflit ou capacité ? **CAPACITÉ — mesurée**

**Mesure (A2)** : cosinus moyen entre gradients par paire de tâches sur le
corps partagé (6 batches × 7 tâches) : **tous ≈ 0** (extrêmes +0,06 / −0,006).
Aucune tâche ne tire systématiquement contre le RUL — l'« interférence »
diagnostiquée au Step-6 (pente −0,94 seule vs −0,61/−0,78 en multi-tâches)
n'est **pas un conflit directionnel** mais une **dilution du budget de
représentation**.

**Conséquences** : PCGrad/chirurgie de gradients serait sans objet (rien à
projeter) — écarté sur mesure. Remède adopté : **hidden 96 → 128** (budget,
pas direction). La couche privée par tâche reste l'option suivante si la
dilution persiste (documentée, non faite).

## 3. Pseudo-labels vibration — le banding est-il le bon proxy ? **PLAFONNÉ — inchangé, en connaissance de cause**

**Mesure (A3)** sur les 3 067 fenêtres défaillantes CWRU, accord
argmax/vérité :

| Variante | Accord |
|---|---|
| Actuelle (harmoniques 1-2) | 73,6 % |
| Harmoniques 1-3 | 73,7 % |
| SNR local (bandes de garde) | 71,1 % |
| + bandes latérales BPFI±f_r (signature IR, Randall) | 72,9 % |

Aucun raffinement du banding ne dépasse le bruit : **le plafond (~74 %) est le
proxy lui-même**, pas son paramétrage. Décision : conservé tel quel dans son
rôle mesuré d'enseignant faible/régularisateur (λ=0,1 ; gradients réels,
ratio 0,15 — cf. diagnostic Step-6). Alternatives lourdes (filtres adaptés,
cepstre, features apprises) = v3, coût/bénéfice non justifié au stade démo.

## 4. Constantes (c) — lesquelles comptent ? **PRESQUE AUCUNE — quantifié**

**Kamal (m, n, k₂/k₁)** : grille 27 combinaisons (m∈{0,5-2}, n∈{1-2},
r∈{1-4}) avec **re-calibration de A à chaque fois** (ancrage t99(195 °C)=12 min
par construction) : α_end du scénario de fuite fixe varie de **0,967 à 0,988**
et t99(180 °C) reste **28,1 min partout**. → Le design auto-calibré absorbe
l'incertitude (c) de la cinétique ; le choix m=1, n=1,5, r=2 est immatériel à
l'échelle démo. Inchangé, tagué (c) avec cette preuve.

**Hertz (longueur de rouleau, part de charge)** : σ_H varie en √(part/L)
(1,2 → 2,5 GPa sur la grille) mais σf′ recalibré sur les mêmes N_f absorbe
exactement l'échelle → **D et heures-avant-remplacement insensibles** à ces
deux (c). Inchangés, absorption documentée.

**Ea/R (brevet US4022555, 10-14 kK)** : l'ancrage refixe t99(195°)=12 min ;
seule la *sensibilité en température* change — ratio t99(180°)/t99(210°) de
**3,96 à 6,87** sur la plage. → Sans conséquence sur le pipeline (les cycles
sont asservis à la recette), mais **à flaguer si on cite un temps froid
précis**. Valeur médiane 12 kK conservée, plage citée.

## 5. Pression — l'ODE avait-elle besoin de structure ? **OUI — intégrée au forward**

Le Step-6 avait mesuré le résidu ODE **vacuëment satisfait** (ratio de
gradient 4×10⁻⁴) : une reconstruction canonique moyenne respecte la *forme*
sans suivre aucun cycle réel. Réponse structurelle (pas une rustine) : la tête
**estime les paramètres physiques du cycle observé** (P_set, τ_ramp,
τ_release) et la reconstruction nominale est produite en **intégrant l'ODE**
avec ces paramètres — pas exact exponentiel par phase (l'ODE est linéaire par
morceaux : `P⁺ = P_set + (P−P_set)·e^(−dt/τ)`), inconditionnellement stable
(un Euler explicite divergerait : dt_token ≈ 56 s > τ ≈ 25 s) et
différentiable de bout en bout. Le terme « physique » de la loss devient
l'erreur d'**estimation de paramètres** vs la vérité du générateur (luxe
possible uniquement parce que ces données sont SYNTHÉTIQUES et étiquetées
comme telles). Anomalie et α_end restent lus sur le résidu
(observé − nominal intégré), désormais signifiant par cycle.

## Laisser tel quel — avec raison

- **Pseudo-labels vibration** (cf. §3 — plafond mesuré du proxy).
- **Toutes les constantes (c)** (cf. §4 — insensibilité quantifiée).
- **Tête RUL standard** (la variante monotone dure reste dans le code,
  strictement dominée au Step-6 : RMSE 17,64 vs 15,10).
- **Cible RUL Heimes-125** pour l'évaluation (comparabilité littérature) — le
  genou ne modifie que la portée de la *contrainte*.

## Validation — les changements tiennent en multi-tâches

Tête seule (`runs/diag_press_v2/`, `runs/diag_rul_knee/`) :
- Pression intégrée : loss d'entraînement **17,5 → 3,3** (le plateau de
  reconstruction canonique disparaît — le suivi par cycle fonctionne).
- RUL au genou : pente sous cap125 **−1,004**, RMSE 15,66.

Run multi-tâches complet (`runs/v2/metrics.json`, hidden 128 + genou +
pression intégrée) :

| Métrique | Avant Step-7 | Après | Trajectoire complète |
|---|---|---|---|
| RUL pente sous cap125 (multi-tâches) | −0,775 | **−0,99** | −0,61 (Step-6) → −0,78 (λ=5) → −0,99 |
| RUL RMSE test | 13,69 | **13,74** | la physique n'a rien coûté en précision |
| Pression : pente de calibration α̂→α | 1,73 (compression 0,58) | **1,11 (quasi-identité)** | l'intégration ODE a éliminé la compression |
| Pression α MAE / corr fuites | 0,0182 / 0,917 | **0,0106** / 0,894 | catch 3/6 + 1 fausse alarme — les 3 ratées sont à ±MAE du seuil (limite de mesure à n=6) |
| Fatigue (dernière fenêtre avant casse réelle) | D=0,877 → « 12,8 h » | **D=0,935 → « remplacer maintenant »** | et le roulement a réellement cassé juste après |
| Démo mono-capteur | stable | **0,56 K / 7,4 %** (AI4I) ; **12,8 %** RMS fan-end (RÉEL) | inchangée par les refontes — robuste |

Conclusion Step-7 : le modèle ne « penchait » pas uniformément sur les
données ; là où c'était vrai (RUL, pression), la cause n'était pas λ mais la
**forme** (portée de contrainte non observable, ODE sans structure par cycle)
— corrigée par la donnée, et les corrections tiennent sans coût de précision.
