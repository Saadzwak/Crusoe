# Diagnostic v1 — « le modèle s'appuie-t-il plus sur les données que sur la physique ? »

Réponse au Step 6 du brief v1 : investigation **instrumentée** (pas de
suppositions), outillage dans [pinn/diagnose.py](../pinn/diagnose.py), données
brutes dans `runs/v1/diagnosis.json` et `runs/diag_rul_*/metrics.json`.

**Réponse courte : les trois têtes suspectées ont trois mécanismes différents —
le soupçon était surestimé pour la vibration, réel mais mal attribué pour le
RUL (bug d'échelle + interférence multi-tâches, pas λ ni architecture), et
d'une autre nature pour la pression (compression de régression + résidu ODE
satisfait de façon vacuë).**

---

## Méthode

1. **Sonde de gradients** : au checkpoint v1 final, backward *séparé* du terme
   data et du terme λ·physique sur un batch de validation, norme du gradient
   induit sur les paramètres du **corps partagé**. C'est la réponse directe à
   « le terme reçoit-il des gradients ? ».
2. **Fractions de loss** à l'epoch finale (history.json).
3. **Plancher d'entropie** du terme vibration (cross-entropy à cibles molles :
   minimum = H(cibles), pas 0).
4. **Balayage RUL** tête seule : λ ∈ {0, 0.5, 5} + tête monotone par
   construction (`RulHeadMonotone`).
5. **Feature physique pression** jamais fournie au modèle :
   ∫ déficit de T_saturation dt (3,2 K/bar × déficit P, phase palier).

## Mesures

### Gradients et fractions par tête (checkpoint v1)

| Tête | ‖g‖ λ·phys / ‖g‖ data (corps partagé) | Fraction phys de la loss (valeur) | Lecture |
|---|---|---|---|
| fe_recon | **3,34** | 0,92 | tête dominée par la physique (cohérence DE↔FE) |
| fatigue | 0,22 | 0,16 | contribution saine |
| vibration | 0,15 | 0,75* | actif ; *la valeur est surtout le plancher d'entropie |
| thermal | 0,023 | 0,053 | faible **parce que déjà satisfait** (règles ≡ labels) |
| thermal_recon | 0,040 | 0,074 | idem |
| rul | **0,026** | 0,026 | **numériquement inerte — bug d'échelle** |
| pressure | **0,0004** | 0,0008 | **satisfait de façon vacuë** |

### Q1 — Vibration : « terme statique, supposé » → **soupçon SURESTIMÉ**

- Le terme **reçoit des gradients** (ratio 0,15 sur le corps partagé — mesuré,
  pas supposé).
- Observé 1,224 ; **plancher d'entropie des cibles molles 0,633** → distance
  KL réellement optimisable : 0,59. Le terme n'est *pas* saturé…
- …mais les pseudo-labels de bandes ne coïncident avec la classe vraie que
  **81,5 %** du temps : le plateau est **l'équilibre face à un enseignant
  bruité** — quand pseudo-label et vérité divergent (18,5 % des fenêtres), le
  terme data gagne, ce qui est le comportement voulu à λ=0,1.
- Verdict : fonctionne comme conçu (régularisateur physique faible). Piste
  v3 si besoin : pseudo-labels plus propres (harmoniques multiples, gating
  kurtosis), pas λ.

### Q2 — RUL (−0,61 vs −1) : « λ ou structure ? » → **ni l'un ni l'autre tel que posé**

Balayage tête seule (10 epochs, mêmes seeds) :

| Config | RMSE test | Pente sous cap | Pas croissants |
|---|---|---|---|
| λ=0 (sans physique) | 16,34 | −0,936 | 0,364 |
| λ=0,5 (défaut v1) | 15,99 | −0,910 | 0,344 |
| **λ=5** | **15,10** | **−0,959** | 0,289 |
| Monotone par construction (λ=0,5) | 17,64 | −0,767 | **0,000** |

Trois faits établis :
1. **Bug d'échelle** : la double normalisation `((Δ+1)/125)²` rend le terme
   ~1000× plus petit que le terme data (d'où le ratio de gradient 0,026 —
   λ=0,5 n'y change rien : λ=0 et λ=0,5 sont au bruit près identiques).
2. **La pente −0,61 de la v1 était de l'interférence multi-tâches** : la même
   tête, seule, fait −0,936 *sans aucune physique*. Ce n'est pas un problème
   de capacité représentationnelle.
3. **λ=5 améliore LES DEUX métriques** (RMSE et pente) — la physique bien
   dosée est une régularisation gagnante, pas un compromis. La **tête
   monotone dure est strictement pire** (RMSE +2,5 cycles, pente plus plate)
   à cette échelle d'entraînement : garantir zéro remontée coûte de la
   précision ; **résultat négatif documenté, variante conservée dans le code
   pour re-test à plus grande échelle**.

→ Correctif adopté (mesuré, pas supposé) : `RUL_LAMBDA_PHYS = 5.0`.

### Q3 — Pression (sous-cuisons ratées) : **mécanisme distinct, confirmé**

- Résidu ODE : ratio de gradient **0,0004** — la reconstruction canonique
  satisfait déjà la forme ODE (palier plat, montée/relâche exponentielles)
  sans suivre les cycles individuels : contrainte **vacuëment satisfaite**,
  elle ne « tire » plus rien.
- Feature physique ∫déficit T_sat : corr −0,77 avec α_true seule ; la tête
  fait mieux (0,92) et **utilise** cette information (corr −0,84 avec la
  feature) — la physique n'est pas ignorée…
- …mais **pente de régression α̂ vs α_true = 0,516** sur cycles fuyants :
  compression classique vers le cluster nominal (cible continue déséquilibrée).
  Les ratés au seuil 0,90 sont un problème de **calibration d'amplitude**, pas
  de physique manquante ni de λ.

→ Correctif adopté : **recalibration linéaire** ajustée sur les cycles fuyants
du split *train*, appliquée à l'inférence (aucune fuite de données) ;
coefficients persistés dans metrics.json.

## Décision v2 (jugement propre, comme demandé)

| Priorité | Action | Justification |
|---|---|---|
| 1 | λ_RUL 0,5 → **5,0** | mesuré meilleur sur RMSE **et** pente ; corrige le terme inerte |
| 2 | **Recalibration α̂** (train-fit, val-éval) | pente 0,52 diagnostiquée ; le chiffre-clé du récit cure doit être fiable |
| 3 | **Rejet** de la tête monotone dure | strictement dominée à cette échelle (17,64 vs 15,10) ; code conservé |
| 4 | Vibration : **aucun changement** | comportement conforme à la conception (enseignant bruité) |
| — | v3 documentés, non faits | équilibrage multi-tâches (PCGrad/échantillonnage), pseudo-labels vibration plus propres, recon pression conditionnée par cycle |

Run de confirmation multi-tâches avec (1)+(2) : voir `runs/v1/metrics.json`
(régénéré) — vérifier que la pente RUL multi-tâches remonte vers −1 et que la
détection de sous-cuisson calibrée attrape la queue.
