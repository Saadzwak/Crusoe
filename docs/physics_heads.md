# MH-PINN — physique par tête : équations, sources, confiance

> **Mission** (pourquoi un PINN et pas une boîte noire) : *prédire des grandeurs
> physiques importantes en des points où il n'y a pas de capteur*, à partir des
> équations + des capteurs existants. Chaque tête est évaluée contre cet
> objectif — voir [la démo mono-capteur](#8-démo-phare--reconstruction-mono-capteur).

**Catégories de confiance** — chaque équation/constante de ce document et du
code (`pinn/physics/`) est taguée :

| Tag | Signification |
|---|---|
| **(a)** | Loi physique établie, niveau manuel — pas en question |
| **(b)** | Tirée d'une source précise citée (papier / brevet / norme) |
| **(c)** | Choix plausible **non confirmé**, en attente de validation — jamais présenté comme un fait |

Provenance des données : **RÉELLES** (CWRU, IMS — mesures), **SIMULÉES**
(C-MAPSS — simulateur NASA ; AI4I — règles génératives documentées),
**SYNTHÉTIQUES** (cycles de presse générés in-repo, étiquetés partout).
Aucune donnée confidentielle Michelin nulle part.

Architecture : adaptateurs d'entrée par tête (taux/canaux hétérogènes) → corps
LSTM partagé → têtes. La physique entre par les **résidus de loss**
([pinn/losses.py](../pinn/losses.py)).

---

## 1. Tête vibration — cinématique des roulements

**Équations (a)** — cinématique classique (roulement sans glissement ;
réf. Randall & Antoni, *MSSP* 2011 **(b)** pour la pratique diagnostique) :

```
BPFO = (n/2)·f_r·(1 − (d/D)·cos φ)      bague externe
BPFI = (n/2)·f_r·(1 + (d/D)·cos φ)      bague interne
BSF  = (D/2d)·f_r·(1 − ((d/D)·cos φ)²)  élément roulant
```

**Pourquoi c'est légitime** : un défaut localisé est percuté à chaque passage
d'élément → impulsions périodiques à ces fréquences exactes, fonction de la
seule géométrie publiée + vitesse d'arbre. La loss contraint la croyance
{IR, OR, B} du modèle à suivre l'énergie du **spectre d'enveloppe** (Hilbert —
le défaut module en amplitude des résonances HF) aux bandes {BPFI, BPFO, BSF}.

**Constantes (b), cross-checkées à chaque run** : géométries SKF 6205/6203
(CWRU « Apparatus & Procedures ») et Rexnord ZA-2115 (readme IMS / Qiu et al.
2006, *J. Sound & Vibration*). `verify_against_published()` recalcule les
multiplicateurs publiés (5.4152, 3.5848, 236.4 Hz…) à ~10⁻⁴ — une constante
fausse ne peut pas entrer silencieusement à l'entraînement.

Supervision : CWRU → classes ; IMS → `health` (proxy position de vie **(c)**,
standard en run-to-failure sans inspections intermédiaires, monotone par
construction).

## 2. Tête thermique/puissance — règles génératives d'AI4I

**Équations (b)** — documentation du dataset (Matzka 2020, UCI #601) :

```
HDF : (T_process − T_air) < 8,6 K  ET  vitesse < 1380 rpm
PWF : P = τ·ω (a) ;  défaut si P < 3500 W ou P > 9000 W
OSF : usure·τ > 11000/12000/13000 min·Nm (types L/M/H)
```

Lecture physique : refroidissement de Newton **(a)** (dissipation ∝ gradient,
aidée par la convection liée à la rotation) ; puissance d'arbre τ·ω exacte
**(a)**. Vérifié sur données : les règles reproduisent **115/115 HDF, 95/95
PWF** — ce dataset satisfait sa propre physique, c'est précisément pourquoi une
tête physics-informed s'y prête (la loss pénalise le désaccord avec les règles
relâchées en sigmoïdes).

**Distinction explicite exigée par le standard documentaire** : AI4I est un
**proxy empirique** du comportement thermique/puissance *machine*. La chimie de
cuisson (§3) justifie physiquement pourquoi une tête thermique importe pour une
presse ; elle n'est **pas** ce que les labels AI4I encodent. Les deux ne sont
jamais confondues dans le code.

## 3. Chimie de cuisson (vulcanisation) — [pinn/physics/cure_kinetics.py](../pinn/physics/cure_kinetics.py)

**Conduction avec source réactive (a)** — Fourier :
`ρ·c_p·∂T/∂t = ∇·(k∇T) + Q`, Q incluant l'exotherme de réaction (la cuisson est
*couplée*, pas un simple chauffage). Non résolue spatialement en v1 ; énoncée
parce que c'est *pourquoi* température de moule ≠ état de cuisson interne — et
l'état interne est la grandeur sans capteur que la tête estime.

**Arrhenius** : `k(T) = A·exp(−Ea/RT)` — forme **(a)** ; plage
**Ea/R ≈ 10 000–14 000 K** : brevet **US4022555 (b)** — source non peer-reviewed,
citée comme ordre de grandeur indicatif, pas comme constante précise.

**Kamal-Sourour (b)** — modèle autocatalytique standard de la littérature cure
(Kamal & Sourour 1976, *Polym. Eng. Sci.* ; application caoutchouc p.ex. Rafei,
Ghoreishy & Naderi, *Comput. Mater. Sci.* 2009) :

```
dα/dt = (k₁ + k₂·α^m)·(1−α)^n,   k₁, k₂ Arrhenius
```

**m = 1, n = 1,5, k₂/k₁ = 2 : (c)** — spécifiques au compound dans la
littérature (fittés par rhéométrie/DSC) ; nos valeurs sont des placeholders
dans les plages usuelles, à re-fitter sur données ODR (**ASTM D2084 /
ISO 3417 (b)**) si un compound réel est un jour caractérisé.

**Pré-facteur A : résolu, pas inventé (c ancré)** — A est calculé par bissection
pour que t99(195 °C) = 12 min (milieu de la plage **vérifiée** 10-15 min).
Résultat : A ≈ 1,23×10⁹ s⁻¹, et le modèle prédit alors t99(180 °C) ≈ 28 min —
cohérent avec le « jusqu'à 30 min » documenté pour les gros pneus, une
validation qu'on n'a pas imposée.

**α pratique (b)** : α ≈ t_cure/t99 (pratique industrielle, ASTM D2084).

**Couplage fuite→température (b)** : en vulcanisation vapeur directe, la cavité
est à la température de **saturation** — pente ≈ 3,2 K/bar autour de 15 bar
(tables vapeur standard : 10 bar/180 °C, 15/198, 20/212). Une fuite mi-palier
fait donc chuter la cuisson : la sortie physiquement signifiante est
**α_end (sous-cuisson)**, pas seulement « pression basse ».

## 4. Tête fatigue — [pinn/physics/fatigue.py](../pinn/physics/fatigue.py) (« remplacer avant la casse »)

Chaîne : charge + géométrie —(Hertz)→ σ_H —(Basquin)→ N_f —(Miner)→ D∈[0,1].

**Hertz, contact linéique (a)** (Harris, *Rolling Bearing Analysis* **(b)**) :
`σ_H = √(P_lin·E*/(π·R_eff))`, E* acier/acier = 115,4 GPa **(a)** ;
charge max par rouleau via Stribeck `P_max = 4,6·F_r/(i·Z·cos α)` **(b)**
(4,6 = jeu nul ; ~5,0 avec jeu — on porte 4,6 et on le dit).
- **Longueur de rouleau 0,35″ : (c)** — absente du readme IMS ; hypothèse
  typique pour d = 0,331″, signalée.
- **Répartition de charge : (c)** — les 6000 lbs s'appliquent sur les paliers
  2-3 (readme) ; part exacte vue par chaque palier inconnue → F/2 nominal,
  incertitude absorbée par la calibration de σf′.
- Résultat : **σ_H ≈ 1,49 GPa** — bas de plage typique des bancs (1,5-3,5 GPa) ✓.

**Basquin (a en forme)** : `σ_a = σf′·(2N_f)^b`.
- **b acier : −0,05…−0,12 (b)** (littérature fatigue générale, p.ex. Dowling).
- **AISI 52100 : (c)** — le readme IMS ne précise pas l'acier ; 52100 est le
  standard roulements, hypothèse signalée.
- **Exposant 10,34 (b, étude torsion 52100 HRC 58-62, pas de limite de fatigue
  observée)** : paramétrisation ET essai **différents** — jamais interchangé ni
  moyenné avec b. On note (sans assimiler) que 1/10,34 ≈ 0,097 tombe dans la
  plage |b| : consonance, pas validation.

**Calibration sur les VRAIES défaillances IMS plutôt qu'import de constantes** —
limitation dite telle quelle : IMS a tourné à **un seul point de charge** ; un
seul niveau de contrainte ne peut identifier σf′ **et** b (2 inconnues,
1 abscisse). Procédure honnête : N_f = durée réelle × fréquence de passage des
rouleaux (BPFO/BPFI — chaque passage = 1 cycle de Hertz au point défaillant),
b fixé sur la grille littérature, σf′ résolu :

| Test | Durée → N_f (cycles) | σf′ à b=−0,05 | à b=−0,0713 | à b=−0,12 |
|---|---|---|---|---|
| 1er (IR, palier 3) | 828 h → 8,9×10⁸ | 4,3 GPa | 6,8 GPa | 19,2 GPa |
| 2e (OR, palier 1) | 164 h → 1,4×10⁸ | 3,9 GPa | 6,0 GPa | 15,3 GPa |
| 3e (OR, palier 3) | 1073 h → 9,1×10⁸ | 4,3 GPa | 6,8 GPa | 19,2 GPa |

Le bas de plage |b| donne les σf′ les plus plausibles pour un acier durci —
c'est le *plausibility check* littérature, pas une vérité importée. Dispersion
×6,5 entre tests = scatter de fatigue attendu (Weibull) **(a)**.

**Miner (a, règle d'ingénierie standard)** : `D = Σ nᵢ/N_f(σᵢ)`, casse vers
D ≈ 1. Sous charge constante D(t) = t/T_f — c'est la supervision de la tête ;
Miner est ce qui permet d'extrapoler sous charge variable. **OSF d'AI4I**
(usure×couple > seuil) = analogue conceptuel grossier, utilisé comme
cross-check documenté, **jamais** comme donnée de fatigue.

Sortie démo : `hours_to_replacement(D)` — heures avant D = 0,9 au rythme de
cyclage courant (Basquin/Hertz + N_f empirique).

## 5. Tête dégradation/RUL — C-MAPSS FD001

Cible **(b)** : RUL par morceaux cappé à 125 (Heimes 2008, convention FD001).
Contrainte physique : dommage irréversible → `dRUL/dcycle = −1` **sous le cap**
(définition de la grandeur en run-to-failure, pas une loi du turbofan — dit tel
quel) ; **pente 0 dans la zone cappée** (la v0 forçait −1 partout : bug
détecté à l'éval, corrigé en contrainte par morceaux).

**Colonnes — correction du brief v1 intégrée** : les 7 colonnes listées
(op_setting_3, s1, s5, s10, s16, s18, s19) sont **exactement constantes**
(std = 0, vérifié) et en dropper 7 laisse **15** capteurs. Nous droppons **en
plus** `sensor_6` (std ≈ 0,0014, vérifié quasi-constant) → 14 entrées :
**jugement assumé (c)**, pas une exigence — un capteur z-scoré à variance
~0 n'injecte que du bruit amplifié.

## 6. Tête pression — SYNTHÉTIQUE ⚠️

Aucun dataset public n'existe (pratique acceptée par les mentors). Ancrage
**vérifié (b)** : 180-210 °C ; pression **> 20 bar** (serrage) **et**
**1,6-1,9 MPa** (vapeur directe) — deux chiffres de sources différentes,
**cités séparément, jamais moyennés** ; cycles 10-15 min (jusqu'à ~30 pour
gros pneus). ODE de cycle 1er ordre **(a en forme, τ en (c))** :

```
montée : dP/dt = (P_set−P)/τ_ramp   palier : ≈0   relâche : dP/dt = −P/τ_rel
fuite  : dP/dt += −k_leak·P  (anomalie injectée, joint défaillant)
```

Les τ d'une vraie presse sont inconnus/confidentiels — la *forme* est textbook,
les constantes de temps sont **(c)**. Durée de cycle **asservie à la recette**
(t99(T)×marge) comme en production réelle **(c motivé)** — sans cela le signal
fuite→sous-cuisson serait noyé dans la variation de consigne. Étiquetage
SYNTHETIC : code, docs, plots, et champ `provenance` **signé** dans la
télémétrie.

## 7. Seuils d'anomalie vérifiés (pour la couche information)

Machine-readable : [docs/anomaly_thresholds.json](anomaly_thresholds.json).

| Paramètre | Plage normale | Confiance |
|---|---|---|
| Température | 180–210 °C | **(b)** littérature cuisson pneus, vérifié |
| Pression | > 20 bar (serrage) / 1,6–1,9 MPa (vapeur) | **(b)** deux sources, citées séparément |
| Durée de cycle | 10–15 min (jusqu'à ~30) | **(b)** littérature cuisson pneus |
| Vibration | Zones A ≤ 1,12 · B ≤ 2,8 · C ≤ 7,1 · D > 7,1 mm/s RMS | **(b)** ISO 10816-3/20816-3, machines Groupe 2 (15–300 kW) — **norme industrielle générale, PAS spécifique presse de cuisson** (à rappeler partout où citée) |
| Sous-cuisson | α_end < 0,90 | **(c)** seuil démo sur données synthétiques |

## 8. Démo phare — reconstruction mono-capteur

**La preuve directe de la mission** : un papier 2025 (**arXiv 2506.14236 (b)**)
montre l'estimation de paramètres PINN tenant à 5-8 % d'erreur avec **un seul
point de capteur**, la contrainte physique effondrant l'espace des solutions.
Notre implémentation (`virtual_sensor_demo` dans [pinn/train.py](../pinn/train.py)) :

1. **AI4I — température process** : colonne masquée en entrée, reconstruite
   depuis T_air + vitesse + couple + usure ; physique = la règle HDF évaluée
   sur le ΔT *reconstruit* doit reproduire le label réel. Erreur rapportée en
   K contre la vérité retenue (SIMULÉE).
2. **CWRU — accéléromètre fan-end** (RÉEL) : jamais donné en entrée ;
   ses énergies de bande {BPFO, BPFI, BSF} + RMS reconstruites depuis le
   drive-end seul — même arbre, même cinématique de défaut (physique = KL entre
   distributions de bandes DE/FE). Erreur contre la mesure réelle retenue.

Chiffres et PNG : `runs/v1/` (metrics.json, virtual_sensor_demo.png) — branché
sur l'outil `get_pinn_reconstruction` prévu par la couche information.

## 9. Entraînement — choix documentés (Step 3 du brief : « on propose, tu décides »)

- **Optimiseur** : Adam (proposition acceptée — bas risque, standard), lr 1e-3,
  grad-clip 5. A convergé proprement en v0.
- **Pondération des losses** : **λ fixes, équilibrés par ordre de grandeur** —
  chaque terme pré-normalisé à O(1) (RUL/cap, températures/σ, α×10), puis
  λ_phys ∈ [0,1-0,5]. Baseline standard de la littérature PINN ; les schémas
  adaptatifs — ReLoBRaLo ([Bischof & Kraus, CMAME 2025](https://www.sciencedirect.com/science/article/pii/S0045782525001860)),
  self-adaptive ([arXiv 2104.06217](https://arxiv.org/pdf/2104.06217)),
  annealing par statistiques de gradient — sont le raffinement v2 documenté,
  non nécessaire à cette échelle.
- **Écarts motivés par la mesure** : λ_phys(RUL) 0,1 → 0,5 (v1, résidu observé
  −0,40) puis **0,5 → 5,0** après le diagnostic instrumenté Step-6 (terme
  ~1000× sous-échelle, balayage λ : meilleur RMSE **et** meilleure pente à
  λ=5 ; tête monotone dure rejetée sur mesure). Détails, sonde de gradients et
  verdicts par tête : [v1_physics_diagnosis.md](v1_physics_diagnosis.md).
- Courbes loss/métriques par tête et par epoch : `runs/v1/curves_*.png`.

## 10. Contrat de sortie (FIGÉ, aligné sur la branche du coéquipier)

`SignedReading{payload: PinnReading, signature}` — HMAC-SHA256 hex sur la
sérialisation canonique (clés triées, pas d'espaces), clé `PINN_HMAC_SECRET`,
champs `machine_id, department, epoch, timestamp, signals{mould_temp_C,
coil_power_kW, vibration_rms_mm_s, pressure_bar}, pinn{health_index (1=SAIN),
rul_cycles, residual, failure_mode_probs}, note`. Émetteur :
[pinn/telemetry.py](../pinn/telemetry.py) (`to_praetor_signed_reading`) —
**cross-vérifié** contre `backend/agent/hmac_auth.py` de la branche
`feat/llm-information-layer`. ⚠️ `health_index` est inversé vs nos proxys de
dommage (D=1 → health=0) — conversion faite à l'émission, testée.
