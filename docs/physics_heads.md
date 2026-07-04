# MH-PINN — physique par tête (équations, sources, justification)

Modèle : un « corps » récurrent partagé (LSTM) + un adaptateur d'entrée par
tête (taux d'échantillonnage et nombres de canaux hétérogènes — de 1 ligne
tabulaire à des formes d'onde 12 kHz) + une tête de sortie par phénomène.
La physique entre par les **résidus dans la loss** (`pinn/losses.py`), chaque
équation étant documentée dans `pinn/physics/`.

Machine cible (cadre narratif) : **presse de cuisson/vulcanisation** de pneus,
inspirée Michelin. Aucune donnée confidentielle Michelin ; jeux de données
publics utilisés comme substituts physiquement plausibles, toujours étiquetés.

| Tête | Données | Provenance | Équation gouvernante |
|---|---|---|---|
| Vibration | CWRU + IMS | RÉELLES | Fréquences caractéristiques de défaut de roulement |
| Thermique/puissance | AI4I 2020 | SIMULÉES (règles documentées) | Règles de défaillance en forme fermée du dataset |
| Dégradation/RUL | C-MAPSS FD001 | SIMULÉES (NASA) | Monotonie du dommage : dRUL/dcycle = −1 sous le cap |
| Pression | générées in-repo | **SYNTHÉTIQUES** | ODE de cycle du 1er ordre (montée/palier/relâche) |

---

## 1. Tête vibration — cinématique des roulements

**Équations** (kinématique classique d'un roulement, condition de roulement
sans glissement ; réf. Randall & Antoni, *MSSP* 2011) :

```
BPFO = (n/2)·f_r·(1 − (d/D)·cos φ)        défaut bague externe
BPFI = (n/2)·f_r·(1 + (d/D)·cos φ)        défaut bague interne
BSF  = (D/2d)·f_r·(1 − ((d/D)·cos φ)²)    défaut d'élément roulant
```

avec n = nombre d'éléments roulants, f_r = fréquence de rotation de l'arbre,
d = diamètre d'élément, D = diamètre primitif, φ = angle de contact.

**Pourquoi c'est une contrainte légitime** : un défaut localisé est percuté à
chaque passage d'élément roulant → il injecte des impulsions périodiques à
exactement ces fréquences. La *classe* de défaut a donc une signature
fréquentielle en forme fermée qui ne dépend que de la géométrie publiée du
roulement et de la vitesse d'arbre — pas de paramètres appris. La loss
contraint la croyance relative du modèle entre {IR, OR, B} à suivre l'énergie
du **spectre d'enveloppe** observée aux bandes {BPFI, BPFO, BSF} (enveloppe de
Hilbert et non FFT brute : le défaut module en amplitude des résonances haute
fréquence — la signature vit dans l'enveloppe).

**Constantes non inventées** : géométries SKF 6205/6203 (CWRU) et Rexnord
ZA-2115 (IMS) prises dans la documentation des datasets ;
`pinn.physics.bearing.verify_against_published()` re-calcule les
multiplicateurs publiés par les auteurs (5.4152, 3.5848, …) à ~10⁻⁴ près **à
chaque lancement d'entraînement** — une constante fausse ne peut pas entrer
silencieusement.

Supervision : CWRU → classes de défaut ; IMS (run-to-failure réel) → sortie
`health` avec proxy de position de vie (0 = début, 1 = dernière mesure avant
défaillance physique), monotone par construction.

## 2. Tête thermique/puissance — règles génératives d'AI4I

**Équations** (documentation du dataset : Matzka 2020, UCI #601) :

```
HDF :  (T_process − T_air) < 8,6 K  ET  vitesse < 1380 rpm
PWF :  P = τ·ω,  ω = 2π·rpm/60 ;  défaillance si P < 3500 W ou P > 9000 W
OSF :  usure·τ > 11000/12000/13000 min·Nm  (types L/M/H)
```

**D'où ça vient / pourquoi c'est légitime** : AI4I est synthétique *par
conception* et ses auteurs publient les règles génératives exactes. Lecture
physique : dissipation thermique ∝ gradient de température (loi de
refroidissement de Newton) aidée par la convection liée à la rotation ;
puissance mécanique d'un arbre = τ·ω exactement. La loss pénalise le
désaccord entre les probabilités prédites HDF/PWF et les règles relâchées en
sigmoïdes (différentiables) évaluées sur les entrées physiques brutes.
Vérifié sur données : les règles reproduisent 115/115 HDF. TWF (usure tirée
au hasard 200-240 min) et RNF (0,1 % aléatoire) sont stochastiques → têtes
data-only, pas de contrainte physique.

## 3. Tête dégradation/RUL — monotonie du dommage (C-MAPSS FD001)

**Équations** :

```
cible : RUL(cycle) = min(125, N_total − cycle)     (Heimes 2008, convention FD001)
contrainte : RUL[t+1] − RUL[t] = −1  sous le cap ;  jamais > 0 dans la zone cappée
```

**Pourquoi c'est légitime — et honnêtement qualifié** : sur une trajectoire
run-to-failure mesurée en cycles, le dommage est irréversible et le RUL décroît
d'exactement 1 par cycle une fois la dégradation observable (c'est la
*définition* de la quantité, pas une loi thermodynamique du turbofan — nous le
disons tel quel). En début de vie les capteurs ne montrent rien → cible
plafonnée, pente vraie = 0 : la contrainte −1 ne s'applique que sous le cap
(la v0 initiale l'appliquait partout — bug physique détecté à l'éval et
corrigé, voir rapport). Colonnes : les 7 listées par le brief sont
exactement constantes (std = 0, vérifié) ; `sensor_6` quasi-constant
(std ≈ 0,0014) est droppé en plus pour retrouver les « 14 capteurs informatifs ».

## 4. Tête pression — ODE de cycle de presse ⚠️ DONNÉES SYNTHÉTIQUES

**Équations** (modèle du 1er ordre d'un volume rempli à travers une
restriction — bilan de masse linéarisé, même forme qu'une charge RC) :

```
montée  : dP/dt = (P_set − P)/τ_ramp
palier  : dP/dt ≈ 0
relâche : dP/dt = −P/τ_release
fuite (anomalie) : dP/dt += −k_leak·P  (mi-palier → fin)
```

**Ancrage réel** : enveloppe opératoire publique et vérifiée des presses de
cuisson — 180-210 °C, > 20 bar de fermeture / 1,6-1,9 MPa de vapeur, cycles
10-15 min. **Les constantes de temps τ d'une vraie presse Michelin nous sont
inconnues** (et confidentielles) ; c'est la *forme* de la dynamique qui est
textbook. Le modèle apprend à reconstruire le cycle sain sous contrainte de
l'ODE (résidu en différences finies sur la grille de tokens) + détecte
l'anomalie de fuite de joint injectée. **Aucun dataset public n'existe pour
cette grandeur : tout est généré in-repo et étiqueté SYNTHETIC partout**
(code, docs, sorties télémétrie — champ `provenance` signé dans le payload).

---

## Où c'est implémenté

- Équations & constantes : [pinn/physics/](../pinn/physics/)
- Résidus dans la loss : [pinn/losses.py](../pinn/losses.py)
- Adaptateurs hétérogènes + corps partagé + têtes : [pinn/model/mh_pinn.py](../pinn/model/mh_pinn.py)
- Télémétrie signée HMAC (schéma DRAFT à verrouiller avec la couche info) : [pinn/telemetry.py](../pinn/telemetry.py)
