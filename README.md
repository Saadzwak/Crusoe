# Crusoe — Factory Digital Twin (RAISE Summit 2026, track Crusoe)

Jumeau numérique d'usine inspiré du cas **Michelin Roanne** (pneus UHP, ligne de
cuisson C3M — le goulot d'étranglement) : une couche **physique** comprend l'état
réel de chaque machine, une couche **information** transforme cette physique en
avis qu'un humain peut questionner et outrepasser, et une **UI opérateur** montre
l'agent en train d'agir.

> **Aucune donnée confidentielle Michelin** n'est utilisée nulle part. Jeux de
> données publics vérifiés, utilisés comme substituts physiquement plausibles et
> **toujours étiquetés** (réel / simulé / synthétique — jusque dans la signature
> des messages).

## Les trois blocs

| Bloc | Rôle | Code | Docs |
|---|---|---|---|
| **MH-PINN** (couche physique) | Corps LSTM partagé + têtes par phénomène (vibration, thermique, RUL, pression/cuisson, fatigue) ; démo phare : reconstruction de capteur retenu | [pinn/](pinn/) | [docs/MODEL_CARD.md](docs/MODEL_CARD.md) · [docs/physics_heads.md](docs/physics_heads.md) |
| **PRAETOR** (couche information) | Vérif HMAC → triage 3 étages → advisory citée → débat Advocate/Skeptic → Jury → opérateur outillé ; app intervention (notifié→vérifié VLM) | [backend/agent/](backend/agent/) | [docs/LLM_LAYER_README.md](docs/LLM_LAYER_README.md) · [CLAUDE.md](CLAUDE.md) |
| **CureWatch** (UI) | Interface opérateur 3D (Claude Design) servie par le backend — vues LEGION / Floor / Plant / Agent / Log / Flow | [backend/agent/static/](backend/agent/static/) | [docs/OPERATOR_APP.md](docs/OPERATOR_APP.md) |

**Interface entre couches (contrat figé)** : `SignedReading{payload: PinnReading,
signature}` — HMAC-SHA256 sur sérialisation canonique, stdlib pur. Testé **en
conditions réelles** (les deux codes du même arbre) :
[scripts/test_cross_layer_contract.py](scripts/test_cross_layer_contract.py).

## L'interface opérateur (CureWatch) — ce que montre la démo

Un seul fichier servi par le backend (`backend/agent/static/index.html`,
Three.js vendorisé, aucun asset externe). Deux rôles (Opérateur / Manager),
chacun avec son propre fil de chat.

### LEGION — la vue de commandement (écran d'accueil)

La hiérarchie d'agents rendue en pyramide 3D néon : **CAESAR** (agent-cerveau
de l'usine) → **FORVM** (conseil Advocate / Skeptic / Jury, avec la dernière
décision du jury affichée) → **CENTVRIO I–III** (agents de ligne, 4 presses
chacun) → les 12 presses-soldats avec leur état en direct. Les canaux animés
matérialisent la couche d'information : physique (jumeau PINN), information
(custody HMAC), défaillance. En cas d'alerte, le canal de la presse fautive
passe au rouge et « Caesar pèse une intervention ».

### Floor — l'atelier 3D

Hall de cuisson complet (12 presses, convoyeurs, pneus) en Three.js avec bloom
et ombres. Clic sur une presse → panneau détaillé : lectures en direct
comparées aux **limites vérifiées** ([backend/agent/limits.py](backend/agent/limits.py)),
santé PINN, RUL, modes de défaillance, recommandation de l'agent avec
Accept / Override (chaque décision est journalisée et nourrit les prompts
suivants — la boucle d'apprentissage). Panneau d'**injection de fautes** pour
la démo : dérive thermique, usure de roulement, retour au nominal — la presse
héroïne CP-07 est pilotée par le **vrai MH-PINN entraîné** (poids commités),
CP-01/03/10 rejouent d'honnêtes trajectoires C-MAPSS FD001.

### X-ray couche par couche (panneau presse → « Layer-by-layer X-ray »)

Double-clic sur une presse ou bouton dédié : **jumeau MH-PINN filaire** (ce
que le modèle croit) et **modèle physique** côte à côte, en vues éclatées
synchronisées — 5 couches pelables (dôme vapeur → système de chauffe → vessie
& moule → hydraulique → réseau de capteurs).

- **Localisation honnête** : la panne est localisée au niveau **sous-système**,
  avec la traçabilité affichée (limites dépassées + probabilités de mode du
  PINN + résidu de reconstruction — « comment on le sait »). Une excursion
  **thermique** reste volontairement **non localisée** : le jumeau détecte le
  mode, pas l'emplacement de la source de chaleur — la machine entière «
  respire » en halo chaud, aucun composant n'est accusé (trouver le point
  chaud est un travail de terrain, caméra IR).
- **Correction auto-optimisante** : plan en 5 étapes que l'agent exécute
  (dérater → rééquilibrer → appliquer les consignes → **vérifier la
  récupération contre le jumeau** → journaliser). L'humain décide, l'agent
  propose et rend compte.

### Passeport numérique (panneau presse → « Digital passport »)

La carte d'identité de la couche physique, par presse : schéma technique en
élévation + état des composants en direct (chauffe, vessie, entraînement,
sondes) à gauche ; à droite le **graphe du résidu physique PINN** — mesuré vs
bande prédite vs enveloppe des limites, bande σ avec seuil +3σ, verdict
(stable / en dérive / enveloppe physique violée). Sceau de custody
HMAC-SHA256 en pied de carte.

### Agent — le chat opérateur

Lane **deep** par défaut : agent tool-calling (LangChain `bind_tools`) qui
consulte réellement diagnostic, historique capteurs, limites, dérive et
custody avant de répondre — chaque réponse cite ses outils (« Data
Provenance » construite programmatiquement, jamais confiée au modèle). La
forme de la réponse suit la question (statut / tendance / risque / limites),
avec **mémoire de conversation** (pas de redite des chiffres déjà donnés).
Lane **quick** (DeepSeek Flash, secondes) pour l'app intervention.
En mode mock (sans clé), les réponses sont composées déterministiquement des
mêmes outils — utile hors-ligne, mais c'est la lane live qui montre le vrai
comportement.

### Manager — FLOW, décisions, alertes

Vue **FLOW** : la topologie de l'information (PINN → GATE → TRIAGE → DRAFT →
DEBATE → JURY → ADVISORY → OPÉRATEUR / NOTIFY / BOSS) animée depuis les
événements réels du flight-recorder du pipeline. Journal des décisions
(proposé vs décidé, `GET /api/history`). Alertes HIGH/CRITICAL : paging de
l'opérateur responsable via Teams Workflows + réservation du créneau
d'intervention sur son calendrier (optionnel, `TEAMS_WEBHOOK_URL`).

## Démarrage rapide

```bash
# 1) Secrets — le repo est destiné à être public : AUCUNE clé committée, jamais
cp .env.example .env          # remplir CRUSOE_API_KEY (vide = mode mock complet)

# 2) Couche information + UI
pip install -r backend/requirements.txt
python scripts/crusoe_sanity.py                # si clé : vérifie les modèles + tool-calling
uvicorn backend.agent.main:app --port 8000     # UI sur http://localhost:8000
# démarrer la simulation : POST /api/loop/start?interval=5&reset=1 (ou bouton UI)

# 3) Couche physique (torch CPU : voir note dans requirements.txt racine)
pip install -r requirements.txt
python -m pinn.train --smoke                   # vérification rapide
```

**Poids démo commités exprès** : [pinn/models/mh_pinn_v2.pt](pinn/models/)
(~0,6 Mo — un clone frais fait tourner la démo sans réentraînement ; décision
d'équipe, exception ciblée du .gitignore).

**Mock vs live** : sans `CRUSOE_API_KEY`, tout tourne en mock déterministe
(scores de jury figés à 4.1 — la signature du mock). Avec la clé, triage,
advisory, débat, jury et chat passent par les modèles Crusoe
(`GET /api/health` affiche le mode). Les 5 suites de tests sont conçues pour
le mock : `MOCK_LLM=1 python backend/agent/tests/test_scenarios.py` etc.

## API (contrat complet : [backend/agent/PIPELINE_CONTRACT.md](backend/agent/PIPELINE_CONTRACT.md))

| Route | Rôle |
|---|---|
| `GET /` · `GET /operator` | CureWatch · app intervention |
| `GET /api/health` | mode mock/live, modèles, boucle, epoch |
| `POST /api/loop/start?interval=&reset=` · `/api/loop/stop` | simulation |
| `POST /api/scenario/fault?kind=thermal\|bearing` · `/api/scenario/reset` | injection de fautes (démo) |
| `GET /api/stream` | SSE : tick / advisory / boss / override / tool_call / tool_result / notify / intervention |
| `GET /api/factory/state` | instantané par machine (contrat de boot des vues 3D) |
| `POST /api/chat` | `{question, stream, mode:"deep"\|"quick", machine_id, history[]}` |
| `POST /api/advisory/{id}/accept\|override` | décision opérateur (journalisée, boucle d'apprentissage) |
| `GET /api/history` · `GET /api/flow` | journal des décisions · topologie de l'information |
| `GET /api/operators` · `POST /api/machine/{id}/take_charge\|repair_done` | app intervention |

## Données

- **Datasets bruts NON versionnés** (`/data/`, multi-Go, retéléchargeables —
  procédure dans [pinn_data_exploration/](pinn_data_exploration/)).
- **Stand-ins démo commités** dans [pinn/data/](pinn/data/) (C-MAPSS FD001 +
  AI4I, petits fichiers publics rejoués par `telemetry_source.py`) — le même
  dossier contient le **package Python des loaders** (`*.py`).
- Seuils machine sourcés : [backend/agent/limits.py](backend/agent/limits.py) ·
  [docs/anomaly_thresholds.json](docs/anomaly_thresholds.json).

## Travailler en équipe

Branche dédiée par chantier, PR vers `main`, pas de push direct. Historique
complet des investigations physiques : `docs/v1_physics_diagnosis.md`,
`docs/v2_equation_stress_test.md`.
