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
| **CureWatch** (UI) | Interface opérateur (Claude Design) servie par le backend | [backend/agent/static/](backend/agent/static/) | [docs/OPERATOR_APP.md](docs/OPERATOR_APP.md) |

**Interface entre couches (contrat figé)** : `SignedReading{payload: PinnReading,
signature}` — HMAC-SHA256 sur sérialisation canonique, stdlib pur. Testé **en
conditions réelles** (les deux codes du même arbre) :
[scripts/test_cross_layer_contract.py](scripts/test_cross_layer_contract.py).

## Démarrage rapide

```bash
# 1) Secrets — le repo est destiné à être public : AUCUNE clé committée, jamais
cp .env.example .env          # remplir CRUSOE_API_KEY (vide = mode mock complet)

# 2) Couche information + UI
pip install -r backend/requirements.txt
uvicorn backend.agent.main:app --port 8000     # UI sur http://localhost:8000

# 3) Couche physique (torch CPU : voir note dans requirements.txt racine)
pip install -r requirements.txt
python -m pinn.train --smoke                   # vérification rapide
```

**Poids démo commités exprès** : [pinn/models/mh_pinn_v2.pt](pinn/models/)
(~0,6 Mo — un clone frais fait tourner la démo sans réentraînement ; décision
d'équipe, exception ciblée du .gitignore).

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
