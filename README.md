# Crusoe — Factory Digital-Twin

Projet d'équipe pour le **RAISE Summit 2026 hackathon, track Crusoe** : un système de jumeau
numérique d'usine, construit par une équipe de 5.

Le système s'articule autour de deux couches développées en parallèle :

1. **Information layer** — compression, raisonnement, débat, sécurité (HMAC) et
   observabilité/audit de la télémétrie. Tourne aujourd'hui sur des données synthétiques
   de substitution en attendant le modèle physique.
2. **Physical layer (MH-PINN)** — un ensemble de **Multi-Head Physics-Informed Neural
   Networks**, un par machine surveillée. Chaque PINN partage un « corps » récurrent
   (type PI-LSTM) et plusieurs « têtes » de sortie, une par phénomène physique surveillé
   (vibration, thermique/puissance, dégradation / durée de vie restante). Objectif :
   estimer l'état physique réel d'une machine à partir de capteurs, puis alimenter la
   couche d'information en télémétrie structurée et signée (HMAC).

## Cadre narratif / démo

Le système s'inspire du procédé de fabrication de pneus de Michelin. La machine ciblée
pour la physique est la **presse de cuisson (vulcanisation)** — le goulot d'étranglement
reconnu en fabrication de pneus (températures jusqu'à 180 °C, pressions > 20 bar, cycles
de 10–15 min).

> **Aucune donnée confidentielle Michelin** n'est utilisée nulle part. Nous utilisons
> délibérément des jeux de données publics, vérifiés et physiquement plausibles comme
> substituts, et nous le signalons explicitement à chaque présentation.

## Structure du dépôt

```
.
├── docs/        Briefs et documentation de projet
│   └── claude-code-prompt-data-exploration.md   Brief de la tâche d'exploration des données
└── README.md
```

Les dossiers de travail (couche d'information, `pinn_data_exploration/`, etc.) seront
ajoutés au fur et à mesure, chacun sur sa branche dédiée.

## Travailler en équipe

- Travaillez sur une **branche dédiée** par chantier ; ne poussez pas directement sur
  `main`. Ouvrez une Pull Request pour fusionner.
- Ne touchez pas au code d'un·e coéquipier·ère sans concertation (les deux couches
  évoluent en parallèle).
- Les **jeux de données brutes ne sont pas versionnés** (voir `.gitignore`) : ils sont
  volumineux et se retéléchargent. Documentez les sources plutôt que de committer les
  fichiers.

## Prochaine étape

La première tâche planifiée est l'**exploration et la vérification des jeux de données**
avant de concevoir l'architecture PINN. Le brief détaillé se trouve dans
[`docs/claude-code-prompt-data-exploration.md`](docs/claude-code-prompt-data-exploration.md).
