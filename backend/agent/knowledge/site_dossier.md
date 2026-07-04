# Michelin Roanne — Site Selection Dossier (extracted)

Source: Michelin_Roanne_Site_Selection_Dossier.pdf


---
[page 1]

Prepared for the project demo
SITE & PRODUCTION-LINE SELECTION DOSSIER
                                                                                                                   July 2026 · v1




P R E D I C T I V E M A I N T E N A N C E & S A F E T Y · D I G I TA L-T W I N C A S E S T U DY



The line to model:
Michelin Roanne (France)
Recommended target — the Ultra-High-Performance passenger-tyre
line built on Michelin's automated, digital C3M process.
One product, one continuous chain of machines, high human-risk stages, and a cost of
stoppage large enough to prove the value of predictive maintenance.



  SITE                            PRODUCT                          OUTPUT                         PROCESS

  Roanne, Loire — France          UHP passenger tyres              ≈ 5,000 tyres / day            C3M — automated, digital,
                                  (Porsche, BMW, AMG,                                             electric curing
                                  Tesla…)




Contents — 01 Michelin at a glance · 02 The Roanne site · 03 The production chain · 04 Cost of              Public sources
downtime                                                                                                    only




Illustrative selection dossier · public data only                                                                        p. 1 / 5

---
[page 2]

MICHELIN ROANNE — SITE & LINE SELECTION                                                                               Predictive-maintenance & safety case study



01 — MICHELIN, THE BIG PICTURE


How the organisation is built
Michelin is the world's #1–#2 tyre maker and a materials group. It runs as a matrix: three customer-facing
business segments, a set of transversal functions shared across all of them, and a layer of geographic regions. Our
project lives inside three of those functions — Industrial Operations, Health-Safety-Environment, and Data & AI.


    €27.2 bn                               ≈125,400                         26                               >200 M                                 6,000
    2024 SALES                             EMPLOYEES                        COUNTRIES WITH                   TYRES / YEAR                           R&D STAFF
                                                                            PLANTS




                                                            MANAGING CHAIRMAN — FLORENT MENEGAUX
                                                         Group Executive Committee · Strategy “Michelin in Motion 2030”




           SR1 · AUTOMOTIVE & TWO-WHEEL                                   SR2 · ROAD TRANSPORTATION                                SR3 · SPECIALTY BUSINESSES

     Passenger · light-truck · two-wheel                          Truck & bus tyres +                                     Mining/OTR · Aircraft ·
     tyres — OE + replacement                                     MICHELIN Connected Fleets                               Agriculture · High-Tech Materials


     €14.7 bn · margin 13.1% (2024)                               €6.6 bn · margin 9.0% (2024)                            €5.9 bn · margin 14.6% (2024)




     TRANSVERSAL FUNCTIONS — SHARED ACROSS ALL SEGMENTS


               R&D (Design)                  Manufacturing / Industrial ★                Supply Chain                 Purchasing                    Sales & Marketing



              Finance                      People / HR              Data & AI ★                  Health · Safety · Environment ★                          Quality


     ★ where this project plugs in — Industrial Ops × Data & AI × Health-Safety-Environment




     GEOGRAPHIC REGIONS — Europe · North America · Asia · South America · Africa / India / Middle East




      Fig. 1 — Michelin operating model: three business segments (SR1–SR3) over shared functions, run across regions.


Michelin already treats predictive maintenance and digital twins as strategic: the group runs Industry-4.0 “digital leader” plants
and multiple digital-twin and condition-/predictive-maintenance demonstrators, and lists safety as a core value. A predictive-
maintenance + safety layer on a production line is therefore aligned with the group's real direction, not a bolt-on.




Illustrative selection dossier · public data only                                                                                                                       p. 2 / 5

---
[page 3]

MICHELIN ROANNE — SITE & LINE SELECTION                                               Predictive-maintenance & safety case study



02 — THE SELECTED SITE


Michelin Roanne — a premium, digital tyre plant
Roanne (Loire, France) makes only Ultra-High-Performance passenger tyres, exclusively for prestige carmakers, on
Michelin's automated C3M process. It is a rare case of a single-product, already-digital line — clean to model —
with strong human-risk stages and a high, quantifiable cost of stoppage.


    1974                     ≈833                     ≈5,000                   100%                        €300 M
    PLANT OPENED             EMPLOYEES                TYRES / DAY              C3M BY 2025                 INVESTED SINCE
                                                                                                           2014



                                                                     ▸ SITE DEPARTMENTS (PLANT ORGANISATION)

     Product & customers
                                                                      DEPARTMENT              SCOPE
     Ultra-High-Performance passenger car tyres for Porsche,
                                                                      Mixing                  Rubber compounding
     BMW, Mercedes-AMG, Aston Martin and Tesla. High unit
     value, original-equipment (just-in-time) supply — one of only    Semi-finished           Extrusion, calendering, beads
     two European sites producing Michelin's very-high-end tyres.
     Runs a 5×8 pattern, so the machines operate six days out of      Building                Green-tyre assembly (C3M)
     seven.
                                                                      Curing                  Vulcanisation (electric
                                                                                              presses)

                                                                      Finishing / Quality     Trimming, uniformity,
     Why Roanne wins on all three criteria                                                    inspection

         Documented chain. Single product, known machine              Maintenance / Eng.      Asset reliability, tooling
         sequence, publicly-communicated process (C3M) —
         easy to map end-to-end.                                      Industrial Logistics    Material & WIP flow

         Human risk is real. Mixing dust, calender nip points and     HSE                     Health, safety, environment
         hot high-pressure curing presses put safety squarely in
                                                                      Digital / Industry      Data, C3M automation
         play.                                                        4.0
         Stoppage is expensive & provable. Premium output +
         prestige OEM just-in-time supply make the cost of
         downtime both high and easy to argue.




Illustrative selection dossier · public data only                                                                          p. 3 / 5

---
[page 4]

MICHELIN ROANNE — SITE & LINE SELECTION                                                                                            Predictive-maintenance & safety case study



03 — THE PRODUCTION CHAIN · THE CORE OF THIS DOSSIER


Machine-by-machine, and where a stop propagates
A tyre flows through a fixed succession of machines. Component preparation fans out into three parallel streams
that converge at building; the green tyre is then cured and inspected. The two stages that carry the highest human
risk and the highest cost of stoppage are calendering (nip points) and curing (the bottleneck).



                                                                                02 · EXTRUSION
                                                                                Tread & sidewall profiles
                                                                                ▲ hot rubber (burns)




             RAW                     01 · MIXING                                03 · CALENDERING                        05 · BUILDING                           06 · CURING
        rubber · black               Banbury internal                                                                   C3M assembly →                          Electric presses (C3M)
                                                                                Textile & steel plies
        silica · chem.               mixers → compound                                                                  green (uncured) tyre                    heat + pressure
                                                                                ▲ NIP POINTS — severe
                                     ▲ dust · heat · UPSTREAM                                                           ▲ robotics · pinch                      ▲ burns · pressure · BOTTLENECK




                                                                                04 · BEAD BUILDING
                                                                                Bead-wire assemblies
                                                                                ▲ wire handling
                                                                                                                       DISPATCH                        07 · INSPECTION & FINISHING
                                                                                                                   to OEM (just-in-time)               Uniformity · X-ray · AI vision
                                                                                                                                                       ▲ low · quality gate



      IMPACT IF CURING (06) STOPS                                                                                    HAZARD
      In-process green tyres are scrapped, Building (05) backs up, and                                                  Mechanical           Thermal           Dust
      Inspection (07) is starved — the single costliest stoppage on the line.
                                                                                                                        High-severity        Nominal           ▲ = risk marker


     Fig. 2 — Roanne UHP tyre chain (C3M). Node borders encode the dominant human hazard; the dashed ring marks the curing
                                                                                               bottleneck.


                                                                                                                                                 WHAT THE AGENT / PIN
 #        STAGE                                  FUNCTION                                 HUMAN RISK                      SEVERITY
                                                                                                                                                 WATCHES


 01       Mixing                                 Compound rubber, black,                  Dust (respiratory +               Medium               Motor current · vibration · dust /
                                                 silica, chemicals                        explosion), heat, rotors                               LEL · temperature

 02       Extrusion                              Tread & sidewall profiles                Hot rubber — burns                Medium               Temperature · vibration ·
                                                                                                                                                 current

 03       Calendering                            Coat textile & steel plies               Nip points — hand/arm             Severe               Light-curtain / interlock · roll
                                                                                          entrapment                                             vibration · current

 04       Bead building                          Bead-wire assemblies                     Wire handling, cutting            Medium               Tension · vibration · current

 05       Tyre building                          Assemble green (uncured)                 Robotics / pinch points           Medium               Robot faults · cycle time ·
                                                 tyre — C3M                                                                                      vibration

 06       Curing                                 Vulcanise — electric                     Burns, high-pressure              Severe               Mould temp · pressure · press
                                                 presses (C3M)                            media, crush                                           vibration · cycle drift

 07       Inspection &                           Uniformity, balance, X-ray,              Low (contained)                   Low                  Quality data · reject rate (quality
          finishing                              AI vision                                                                                       gate)

Branch & impact logic. Stages 02–04 run in parallel and converge at building (05); a fault in any one stalls building. Curing (06) is the
constraint: because a green tyre cannot wait, a curing outage scraps work-in-process, backs up building, and starves inspection —
which is exactly the failure the predictive layer is meant to catch early.




Illustrative selection dossier · public data only                                                                                                                                        p. 4 / 5

---
[page 5]

MICHELIN ROANNE — SITE & LINE SELECTION                                                          Predictive-maintenance & safety case study



04 — THE ECONOMICS


What a stoppage costs — and what the system saves
Michelin does not publish per-line downtime figures, so the model below is built bottom-up from public output data
and industry benchmarks. It is illustrative but defensible, and every input can be swapped for the team's own
numbers.

  ▸ BOTTOM-UP MODEL (ROANNE)                                             ▸ INDUSTRY BENCHMARK — COST OF ONE IDLE HOUR

    INPUT / STEP                                                VALUE     Roanne — direct output (est.)                            €20–30k

    Output                                     ≈5,000 tyres/day           Mid-size plant                                              $25k

    ≈ per hour (6 days/7)                            ≈200 tyres/h         Industrial median (ABB)                                    $125k

    Conservative production value                 €100–150 /tyre          General mfg avg (Aberdeen)                                 $260k

    Output is unrecoverable                               yes (OT)        Automotive (Siemens 2024)                                  $2.3M

                                                                         Bars are indicative (automotive sets the scale). Roanne's direct
    Direct loss / hour                                   ≈€20–30 k
                                                                         output loss is modest per hour, but its just-in-time coupling to
    Direct loss / 4-h stop                              ≈€80–120 k       prestige assembly lines pushes the tail risk toward the automotive
                                                                         figure. Equipment failure is the largest single cause of unplanned
    Direct loss / lost day                           ≈€0.5–0.75 M        downtime (~42%) — precisely the failure mode predictive
                                                                         maintenance addresses.
  Amplifiers on top of direct loss: hidden costs ×2–3 (restart scrap
  of green tyres, recalibration, ~833 idle staff, energy) and just-in-
  time exposure — OEM late-delivery penalties (~0.5–1%/week of
  order value, capped 5–10%), expedited freight and preferred-
  supplier risk.



   THE DEMO SCENARIO — ONE CASE THAT WORKS

   A curing press's vibration/thermal signature drifts. The system flags it ~48 h ahead,
   the fix moves to planned maintenance, and an unplanned ~4-hour stop (≈€80–120k
   direct, several hundred k€ all-in) plus a burns/pressure safety incident are both
   avoided.


   VALUE IN ONE LINE

   Convert unplanned stoppages on the highest-risk, highest-cost stage (curing) into scheduled maintenance —
   protecting both people and a six-figure-per-incident cost, on a line that already feeds just-in-time to Porsche, BMW
   and Tesla.



Method & sources. Company & site figures: Michelin 2024 Annual Results and Michelin “Roanne 50 years” release; segment
figures from Michelin 2024 reporting. Downtime benchmarks: Siemens True Cost of Downtime 2024 (automotive ≈$2.3M/h;
equipment failure ≈42% of unplanned downtime), ABB Value of Reliability (≈$125k/h median), Aberdeen (≈$260k/h
general average). No confidential Michelin data is used; all figures are public or clearly-labelled estimates for
illustration. Unit-value and per-hour figures are conservative placeholders to be replaced with the team's internal
numbers.




Illustrative selection dossier · public data only                                                                                      p. 5 / 5