# Troubleshooting log — simple words

What we tested, what was wrong, what we changed, and how we know it works now.
(Technical version with code details: `backend/agent/NOTES-D.md`.)

## How we tested
We fed the system fake but realistic sensor data built from the verified
machine limits (temperature 180–210 °C, pressure above 20 bar or 16–19 bar in
steam mode, cycle 10–15 min up to 30, ISO vibration zones A/B/C/D). Ten
scenarios: a healthy press, a slowly degrading one, an overheating one, a
pressure loss, a too-long cycle, a faked message, a slow drift, plus questions
to the chatbot and one operator override. After each scenario we compared what
the system SHOULD do with what it ACTUALLY did.

## Problem 1 — false alarm on a healthy press
- What happened: a perfectly normal cure at 196 °C set off an alarm.
- Why: the old temperature limit in the code was made up (alarm above 195 °C).
  The real, verified range says up to 210 °C is normal.
- Fix: all limits now live in ONE file (`backend/agent/limits.py`) that uses
  the verified table, with the source written next to each number.
- Proof: the healthy-press scenario now passes with zero alarms.

## Problem 2 — real overheating gave a vague alarm
- What happened: at 214 °C the alarm fired, but the reason did not say which
  limit was broken.
- Why: the first check (tier 1) never looked at the curing signals at all —
  only at the health score.
- Fix: tier 1 now checks temperature, pressure, vibration zone and cycle time,
  and writes the limit into the reason, e.g. "mould_temp 214.0 °C above 210
  normal ceiling".
- Proof: the overheating scenario shows the limit and the value in the alarm.

## Problem 3 — a 31-minute cycle raised no flag
- What happened: a cycle far past the 30-minute maximum was treated as normal.
- Why: nobody had written a cycle-time check. The test data never had long
  cycles, so the gap stayed invisible.
- Fix: cycle check added (10–15 min normal, 15–30 watch, over 30 alarm).
- Proof: the long-cycle scenario now raises HIGH with a clear reason.

## Problem 4 — one test protected the OLD wrong number
- What happened: after fixing the limits, one old test failed.
- Why: that test had the made-up vibration limit (7.0) written inside it. The
  verified ISO danger line is 7.1.
- Fix: the test now checks 7.1.
- Lesson: tests must read limits from the limits file, not carry their own copies.

## Result
All 10 scenarios pass. 11 of 11 components do what they should: threshold
check, fast classifier, advisory writer, debate, jury, message authentication,
storage, drift detector, chatbot with tools, operator override loop, plant
summary. Re-run anytime with:

    MOCK_LLM=1 python backend/agent/tests/test_scenarios.py
