"""
Static repair knowledge base, keyed by root_cause label.

Deliberately NOT model-generated: recommendations, checklists, and risk levels
are fixed reference data a service-ops lead would sign off on, same as a
workshop manual. The ML model only ranks *which* cause is likely; this table
supplies what a technician does once a cause is suspected.
"""

# risk: "high" (safety-critical -- vehicle should not be released until resolved),
# "medium" (should be fixed soon, not an immediate safety issue),
# "low" (comfort/efficiency issue)
REPAIR_KNOWLEDGE = {
    "12V battery weak": {
        "risk": "medium",
        "recommendation": "Load-test the 12V battery; replace if it fails under load or is past rated cycle life.",
        "checklist": ["Measure resting + cranking voltage", "Inspect terminals for corrosion", "Load-test battery"],
    },
    "Alternator fault": {
        "risk": "medium",
        "recommendation": "Bench-test alternator output; replace or rebuild if output is below spec at idle and load.",
        "checklist": ["Check charging voltage at idle/2000rpm", "Inspect drive belt tension", "Test alternator on bench if removed"],
    },
    "BMS communication fault": {
        "risk": "high",
        "recommendation": "Do not operate until BMS communication is restored -- check harness/connector before assuming BMS failure.",
        "checklist": ["Inspect BMS harness + connector pins", "Check for stored comms fault codes", "Verify BMS firmware version"],
    },
    "Battery cell imbalance": {
        "risk": "high",
        "recommendation": "Run a balancing cycle; if imbalance persists, isolate and test individual cell groups.",
        "checklist": ["Pull per-cell voltage log", "Run balancing cycle", "Re-check spread after full charge"],
    },
    "Battery pack degraded": {
        "risk": "medium",
        "recommendation": "Run a capacity test against rated spec; quote pack replacement if capacity has dropped materially.",
        "checklist": ["Run full capacity test", "Compare to rated Ah", "Check for swelling/thermal damage"],
    },
    "Battery voltage low": {
        "risk": "medium",
        "recommendation": "Charge fully and re-test; if voltage sags under load, suspect a weak cell group or bad connection.",
        "checklist": ["Full charge then re-measure", "Load-test", "Check connector/fuse resistance"],
    },
    "Catalytic converter worn": {
        "risk": "low",
        "recommendation": "Confirm with back-pressure test before replacing converter; rule out upstream sensor fault first.",
        "checklist": ["Exhaust back-pressure test", "Check upstream O2 sensor readings", "Visual check for rattling/damage"],
    },
    "Cell voltage imbalance": {
        "risk": "high",
        "recommendation": "Same as battery cell imbalance -- balance first, isolate affected cell group if it recurs.",
        "checklist": ["Pull per-cell voltage log", "Run balancing cycle", "Flag pack if imbalance recurs within days"],
    },
    "Charging cable fault": {
        "risk": "low",
        "recommendation": "Inspect cable/connector for damage or bent pins; swap cable to confirm before replacing onboard charger.",
        "checklist": ["Visual check cable + connector", "Test with known-good cable", "Check pin continuity"],
    },
    "Charging port fault": {
        "risk": "low",
        "recommendation": "Inspect port for debris/corrosion/bent pins; reseat and retest before replacing.",
        "checklist": ["Visual inspect port", "Clean contacts", "Retest charge session"],
    },
    "Charging system fault": {
        "risk": "medium",
        "recommendation": "Trace charging path from source to battery; check fuses and charge controller before replacing components.",
        "checklist": ["Check charge controller status/fault log", "Test fuses/relays in charge path", "Verify charge current reaches battery"],
    },
    "Coolant pump fault": {
        "risk": "medium",
        "recommendation": "Check pump flow/pressure; replace if not meeting spec, otherwise check for airlock first.",
        "checklist": ["Check coolant level + for airlock", "Test pump flow/pressure", "Inspect for leaks at pump seal"],
    },
    "Cooling fan fault": {
        "risk": "medium",
        "recommendation": "Test fan operation directly (bypass thermostat control); check relay/fuse if fan doesn't spin.",
        "checklist": ["Manually trigger fan", "Check fan relay/fuse", "Inspect wiring to fan motor"],
    },
    "Exhaust leak": {
        "risk": "medium",
        "recommendation": "Locate leak point (manifold, joint, muffler) with smoke test or by ear at idle; repair or replace affected section.",
        "checklist": ["Inspect manifold/joints for soot staining", "Smoke test if source unclear", "Check gasket condition"],
    },
    "Fuel injector clog": {
        "risk": "low",
        "recommendation": "Run injector cleaning cycle first; replace only if flow test still fails after cleaning.",
        "checklist": ["Injector flow/spray pattern test", "Run cleaning additive cycle", "Re-test flow after cleaning"],
    },
    "Fuel pump weak": {
        "risk": "medium",
        "recommendation": "Measure fuel pressure at idle and under load; replace pump if it can't hold spec pressure.",
        "checklist": ["Fuel pressure test at idle", "Fuel pressure test under load", "Check for pump noise/whine"],
    },
    "Fuel filter clogged": {
        "risk": "low",
        "recommendation": "Check fuel pressure before/after the filter to confirm restriction; replace filter rather than guessing from the code alone.",
        "checklist": ["Compare pressure pre/post filter", "Inspect filter for debris on removal", "Check for restricted fuel flow at tank"],
    },
    "Fuel pressure regulator fault": {
        "risk": "medium",
        "recommendation": "Check regulator vacuum line and fuel pressure with/without vacuum applied; replace if it can't hold rated pressure.",
        "checklist": ["Inspect regulator vacuum line for cracks/disconnection", "Test fuel pressure with vacuum applied vs removed", "Check for fuel in the vacuum line (failed diaphragm)"],
    },
    "Gear sensor fault": {
        "risk": "medium",
        "recommendation": "Check sensor signal and connector before replacement; recalibrate if a reset resolves it.",
        "checklist": ["Check sensor connector/wiring", "Read live sensor signal", "Recalibrate if applicable"],
    },
    "Ignition coil fault": {
        "risk": "medium",
        "recommendation": "Swap-test coil with a known-good unit on a different cylinder to confirm before replacing.",
        "checklist": ["Check spark strength", "Swap-test with known-good coil", "Inspect coil connector/wiring"],
    },
    "MAF sensor dirty": {
        "risk": "low",
        "recommendation": "Clean MAF sensor with proper cleaner and re-test; replace only if readings stay out of spec after cleaning.",
        "checklist": ["Visual inspect sensor element", "Clean with MAF-safe cleaner", "Re-check live airflow reading"],
    },
    "Motor controller overheat": {
        "risk": "high",
        "recommendation": "Check cooling path and derate history before assuming controller failure; do not run under load until confirmed.",
        "checklist": ["Check controller temp log/derate events", "Inspect cooling fins/fan for controller", "Verify no loose high-current connections"],
    },
    "Onboard charger fault": {
        "risk": "medium",
        "recommendation": "Check charger fault codes and output voltage/current; test with alternate outlet before replacing unit.",
        "checklist": ["Read charger fault log", "Test on a different outlet/circuit", "Measure charger output"],
    },
    "Oxygen sensor fault": {
        "risk": "low",
        "recommendation": "Check sensor response time on live data; replace if sluggish or stuck rather than guessing from code alone.",
        "checklist": ["Read live O2 sensor voltage/response", "Check sensor wiring/connector", "Verify exhaust leak isn't the real cause"],
    },
    "Spark plug worn": {
        "risk": "low",
        "recommendation": "Inspect gap and electrode wear; replace set (not just one) if worn or fouled.",
        "checklist": ["Check plug gap/wear", "Inspect for fouling (oil/carbon)", "Replace full set if any are worn"],
    },
    "Thermal sensor fault": {
        "risk": "medium",
        "recommendation": "Compare sensor reading against a second reference thermometer before replacing; check connector first.",
        "checklist": ["Cross-check reading against reference", "Inspect sensor connector", "Check for intermittent fault under vibration"],
    },
    "Transmission control fault": {
        "risk": "medium",
        "recommendation": "Pull TCU fault history and check shift-solenoid wiring before replacing the control unit itself.",
        "checklist": ["Pull TCU fault history", "Check shift solenoid wiring", "Road-test to confirm symptom matches code"],
    },
    "Vacuum leak": {
        "risk": "low",
        "recommendation": "Smoke-test intake system to locate leak point; repair hose/gasket rather than guessing.",
        "checklist": ["Smoke test intake system", "Inspect vacuum hoses for cracking", "Check intake gasket seating"],
    },
    "Wiring corrosion": {
        "risk": "medium",
        "recommendation": "Inspect affected harness section for moisture ingress; clean/reseal or replace corroded segment.",
        "checklist": ["Visual inspect harness/connectors", "Check for moisture ingress point", "Test continuity across corroded section"],
    },
    "Wiring harness fault": {
        "risk": "medium",
        "recommendation": "Wiggle-test harness while monitoring live signal to localize intermittent fault before replacing harness.",
        "checklist": ["Wiggle-test harness under live monitoring", "Check ground points", "Inspect for chafing/damage points"],
    },
    "Fuel injector stuck open": {
        "risk": "medium",
        "recommendation": "Check injector flow/leak-down; a stuck-open injector causes rich running and can wash oil off cylinder walls if left unresolved.",
        "checklist": ["Injector leak-down test", "Check spark plug for fuel fouling", "Inspect injector for debris preventing seal"],
    },
    "MAP sensor fault": {
        "risk": "low",
        "recommendation": "Compare MAP reading against known-vacuum reference before replacing; check vacuum line to sensor first.",
        "checklist": ["Check vacuum line to MAP sensor", "Compare live reading to reference", "Inspect sensor connector"],
    },
    "IAT sensor fault": {
        "risk": "low",
        "recommendation": "Cross-check IAT reading against ambient temp; clean or replace only if reading is clearly wrong, not just borderline.",
        "checklist": ["Compare reading to ambient temp", "Inspect sensor connector/wiring", "Check for physical damage to sensor tip"],
    },
    "Coolant temp sensor fault": {
        "risk": "medium",
        "recommendation": "Cross-check against a physical thermometer at the radiator/block before replacing; a false-cold reading can mask overheating.",
        "checklist": ["Compare reading to physical thermometer", "Inspect sensor connector", "Check for coolant leak at sensor seat"],
    },
    "Throttle position sensor fault": {
        "risk": "medium",
        "recommendation": "Check for smooth, linear voltage sweep across full throttle travel; replace if it jumps or drops out at any point.",
        "checklist": ["Check voltage sweep through full throttle travel", "Inspect connector/wiring", "Check for mechanical play at sensor mount"],
    },
    "O2 sensor heater fault": {
        "risk": "low",
        "recommendation": "Check heater circuit resistance and fuse before replacing sensor; a blown fuse is cheaper and more common than a failed sensor.",
        "checklist": ["Check heater circuit fuse", "Measure heater element resistance", "Verify heater voltage at connector"],
    },
    "Fuel pressure sensor fault": {
        "risk": "medium",
        "recommendation": "Cross-check sensor reading against a mechanical fuel pressure gauge before condemning the sensor.",
        "checklist": ["Compare to mechanical gauge reading", "Inspect sensor connector/wiring", "Check for fuel leak at sensor seal"],
    },
    "Fuel pump relay fault": {
        "risk": "medium",
        "recommendation": "Swap-test relay with an identical known-good relay before replacing the fuel pump itself.",
        "checklist": ["Swap-test with known-good relay", "Check relay socket for corrosion", "Verify pump runs when relay is bypassed"],
    },
    "Knock sensor fault": {
        "risk": "medium",
        "recommendation": "Check sensor torque and connector before replacement; a loose sensor mount is a common false trigger.",
        "checklist": ["Check sensor mounting torque", "Inspect connector/wiring", "Verify with live knock signal if scan tool supports it"],
    },
    "Crank position sensor fault": {
        "risk": "high",
        "recommendation": "This can cause a no-start or stall while riding -- check air gap and connector before replacing; do not release the vehicle if it stalls unpredictably.",
        "checklist": ["Check sensor air gap to trigger wheel", "Inspect connector/wiring for damage", "Check for correct trigger wheel condition"],
    },
    "Cam position sensor fault": {
        "risk": "medium",
        "recommendation": "Check sensor air gap and connector before replacing; verify timing components are correctly indexed first.",
        "checklist": ["Check sensor air gap", "Inspect connector/wiring", "Verify cam timing is correctly indexed"],
    },
    "EVAP system leak": {
        "risk": "low",
        "recommendation": "Smoke-test the EVAP system to locate the actual leak point rather than replacing components on guesswork.",
        "checklist": ["Smoke test EVAP system", "Inspect hoses/canister for cracking", "Check gas cap seal if equipped"],
    },
    "EVAP purge valve fault": {
        "risk": "low",
        "recommendation": "Check valve opens/closes on command before replacing; a stuck-open valve causes rough idle, stuck-closed causes pressure codes.",
        "checklist": ["Command valve open/closed and listen/feel for actuation", "Check valve connector/wiring", "Inspect for carbon buildup preventing seal"],
    },
    "Idle air control valve fault": {
        "risk": "low",
        "recommendation": "Clean valve and passage before replacing; carbon buildup causing a stuck valve is far more common than valve failure.",
        "checklist": ["Clean valve and idle air passage", "Check valve connector/wiring", "Verify idle stabilizes after cleaning"],
    },
    "Throttle body dirty": {
        "risk": "low",
        "recommendation": "Clean throttle body and relearn idle; do not replace the whole assembly for a cleaning-fixable issue.",
        "checklist": ["Clean throttle bore and idle passage", "Perform idle relearn/reset after cleaning", "Check for vacuum leak at throttle body gasket"],
    },
    "Voltage regulator fault": {
        "risk": "medium",
        "recommendation": "Measure regulated output voltage under load; replace if it can't hold spec voltage rather than assuming from the code alone.",
        "checklist": ["Measure output voltage at idle and under load", "Check regulator ground connection", "Inspect for overheating damage on regulator"],
    },
    "Input speed sensor fault": {
        "risk": "medium",
        "recommendation": "Check sensor air gap and connector before replacing; verify with live sensor data during a road test if possible.",
        "checklist": ["Check sensor air gap/mounting", "Inspect connector/wiring", "Road-test with live sensor data if available"],
    },
    "Output speed sensor fault": {
        "risk": "medium",
        "recommendation": "Check sensor air gap and connector before replacing; verify with live sensor data during a road test if possible.",
        "checklist": ["Check sensor air gap/mounting", "Inspect connector/wiring", "Road-test with live sensor data if available"],
    },
    "Regen braking control fault": {
        "risk": "high",
        "recommendation": "Do not rely on regen braking until resolved -- verify mechanical brakes are fully functional as primary stopping power in the meantime.",
        "checklist": ["Confirm mechanical brakes are fully functional first", "Check regen control wiring/connector", "Read regen-specific fault codes if available"],
    },
    "Motor phase wiring fault": {
        "risk": "high",
        "recommendation": "Inspect phase wiring and connectors for damage before assuming motor failure; do not operate under load until confirmed safe.",
        "checklist": ["Inspect phase wiring/connectors for damage", "Check phase-to-phase resistance balance", "Verify motor controller fault log"],
    },
    "Main contactor fault": {
        "risk": "high",
        "recommendation": "Do not operate the vehicle until contactor operation is confirmed -- a stuck or failing contactor is a safety-critical fault.",
        "checklist": ["Check contactor click/engagement on power-up", "Inspect contactor contacts for pitting", "Check contactor coil wiring/connector"],
    },
    "HV isolation fault": {
        "risk": "high",
        "recommendation": "Treat as an electric-shock hazard until cleared -- isolate the high-voltage system and use proper PPE before any further inspection.",
        "checklist": ["Isolate HV system per safety procedure before touching anything", "Measure isolation resistance with appropriate tester", "Inspect for moisture ingress at HV connectors"],
    },
    "DC-DC converter fault": {
        "risk": "medium",
        "recommendation": "Measure converter output voltage under load; a failing DC-DC converter will present as intermittent 12V accessory issues.",
        "checklist": ["Measure DC-DC output voltage under load", "Check converter connector/wiring", "Check converter cooling/ventilation"],
    },
}

DEFAULT_ENTRY = {
    "risk": "medium",
    "recommendation": "No reference entry for this cause yet -- inspect based on technician judgment and log findings.",
    "checklist": ["Visual inspection", "Check related sensor/connector wiring", "Confirm with technician judgment"],
}


def lookup(root_cause: str) -> dict:
    return REPAIR_KNOWLEDGE.get(root_cause, DEFAULT_ENTRY)
