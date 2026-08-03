"""
Generates a synthetic repair-history dataset for the AI Repair Intelligence POC.
Fault codes are real OBD-II / EV diagnostic codes; root-cause distributions are
synthetic but built to shift realistically with mileage, age, and vehicle type.
"""
import random
import csv
import os

random.seed(42)

VEHICLE_TYPES = ["ICE", "Scooter", "EV"]

# fault_code -> (possible root causes with base weights, applies_to vehicle types)
FAULT_CODES = {
    # --- Misfire family ---
    "P0300": (["Spark plug worn", "Fuel injector clog", "Ignition coil fault"], ["ICE", "Scooter"]),
    "P0301": (["Spark plug worn", "Fuel injector clog", "Ignition coil fault"], ["ICE", "Scooter"]),
    "P0302": (["Spark plug worn", "Fuel injector clog", "Ignition coil fault"], ["ICE", "Scooter"]),
    "P0303": (["Spark plug worn", "Fuel injector clog", "Ignition coil fault"], ["ICE", "Scooter"]),
    "P0304": (["Spark plug worn", "Fuel injector clog", "Ignition coil fault"], ["ICE", "Scooter"]),

    # --- Fuel/air metering ---
    "P0171": (["Vacuum leak", "Fuel pump weak", "MAF sensor dirty"], ["ICE", "Scooter"]),
    "P0172": (["Fuel injector stuck open", "MAF sensor dirty", "Fuel pressure regulator fault"], ["ICE", "Scooter"]),
    "P0100": (["MAF sensor dirty", "Wiring harness fault"], ["ICE", "Scooter"]),
    "P0105": (["MAP sensor fault", "Wiring harness fault"], ["ICE", "Scooter"]),
    "P0110": (["IAT sensor fault", "Wiring harness fault"], ["ICE", "Scooter"]),
    "P0115": (["Coolant temp sensor fault", "Wiring harness fault"], ["ICE", "Scooter"]),
    "P0120": (["Throttle position sensor fault", "Wiring harness fault"], ["ICE", "Scooter"]),

    # --- O2 sensor family ---
    "P0130": (["Oxygen sensor fault", "Wiring harness fault"], ["ICE", "Scooter"]),
    "P0135": (["O2 sensor heater fault", "Wiring harness fault"], ["ICE", "Scooter"]),
    "P0141": (["O2 sensor heater fault", "Wiring harness fault"], ["ICE", "Scooter"]),

    # --- Fuel pressure/delivery ---
    "P0087": (["Fuel pump weak", "Fuel filter clogged", "Fuel pressure regulator fault"], ["ICE", "Scooter"]),
    "P0088": (["Fuel pressure regulator fault", "Fuel pressure sensor fault"], ["ICE", "Scooter"]),
    "P0191": (["Fuel pressure sensor fault", "Wiring harness fault"], ["ICE", "Scooter"]),
    "P0230": (["Fuel pump relay fault", "Wiring harness fault"], ["ICE", "Scooter"]),

    # --- Ignition/position sensors ---
    "P0325": (["Knock sensor fault", "Wiring harness fault"], ["ICE", "Scooter"]),
    "P0335": (["Crank position sensor fault", "Wiring harness fault"], ["ICE", "Scooter"]),
    "P0340": (["Cam position sensor fault", "Wiring harness fault"], ["ICE", "Scooter"]),

    # --- Catalyst/emissions ---
    "P0420": (["Catalytic converter worn", "Oxygen sensor fault", "Exhaust leak"], ["ICE", "Scooter"]),
    "P0440": (["EVAP system leak", "EVAP purge valve fault"], ["ICE", "Scooter"]),
    "P0442": (["EVAP system leak"], ["ICE", "Scooter"]),
    "P0455": (["EVAP system leak", "EVAP purge valve fault"], ["ICE", "Scooter"]),

    # --- Idle/throttle ---
    "P0505": (["Idle air control valve fault", "Throttle body dirty"], ["ICE", "Scooter"]),
    "P0507": (["Idle air control valve fault", "Throttle body dirty"], ["ICE", "Scooter"]),

    # --- Charging/voltage ---
    "P0562": (["Battery voltage low", "Alternator fault", "Wiring corrosion"], ["ICE", "Scooter"]),
    "P0563": (["Voltage regulator fault", "Alternator fault"], ["ICE", "Scooter"]),
    "P0620": (["Alternator fault", "Wiring harness fault"], ["ICE", "Scooter"]),
    "B1318": (["12V battery weak", "Charging system fault"], ["ICE", "Scooter", "EV"]),

    # --- Cooling ---
    "P0480": (["Cooling fan fault"], ["ICE", "Scooter"]),

    # --- Transmission (CVT/auto-clutch scooters) ---
    "P0700": (["Transmission control fault", "Gear sensor fault", "Wiring harness fault"], ["ICE", "Scooter"]),
    "P0705": (["Gear sensor fault", "Wiring harness fault"], ["Scooter"]),
    "P0715": (["Input speed sensor fault", "Wiring harness fault"], ["Scooter"]),
    "P0720": (["Output speed sensor fault", "Wiring harness fault"], ["Scooter"]),
    "P0730": (["Transmission control fault", "Wiring harness fault"], ["Scooter"]),

    # --- EV (no universal EV DTC standard exists like SAE J2012 for OBD-II;
    # these are representative fault categories, not from a published standard) ---
    "EV0101": (["Battery cell imbalance", "BMS communication fault", "Thermal sensor fault"], ["EV"]),
    "EV0102": (["Regen braking control fault", "Wiring harness fault"], ["EV"]),
    "EV0204": (["Motor controller overheat", "Cooling fan fault", "Coolant pump fault"], ["EV"]),
    "EV0205": (["Motor phase wiring fault"], ["EV"]),
    "EV0310": (["Charging port fault", "Onboard charger fault", "Charging cable fault"], ["EV"]),
    "EV0311": (["Main contactor fault"], ["EV"]),
    "EV0412": (["HV isolation fault"], ["EV"]),
    "EV0501": (["DC-DC converter fault"], ["EV"]),
    "P0A80": (["Battery pack degraded", "Cell voltage imbalance"], ["EV"]),
}

PARTS = {
    "Spark plug worn": "Spark plug",
    "Fuel injector clog": "Fuel injector",
    "Ignition coil fault": "Ignition coil",
    "Vacuum leak": "Vacuum hose",
    "Fuel pump weak": "Fuel pump",
    "Fuel filter clogged": "Fuel filter",
    "Fuel pressure regulator fault": "Fuel pressure regulator",
    "MAF sensor dirty": "MAF sensor",
    "Catalytic converter worn": "Catalytic converter",
    "Oxygen sensor fault": "O2 sensor",
    "Exhaust leak": "Exhaust gasket",
    "Battery voltage low": "12V battery",
    "Alternator fault": "Alternator",
    "Wiring corrosion": "Wiring harness",
    "12V battery weak": "12V battery",
    "Charging system fault": "Charging module",
    "Battery cell imbalance": "Battery module",
    "BMS communication fault": "BMS unit",
    "Thermal sensor fault": "Thermal sensor",
    "Motor controller overheat": "Motor controller",
    "Cooling fan fault": "Cooling fan",
    "Coolant pump fault": "Coolant pump",
    "Charging port fault": "Charging port",
    "Onboard charger fault": "Onboard charger",
    "Charging cable fault": "Charging cable",
    "Battery pack degraded": "Battery pack",
    "Cell voltage imbalance": "Battery module",
    "Transmission control fault": "TCU",
    "Gear sensor fault": "Gear sensor",
    "Wiring harness fault": "Wiring harness",
    "Fuel injector stuck open": "Fuel injector",
    "MAP sensor fault": "MAP sensor",
    "IAT sensor fault": "IAT sensor",
    "Coolant temp sensor fault": "Coolant temp sensor",
    "Throttle position sensor fault": "Throttle position sensor",
    "O2 sensor heater fault": "O2 sensor",
    "Fuel pressure sensor fault": "Fuel pressure sensor",
    "Fuel pump relay fault": "Fuel pump relay",
    "Knock sensor fault": "Knock sensor",
    "Crank position sensor fault": "Crank position sensor",
    "Cam position sensor fault": "Cam position sensor",
    "EVAP system leak": "EVAP hose/canister",
    "EVAP purge valve fault": "EVAP purge valve",
    "Idle air control valve fault": "Idle air control valve",
    "Throttle body dirty": "Throttle body",
    "Voltage regulator fault": "Voltage regulator",
    "Input speed sensor fault": "Input speed sensor",
    "Output speed sensor fault": "Output speed sensor",
    "Regen braking control fault": "Regen braking controller",
    "Motor phase wiring fault": "Motor phase wiring",
    "Main contactor fault": "Main contactor",
    "HV isolation fault": "HV wiring/insulation",
    "DC-DC converter fault": "DC-DC converter",
}

NOTE_TEMPLATES = [
    "Customer reported {symptom}. Checked {part_area}, found issue consistent with {cause}.",
    "Bike came in with {symptom}. {cause} suspected after inspection.",
    "Routine service flagged {symptom}. Root cause traced to {cause}.",
    "No prior complaints, code triggered during diagnostic scan. {cause} confirmed on teardown.",
]

SYMPTOMS = ["rough idling", "reduced range", "warning light on dash", "slow charging",
            "unusual noise", "power loss on acceleration", "battery draining fast"]


def weighted_cause(causes, mileage, age_months):
    # bias toward wear-related causes as mileage/age increase
    weights = []
    for c in causes:
        w = 1.0
        if "worn" in c or "degraded" in c or "weak" in c:
            w += (mileage / 20000) + (age_months / 24)
        weights.append(w)
    return random.choices(causes, weights=weights, k=1)[0]


def gen_row():
    fault_code, (causes, vtypes) = random.choice(list(FAULT_CODES.items()))
    vehicle_type = random.choice(vtypes)
    age_months = random.randint(1, 60)
    mileage = random.randint(500, 60000)
    prior_services = max(0, int(mileage / 8000) + random.randint(-1, 1))
    root_cause = weighted_cause(causes, mileage, age_months)
    replaced_part = PARTS.get(root_cause, "Unknown part")
    note_template = random.choice(NOTE_TEMPLATES)
    note = note_template.format(
        symptom=random.choice(SYMPTOMS),
        part_area=replaced_part.lower(),
        cause=root_cause.lower(),
    )
    return [vehicle_type, fault_code, age_months, mileage, prior_services, note, replaced_part, root_cause]


def main(n=6000, out_path=None):
    out_path = out_path or os.path.join(os.path.dirname(__file__), "repair_history.csv")
    header = ["vehicle_type", "fault_code", "vehicle_age_months", "mileage_km",
              "prior_services", "technician_notes", "replaced_part", "root_cause"]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for _ in range(n):
            writer.writerow(gen_row())
    print(f"Wrote {n} rows to {out_path}")


if __name__ == "__main__":
    main()
