CREATE TABLE IF NOT EXISTS repair_records (
    id SERIAL PRIMARY KEY,
    vehicle_type VARCHAR(20) NOT NULL,        -- ICE, Scooter, EV
    fault_code VARCHAR(20) NOT NULL,
    vehicle_age_months INT NOT NULL,
    mileage_km INT NOT NULL,
    prior_services INT NOT NULL,
    technician_notes TEXT,
    replaced_part VARCHAR(100),
    root_cause VARCHAR(100) NOT NULL,          -- label
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    fault_code VARCHAR(20) NOT NULL,
    vehicle_type VARCHAR(20) NOT NULL,
    vehicle_age_months INT,
    mileage_km INT,
    prior_services INT,
    vin VARCHAR(17),
    make VARCHAR(50),
    model VARCHAR(50),
    year INT,
    predicted_cause VARCHAR(100),
    confidence FLOAT,
    explanation TEXT,
    technician_override VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);
