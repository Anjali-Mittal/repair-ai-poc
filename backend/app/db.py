import os
import time
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://repair:repair@postgres:5432/repair_ai")


def get_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def run_migrations(max_retries: int = 10, delay_seconds: float = 2.0):
    """
    Additive, idempotent schema patches for volumes created before a column was
    added -- so upgrading doesn't require wiping the Postgres volume. init.sql
    only runs once, on first container creation.

    Retries on connection failure: Postgres briefly restarts after its own
    initdb step on first boot, and depends_on/healthcheck don't fully close
    that race on every platform, so this is a second line of defense rather
    than the primary fix.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            conn = get_conn()
            with conn, conn.cursor() as cur:
                # Idempotent -- recreates schema on platforms (e.g. Render managed
                # Postgres) that don't run docker/init.sql on container start.
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS repair_records (
                        id SERIAL PRIMARY KEY,
                        vehicle_type VARCHAR(20) NOT NULL,
                        fault_code VARCHAR(20) NOT NULL,
                        vehicle_age_months INT NOT NULL,
                        mileage_km INT NOT NULL,
                        prior_services INT NOT NULL,
                        technician_notes TEXT,
                        replaced_part VARCHAR(100),
                        root_cause VARCHAR(100) NOT NULL,
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
                    ALTER TABLE predictions
                        ADD COLUMN IF NOT EXISTS vin VARCHAR(17),
                        ADD COLUMN IF NOT EXISTS make VARCHAR(50),
                        ADD COLUMN IF NOT EXISTS model VARCHAR(50),
                        ADD COLUMN IF NOT EXISTS year INT
                """)
            conn.close()
            return
        except psycopg2.OperationalError as e:
            last_err = e
            time.sleep(delay_seconds)
    raise last_err
