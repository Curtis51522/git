-- Bakery AI System - New Tables (Phase 2: Commercial Grade)
-- Run in Supabase SQL Editor

-- ============================================
-- 1. Products Master Data
-- ============================================
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    product_name VARCHAR(50) UNIQUE NOT NULL,
    category VARCHAR(20) NOT NULL CHECK (category IN ('bakery', 'coffee')),
    unit_price DECIMAL(8,2) NOT NULL,
    material_cost DECIMAL(8,2) DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Seed: 16 bakery + 8 coffee
INSERT INTO products (product_name, category, unit_price, material_cost) VALUES
    ('donut', 'bakery', 4.50, 1.35),
    ('croissant', 'bakery', 5.50, 1.90),
    ('bread_coconut', 'bakery', 3.50, 1.05),
    ('bread_roll', 'bakery', 3.50, 1.05),
    ('chiffon', 'bakery', 8.00, 2.40),
    ('croissant_chocolate', 'bakery', 5.50, 1.90),
    ('eggtart', 'bakery', 3.50, 1.20),
    ('cream_horn', 'bakery', 4.00, 1.50),
    ('melon_bread', 'bakery', 3.50, 1.30),
    ('pizza_bread', 'bakery', 4.50, 1.80),
    ('soboru_bread', 'bakery', 4.00, 1.40),
    ('chocopie', 'bakery', 4.50, 1.60),
    ('stickbread', 'bakery', 3.00, 1.00),
    ('baguette', 'bakery', 4.00, 1.10),
    ('pandesal', 'bakery', 3.00, 0.90),
    ('sourdough', 'bakery', 6.00, 1.80),
    ('latte', 'coffee', 8.50, 2.50),
    ('americano', 'coffee', 6.50, 1.80),
    ('cappuccino', 'coffee', 9.00, 2.80),
    ('cold_brew', 'coffee', 10.00, 3.00),
    ('iced_americano', 'coffee', 7.20, 2.00),
    ('mocha', 'coffee', 10.50, 3.20)
ON CONFLICT (product_name) DO NOTHING;

-- ============================================
-- 2. Orders (checkout header)
-- ============================================
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    order_time TIMESTAMPTZ DEFAULT NOW(),
    cashier_id VARCHAR(10) REFERENCES employees(id),
    payment_method VARCHAR(10) CHECK (payment_method IN ('cash', 'card', 'qr')),
    cash_received DECIMAL(8,2),
    change_given DECIMAL(8,2),
    subtotal DECIMAL(8,2),
    discount_total DECIMAL(8,2) DEFAULT 0,
    total_amount DECIMAL(8,2),
    total_profit DECIMAL(8,2) DEFAULT 0
);

-- ============================================
-- 3. Order Items (checkout line items)
-- ============================================
CREATE TABLE IF NOT EXISTS order_items (
    id SERIAL PRIMARY KEY,
    order_id INT REFERENCES orders(id),
    product_name VARCHAR(50),
    quantity INT NOT NULL,
    unit_price DECIMAL(8,2),
    discount_rate DECIMAL(5,3) DEFAULT 0,
    line_total DECIMAL(8,2),
    line_profit DECIMAL(8,2) DEFAULT 0,
    freshness VARCHAR(10) CHECK (freshness IN ('Fresh', 'Day-1')),
    coffee_temp VARCHAR(10),      -- 'hot' or 'iced'
    coffee_ice VARCHAR(10),       -- 'normal', 'less', 'none'
    coffee_sugar VARCHAR(10)      -- 'normal', 'less', 'half', 'none'
);

-- ============================================
-- 4. Attendance (check-in/out)
-- ============================================
CREATE TABLE IF NOT EXISTS attendance (
    id SERIAL PRIMARY KEY,
    employee_id VARCHAR(10) REFERENCES employees(id),
    date DATE NOT NULL,
    clock_in TIMESTAMPTZ,
    clock_out TIMESTAMPTZ,
    status VARCHAR(10) DEFAULT 'present' CHECK (status IN ('present', 'late', 'absent', 'leave'))
);

-- ============================================
-- 5. Raw Materials (ingredient inventory)
-- ============================================
CREATE TABLE IF NOT EXISTS raw_materials (
    id SERIAL PRIMARY KEY,
    material_name VARCHAR(50) UNIQUE NOT NULL,
    category VARCHAR(20),         -- flour, dairy, sugar, coffee, packaging
    unit VARCHAR(20),             -- kg, L, pcs, g
    stock_quantity DECIMAL(8,2) DEFAULT 0,
    unit_price DECIMAL(8,2),      -- purchase cost per unit
    reorder_point DECIMAL(8,2),
    supplier VARCHAR(50)
);

-- Seed: basic bakery ingredients
INSERT INTO raw_materials (material_name, category, unit, stock_quantity, unit_price, reorder_point, supplier) VALUES
    ('Bread Flour', 'flour', 'kg', 25.0, 3.50, 5.0, 'BakeSupply KL'),
    ('Cake Flour', 'flour', 'kg', 15.0, 4.20, 3.0, 'BakeSupply KL'),
    ('Sugar', 'sugar', 'kg', 20.0, 2.80, 5.0, 'BakeSupply KL'),
    ('Butter', 'dairy', 'kg', 8.0, 18.00, 2.0, 'DairyFresh MY'),
    ('Eggs', 'dairy', 'pcs', 120, 0.45, 30, 'DairyFresh MY'),
    ('Milk', 'dairy', 'L', 12.0, 6.50, 3.0, 'DairyFresh MY'),
    ('Coffee Beans', 'coffee', 'kg', 5.0, 45.00, 1.0, 'Barista Supply'),
    ('Chocolate Powder', 'coffee', 'kg', 3.0, 22.00, 0.5, 'Barista Supply'),
    ('Coconut Flakes', 'flour', 'kg', 4.0, 8.00, 1.0, 'BakeSupply KL'),
    ('Packaging Box', 'packaging', 'pcs', 200, 0.30, 50, 'PackPro'),
    ('Paper Cup', 'packaging', 'pcs', 300, 0.15, 80, 'PackPro')
ON CONFLICT (material_name) DO NOTHING;

-- ============================================
-- 6. Material Transactions (ingredient in/out)
-- ============================================
CREATE TABLE IF NOT EXISTS material_transactions (
    id SERIAL PRIMARY KEY,
    material_name VARCHAR(50),
    transaction_type VARCHAR(10) CHECK (transaction_type IN ('inflow', 'outflow')),
    quantity DECIMAL(8,2),
    unit_price DECIMAL(8,2),
    total_cost DECIMAL(8,2),
    transaction_time TIMESTAMPTZ DEFAULT NOW(),
    notes TEXT
);

-- ============================================
-- 7. Alert Log (S5 AI decisions + alerts)
-- ============================================
CREATE TABLE IF NOT EXISTS alert_log (
    id SERIAL PRIMARY KEY,
    alert_type VARCHAR(30),       -- stock_alert, staffing_alert, profit_alert, demand_spike
    severity VARCHAR(10) CHECK (severity IN ('info', 'warning', 'critical')),
    module VARCHAR(20),           -- demand, inventory, production, staffing, promo, profit
    title VARCHAR(100),
    detail TEXT,
    recommendation TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMPTZ
);

-- ============================================
-- 8. Daily Summary (pre-aggregated for Dashboard)
-- ============================================
CREATE TABLE IF NOT EXISTS daily_summary (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    total_sales DECIMAL(10,2) DEFAULT 0,
    total_profit DECIMAL(10,2) DEFAULT 0,
    order_count INT DEFAULT 0,
    avg_order_value DECIMAL(8,2) DEFAULT 0,
    cash_total DECIMAL(10,2) DEFAULT 0,
    card_total DECIMAL(10,2) DEFAULT 0,
    qr_total DECIMAL(10,2) DEFAULT 0,
    top_product VARCHAR(50),
    top_product_qty INT DEFAULT 0,
    bakery_sales DECIMAL(10,2) DEFAULT 0,
    coffee_sales DECIMAL(10,2) DEFAULT 0,
    day1_discount_total DECIMAL(10,2) DEFAULT 0,
    staff_count INT DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- 9. Employee KPI (monthly, auto-computed)
-- ============================================
-- Sources:
--   attendance_rate = COUNT(attendance where status IN ('present','late')) / total_scheduled_days
--   punctuality_rate = COUNT(attendance where status='present') / COUNT(attendance where status IN ('present','late'))
--   hours_compliance = SUM(actual_hours from clock_in/clock_out) / S3 scheduled_hours * 100
--   team_goal_rate   = team_actual_production / S2 forecast * 100 (same score for all in same role group)
--   z_score          = (sum of 4 normalized metrics - mean) / stddev, computed per role group
--   rank_in_team     = RANK() by z_score within role group
-- All metrics auto-computed — no manager input needed.

CREATE TABLE IF NOT EXISTS employee_kpi (
    id SERIAL PRIMARY KEY,
    employee_id VARCHAR(10) NOT NULL REFERENCES employees(id),
    month VARCHAR(7) NOT NULL,                          -- 'YYYY-MM'
    attendance_rate DECIMAL(5,2) DEFAULT 100.00,
    punctuality_rate DECIMAL(5,2) DEFAULT 100.00,
    hours_compliance DECIMAL(5,2) DEFAULT 100.00,       -- actual vs S3 scheduled
    team_goal_rate DECIMAL(5,2) DEFAULT 100.00,         -- team output vs S2 forecast
    z_score DECIMAL(6,3) DEFAULT 0,
    rank_in_team INT DEFAULT 0,
    UNIQUE (employee_id, month)
);

-- ============================================
-- 10. batch_inventory extension: baker tracking
-- ============================================
ALTER TABLE batch_inventory ADD COLUMN IF NOT EXISTS baker_id VARCHAR(10) REFERENCES employees(id);

-- ============================================
-- 11. Product Recipes (ingredient per unit)
-- ============================================
CREATE TABLE IF NOT EXISTS product_recipes (
    id SERIAL PRIMARY KEY,
    product_name VARCHAR(50) NOT NULL,
    material_name VARCHAR(50) NOT NULL,
    quantity_per_unit DECIMAL(8,3) NOT NULL,    -- grams / mL / pcs per 1 unit of product
    unit VARCHAR(10) NOT NULL DEFAULT 'g'        -- g, mL, pcs
);

-- Seed: standard bakery recipes (per piece)
INSERT INTO product_recipes (product_name, material_name, quantity_per_unit, unit) VALUES
    ('donut',          'Bread Flour',        45,  'g'),
    ('donut',          'Sugar',              10,  'g'),
    ('donut',          'Butter',              8,  'g'),
    ('donut',          'Eggs',             0.25, 'pcs'),
    ('croissant',      'Bread Flour',        55,  'g'),
    ('croissant',      'Butter',             22,  'g'),
    ('croissant',      'Sugar',               5,  'g'),
    ('croissant',      'Eggs',             0.30, 'pcs'),
    ('bread_coconut',  'Cake Flour',         50,  'g'),
    ('bread_coconut',  'Sugar',              12,  'g'),
    ('bread_coconut',  'Butter',             10,  'g'),
    ('bread_coconut',  'Coconut Flakes',     15,  'g'),
    ('bread_coconut',  'Eggs',             0.25, 'pcs'),
    ('bread_roll',     'Bread Flour',        40,  'g'),
    ('bread_roll',     'Butter',              6,  'g'),
    ('bread_roll',     'Sugar',               5,  'g'),
    ('chiffon',        'Cake Flour',         60,  'g'),
    ('chiffon',        'Sugar',              18,  'g'),
    ('chiffon',        'Butter',             15,  'g'),
    ('chiffon',        'Eggs',             0.50, 'pcs'),
    ('croissant_chocolate', 'Bread Flour',   55,  'g'),
    ('croissant_chocolate', 'Butter',        22,  'g'),
    ('croissant_chocolate', 'Chocolate Powder', 8, 'g'),
    ('croissant_chocolate', 'Sugar',          5,  'g'),
    ('croissant_chocolate', 'Eggs',        0.30, 'pcs'),
    -- 10 new bakery products
    ('eggtart',       'Cake Flour',         35,  'g'),
    ('eggtart',       'Butter',             15,  'g'),
    ('eggtart',       'Sugar',               8,  'g'),
    ('eggtart',       'Eggs',             0.50, 'pcs'),
    ('cream_horn',    'Bread Flour',        40,  'g'),
    ('cream_horn',    'Butter',             18,  'g'),
    ('cream_horn',    'Sugar',              10,  'g'),
    ('cream_horn',    'Eggs',             0.30, 'pcs'),
    ('melon_bread',   'Bread Flour',        45,  'g'),
    ('melon_bread',   'Butter',             12,  'g'),
    ('melon_bread',   'Sugar',              15,  'g'),
    ('melon_bread',   'Eggs',             0.30, 'pcs'),
    ('pizza_bread',   'Bread Flour',        50,  'g'),
    ('pizza_bread',   'Butter',              8,  'g'),
    ('pizza_bread',   'Sugar',               5,  'g'),
    ('pizza_bread',   'Eggs',             0.20, 'pcs'),
    ('soboru_bread',  'Bread Flour',        42,  'g'),
    ('soboru_bread',  'Butter',             14,  'g'),
    ('soboru_bread',  'Sugar',              12,  'g'),
    ('soboru_bread',  'Eggs',             0.30, 'pcs'),
    ('chocopie',      'Cake Flour',         35,  'g'),
    ('chocopie',      'Butter',             12,  'g'),
    ('chocopie',      'Chocolate Powder',   10,  'g'),
    ('chocopie',      'Sugar',              10,  'g'),
    ('chocopie',      'Eggs',             0.30, 'pcs'),
    ('stickbread',    'Bread Flour',        50,  'g'),
    ('stickbread',    'Butter',              5,  'g'),
    ('stickbread',    'Sugar',               3,  'g'),
    ('baguette',      'Bread Flour',        80,  'g'),
    ('baguette',      'Butter',              5,  'g'),
    ('baguette',      'Sugar',               3,  'g'),
    ('pandesal',      'Bread Flour',        40,  'g'),
    ('pandesal',      'Butter',              8,  'g'),
    ('pandesal',      'Sugar',              10,  'g'),
    ('pandesal',      'Eggs',             0.30, 'pcs'),
    ('sourdough',     'Bread Flour',       100,  'g'),
    ('sourdough',     'Butter',              5,  'g'),
    ('sourdough',     'Sugar',               2,  'g')
ON CONFLICT DO NOTHING;

-- ============================================
-- 12. Material Wastage Log (weekly, auto-computed)
-- ============================================
-- Flow:
--   Week starts → baker scans batch → system deducts (recipe × quantity), no wastage factor
--   End of week → manager inputs actual remaining stock
--   System computes: wastage_rate = (theoretical - actual) / total consumed
--   Next week's purchase forecast: recipe_usage × (1 + last_week_wastage_rate)
-- Default wastage starts at 5% until first week of data collected.

CREATE TABLE IF NOT EXISTS material_wastage_log (
    id SERIAL PRIMARY KEY,
    material_name VARCHAR(50) NOT NULL,
    week_start DATE NOT NULL,                     -- Monday
    week_end DATE NOT NULL,                       -- Sunday
    total_consumed DECIMAL(8,2) DEFAULT 0,        -- sum of (recipe × scan_qty) for the week
    theoretical_remaining DECIMAL(8,2),           -- opening + inflow - consumed
    actual_remaining DECIMAL(8,2),                -- manager input at week end
    wastage_rate DECIMAL(5,3) DEFAULT 0.050,      -- default 5%, auto-updated
    UNIQUE (material_name, week_start)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_orders_time ON orders(order_time);
CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
CREATE INDEX IF NOT EXISTS idx_alert_created ON alert_log(created_at);
CREATE INDEX IF NOT EXISTS idx_daily_summary_date ON daily_summary(date);
CREATE INDEX IF NOT EXISTS idx_employee_kpi_month ON employee_kpi(month);
CREATE INDEX IF NOT EXISTS idx_employee_kpi_rank ON employee_kpi(month, rank_in_team);
