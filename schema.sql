-- Bakery AI System canonical MySQL schema
-- Structure only: no operational or seed data is included.

CREATE DATABASE IF NOT EXISTS `bakery_ai` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `bakery_ai`;

CREATE TABLE `attendance_records` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `emp_id` varchar(20) NOT NULL,
  `emp_name` varchar(50) NOT NULL,
  `emp_role` varchar(30) NOT NULL,
  `date` date NOT NULL,
  `punch_in` time DEFAULT NULL,
  `punch_out` time DEFAULT NULL,
  `status` varchar(20) DEFAULT 'present',
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_attendance_emp_date` (`emp_id`,`date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `batch_inventory` (
  `batch_id` varchar(50) NOT NULL,
  `product_name` varchar(50) NOT NULL,
  `quantity` int(11) NOT NULL DEFAULT 0,
  `production_time` datetime NOT NULL,
  `tray_color` varchar(20) DEFAULT 'green',
  `freshness_status` varchar(30) DEFAULT 'Fresh',
  `quantity_initial` int(11) DEFAULT NULL,
  `quantity_remaining` int(11) DEFAULT NULL,
  `sales_area` varchar(30) DEFAULT 'Fresh Area',
  PRIMARY KEY (`batch_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `business_events` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `event_type` varchar(40) NOT NULL,
  `start_date` date NOT NULL,
  `end_date` date NOT NULL,
  `products` json DEFAULT NULL,
  `discount_pct` float DEFAULT NULL,
  `note` text DEFAULT NULL,
  `active` tinyint(4) NOT NULL DEFAULT 1,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `detection_log` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `model_version` varchar(30) NOT NULL,
  `image_id` varchar(100) DEFAULT NULL,
  `scenario` varchar(30) NOT NULL DEFAULT 'checkout',
  `predicted_class` varchar(50) DEFAULT NULL,
  `bbox` json DEFAULT NULL,
  `confidence` float DEFAULT NULL,
  `inference_time` float DEFAULT NULL,
  `manual_check_required` tinyint(4) DEFAULT 0,
  `final_class` varchar(50) DEFAULT NULL,
  `error_type` varchar(30) DEFAULT 'none',
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `employees` (
  `id` varchar(10) NOT NULL,
  `name` varchar(50) NOT NULL,
  `skills` json NOT NULL DEFAULT (JSON_ARRAY('bakery')),
  `min_hours_per_week` float NOT NULL DEFAULT 15,
  `max_hours_per_week` float NOT NULL DEFAULT 40,
  `available` tinyint(4) NOT NULL DEFAULT 1,
  `rest_days_per_week` int(11) NOT NULL DEFAULT 1,
  `unavailable_dates` json NOT NULL DEFAULT (JSON_ARRAY()),
  `pin` varchar(10) DEFAULT '1234',
  `role` varchar(30) DEFAULT 'bakery',
  `role2` varchar(30) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `inventory_transactions` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `transaction_type` varchar(20) NOT NULL,
  `batch_id` varchar(50) DEFAULT NULL,
  `product_name` varchar(50) NOT NULL,
  `quantity` int(11) NOT NULL,
  `unit_price` float DEFAULT 0,
  `discount_applied` float DEFAULT 0,
  `freshness_status` varchar(30) DEFAULT NULL,
  `transaction_time` timestamp NOT NULL DEFAULT current_timestamp(),
  `receipt_id` varchar(50) DEFAULT NULL,
  `beverage_size` varchar(20) DEFAULT NULL,
  `beverage_temp` varchar(20) DEFAULT NULL,
  `beverage_sweetness` varchar(30) DEFAULT NULL,
  `beverage_ice` varchar(20) DEFAULT NULL,
  `reversal_of_transaction_id` int(11) DEFAULT NULL,
  `disposition` varchar(30) DEFAULT NULL,
  `reason` varchar(255) DEFAULT NULL,
  `performed_by` varchar(50) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_inventory_transactions_reversal` (`reversal_of_transaction_id`),
  KEY `idx_inventory_transactions_receipt_id` (`receipt_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `material_inventory` (
  `material_name` varchar(50) NOT NULL,
  `current_stock` decimal(12,6) DEFAULT 0.000000,
  `unit` varchar(20) DEFAULT 'kg',
  `threshold` decimal(12,6) DEFAULT 1.000000,
  `cost_per_unit` decimal(8,2) DEFAULT 0.00,
  PRIMARY KEY (`material_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `material_transactions` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `material_name` varchar(50) NOT NULL,
  `transaction_type` varchar(20) NOT NULL,
  `quantity` decimal(12,6) DEFAULT 0.000000,
  `unit` varchar(20) DEFAULT 'g',
  `reference` varchar(100) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  KEY `idx_mat_trans` (`material_name`,`transaction_type`,`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `material_wastage_log` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `material_name` varchar(50) NOT NULL,
  `check_date` date NOT NULL,
  `theoretical_stock` decimal(12,6) DEFAULT 0.000000,
  `actual_stock` decimal(12,6) DEFAULT 0.000000,
  `theoretical_consumed` decimal(12,6) DEFAULT 0.000000,
  `actual_consumed` decimal(12,6) DEFAULT 0.000000,
  `wastage_qty` decimal(12,6) DEFAULT 0.000000,
  `wastage_rate` decimal(10,4) DEFAULT 0.0000,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  KEY `idx_mat_date` (`material_name`,`check_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `order_items` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `order_id` int(11) DEFAULT NULL,
  `product_name` varchar(50) DEFAULT NULL,
  `quantity` int(11) DEFAULT 1,
  `line_total` decimal(10,2) DEFAULT 0.00,
  `line_profit` decimal(10,2) DEFAULT 0.00,
  `unit_price` decimal(8,2) DEFAULT 0.00,
  `discount_rate` decimal(5,4) DEFAULT 0.0000,
  `freshness` varchar(30) DEFAULT 'Fresh',
  `coffee_temp` varchar(10) DEFAULT NULL,
  `coffee_ice` varchar(10) DEFAULT NULL,
  `coffee_sugar` varchar(10) DEFAULT NULL,
  `coffee_size` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `attendance_correction_log` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `emp_id` varchar(20) NOT NULL,
  `attendance_date` date NOT NULL,
  `previous_punch_in` time DEFAULT NULL,
  `previous_punch_out` time DEFAULT NULL,
  `previous_status` varchar(20) DEFAULT NULL,
  `corrected_punch_in` time NOT NULL,
  `corrected_punch_out` time NOT NULL,
  `corrected_status` varchar(20) NOT NULL,
  `reason` varchar(255) NOT NULL,
  `corrected_by` varchar(50) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  KEY `idx_attendance_correction_date` (`attendance_date`,`emp_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `sick_leave_log` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `employee_id` varchar(20) NOT NULL,
  `leave_date` date NOT NULL,
  `action` varchar(20) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  KEY `idx_sick_leave_date` (`leave_date`,`employee_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `sick_replacements` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `original_employee_id` varchar(20) NOT NULL,
  `original_employee_name` varchar(50) NOT NULL,
  `schedule_date` date NOT NULL,
  `time_slot` varchar(20) NOT NULL,
  `role` varchar(30) NOT NULL,
  `demand_level` varchar(20) DEFAULT 'normal',
  `production_target` int(11) DEFAULT NULL,
  `replaced_at` datetime NOT NULL,
  `replacement_employee_id` varchar(20) DEFAULT NULL,
  `is_undone` tinyint(1) NOT NULL DEFAULT 0,
  PRIMARY KEY (`id`),
  KEY `idx_sick_replacement_date` (`schedule_date`,`original_employee_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `recommendation_events` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `request_id` varchar(50) NOT NULL,
  `operation_date` date NOT NULL,
  `shown_at` datetime NOT NULL,
  `rank_position` int(11) NOT NULL,
  `bakery_product` varchar(50) NOT NULL,
  `beverage_product` varchar(50) DEFAULT NULL,
  `score` decimal(10,4) DEFAULT NULL,
  `discount_rate` decimal(5,4) NOT NULL DEFAULT 0.0000,
  `discount_source` varchar(40) DEFAULT NULL,
  `discount_strategy` varchar(80) DEFAULT NULL,
  `selected_at` datetime DEFAULT NULL,
  `purchased_order_id` int(11) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  KEY `idx_recommendation_events_date` (`operation_date`),
  KEY `idx_recommendation_events_request` (`request_id`),
  KEY `idx_recommendation_events_order` (`purchased_order_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `orders` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `order_date` date NOT NULL,
  `product_name` varchar(50) DEFAULT NULL,
  `quantity` int(11) DEFAULT 1,
  `unit_price` decimal(8,2) DEFAULT 0.00,
  `total` decimal(8,2) DEFAULT 0.00,
  `total_amount` decimal(10,2) DEFAULT 0.00,
  `total_profit` decimal(10,2) DEFAULT 0.00,
  `discount_total` decimal(10,2) DEFAULT 0.00,
  `state` varchar(20) DEFAULT 'completed',
  `payment_method` varchar(20) DEFAULT 'Cash',
  `dine_type` varchar(10) DEFAULT 'dine_in',
  `order_time` time DEFAULT '12:00:00',
  `ticket_id` varchar(50) DEFAULT NULL,
  `subtotal` decimal(10,2) DEFAULT 0.00,
  `item_count` int(11) DEFAULT 1,
  `beverage_size` varchar(10) DEFAULT NULL,
  `beverage_temp` varchar(10) DEFAULT NULL,
  `beverage_sweetness` varchar(10) DEFAULT NULL,
  `beverage_ice` varchar(10) DEFAULT NULL,
  `is_rainy` tinyint(4) DEFAULT 0,
  `temp_mean` decimal(5,2) DEFAULT NULL,
  `temp_range` decimal(5,2) DEFAULT NULL,
  `is_cold_day` tinyint(4) DEFAULT 0,
  `is_hot_day` tinyint(4) DEFAULT 0,
  `is_member_day` tinyint(4) DEFAULT 0,
  `is_competitor` tinyint(4) DEFAULT 0,
  `is_new_product` tinyint(4) DEFAULT 0,
  `is_day1` tinyint(4) DEFAULT 0,
  `is_top3` tinyint(4) DEFAULT 0,
  `discount_pct` decimal(5,4) DEFAULT 0.0000,
  `refund_reason` varchar(255) DEFAULT NULL,
  `refunded_by` varchar(50) DEFAULT NULL,
  `refunded_at` datetime DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `payments` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `order_id` int(11) DEFAULT NULL,
  `payment_method` varchar(20) DEFAULT 'Cash',
  `amount` decimal(10,2) DEFAULT 0.00,
  `payment_date` date DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `product_inventory` (
  `product_name` varchar(50) NOT NULL,
  `stock` int(11) NOT NULL DEFAULT 0,
  `day1_stock` int(11) NOT NULL DEFAULT 0,
  `unit_price` decimal(8,2) DEFAULT 0.00,
  PRIMARY KEY (`product_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `product_recipes` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `product_name` varchar(50) NOT NULL,
  `material_name` varchar(50) NOT NULL,
  `quantity_per_unit` decimal(12,6) DEFAULT 0.000000,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `products` (
  `product_name` varchar(50) NOT NULL,
  `material_cost` decimal(8,2) DEFAULT 0.00,
  `wastage_pct` decimal(5,2) DEFAULT 0.05,
  `unit_price` decimal(8,2) DEFAULT 0.00,
  `category` varchar(20) DEFAULT 'bread',
  `selling_price` decimal(8,2) DEFAULT 0.00,
  `stock_day1` int(11) DEFAULT 0,
  `daily_capacity` int(11) DEFAULT 50,
  PRIMARY KEY (`product_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `raw_materials` (
  `material_name` varchar(50) NOT NULL,
  `stock_quantity` decimal(12,6) DEFAULT 0.000000,
  `unit` varchar(20) DEFAULT 'kg',
  `unit_price` decimal(8,2) DEFAULT 0.00,
  `category` varchar(30) DEFAULT 'baking',
  `reorder_point` decimal(12,6) DEFAULT 1.000000,
  `track_inventory` tinyint(1) NOT NULL DEFAULT 1,
  PRIMARY KEY (`material_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `receipts` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `receipt_id` varchar(50) NOT NULL,
  `items` json NOT NULL,
  `subtotal` float NOT NULL DEFAULT 0,
  `discount_total` float NOT NULL DEFAULT 0,
  `total` float NOT NULL DEFAULT 0,
  `savings` float NOT NULL DEFAULT 0,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  UNIQUE KEY `receipt_id` (`receipt_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `shift_schedule` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `schedule_date` date NOT NULL,
  `time_slot` varchar(20) NOT NULL,
  `employee_id` varchar(10) DEFAULT NULL,
  `employee_name` varchar(50) DEFAULT NULL,
  `role` varchar(30) DEFAULT 'bakery',
  `staff_count` int(11) NOT NULL DEFAULT 1,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `demand_level` varchar(10) DEFAULT 'normal',
  `production_target` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `schedule_history` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `change_group` varchar(36) NOT NULL,
  `snapshot_type` varchar(24) NOT NULL,
  `change_type` varchar(32) NOT NULL,
  `change_reason` varchar(255) NOT NULL,
  `source_schedule_id` int DEFAULT NULL,
  `schedule_date` date NOT NULL,
  `time_slot` varchar(20) NOT NULL,
  `employee_id` varchar(10) DEFAULT NULL,
  `employee_name` varchar(50) DEFAULT NULL,
  `role` varchar(30) DEFAULT NULL,
  `staff_count` int NOT NULL DEFAULT 1,
  `demand_level` varchar(10) DEFAULT 'normal',
  `production_target` int DEFAULT NULL,
  `recorded_at` timestamp NOT NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  KEY `idx_schedule_history_date` (`schedule_date`,`recorded_at`),
  KEY `idx_schedule_history_baseline` (`snapshot_type`,`schedule_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `users` (
  `username` varchar(50) NOT NULL,
  `password_hash` varchar(255) NOT NULL,
  `role` varchar(20) NOT NULL DEFAULT 'staff',
  PRIMARY KEY (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

ALTER TABLE `order_items`
  MODIFY `order_id` int(11) NOT NULL,
  ADD KEY `idx_order_items_order_id` (`order_id`),
  ADD CONSTRAINT `fk_order_items_order`
    FOREIGN KEY (`order_id`) REFERENCES `orders` (`id`) ON DELETE CASCADE;

ALTER TABLE `payments`
  MODIFY `order_id` int(11) NOT NULL,
  ADD KEY `idx_payments_order_id` (`order_id`),
  ADD CONSTRAINT `fk_payments_order`
    FOREIGN KEY (`order_id`) REFERENCES `orders` (`id`) ON DELETE CASCADE;
