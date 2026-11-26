-- SPDX-FileCopyrightText: Copyright (c) 2026, AkilsuryaS.
-- All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

CREATE SCHEMA sales;
CREATE VIEW sales.customers AS
SELECT * FROM (VALUES (1, 'Tailspin Toys'), (2, 'Wingtip Toys')) AS customers(customer_id, customer_name);
CREATE VIEW sales.orders AS
SELECT * FROM (VALUES (101, 1, DATE '2026-01-01'), (102, 2, DATE '2026-01-02')) AS orders(order_id, customer_id, order_date);
CREATE VIEW sales.order_lines AS
SELECT * FROM (VALUES (1001, 101, 3, 12.50::numeric), (1002, 102, 2, 25.00::numeric)) AS order_lines(order_line_id, order_id, quantity, unit_price);
