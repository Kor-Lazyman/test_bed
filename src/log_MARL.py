# Log daily repots: Inventory level for each item; In-transition inventory for each material; Remaining demand (demand - product level)
STATE_ACTION_REPORT_REAL = []  # Real State
COST_RATIO_HISTORY = []

# Record the cumulative value of each cost component
LOG_TOTAL_COST_COMP = {
    'Holding cost': 0,
    'Process cost': 0,
    'Delivery cost': 0,
    'Order cost': 0,
    'Shortage cost': 0
}
