from math import e
import simpy
import numpy as np
from config import *
import random


class Inventory:
    def __init__(self, env, item_id, holding_cost):
        self.env = env
        self.item_id = item_id  # 0: product; others: WIP or raw material
        self.on_hand_inventory = INIT_LEVEL  # capacity=infinity\
        self.inventory = 0  # inventory=on_hand_inventory+intransition_inventory
        self.intransition_inventory = 0
        self.unit_holding_cost = holding_cost/24  # $/unit*hour
        self.holding_cost_last_updated = 0.0
        self.daily_inven_cost = 0
        self.capacity_limit = INVEN_LEVEL_MAX
        # self.unit_shortage_cost = shortage_cost
        # self.level_over_time = []  # Data tracking for inventory level
        # self.inventory_cost_over_time = []  # Data tracking for inventory cost
        # self.total_inven_cost = []

    def _cal_holding_cost(self, daily_events):
        holding_cost = self.on_hand_inventory * self.unit_holding_cost * \
            (self.env.now - self.holding_cost_last_updated)
        self.holding_cost_last_updated = self.env.now
        daily_events.append(
            f"{self.env.now}: {I[self.item_id]['NAME']}\'s On_Hand_Inventory level                 : {self.on_hand_inventory} units")
        daily_events.append(
            f"{self.env.now}: {I[self.item_id]['NAME']}\'s Daily holding cost updated              : {holding_cost}")
        self.daily_inven_cost += holding_cost

    def update_inven_level(self, quantity_of_change, daily_events):

        self.on_hand_inventory += quantity_of_change

        if self.on_hand_inventory > self.capacity_limit:
            daily_events.append(
                f"{self.env.now}: Due to the upper limit of the inventory, {I[self.item_id]['NAME']} is wasted: {self.on_hand_inventory - self.capacity_limit}")
            self.on_hand_inventory = self.capacity_limit
        if self.on_hand_inventory < 0:
            daily_events.append(
                f"{self.env.now}: Shortage of {I[self.item_id]['NAME']}: {self.capacity_limit - self.on_hand_inventory}")
            self.on_hand_inventory = 0
        self._cal_holding_cost(daily_events)

        self.inventory = self.on_hand_inventory+self.intransition_inventory
    # def cal_inventory_cost(self, daily_events):
    #     if self.current_level > 0:
    #         self.inventory_cost_over_time.append(
    #             self.holding_cost * self.current_level)
    #     elif self.current_level < 0:
    #         self.inventory_cost_over_time.append(
    #             self.unit_shortage_cost * abs(self.current_level))
    #     else:
    #         self.inventory_cost_over_time.append(0)
    #     daily_events.append(
    #         f"[Inventory Cost of {I[self.item_id]['NAME']}]  {self.inventory_cost_over_time[-1]}")


class Provider:
    def __init__(self, env, name, item_id):
        self.env = env
        self.name = name
        self.item_id = item_id

    def deliver_to_manufacturer(self, demand_size, inventory, daily_events):
        # Lead time
        lead_time = I[self.item_id]["SUP_LEAD_TIME"]

        daily_events.append(
            f"{self.env.now}: {I[self.item_id]['NAME']} will be delivered at {lead_time} days after         : {I[self.item_id]['LOT_SIZE_ORDER']} units")

        yield self.env.timeout(lead_time*24)
        daily_events.append(
            f"===============PRODUCT\'s Inventory ===============")
        inventory.intransition_inventory = inventory.intransition_inventory - \
            demand_size  # update intransition_inventory
        # update on_hand_inventory
        inventory.update_inven_level(demand_size, daily_events)
        daily_events.append(
            f"{self.env.now}: {I[self.item_id]['NAME']} has delivered                             : {demand_size} units ")  # Record when Material provide


class Procurement:
    def __init__(self, env, item_id, purchase_cost, setup_cost):
        self.env = env
        self.item_id = item_id
        self.unit_purchase_cost = purchase_cost
        self.unit_setup_cost = setup_cost
        self.daily_procurement_cost = 0
        # self.purchase_cost_over_time = []  # Data tracking for purchase cost
        # self.setup_cost_over_time = []  # Data tracking for setup cost

    def _cal_procurement_cost(self, order_size, daily_events):
        self.daily_procurement_cost += self.unit_purchase_cost * \
            order_size + self.unit_setup_cost

    def order_material(self, provider, inventory, daily_events):
        time = I[self.item_id]["MANU_ORDER_CYCLE"] * \
            24  # Control timeout function
        while True:
            yield self.env.timeout(time)
            daily_events.append(
                f"==============={I[self.item_id]['NAME']}\'s Inventory ===============")  # Change timeout function to cycle 24 hours

            # THIS WILL BE AN ACTION OF THE AGENT
            order_size = I[self.item_id]["LOT_SIZE_ORDER"]
            # order_size = agent.choose_action_tmp(inventory)
            if order_size > 0:

                  inventory.intransition_inventory += order_size  # update intansition_inventory
                   inventory.inventory += order_size  # update inventory

                    self.env.process(provider.deliver_to_manufacturer(
                        order_size, inventory, daily_events))
                    self._cal_procurement_cost(order_size, daily_events)
                    # Record intransition_inventory
                    daily_events.append(
                        f"{self.env.now}: RAW MATERIAL 1\'s Intransition_inventory                  : {inventory.intransition_inventory} units ")

                    # Record inventory
                    daily_events.append(
                        f"{self.env.now}: RAW MATERIAL 1\'s Real_Inventory                          : {inventory.inventory} units  ")
            daily_events.append(
                f"{self.env.now}: {I[self.item_id]['NAME']}\'s daily procurement cost                  : {self.daily_procurement_cost}")  # Change timeout function to cycle 24 hours


class Production:
    def __init__(self, env, name, process_id, production_rate, output, input_inventories, qnty_for_input_item, output_inventory, processing_cost, process_stop_cost):
        self.env = env
        self.name = name
        self.process_id = process_id
        self.production_rate = production_rate
        self.output = output
        self.input_inventories = input_inventories
        self.qnty_for_input_item = qnty_for_input_item
        self.output_inventory = output_inventory
        self.unit_processing_cost = processing_cost
        # self.processing_cost_over_time = []  # Data tracking for processing cost
        self.unit_process_stop_cost = process_stop_cost
        self.daily_production_cost = 0

    def _cal_processing_cost(self, processing_time, daily_events):
        processing_cost = self.unit_processing_cost * processing_time
        self.daily_production_cost += processing_cost
        daily_events.append(
            f"{self.env.now}: {self.name}\'s Daily production cost updated         : {self.daily_production_cost}")

    def process(self, daily_events):
        while True:
            # self.daily_production_cost += self.unit_process_stop_cost
            daily_events.append(
                "===============Process Phase===============")
            # Check the current state if input materials or WIPs are available
            shortage_check = False
            for inven, input_qnty in zip(self.input_inventories, self.qnty_for_input_item):
                if inven.on_hand_inventory < input_qnty:  # SHORTAGE
                    shortage_check = True
            # Check the current state if the output inventory is full
            inven_upper_limit_check = False
            if self.output_inventory.on_hand_inventory >= self.output_inventory.capacity_limit:
                inven_upper_limit_check = True

            if shortage_check:
                # self.daily_production_cost += self.unit_process_stop_cost
                daily_events.append(
                    f"{self.env.now}: Stop {self.name} due to a shortage of input materials or WIPs")
                # daily_events.append(f"{self.env.now}: Process stop cost : {self.unit_process_stop_cost}")
                # Check again next day
                yield self.env.timeout(24 - (self.env.now % 24))
                # continue
            elif inven_upper_limit_check:
                # self.daily_production_cost += self.unit_process_stop_cost
                daily_events.append(
                    f"{self.env.now}: Stop {self.name} due to the upper limit of the inventory. The output inventory is full")
                # daily_events.append(f"{self.env.now}: Process stop cost : {self.unit_process_stop_cost}")
                # Check again next day
                yield self.env.timeout(24 - (self.env.now % 24))
                # continue
            else:
                # Consuming input materials or WIPs and producing output WIP or Product
                daily_events.append(
                    f"{self.env.now}: Process {self.process_id} begins")
                # Update the inventory level for input items
                for inven, input_qnty in zip(self.input_inventories, self.qnty_for_input_item):
                    inven.update_inven_level(-input_qnty, daily_events)
                # Processing time has elapsed
                processing_time = 24 / self.production_rate

                yield self.env.timeout(processing_time)
                daily_events.append(
                    "===============Result Phase================")
                daily_events.append(
                    f"{self.env.now}: {self.output['NAME']} has been produced                         : 1 units")
                # Update the inventory level for the output item
                self.output_inventory.update_inven_level(1, daily_events)
                # Calculate the processing cost
                self._cal_processing_cost(processing_time, daily_events)


class Sales:
    def __init__(self, env, item_id, delivery_cost, setup_cost, backorder, due_date):
        self.env = env
        self.item_id = item_id
        self.unit_delivery_cost = delivery_cost
        self.unit_setup_cost = setup_cost
        self.unit_backorder_cost = backorder
        self.due_date = due_date
        # self.selling_cost_over_time = []  # Data tracking for selling cost
        self.daily_selling_cost = 0
        self.daily_penalty_cost = 0

    def _cal_selling_cost(self, demand_size, daily_events):
        self.daily_selling_cost += self.unit_delivery_cost * \
            demand_size + self.unit_setup_cost
        daily_events.append(
            f"{self.env.now}: {I[self.item_id]['NAME']}\'s daily selling cost                      : {self.daily_selling_cost}")

    def _cal_penalty_cost(self, num_shortages, daily_events):
        self.daily_penalty_cost += self.unit_backorder_cost * num_shortages
        daily_events.append(
            f"{self.env.now}: {I[self.item_id]['NAME']}\'s daily penalty cost                      : {self.daily_penalty_cost}")

    def deliver_to_cust(self, demand_size, product_inventory, daily_events):
        # Lead time
        # yield self.env.timeout(self.due_date * 24)
        # BACKORDER: Check if products are available
        if product_inventory.on_hand_inventory < demand_size:
            num_shortages = abs(
                product_inventory.on_hand_inventory - demand_size)
            if product_inventory.on_hand_inventory > 0:  # Delivering the remaining products to the customer
                daily_events.append(
                    f"{self.env.now}: PRODUCT have been delivered to the customer       : {product_inventory.on_hand_inventory} units ")
                # yield self.env.timeout(DELIVERY_TIME)
                product_inventory.update_inven_level(
                    -product_inventory.on_hand_inventory, daily_events)
                self._cal_selling_cost(
                    product_inventory.on_hand_inventory, daily_events)
            self._cal_penalty_cost(num_shortages, daily_events)
            daily_events.append(
                f"[Daily penalty cost] {self.daily_penalty_cost}")
            daily_events.append(
                f"{self.env.now}: Unable to deliver {num_shortages} units to the customer due to product shortage")
            # Check again after 24 hours (1 day)
            # yield self.env.timeout(24)
        # Delivering products to the customer
        else:
            product_inventory.update_inven_level(-demand_size, daily_events)
            daily_events.append(
                f"{self.env.now}: PRODUCT have been delivered to the customer       : {demand_size} units  ")
            self._cal_selling_cost(demand_size, daily_events)

    def sales_cycle(self, env, product_inventory, daily_events):
         cont = 0
          while (True):
               if cont == 0:
                    yield env.timeout(I[0]['DUE_DATE']*24)
                    self.deliver_to_cust(ORDER_HISTORY.pop(
                        0), product_inventory, daily_events)
                    cont = 1
                else:
                    yield env.timeout(I[0]["CUST_ORDER_CYCLE"]*24)
                    self.deliver_to_cust(ORDER_HISTORY.pop(
                        0), product_inventory, daily_events)


class Customer:
    def __init__(self, env, name, item_id):
        self.env = env
        self.name = name
        self.item_id = item_id

    def order_product(self, env, product_inventory, daily_events):
        while True:
            yield self.env.timeout(I[0]["CUST_ORDER_CYCLE"] * 24)
            # THIS WILL BE A RANDOM VARIABLE
            order_size = I[0]["DEMAND_QUANTITY"]
            # order_size = random.randint(DEMAND_QTY_MIN, DEMAND_QTY_MAX)


def create_env(I, P, daily_events):
    # Create a SimPy environment
    simpy_env = simpy.Environment()

    # Create an inventory for each item
    inventoryList = []
    for i in I.keys():
        inventoryList.append(
            Inventory(simpy_env, i, I[i]["HOLD_COST"]))
    # Create stakeholders (Customer, Providers)
    customer = Customer(simpy_env, "CUSTOMER", I[0]["ID"])
    providerList = []
    procurementList = []
    for i in I.keys():
        # Create a provider and the corresponding procurement if the type of the item is Raw Material
        if I[i]["TYPE"] == 'Raw Material':
            providerList.append(Provider(simpy_env, "PROVIDER_"+str(i), i))
            procurementList.append(Procurement(
                simpy_env, I[i]["ID"], I[i]["PURCHASE_COST"], I[i]["SETUP_COST_RAW"]))
    # Create managers for manufacturing process, procurement process, and delivery process
    sales = Sales(simpy_env, customer.item_id,
                  I[0]["DELIVERY_COST"], I[0]["SETUP_COST_PRO"], I[0]["BACKORDER_COST"], I[0]["DUE_DATE"])
    productionList = []
    for i in P.keys():
        output_inventory = inventoryList[P[i]["OUTPUT"]["ID"]]
        input_inventories = []
        for j in P[i]["INPUT_TYPE_LIST"]:
            input_inventories.append(inventoryList[j["ID"]])
        productionList.append(Production(simpy_env, "PROCESS_"+str(i), P[i]["ID"],
                                         P[i]["PRODUCTION_RATE"], P[i]["OUTPUT"], input_inventories, P[i]["QNTY_FOR_INPUT_ITEM"], output_inventory, P[i]["PROCESS_COST"], P[i]["PROCESS_STOP_COST"]))

    return simpy_env, inventoryList, procurementList, productionList, sales, customer, providerList, daily_events


def simpy_event_processes(simpy_env, inventoryList, procurementList, productionList, sales, customer, providerList, daily_events, I):
    # Event processes for SimPy simulation
    # Customer

    for production in productionList:  # Processes
        simpy_env.process(production.process(daily_events))
    for i in range(len(providerList)):  # Procurements
       simpy_env.process(procurementList[i].order_material(
           providerList[i], inventoryList[providerList[i].item_id], daily_events))
    simpy_env.process(customer.order_product(
        sales, inventoryList[I[0]["ID"]], daily_events))
    simpy_env.process(sales.sales_cycle(
        simpy_env, inventoryList[0], daily_events))


# The total cost is accumulated every hour.
def cal_daily_cost_ACC(inventoryList, procurementList, productionList, sales):
    daily_total_cost = 0
    for inven in inventoryList:
        daily_total_cost += inven.daily_inven_cost
        inven.daily_inven_cost = 0
    for production in productionList:
        daily_total_cost += production.daily_production_cost
        production.daily_production_cost = 0
    for procurement in procurementList:
        daily_total_cost += procurement.daily_procurement_cost
        procurement.daily_procurement_cost = 0
    daily_total_cost += sales.daily_selling_cost
    sales.daily_selling_cost = 0
    daily_total_cost += sales.daily_penalty_cost
    sales.daily_penalty_cost = 0

    return daily_total_cost

# The total cost is calculated based on the inventory level for every 24 hours.


# def cal_daily_cost_DESC(s1, s2, agent_action):
#     daily_total_cost = 0
#     HoldingCost = s1 * I[0]['HOLD_COST'] + s2 * I[1]['HOLD_COST']
#     ProductionCost = P[0]['PROCESS_COST']
#     ProcurementCost = I[1]['PURCHASE_COST'] * agent_action
#     SellingCost = 0
#     # SellingCost += I[0]['DELIVERY_COST'] * I[0]['DEMAND_QUANTITY'] + I[0]['SETUP_COST_PRO']
#     if s1 < I[0]['DEMAND_QUANTITY']:
#         PenaltyCost = I[0]['BACKORDER_COST'] * (I[0]['DEMAND_QUANTITY'] - s1)
#     else:
#         PenaltyCost = 0
#     daily_total_cost = HoldingCost+ProductionCost + \
#         ProcurementCost+SellingCost+PenaltyCost
#     return daily_total_cost


def cap_current_state(inventoryList):
    # State space: inventory level
    state = np.array([inven.on_hand_inventory for inven in inventoryList])
    # # State space: inventory level + demand quantity
    if STATE_DEMAND:
        state = np.append(state, I[0]['DEMAND_QUANTITY'])

    return state
