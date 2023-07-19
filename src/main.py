import simpy
import random
import visualization
# Items:
# ID: Index of the element in the dictionary
# TYPE: Product, Raw Material, WIP;
# NAME: Item's name or model;
# REMOVE## FROM: process key; TO: process key;
# CUST_ORDER_CYCLE: Customer ordering cycle [days]
# MANU_ORDER_CYCLE: Manufacturer ordering cycle to providers [days]
# DEMAND_QUANTITY: Demand quantity for the final product [units]
# DELIVERY_TIME_TO_CUST: Delivery time to the customer [days]
# DELIVERY_TIME_FROM_SUP: Delivery time from a supplier [days]
# MANU_LEAD_TIME: The total processing time for the manufacturer to process and deliver the customer's order [days]
# SUP_LEAD_TIME: The total processing time for a supplier to process and deliver the manufacturer's order [days]
# LOT_SIZE_ORDER: Lot-size for the order of raw materials (Q) [units]
# SETUP_COST_PRO: Setup cost for the delivery of the products to the customer [$/delivery]
# SETUP_COST_RAW: Setup cost for the ordering of the raw materials to a supplier [$/order]
# HOLD_COST: Holding cost of the items [$/unit*day]
# DELIVERY_COST_PRO: Holding cost of the products [$/unit]
# PURCHASE_COST_RAW: Holding cost of the raw materials [$/unit]
I = {0: {"ID": 0, "TYPE": "Product",      "NAME": "PRODUCT",          "CUST_ORDER_CYCLE": 7, "DEMAND_QUANTITY": 21, "MANU_LEAD_TIME": 7,                      "SETUP_COST_PRO": 500, "HOLD_COST": 35, "DELIVERY_COST_PRO": 5},
     1: {"ID": 1, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 1.1", "MANU_ORDER_CYCLE": 7,                        "SUP_LEAD_TIME": 7, "LOT_SIZE_ORDER": 21, "SETUP_COST_RAW": 300, "HOLD_COST": 10, "PURCHASE_COST_RAW": 3},
     2: {"ID": 2, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 2.1", "MANU_ORDER_CYCLE": 7,                        "SUP_LEAD_TIME": 7, "LOT_SIZE_ORDER": 21, "SETUP_COST_RAW": 300, "HOLD_COST": 10, "PURCHASE_COST_RAW": 3},
     3: {"ID": 3, "TYPE": "Raw Material", "NAME": "RAW MATERIAL 2.2", "MANU_ORDER_CYCLE": 7,                        "SUP_LEAD_TIME": 7, "LOT_SIZE_ORDER": 21, "SETUP_COST_RAW": 300, "HOLD_COST": 10, "PURCHASE_COST_RAW": 3},
     4: {"ID": 4, "TYPE": "WIP",          "NAME": "WIP 1",                                                                                                                           "HOLD_COST": 10}}

# Processes:
# ID: Index of the element in the dictionary
# PRODUCTION_RATE [units/day]
# INPUT_LIST: List of input materials or WIPs
# OUTPUT: Output WIP or Product
P = {0: {"ID": 0, "PRODUCTION_RATE": 3, "INPUT_LIST": [I[1]], "OUTPUT": I[4]},
     1: {"ID": 1, "PRODUCTION_RATE": 2, "INPUT_LIST": [I[2], I[3], I[4]], "OUTPUT": I[0]}}

# Demand quantity for the final product [units]
'''
MIN_ORDER_SIZE = 80
MAX_ORDER_SIZE = 120
'''
'''
HOLDING_COST = 1  # 단위당 보유 비용
BACKORDER_COST = 10  # 단위당 백오더 비용
'''
# Simulation
SIM_TIME = 20  # [days]
INITIAL_INVENTORY = 30  # [units]

'''
# Data tracking for inventory level
inventory_level_over_time = {}
inventory_cost_over_time = {}
for i in range(len(I)):
    inventory_level_over_time[i] = [0]
    inventory_cost_over_time[i] = [0]
'''


class Inventory:
    def __init__(self, env, item_id):
        self.item_id = item_id  # 0: product; others: WIP or raw material
        self.store = simpy.Store(env)  # capacity=infinity
        for _ in range(INITIAL_INVENTORY):  # initial inventory
            self.store.put(1)
        # self.cost = 0
        self.level_over_time = [0]  # Data tracking for inventory level
        self.cost_over_time = [0]  # Data tracking for inventory cost


class Provider:
    def __init__(self, env, name, item_id):
        self.env = env
        self.name = name
        self.item_id = item_id

    def deliver(self, order_size, inventory):
        # Lead time
        yield self.env.timeout(I[self.item_id]["SUP_LEAD_TIME"] * 24)
        for _ in range(order_size):
            yield inventory.store.put(1)
        print(
            f"{self.env.now}: {self.name} has delivered {order_size} units of {I[self.item_id]['NAME']}")


class Procurement:
    def __init__(self, env):
        self.env = env

    def order(self, provider, inventory):
        while True:
            # Place an order to a provider
            yield self.env.timeout(I[provider.item_id]["MANU_ORDER_CYCLE"] * 24)
            # THIS WILL BE AN ACTION OF THE AGENT
            order_size = I[provider.item_id]["LOT_SIZE_ORDER"]
            print(
                f"{self.env.now}: Placed an order for {order_size} units of {I[provider.item_id]['NAME']}")
            self.env.process(provider.deliver(order_size, inventory))
            '''
            # Lead time of the provider
            yield self.env.timeout(I[provider.item_id]["SUP_LEAD_TIME"])
            for _ in range(order_size):
                yield inventory.store.put(1)
            print(
                f"{self.env.now}: {provider.name} has delivered {order_size} units of {I[provider.item_id]['NAME']}")
            '''


class Production:
    def __init__(self, env, name, production_rate, output, input_inventories, output_inventory):
        self.env = env
        self.name = name
        self.production_rate = production_rate
        # self.input_list = input_list
        self.output = output
        self.input_inventories = input_inventories
        self.output_inventory = output_inventory

    def process(self):
        while True:
            # Check the current state if input materials or WIPs are available
            for inven in self.input_inventories:
                if len(inven.store.items) < 1:
                    print(
                        f"{self.env.now}: Stop {self.name} due to a shortage of input materials or WIPs")
                    # Check again after 24 hours (1 day)
                    yield self.env.timeout(24)
                    continue

            # Consuming input materials or WIPs and producing output WIP or Product
            yield self.env.timeout(24 / self.production_rate)
            for inven in self.input_inventories:
                yield inven.store.get()
            yield self.output_inventory.store.put(1)
            print(
                f"{self.env.now}: A unit of {self.output['NAME']} has been produced")


class Sales:
    def __init__(self, env):
        self.env = env

    def delivery(self, item_id, order_size, product_inventory):
        # Lead time
        yield self.env.timeout(I[item_id]["MANU_LEAD_TIME"] * 24)
        # SHORTAGE: Check if products are available
        if len(product_inventory.store.items) < 1:
            print(
                f"{self.env.now}: Unable to deliver to the customer due to product shortage")
            # Check again after 24 hours (1 day)
            yield self.env.timeout(24)
        # Delivering products to the customer
        else:
            for _ in range(order_size):
                yield product_inventory.store.get()
            print(
                f"{self.env.now}: {order_size} units of the product have been delivered to the customer")


class Customer:
    def __init__(self, env, name):
        self.env = env
        self.name = name
        self.item_id = 0
        self.order_history = []

    def order(self, sales, product_inventory):
        while True:
            yield self.env.timeout(I[self.item_id]["CUST_ORDER_CYCLE"] * 24)
            # THIS WILL BE A RANDOM VARIABLE
            order_size = I[self.item_id]["DEMAND_QUANTITY"]
            self.order_history.append(order_size)
            print(
                f"{self.env.now}: The customer has placed an order for {order_size} units of {I[self.item_id]['NAME']}")
            self.env.process(sales.delivery(
                self.item_id, order_size, product_inventory))
    ''' 
    def delivery(self, product_inventory):
        while True:
            # SHORTAGE: Check products are available
            if len(product_inventory.store.items) < 1:
                print(
                    f"{self.env.now}: Unable to deliver to the customer due to product shortage")
                # Check again after 24 hours (1 day)
                yield self.env.timeout(24)
            # Delivering products to the customer
            else:
                demand = I[product_inventory.item_id]["DEMAND_QUANTITY"]
                for _ in range(demand):
                    yield product_inventory.store.get()
                print(
                    f"{self.env.now}: {demand} units of the product have been delivered to the customer")
    '''


'''
def outgoing_delivery():
    while True: 
        yield process.store.get()
'''
'''
def calculate_inventory_cost():
    for item, quantity in inventory.items():
        if quantity > 0:
            inventory_cost[item] += HOLDING_COST * quantity
        else:
            inventory_cost[item] += BACKORDER_COST * abs(quantity)
        inventory_cost_over_time[item].append(inventory_cost[item])
'''
'''
def update_data_trackers():

'''


def main():
    env = simpy.Environment()

    # Print the list of items and processes
    print("\nItem list")
    for i in I.keys():
        print(f"ITEM {i}: {I[i]['NAME']}")
    print("\nProcess list")
    for i in P.keys():
        print(f"Output of PROCESS {i}: {P[i]['OUTPUT']['NAME']}")

    # Create an inventory for each item
    inventoryList = []
    for i in I.keys():
        inventoryList.append(Inventory(env, i))
    '''    
    for inven in inventoryList:
        print(
            f"{env.now}: [ITEM {inven.item_id}] [{inven.store.items.__len__()}]  {inven.store.items}")
    '''
    print("Number of Inventories: ", len(inventoryList))

    # Create stakeholders (Customer, Providers)
    customer = Customer(env, "CUSTOMER")
    providerList = []
    for i in I.keys():
        # Create a provider if the type of the item is Raw Material
        if I[i]["TYPE"] == 'Raw Material':
            providerList.append(Provider(env, "PROVIDER_"+str(i), i))
    print("Number of Providers: ", len(providerList))

    # Create managers for manufacturing process, procurement process, and delivery process
    procurement = Procurement(env)
    sales = Sales(env)
    productionList = []
    for i in P.keys():
        output_inventory = inventoryList[P[i]["OUTPUT"]["ID"]]
        input_inventories = []
        for j in P[i]["INPUT_LIST"]:
            input_inventories.append(inventoryList[j["ID"]])
        productionList.append(Production(env, "PROCESS_"+str(i),
                                         P[i]["PRODUCTION_RATE"], P[i]["OUTPUT"], input_inventories, output_inventory))

    # Event processes for SimPy simulation
    env.process(customer.order(sales, inventoryList[I[0]["ID"]]))
    for production in productionList:
        env.process(production.process())
    for provider in providerList:
        env.process(procurement.order(
            provider, inventoryList[provider.item_id]))
    '''    
    for provider in providerList:
        env.process(provider.supply(inventoryList[provider.item_id]))
    env.process(customer.delivery(inventoryList[0]))
    '''
    # Run the simulation
    for i in range(SIM_TIME*24):
        # Print the inventory level every 24 hours (1 day)
        if i % 24 == 0:
            print(f"\nDAY {int(i/24)+1}")
            for inven in inventoryList:
                inven.level_over_time.append(inven.store.items.__len__())
                print(
                    f"{env.now}: [{I[inven.item_id]['NAME']}]  {inven.store.items.__len__()}")
        env.run(until=i+1)
        # calculate_inventory_cost()

    # 재고 상태 와 비용 출력
    for i in I.keys():
        inventory_visualization = visualization.visualization(
            inventoryList[i], I[i]['NAME'])
        inventory_visualization.inventory_level_graph()
        inventory_visualization.inventory_cost_graph()
        # calculate_inventory_cost()

    '''
    # 재고 상태와 비용 출력
    print("Final inventory status:")
    for item, quantity in inventories.items():
        print(f"{item}: {quantity}")
    
    print("Final inventory cost:")
    for item, cost in inventory_cost.items():
        print(f"{item}: {cost}")
    '''
    '''
    # 시간 경과에 따른 재고 수준 시각화
    plt.figure(figsize=(12, 8))
    for item in inventory.keys():
        plt.plot(inventory_time[item], inventory_level_over_time[item], label=item)
    plt.xlabel('Time')
    plt.ylabel('Inventory level')
    plt.legend()
    plt.show()
    # 시간 경과에 따른 재고 비용 시각화
    plt.figure(figsize=(12, 8))
    for item in inventory.keys():
        plt.plot(inventory_time[item], inventory_cost_over_time[item], label=item)
    plt.xlabel('Time')
    plt.ylabel('Inventory cost')
    plt.legend()
    plt.show()
    '''


if __name__ == "__main__":
    main()
