import gurobipy as gp
from gurobipy import GRB
import xmltodict


"""
This methods reads an xml file and returns all the information that is needed by our program to solve the IP
@:parameter path (String), represents the path to the xml file that will be read
@:returns n_teams (Integer), represents the amount of teams participating
@:returns d_matrix (2x2 Integer), represents the distance between all the distinct team-pairs
@:returns slots (Integer), represents the number of game days that can happen
@:returns upperbound (Integer), represents the maximum of games a player is allowed to play home or away consecutively
@:returns lowerbound (Integer), represents the minimum of games a player has to play home or away consecutively
"""
def load_data(path):
    with open(path, "r") as f:
        data = xmltodict.parse(f.read())
    n_teams = (len(data['Instance']['Resources']['Teams']['team']))
    slots = (len(data['Instance']['Resources']['Slots']['slot']))
    d_matrix = [[0 for _ in range(n_teams)] for _ in range(n_teams)]
    distance_table_data = (data['Instance']['Data']['Distances']['distance'])
    for distance in distance_table_data:
        d_matrix[int(distance['@team1'])][int(distance['@team2'])] = int(distance['@dist'])
    upperbound = int(data['Instance']['Constraints']['CapacityConstraints']['CA3'][0]['@max'])
    lowerbound = int(data['Instance']['Constraints']['CapacityConstraints']['CA3'][0]['@min'])
    return n_teams, d_matrix, slots, upperbound, lowerbound


"""
This methods builds the Integer program inside a gurobi model
@:param distances (2x2 Integer), represents the distance between all the distinct team-pairs
@:param L (Integer), represents the maximum of games a player is allowed to play home or away consecutively
@:param U (Integer), represents the minimum of games a player has to play home or away consecutively
@:param Days (Integer), represents the number of rounds played by each team
@:param Teams (Integer), represents the number of teams playing
@:param Days_X (Range(Integers)), represents a range of integers of all the rounds that will be played
@:param Days_Y (Range(Integers)), represents a range of integers for the travel variable
@:param Days_U (Range(Integers)), represents a range of integers for the upperbound constraint
@:param Days_L (Range(Integers)), represents a range of integers for the lowerbound constraint
@:param max_time (Integer), represents the maximum time the gurobi method is allowed to run, defined in seconds

@:returns model (gp.Model), this represents the Integer Program built into a gurobi application 
"""
def build_base_model(distances, L, U, DAYS, Teams, Days_X, Days_Y, Days_U, Days_L, max_time):
    """
    Constructs the base TTP subproblem model including variables, objective,
    and the core TTP constraints (1) through (9).
    """
    model = gp.Model("TTP_Subproblem")
    model.setParam('OutputFlag', 0)  # makes the output cleaner
    model.setParam('TimeLimit', max_time)
    # --- 3. Decision Variables ---
    # x[i, j, k]: 1 if team i hosts team j on day k (Binary)
    x = model.addVars(Teams, Teams, Days_X, vtype=GRB.BINARY, name="x")

    # z[i, j, k]: Auxiliary variable (Continuous)
    z = model.addVars(Teams, Teams, Days_X, vtype=GRB.BINARY, name="z")

    # y[t, i, j, k]: Travel variable for team t from location i to j on day k (Continuous, LB=0)
    # k is the index of the FIRST day in the transition (k to k+1)
    y = model.addVars(Teams, Teams, Teams, Days_Y, vtype=GRB.BINARY, lb=0, name="y")

    # --- 4. Objective Function ---
    # min sum(i,j) d_ij * x_ij1 + sum(t,i,j) sum(k=1 to 2n-3) d_ij * y_tijk + sum(i,j) d_ij * x_ij,2n-2
    obj = gp.quicksum(distances[i][j] * x[i, j, 1] for i in Teams for j in Teams) + \
          gp.quicksum(distances[i][j] * y[t, i, j, k]
                      for t in Teams for i in Teams for j in Teams for k in Days_Y) + \
          gp.quicksum(distances[i][j] * x[i, j, DAYS] for i in Teams for j in Teams)

    model.setObjective(obj, GRB.MINIMIZE)

    # --- 5. Constraints (TTP Core) ---

    # (1) No self-play
    model.addConstrs((x[i, i, k] == 0 for i in Teams for k in Days_X), name="No_Self_Play")

    # (2) Play one game per day
    model.addConstrs((gp.quicksum(x[i, j, k] + x[j, i, k] for j in Teams) == 1
                      for i in Teams for k in Days_X), name="Play_One_Game_Per_Day")

    # (3) Play each team twice
    model.addConstrs((gp.quicksum(x[i, j, k] for k in Days_X) == 1
                      for i in Teams for j in Teams if i != j), name="Play_Each_Team_Twice")

    # (4) Max home stay
    model.addConstrs((gp.quicksum(x[i, j, k + u] for j in Teams for u in range(U + 1)) <= U
                      for i in Teams for k in Days_U), name="Max_Home_Stay")

    # (5) Max road trip
    model.addConstrs((gp.quicksum(x[i, j, k + u] for i in Teams for u in range(U + 1)) <= U
                      for j in Teams for k in Days_U), name="Max_Road_Trip")

    # (6) No game repeats 2 times back to back
    model.addConstrs((x[i, j, k] + x[j, i, k] + x[i, j, k + 1] + x[j, i, k + 1] <= 1
                      for i in Teams for j in Teams for k in Days_Y if i != j),
                     name="No_Back_to_Back_Game")

    # --- 6. Driving behavior of the teams constraints ---

    # (7)
    model.addConstrs((z[i, i, k] == gp.quicksum(x[i, j, k] for j in Teams)
                      for i in Teams for k in Days_X), name="Def_z_home")

    # SWITHCED I AN J IN THE X COLUMN
    # (8))
    model.addConstrs((z[i, j, k] == x[j, i, k] for i in Teams for j in Teams for k in Days_X if i != j), name="Def_z_game")

    # (9)
    model.addConstrs((y[t, i, j, k] >= z[t, i, k] + z[t, j, k + 1] - 1
                      for t in Teams for i in Teams for j in Teams for k in Days_Y),
                     name="Def_y_travel_link")


    # (10)  Min home stay (Lower Bound L): Forces team i to be home at least L times over L+1 days.
    model.addConstrs((gp.quicksum(x[i, j, k + l] for j in Teams for l in range(L + 1)) >= L
                     for i in Teams for k in Days_L), name="Min_Home_Stay_L")

    # (11) Min road trip (Lower Bound L): Forces team j to be away at least L times over L+1 days.
    model.addConstrs((gp.quicksum(x[i, j, k + l] for i in Teams for l in range(L + 1)) >= L
                     for j in Teams for k in Days_L), name="Min_Road_Trip_L")

    return model


"""
This method will print the computed schedule for every round
@:param model (gp.Model), represents a model with a solution
"""
def show_schedule(model:gp.Model):
    games = []
    for v in model.getVars():
        if v.VarName.startswith("x"):
            if v.X == 1:
                games.append(list(map(int, v.VarName[2:-1].split(','))))
    sorted_games = sorted(games, key=lambda x: x[-1])
    current_round = -1
    for game in sorted_games:
        if game[-1] != current_round:
            print(f"Round {game[-1]}")
            current_round = game[-1]
        print(f"{game[0]} vs {game[1]}")


"""
This is the main method to run the Integer program
@:param complexity_level (Integer), this represent which NL problem will be solved, the complexity needs to be 
                                    a number in the list: 4 - 6 - 8 - 10 - 12 - 14 - 16
@:param maximum_runtime (Integer), this represents the maximum time the model is allowed to try and solve the 
                                    Integer program, if it doesn't reach optimality before this point it will print out
                                    the current solution. It is defined in seconds
"""
def run_exact_program(complexity_level, maximum_runtime, gurobi_solve_method):
    file_name = f"NL{complexity_level}"
    path_file = f"ttp_instances/NL{complexity_level}.xml"

    n, d, days, u, l = load_data(path_file)

    teams = range(0, n)
    days_x = range(1, days + 1)      # Days k for x and z
    days_y = range(1, days)          # Days k for y (1 to 5, as it involves k+1)
    days_u = range(1, days - u + 1)  # Days k for constraints (4) and (5) (1 to 2n-2-L)
    days_l = range(1, days - l + 1)  # Days k for constraints (10) and (11) (1 to 2n-2-L)

    model = build_base_model(d, l, u, days, teams, days_x, days_y, days_u, days_l, maximum_runtime)

    optimal_solutions = {'NL4':8276,
                         'NL6':23916,
                         'NL8':39721,
                         'NL10':59436}

    model.setParam(gp.GRB.Param.BestObjStop, optimal_solutions[file_name])
    model.setParam(GRB.Param.Method, gurobi_solve_method)
    model.optimize()

    show_schedule(model)
    if model.objVal == optimal_solutions[file_name]:
        print(f"The optimal solution of {optimal_solutions[file_name]} has been found in {model.runtime:.4f} seconds")
    else:
        print(f"The optimal solution of {optimal_solutions[file_name]} has not been found.")
        print(f"The model found {model.objVal} in {model.runtime:.4f} seconds")


"""
Change the number of complexity to run a different version of the problem
It needs to be in this list: 4 - 6 - 8 - 10 - 12 - 14 - 16
This file does assume the xml files are located in the same directory in a 'ttp_instances' folder

Change the maximum runtime to put restriction in what time the model has to provide a solution
It is defined in seconds, it needs to be greater than 0

Change the GUROBI solver method to indicate which solver you would like that gurobi uses:
-1) gurobi chooses itself 
 0) barrier, dual, primal simplex
 1) barrier and dual simplex 
 2) barrier and primal simplex
 3) dual and primal simplex
"""
COMPLEXITY_LEVEL = 8
MAXIMUM_RUNTIME = 600
GUROBI_SOLVE_METHOD = -1
if __name__ == "__main__":
    run_exact_program(COMPLEXITY_LEVEL, MAXIMUM_RUNTIME, GUROBI_SOLVE_METHOD)




