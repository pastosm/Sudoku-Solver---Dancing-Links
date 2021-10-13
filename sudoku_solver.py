from gc import collect
from typing import DefaultDict
import numpy as np
import copy
import random
import math
import time

from numpy.core.defchararray import index


class PartialSudokuState:

    def __init__(self, n=9):
        self.n = n
        self.partial_state = np.array([
                                [0,2,0, 0,0,6, 9,0,0],
                                [0,0,0, 0,5,0, 0,2,0],
                                [6,0,0, 3,0,0, 0,0,0],

                                [9,4,0, 0,0,7, 0,0,0],
                                [0,0,0, 4,0,0, 7,0,0],
                                [0,3,0, 2,0,0, 0,8,0],

                                [0,0,9, 0,4,0, 0,0,0],
                                [3,0,0, 9,0,2, 0,1,7],
                                [0,0,8, 0,0,0, 0,0,2]
            ])

        self.cell_number = self.n * self.n
        self.final_values = [-1] * self.cell_number
        self.final_values = np.array(self.final_values)

        # Possible values for each (col,row) pair
        self.possible_values = [[i for i in range(1, 10)] for _ in range(0, self.cell_number)]
        
        self.invalid_initial_state = np.zeros(shape=(9,9))
        self.invalid_initial_state.fill(-1)


    # Checking if the board is in the goal state
    def is_goal(self):
        return all(value != -1 for value in self.final_values)

    
    def apply_configuration(self, partial_state):
        for row in range(0, self.n):
            for col in range(0, self.n):
                if partial_state[row][col] != 0:
                    index = col + (row * self.n)
                    if self.set_board_position(index, partial_state[row][col]) == None:
                        return None
                    else:
                        self = self.set_board_position(index, partial_state[row][col])
        
        return self

    # Return true if the partial state cannot be solved
    def is_invalid(self):
        return any(len(values) == 0 for values in self.possible_values)

    
    # Find the minimum remaining values index
    def mrv_heuristic(self):
        total_vals = np.zeros(self.cell_number)
        for index, _ in enumerate(self.possible_values):
            if self.final_values[index] != -1:
                total_vals[index] = 10
            else:
                total_vals[index] = len(self.possible_values[index])

        return np.argmin(total_vals)

    
    def lcv_heuristic(self, index, values):

        # Convert index to col and rows
        col = index % self.n
        row = index // self.n

        # Least constrained values in order
        lcvs_dict = {}

        for value in values:
            constrains = 0

            # Column and row-wise checking
            for pos in range(0, self.n):
                # row-wise
                index_on_board = col + (pos * self.n)
                if value in self.possible_values[index_on_board] and index_on_board != index:
                    constrains += 1
                # column-wise
                index_on_board = pos + (row * self.n)
                if value in self.possible_values[index_on_board] and index_on_board != index:
                    constrains += 1

            # Subgrid checking
            sub_top_row = row - (row % 3)
            sub_top_col = col - (col % 3)
            for row in range(sub_top_row, sub_top_row+3):
                for col in range(sub_top_col, sub_top_col+3):
                    index_on_board = col + (row * self.n)
                    if value in self.possible_values[index_on_board] and index_on_board != index: 
                        constrains += 1
            
            lcvs_dict[value] = constrains
        
        return lcvs_dict


    def order_lcv(self, lcvs_dict):
        
        lcvs_dict = copy.copy(lcvs_dict)
        values_out = []

        while len(lcvs_dict) != 0:
            value = min(lcvs_dict, key=lcvs_dict.get)
            values_out.append(value)
            del lcvs_dict[value]
        
        return values_out




    # Get the possible values for a specific position on our sudoku board
    def get_possible_values(self, index):
        return self.possible_values[index].copy()

    def get_final_values(self):
        return self.final_values

    
    def get_singleton_positions(self):
        return [index for index, values in enumerate(self.possible_values)
                if len(values) == 1 and self.final_values[index] == -1]


    def __str__(self):
        return str(self.final_values)



    def set_board_position(self, index, number):
        """Returns a new state with this (col, row) position, and the change propagated to other domains"""
        col = index % self.n
        row = index // self.n
        if number not in self.possible_values[index]:
#             raise ValueError(f"{number} is not a valid choice for {index}")
            return None

        # create a deep copy: the method returns a new state, does not modify the existing one   
        state = copy.deepcopy(self)

        # update this cell
        state.possible_values[index] = [number]
        state.final_values[index] = number


        # Update all positions (row-wise and col-wise) which contain our index
        for pos in range(0, self.n):
            # Remove from all positions in this column, the possible value of our index
            index_on_board = col + (pos * self.n)
            if number in state.possible_values[index_on_board] and index_on_board != index:
                state.possible_values[index_on_board].remove(number)    
            # Remove from all positions in this column, the possible value of our index
            index_on_board = pos + (row * self.n)
            if number in state.possible_values[index_on_board] and index_on_board != index:
                state.possible_values[index_on_board].remove(number) 

        # Update all positions in this sub-array that contain this number
        sub_top_row = row - (row % 3)
        sub_top_col = col - (col % 3)
        for row in range(sub_top_row, sub_top_row+3):
            for col in range(sub_top_col, sub_top_col+3):
                index_on_board = col + (row * self.n)
                if number in state.possible_values[index_on_board] and index_on_board != index: 
                    state.possible_values[index_on_board].remove(number)
        


#         singleton_positions= state.get_singleton_positions()
#         while len(singleton_positions) > 0:
#             index = singleton_positions[0]
#             state = state.set_board_position(index, state.possible_values[index][0])
#             singleton_positions = state.get_singleton_positions()




        return state

    def get_final_state(self):
        if self.is_goal():
            return self.final_values
        else:
            return -1




# Pick next position based on the mrv heuristic
def pick_next_position(partial_state):
    return partial_state.mrv_heuristic()


# Order our values based on the lcv heuristic
def order_values(partial_state, index):

    values = partial_state.get_possible_values(index)
    lcv = partial_state.lcv_heuristic(index, values)
    lcv_values = partial_state.order_lcv(lcv)
    
    return lcv_values


def sudoku_solver(sudoku):
    
    partial_state = PartialSudokuState()
    partial_state_copy = copy.deepcopy(partial_state)
    if partial_state_copy.apply_configuration(sudoku) == None:
        return partial_state_copy.invalid_initial_state
    
    result = sudoku_search(partial_state.apply_configuration(sudoku))
    
    if result == None:
        return no_solution
    else:
        return np.reshape(result.final_values,(9,9))
    



def sudoku_search(partial_state):
    next_position = pick_next_position(partial_state)
    values = order_values(partial_state, next_position)

    
    for value in values:
        new_state = partial_state.set_board_position(next_position, value)
        if new_state.is_goal():
            return new_state
        if not new_state.is_invalid():
            deep_state = sudoku_search(new_state)
            if deep_state is not None and deep_state.is_goal():
                return deep_state


    return None


no_solution = np.zeros(shape=(9,9))
no_solution.fill(-1)
test = PartialSudokuState()
test2 = test.apply_configuration()

total_time = 0

start_time = time.process_time()
print(sudoku_solver(test2))
end_time = time.process_time()
total_time += end_time-start_time
    
print(f"Took on average {end_time-start_time} seconds to solve")


#  Took on average 5.559406747359999 seconds to solve (100 runs) 


