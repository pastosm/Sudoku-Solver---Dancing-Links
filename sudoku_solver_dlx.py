import numpy as np
import time


# Representing the data objects as described in Knuth's paper
class Data:

    def __init__(self):
        self.left = self
        self.right = self
        self.up = self
        self.down = self
        self.column = self
        self.row_id = 0


# Representing the column objects as described in Knuth's paper
class Column(Data):

    def __init__(self, ID):
        super().__init__()
        self.size = 0
        self.ID = ID # Symbolic reference 



class DLX:

    def __init__(self, sudoku):
        self.header = Column("Root")
        self.cover_matrix = self.convert_to_exact_cover(sudoku)
        self.num_rows = len(self.cover_matrix)
        self.num_cols = len(self.cover_matrix[0])
        self.matrix = np.zeros((self.num_rows,self.num_cols), dtype="object_")
        self.solutions = []
        self.solution = []



    # Function which converts a 9x9 sudoku board into an exact cover matrix (binary)
    def convert_to_exact_cover(self, sudoku):
        
        # Each possible subset (row) maps to 4 constraints. A solution is found when
        # a collection of disjoint rows (i.e each column has exactly 1 object in it) cover all 324 constraints
        # (81 rows * 4 constraints each = 324)


        # Constaints (9 * 9 * 4) = 324 
        # Rows (9 * 9 * 9) = 729 (Representing all posibilities - for each cell (row,col) and each value (1-9))

        # Cell constrains 0  - 81 (excl.) - each cell must have one of 9 values
        # Row constraints 81 - 162 (excl.) - each row must have each number
        # Column constrains 162 - 243 (excl.) - every column must have each number
        # Box constraints 243 - 324 (excl.) - every box must have each number

        COLUMN_SZ = 324 # The elements in our "universe"
        SUBSETS_SZ = 729 # The subsets of our "universe" (all possible configurations in a sudoku 9x9)


        exact_cover = [[0 for _ in range(COLUMN_SZ)] for _ in range(SUBSETS_SZ + 1)] # One extra row for the column headers

        for r in range(len(sudoku)):
            for c in range(len(sudoku[0])):
                if sudoku[r][c] == 0: # If the cell in "empty" on the sudoku board then all values are possible (1-9)
                    for v in range(9):
                        constraints = self.calculate_mappings(r,c,v)
                        exact_cover = self.apply_to_matrix(exact_cover, constraints)
                        
                else:
                    v = sudoku[r][c] - 1 # (0-8) - otherwise we know that only this value is possible since it's predetermined
                    constraints = self.calculate_mappings(r,c,v)
                    exact_cover = self.apply_to_matrix(exact_cover, constraints)

        return exact_cover


    def calculate_mappings(self, r, c, v):
        # Constraint starting indexes
        CELL_C = 0
        ROW_C = 81
        COL_C = 162
        BOX_C = 243

        # Map row, column, value to the 729 range of subsets (adding one to adjust for the column headers row)
        row_pos = 9 * (r * 9) + (c * 9) + v + 1

        # Calculate mapping of constraints
        cell_constraint = CELL_C + (r * 9) + c
        row_constraint = ROW_C + (r * 9) + v
        col_constraint = COL_C + (c * 9) + v
        box_constraint = BOX_C + (((r - (r % 3)) + c // 3) * 9) + v 

        return (row_pos, cell_constraint, row_constraint, col_constraint, box_constraint)

    def apply_to_matrix(self, matrix, constraints):

        (row_pos, cell_constraint, row_constraint, col_constraint, box_constraint) = constraints
        matrix[row_pos][cell_constraint] = True
        matrix[row_pos][row_constraint] = True
        matrix[row_pos][col_constraint] = True
        matrix[row_pos][box_constraint] = True

        return matrix



    def create_linked_mat(self):

        # Link Columns - First row of matrix is only column objects
        for c in range(self.num_cols):
            col_node = Column(c)
            col_node.left = self.header.left
            col_node.right = self.header
            self.header.left.right = col_node
            self.header.left = col_node
            self.matrix[0][c] = col_node

        
        # Link rows
        for r in range(1,self.num_rows): # Start from 1 (columns are already linked)
            ref_node = None # Used as a reference point when linking rows
            for c in range(self.num_cols):

                if self.cover_matrix[r][c]: # If true

                    column_node = self.matrix[0][c] # Get the column header of this data object

                    data_node = Data()
                    data_node.column = column_node # add reference to column header
                    data_node.up = column_node.up 
                    data_node.down = column_node
                    data_node.row_id = r

                    column_node.up.down = data_node
                    column_node.up = data_node
                    column_node.size += 1

                    if ref_node == None: 
                        ref_node = data_node # The initial reference is the first data object of the row

                    data_node.left = ref_node.left
                    data_node.right = ref_node
                    ref_node.left.right = data_node
                    ref_node.left = data_node

                    self.matrix[r][c] = data_node


    # The cover function as described in Knuth's paper
    def cover(self, target):

        # Get the column of the target node
        target_col = target.column

        # Unlink column header 
        target_col.left.right = target_col.right
        target_col.right.left = target_col.left


        # Unlink rows
        row = target_col.down
        # We unlink all rows of the column
        while (row != target_col):

            row_right = row.right

            # Unlink this row
            while (row_right != row):

                row_right.up.down = row_right.down
                row_right.down.up = row_right.up

                # Decrement the number of nodes this column has
                self.matrix[0][row_right.column.ID].size -= 1
                row_right = row_right.right

            row = row.down  

    # The uncover function as described in Knuth's paper
    def uncover(self, target):
        target_col = target.column


        # Link rows
        row = target_col.up
        
        # We link all rows of this column
        while (row != target_col):

            row_left = row.left

            # Link this row
            while (row_left != row):

                row_left.up.down = row_left
                row_left.down.up = row_left

                # Increment the number of nodes this column has
                self.matrix[0][row_left.column.ID].size += 1
                row_left = row_left.left

            row = row.up  


        target_col.left.right = target_col
        target_col.right.left = target_col


    # Heuristic which choose the column with the least number of nodes (helps reduce the branching factor of our search)
    def get_col_with_least_nodes(self):

        head = self.header

        least_nodes = head.right
        head = head.right

        while (head != self.header):

            if (head.size < least_nodes.size):
                least_nodes = head
            
            head = head.right
        
        return least_nodes

    def convert_to_sudoku(self, solution):
        self.solution = np.zeros((9,9), dtype="int64")

        for row in solution:
            row_id = row.row_id # We have to extract the row, col, value information from the row in the solution
            row_id -= 1 # Take into account the column row
            c = row_id // 81
            r = (row_id % 81) // 9
            n = (row_id % 81) % 9
            self.solution[r][c] = n + 1 # Convert from zero base
        
        print(self.solution)


    # The search function as described in Knuth's paper
    def search(self, k):
        
        # We managed to cover all columns, and therefore found solution(s)
        if (self.header.right == self.header):
            return self.convert_to_sudoku(self.solutions) # Our list of rows is converted to Sudoku array
        
        column = self.get_col_with_least_nodes() # Heuristic that helps us reduce the branching factor of the search

        self.cover(column) # Cover that column

        row = column.down
        while (row != column):
            self.solutions.append(row)

            row_right = row.right

            while (row_right != row):
                self.cover(row_right)
                row_right = row_right.right
            
            self.search(k+1)


            self.solutions.pop()

            column = row.column
            left_node = row.left
            while (left_node != row):
                self.uncover(left_node)
                left_node = left_node.left
            
            row = row.down
        
        self.uncover(column)
        return

TEST_INPUT = np.array([
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


def sudoku_solver(sudoku):
    dlx = DLX(sudoku)
    dlx.create_linked_mat()
    dlx.search(0)
        



st = time.process_time()
sudoku_solver(TEST_INPUT)
et = time.process_time()
print(f"search finished in {et-st} secs")
    

