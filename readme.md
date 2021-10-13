# Sudoku Solver using Dancing Links Algorithm by Donald E. Knuth

# Sudoku solving
Sudoku solving is an NP-complete problem, therefore, finding a way to solve an 'n by n' Sudoku board in polynomial time is very unlikely (Kumar *et al*., 2014). There are various techniques that researchers have developed to solve this problem . 

One such way is to model a sudoku puzzle as a Constraint Satisfaction Problem (CSF) where each cell of the sudoku board must satisfy some constraints. The typical constraints for Sudoku puzzles are:

 - Each cell must contain a value ranging from 1 to 9
 - Each row of the board must not contain duplicates
 - Each column of the board must not contain duplicates
 - Each sub-grid of the board must not contain duplicates

Basic backtracking is one way to solve this problem. This algorithm works as follows. The program places a value in an empty cell. If the value does not conflict with existing values according to the rules of sudoku, the program continues picking the next empty cell (goes one step deeper), and the process is continued until the board is filled or it is unsuccessful when placing an assignment. Once such unsuccessful (conflicting) assignment is found, the algorithm erases the value and tries a new value, essentially backtracking to the previous state. However, the enormous state space makes this algorithm not very practical for large puzzles. For instance, a 9x9 sudoku board has a state space of 9<sup>81-n</sup> (Kumar *et al*., 2014).

This basic approach may be improved using other techniques such as constraint propagation, forward checking and heuristics such as choosing the most contrained value first.

Applying these various techniques can solve sudokus reasonably fast. In the next section, we discuss an implementation of the sudoku solver using techniques of backtracking, forward checking and constraint propagation. Later on, we discuss a more efficient implementation of solving sudoku puzzles using the popular algorithm by Donald E. Knuth  known as "Dancing Links" or "DLX" (for Dancing Links Algorithm "X") (2000). 

# Backtracking implementation with optimization techniques


## Main idea of the backtracking algorithm

The main idea of the backtracking implementation is demonstrated below. The function works by taking a partial state as an argument (we pass in a state of the sudoku matrix) and performs depth-first-search. The next cell to place values to is picked by the Minimum Remaining Values (MRV) heuristic and the potential candidates for the cells are ordered by the Least Constrained Values (LRV) heuristic. These are going to be explained in the subsequent sections.  A new state is created (i.e. a snapshot of the current board) by placing a possible value in the board and propagating the rest of the board. Then, if the placed value turns out to be a solution of the board, we return the state. Otherwise, if we have not reached an unsolvable state, we recurse our function with the new state (i.e go a depth deeper) which performs identical checks on the deep state. Finally, if all the states turn out to be invalid then we have reached a dead end; the board cannot be solved. 

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

*Note*: the function sudoku_search() was adapted from the "Eight Queens Revisited" [lab](https://moodle.bath.ac.uk/mod/page/view.php?id=974677) solution as demonstrated in the Artificial Intelligence module.

## Least Constraining Values (LCV)

The LCV heuristic works by counting the number of times a value appears in its row, column and sub-grid which essentially illustrates how many constraints this value creates. For example, assume we are trying to place the value 2 in the first cell (0,0). We then check how many cells in this row, column and sub-grid contain 2 as a possible value. The higher the number our value appears in these cells, the higher the number of constraints  we are creating by placing this particular value. By ordering our possible values by least constrained to most constrained, this allows our to reach our solution faster by reducing the branching factor of our search since it will avoid partial solutions that cannot be completed (Sadeh and Fox, 1996).


    def lcv_heuristic(self, index, values):

	... (some other code)

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


## Minimum Remaining Values (MRV)

The MRV heuristic is used to prune out impossible configurations early. It works by picking the next position to assign values to, with the least number of possible values. This allows us to "fail" early and save computation resources. In essence, the program does not have to come across an invalid configuration by chance. It is forced to pick positions that are more likely to to lead to an unsolvable state early. 


    def mrv_heuristic(self):
	    total_vals = np.zeros(self.cell_number)
	    for index, _ in enumerate(self.possible_values):
	        if self.final_values[index] != -1:
	            total_vals[index] = 10 # arbitrary
	        else:
	            total_vals[index] = len(self.possible_values[index])

	    return np.argmin(total_vals)

## Forward Checking (FC)

Forward checking allows us to identify invalid configurations early without having to try them by going into a deeper state.  This is done fairly easily by checking if the state has any positions that have no possible values to be placed. It is a consequence of constraint propagation which may force some cells to have no possible values. Forward checking is used in the main search function discussed above.

    def is_invalid(self):
	    return any(len(values) == 0 for values in self.possible_values)

## Constraint Propagation

Constraint Propagation is the process of reducing the domain of a chosen variable to all the constraints stated over this variable (IBM, n.d.). Specifically, when a variable is asssigned to a value, this value is propagated to all the constraints of the variable. In the function below (which is used in the search function), when assigning a value to a given cell, we propagate all the constraints over the cell by removing this value from their list of possible values. This technique is useful for reducing the state space of the Soduku solving problem. 


    def set_board_position(self, index, number):
	    ... (some other code)

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
	   

	    return state

## Further optimisations and conclusions

The  implementation discussed above could be further optimised by extending the constraint propagation techniques by applying further consistency checking which would  assist in reducing the domain of possible values. Popular techniques include ARC consistency and Path consistency. 
Concluding, the techniques applied to solve this problem using backtracking, produced good results. The solver was able to solve easy, medium and the majority of hard problems in under 1.5 seconds. However, it struggled on some hard puzzles taking up to 3.5 seconds. A better choice of data structures and the removal of redundant operations could further improve the time complexity and speed of the solver.  In the next section, we discuss the Dancing Links implementation which is able to solve sudoku puzzles considerably faster. 

# Dancing Links (DLX) implementation

Sudoku puzzles can be formulated as an exact cover problem. As described by Knuth (2000), an exact cover problem can be illustrated by the following binary matrix. He describes the columns of the matrix as the elements of the "universe" and the rows as subsets of the "universe".  The objective of this problem is to find disjoint subsets in the matrix that cover the "universe". A concrete example can be given from the matrix below. Such disjoint subsets that cover the "universe" are rows 1, 4 and 5. Finding solutions for an exact cover problem is tough to compute and is NP-complete. 

| A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|
| 0 | 0 | 1 | 0 | 1 | 1 | 0 |
| 1 | 0 | 0 | 1 | 0 | 0 | 1 |
| 0 | 1 | 1 | 0 | 0 | 1 | 0 |
| 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| 0 | 0 | 0 | 1 | 1 | 0 | 1 |

<br>
The first step in using Knuth's algorithm is converting the Sudoku board into an exact cover matrix. Harrysson and Laestander (2014), explain nicely how this can be achieved. We can imagine that the constraints of the Soduku puzzle are the elements of the "universe" where we need to find a union of disjoint rows that cover the universe. To begin with, we have 81 (in a 9 by 9 board) constraints where each cell should contain a value (1-9). Then, we have row, column and box (or sub-grid) constraints (81 constraints each). Each row, column and box must have 9 unique integers in the range of 1-9. Therefore, we have a total of 324 constraints (4 * 81). The matrix is solved when a collection of disjoint sets (or rows) that cover the universe (i.e our constraints) is found.
<br>The rows in the exact cover matrix represent all possible subsets of the universe. In number terms, we have 729 possible rows  (9 possible values * 9 possible rows * 9 possible columns). Each possible subset satisfies 4 constraints. For example, a row may represent that we have the value 7 in row 2 and column 5. This row "applies" to each of the four constraints. 

## Converting a Sudoku board into an exact cover matrix

In this implementation , we represent our exact cover matrix as a 324 columns x 730 rows array (the first row is reserved for the column header). Then, we convert our 9 by 9 Sudoku board into the exact cover matrix.

First, we map each row, column, value tuple into a row position in our cover matrix. This is achieved by the following calculation:
```
row_position = 9 * (r * 9) + (c * 9) + v + 1
```
*Note*: we add one taking into account the extra column headers row.

Then, we have to calculate the mapping for each of the four constraints.

```
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
```
The cell constraint simply maps a row, column tuple into an index in the range 0-81. The row constraint maps a row, value tuple into an index that represents a value in the given row. Similarly, the column constraint maps a column, value tuple into an index that represents a value in the given column. Finally, the box constraint maps the row, column tuple into a box index (1-9) and the respective value in the cell is added. 

To map the entire Sudoku board we simply iterate through it and map in into the matrix. For cells that contain 0 (i.e. empty) we have to map all 9 possible values into the matrix. Otherwise, if the cell containts  a predetermined value, there is no need to map all 9 values (only the one given) since we know that only the predetermined value is possible.



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


## Algorithm X
In his paper, Knuth (2000), described an algorithm that finds all possible solutions in an exact cover binary matrix. He named this algorithm "Algorithm X".  The pseudocode for the algorithm is as follows (A is the matrix):

    If A is empty, the problem is solved; terminate successfully.
	Otherwise choose a column, c (deterministically).
	Choose a row, r, such that A[r, c] = 1 (nondeterministically).
	Include r in the partial solution.
	For each j such that A[r, j] = 1,
	    delete column j from matrix A;
	    for each i such that A[i, j] = 1,
	        delete row i from matrix A.
	Repeat this algorithm recursively on the reduced matrix A.

The way this algorithm works is by first initializing a data structure that will contain the partial solutions. Then, the algorithm chooses a column to cover deterministically. The nondeterministic choice of row  is where the algorithm essentially copies the current matrix (branches off) and tries to reduce it from there onwards. The row is added to the partial solutions and then the algorithm deletes all the columns that have a 1 in this row and all the rows that have 1 in these columns. The algorithm then recurses on the reduced matrix. The reduced matrix is checked on the first line of the algorithm. If the matrix is empty, it means we found an exact cover and therefore we return the partial solutions. Otherwise, if we find a column with 0 rows with 1 in them (size of column header is 0), this branch of the algorithm is terminated unsuccessfully. Therefore, the algorithm terminates when we have found a solution (empty matrix) or when all branches have been tried out and were unsuccessful.

However, as Knuth (2000) pointed out, this procedure is slow to compute.  Therefore the author introduces the Dancing Links technique. 

## Dancing Links (DLX)

### Matrix representation
Knuth (2000), suggests a good way to implement Algorithm X by having a binary matrix to represent all 1's in the matrix by data objects. Each data object x has five fields, Left(x), Right(x), Up(x), Down(x), Column(x). Each row in the matrix is doubly linked in a circular way by left and right fields of each data object. Columns are linked with each object in the column by the up and down fields. Each column also includes a special data object called its list header. These are part of a larger object called the column object. Column objects include the same fields as data objects with 2 additional fields, a Size field which refers to the number of 1's in this column and a Name field which is a symbolic reference for this column object. The list headers are doubly linked with all the headers (left and right) and serve the purpose of knowing which columns need still to be covered. 

This is how we represent them in our implementation:

    class Data:
    
    def __init__(self):
        self.left = self
        self.right = self
        self.up = self
        self.down = self
        self.column = self
        self.row_id = 0
        
    class  Column(Data):
    
	    def  __init__(self, ID):
		    super().__init__()
		    self.size = 0
		    self.ID = ID  # Symbolic reference

### Linking the matrix

The next step is linking the data and column objects together. In this implementation, we  link the column objects first with left and right pointers and add them to the first row of our matrix (which acts as the list of headers). The header (also known as root) object is a special column object which serves as the master header for all other column objects. 
Then, we start linking the rows in together (only the 1's). Each data object references to the column it belongs to and links by up and down pointers with objects in the column. We also increment the count of nodes for each column in this step. Afterwards, we link the rows with left and right pointers. In this step, we keep a reference node of the previous data object for each row.
```
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
	            
	            data_node = Data()
	            column_node = self.matrix[0][c]
	            data_node.column = column_node
	            data_node.up = column_node.up
	            data_node.down = column_node
	            data_node.row_id = r
	            column_node.up.down = data_node
	            column_node.up = data_node
	            column_node.size += 1

	            if ref_node == None:
	                ref_node = data_node

	            data_node.left = ref_node.left
	            data_node.right = ref_node
	            ref_node.left.right = data_node
	            ref_node.left = data_node

	            self.matrix[r][c] = data_node
```
### Recursive search procedure
The non-deterministic procedure shown in Algorithm X can now be turned into a deterministic recursive search procedure search(k), which is invoked initially with 0 (Knuth, 2000).  The algorithm is shown below (O represents the solutions).
```
If R[h] = h, print the current solution and return. 
Otherwise choose a column object c. 
Cover column c. 
For each r ← D[c], D D[c] , . . . , while r != c, 
	set O(k) ← r; 
	for each j ← R[r], R R[r] , . . . , while j != r, 
		cover column j; 
	search(k + 1); 
	set r ← O(k) and c ← C[r]; 
	for each j ← L[r], L L[r] , . . . , while j != r,
		uncover column j . 
Uncover column c and return.
```
The rows are now chosen deterministically which allows us to perform the efficient dancing links technique by covering and uncovering. 
### Covering
The cover procedure is where a target column c is removed (unlinked) and all the rows in c are removed from all the other columns they are in. In addition, the number of nodes this column has is decremented on each unlinking procedure of the rows. Knuth (2000), describes this procedure in pseudocode.
```
Set L R[c] ← L[c] and R L[c] ← R[c]. 
	For each i ← D[c], D D[c] , . . . , while i != c, 
		for each j ← R[i], R R[i] , . . . , while j != i, 
			set U D[j] ← U[j], D U[j] ← D[j], 
			and set S C[j] ← S C[j] − 1.
```
### Uncovering
The uncovering procedure is the whole point of the algorithm and as Knuth (2000) described it is "where the links do their dance". Uncovering is precicely the reverse order of the covering operation. The pseudocode from the paper is given below.
```
For each i = U[c], U U[c] , . . . , while i != c, 
	for each j ← L[i], L L[i] , . . . , while j != i, 
		set S C[j] ← S [j] + 1, and set U D[j] ← j, D U[j] ← j. 
Set L R[c] ← c and R L[c] ← c.
```
Notice how efficiently we reverse the cover operation by simply setting L R[c] and R L[c] to c.
### Implementation of Cover and Uncover procedures

Our cover and uncover functions perform precicely as Knuth (2000) described.
```
def cover(self, target):

	# Get the column of the target node
	target_col = target.column

	# Unlink column header
	target_col.left.right = target_col.right
	target_col.right.left = target_col.left
	  
	row = target_col.down

	# We unlink all rows in other columns
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

```
    def uncover(self, target):
    
        target_col = target.column
        
        row = target_col.up
        
        # We link all rows in other columns
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

		# Link the column headers back
        target_col.left.right = target_col
        target_col.right.left = target_col

### Branching factor reduction

The search algorithm may be further optimised by reducing the branching factor. One such optimisation can be performed on the choice of column to cover. This heuristic chooses the column with the least size (i.e. the column with the least number of data nodes in it). This is described by Knuth (2000) in pseudocode.
```
c ← R[h]
s ← ∞
for each j ← R[h], R R[h] , . . . , while j != h,
 if S[j] < s 
	 set c ← j and s ← S[j]
```
It is implemented as follows:

```
def get_col_with_least_nodes(self):

    head = self.header

    least_nodes = head.right
    head = head.right

    while (head != self.header):

        if (head.size < least_nodes.size):
            least_nodes = head
        
        head = head.right
    
    return least_nodes
```

### Overall view of the search algorithm

The entry point of the algorithm is the search(k) function. The process is as demonstrated by Knuth (2000).  First step of the algorithm is checking if an empty (i.e. fully covered) matrix has been reached. This is done by checking if the right pointer of the header points to itself. If that is the case, then the rest of the column headers have been unlinked, and therefore covered.  Afterwards, the algorithm chooses the next column to cover based on the heuristic and proceeds to cover it. The algorithm continues by adding each row of the column to the solution set and covers each of those rows (the columns in each row). Then, the algorithm recurses on its self all the way down the search tree until it finds an exact cover solution or finds a column with 0 nodes in it which causes the algorithm to go to the previous level, pop the added row solution from the list and uncover the operations. To illustrate this, let us look at an example. We invoke search(0); at this level the algorithm covers a column with least nodes and covers the next 'row'. Then, the algorithm calls search(1) with the reduced matrix and tries to cover the next column. At this stage, the algorithm covers the column and its next row and recurses on search(2). However, at this point (level 2), a column with 0 nodes is found and therefore it returns back to search(1). Search(1) pops the row from the list and uncovers the columns. It then tries to cover the next row, and the same process continues. 
```
def search(self, k):
    
    # We managed to cover all columns, and therefore found solution(s)
    if (self.header.right == self.header):
        return self.convert_to_sudoku(self.solutions)
    
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
```
When the search function finishes, we either have a list of solutions or an empty list. The last step of the implementation is to convert a list of solutions (containing data objects) to a readable Sudoku array.

### Converting sulutions to a Sudoku array
To convert each element of our solution list into row, column and value information for our Sudoku board, we have to reverse the mapping operation that we did initially.

As shown earlier, we converted row, column and value positions from the Sudoku board to a range of 0-729 using the following operation:
```
position = 9 * (row * 9) + (column * 9) + value + 1
```
The whole solution list is converted by the following function:
```
def convert_to_sudoku(self, solution):
	self.solution = np.zeros((9,9), dtype="int64")

	for row in solution:
	    row_id = row.row_id # We have to extract the row, col, value information from the row in the solution
	    row_id -= 1 # Take into account the column row
	    r = row_id // 81
	    c = (row_id % 81) // 9
	    n = (row_id % 81) % 9
	    self.solution[r][c] = n + 1 # Convert from zero base
        
```

# Discussion

## Optimisations
The Dancing Links implementation could be further optimised by linking the matrix more efficiently.  Currently, the program iterates through 236,196 positions in the array (324 x 729) when linking columns and rows together. The program could possibly be restructured to perform this linking operation when the sudoku is converted into an exact cover matrix. This would allow the program to link only the rows that are possible and therefore not iterate through the whole matrix. As a result the overhead of linking the matrix would reduce the overall runtime of the algorithm.
## Performance
The Dancing Links implementation is significantly faster than the former implementation. On average, the algorithm was able to solve easy, medium and hard Sudokus in 0.035 seconds compared to 1.5 seconds for the former one. 

## Conclusion
This all comes to down to the simple, yet brilliant low level technique of Dancing Links (Knuth, 2000). The removal of an element from a doubly linked list, is very straightforward. 
```
L R[x] ← L[x], R L[x] ← R[x]
```
However, as Knuth (2000) points out, very few programmers have realised that subsequent operations of 
```
L R[x] ← x, R L[x] ← x
```
put x back into the list again. If the programmer does not clean up the data structure after removal (which is not a good practice in most cases)  then this could be utilised by the cover and uncover procedures. Indeed, this process allows us remove elements from the linked list and reverse (backtrack) the process when needed using minimal resources (only pointer adjustments), with the power of dancing links. 



# References

*Harrysson, M., & Laestander, H. 2014. Solving Sudoku efficiently with Dancing Links.*

*IBM. n.d. _IBM Knowledge Center_. [online] Available at: <https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cpo.help/CP_Optimizer/User_manual/topics/propagate_propagate.html> [Accessed 13 March 2021].*

*Kumar Maji, A., Jana, S., Roy, S. and Kumar Pal, R., 2014. An Exhaustive Study on different Sudoku Solving Techniques. _International Journal of Computer Science Issues_, 11(2).*

*Knuth, D. 2000. Dancing Links. Millennial Perspectives in Computer Science. 1.* 


Sadeh, N. and Fox, M., 1996. Variable and value ordering heuristics for the job shop scheduling constraint satisfaction problem. _Artificial Intelligence_, 86(1), pp.1-41.


