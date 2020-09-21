from treelib import Tree, Node
'''
tree = Tree()
tree.create_node("Harry", "harry")  # root node
tree.create_node("Jane", "jane", parent="harry")
tree.create_node("Bill", "bill", parent="harry")
tree.show()
'''

class DecisionTree(Tree):
    def __init__(self, arg1):
        super(DecisionTree, self).__init__()
        self.arg1 = arg1