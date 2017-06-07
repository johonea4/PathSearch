# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions 
to this file that are not part of the classes that we want.
"""

from __future__ import division

import heapq
import pickle
import math

import os


class PriorityQueue(object):
    """A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
        current (int): The index of the current node in the queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue.
        """

        self.queue = []

    def pop(self):
        """Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """
        return heapq.heappop(self.queue)

    def remove(self, node_id):
        """Remove a node from the queue.

        This is a hint, you might require this in ucs,
        however, if you choose not to use it, you are free to
        define your own method and not use it.

        Args:
            node_id (int): Index of node in queue.
        """
        self.queue.remove(node_id)
    
    def __iter__(self):
        """Queue iterator.
        """

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queuer to string.
        """

        return 'PQ:%s' % self.queue

    def append(self, node):
        """Append a node to the queue.
        
        Args:
            node: Comparable Object to be added to the priority queue.
        """
        heapq.heappush(self.queue,node)

    def peek(self,node_id):
        return self.queue[node_id]

    def find(self,key):
        iter=0
        for k in self.queue:
            if(k[1]==key):
                return iter
            iter+=1
        return None


    def __contains__(self, key):
        """Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n for _, n in self.queue]

    def __eq__(self, other):
        """Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self == other

    def size(self):
        """Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)
    
    def clear(self):
        """Reset queue to empty (no nodes).
        """

        self.queue = []
        
    def top(self):
        """Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """

        return self.queue[0]


def Solution(node):
    """This formulates a path list and depends that
    the third value of the triple node is the node's
    parent
    """

    s = list()
    s.append(node[1])
    n = node[2]
    while(n is not None):
        s.append(n[1])
        n = n[2]
    s.reverse()
    return s

def breadth_first_search(graph, start, goal):
    """Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if (start == goal): return []
    treeNode = (0,start,None)
    node=start

    frontierNodes = dict()
    frontier = list()
    explored = set()

    frontierNodes[start]=treeNode
    frontier.append(node)
    # print("Start: %s") %(start)

    while (len(frontier)>0):
        node = frontier.pop(0)
        treeNode = frontierNodes[node]
        explored.add(node)
        children = graph[node]
        for child in children:
            childNode=(treeNode[0]+1,child,treeNode)
            if(child == goal): 
                return Solution(childNode) 
            if(child not in explored and child not in frontier):
                print("Adding: %s") %(child)
 #               childNode=(treeNode[0]+1,child,treeNode)
                frontier.append(child)
                frontierNodes[child]=childNode
            elif(child in frontier):              
                if((treeNode[0]+1) < frontierNodes[child][0]):
                    print("Replacing: %s") %(child)
 #                   childNode=(treeNode[0]+1,child,treeNode)
                    frontierNodes[child]=childNode

    return None

def uniform_cost_search(graph, start, goal):
    """Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    if (start == goal): return []
    node = (0,start,None)

    frontier = PriorityQueue()
    explored = set()

    frontier.append(node)
    # print("Start: %s") %(start)

    while (frontier.size>0):
        node = frontier.pop()
        if(node[1] == goal): 
            return Solution(node) 
        explored.add(node[1])
        children = graph[node[1]]
        for child in children:
            if(child not in explored):
                # print("Adding: %s") %(child)
                h = node[0] + children[child]['weight']
                childNode=(h,child,node)
                frontier.append(childNode)
    return None


def null_heuristic(graph, v, goal ):
    """Null heuristic used as a base line.

    Args:
        graph (explorable_graph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node as a list.
    """

    # TODO: finish this function!
    raise NotImplementedError


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """ Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def bidirectional_ucs(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def bidirectional_a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


# Extra Credit: Your best search method for the race
#
def load_data():
    """Loads data from data.pickle and return the data object that is passed to
    the custom_search method.

    Will be called only once. Feel free to modify.

    Returns:
         The data loaded from the pickle file.
    """

    pickle_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data.pickle")
    data = pickle.load(open(pickle_file_path, 'rb'))
    return data


def custom_search(graph, start, goal, data=None):
    """Race!: Implement your best search algorithm here to compete against the
    other student agents.

    See README.md for exercise description.

    Args:
        graph (explorable_graph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError
