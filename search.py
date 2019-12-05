# -*- coding: utf-8 -*-
# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    from util import Stack
    # 起始节点栈
    opened = Stack()
    # 已经访问过的节点表
    closed = []
    # 得到初始节点
    opened.push((problem.getStartState(), []))
    # 开始遍历
    while not opened.isEmpty():
        # 得到当前的节点
        currentNode0, currentNode1 = opened.pop()
        # 判断是否是目标节点
        if problem.isGoalState(currentNode0):
            return currentNode1
        # 如果当前节点没有被访问过
        if currentNode0 not in closed:
            # 得到后继节点和运行方向以及花费代价
            expand = problem.getSuccessors(currentNode0)
            # 将当前节点加入到closed表中
            closed.append(currentNode0)
            # 遍历后继节点
            for locations, directions, cost in expand:
                if (locations not in closed):
                    opened.push((locations, currentNode1 + [directions]))
    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    # 起始节点队列
    opened = Queue()
    # 已经访问过的节点表
    closed = []
    # 得到初始节点
    opened.push((problem.getStartState(), []))
    # 开始遍历
    while not opened.isEmpty():
        # 得到当前节点
        currentNode0, currentNode1 = opened.pop()
        # 判断是否是目标节点
        if problem.isGoalState(currentNode0):
            return currentNode1
        # 如果当前节点没有访问过
        if currentNode0 not in closed:
            # 得到后继节点和运行方向以及花费代价
            expand = problem.getSuccessors(currentNode0)
            # 将当前节点加入closed表中
            closed.append(currentNode0)
            # 遍历后继节点
            for locations, directions, cost in expand:
                if (locations not in closed):
                    opened.push((locations, currentNode1 + [directions]))
    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # 起始结点
    startNodes = problem.getStartState()
    # 记录下当前队列
    queue = util.PriorityQueueWithFunction(lambda x: x[2])
    # 将起始节点加入队列
    queue.push((startNodes, None, 0))
    # 初始化代价为0
    cost = 0
    # 标记是否记录
    visited = []
    # 记录路径
    path = []
    # 创建字典
    parentSeq = {}
    parentSeq[(startNodes, None, 0)] = None
    while not queue.isEmpty():
        # 得到当前的节点信息
        currentNodes = queue.pop()
        # 判断是否是目标节点
        if (problem.isGoalState(currentNodes[0])):
            break
        else:
            # 得到当前节点信息
            currentState = currentNodes[0]
            # 将该节点加入到visited中
            if currentState not in visited:
                visited.append(currentState)
            else:
                continue
            # 得到表的后继
            successors = problem.getSuccessors(currentState)
            # 遍历后继节点
            for s in successors:
                cost = currentNodes[2] + s[2];
                if s[0] not in visited:
                    queue.push((s[0], s[1], cost))
                    parentSeq[(s[0], s[1])] = currentNodes
    child = currentNodes
    while child != None:
        path.append(child[1])
        if child[0] != startNodes:
            child = parentSeq[(child[0], child[1])]
        else:
            child = None
    path.reverse()
    return path[1:]
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

#郑昊宣
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    fringe = util.PriorityQueue()     # 使用优先队列，每次扩展都是选择当前代价最小的方向，即队头
    actions = []                                    # 操作
    fringe.push((problem.getStartState(),actions),0)   # 把初始化点加入队列，开始扩展
    visited = []                                # 标记已经走过的点
    while not fringe.isEmpty():
        currState,actions = fringe.pop()      # 当前状态
        if problem.isGoalState(currState):
            break
        if currState not in visited:
            visited.append(currState)
            successors = problem.getSuccessors(currState)
            for successor, action, cost in successors:
                tempActions = actions + [action]
                nextCost = problem.getCostOfActions(tempActions) + heuristic(successor,problem)      # 对可选的几个方向，计算代价
                if successor not in visited:
                    fringe.push((successor,tempActions),nextCost)
    return actions                # 返回到达终点的操作顺序

#util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
