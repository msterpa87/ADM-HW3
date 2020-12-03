"""
Longest Increasing Subsequence problem on strings
"""

def recursive_LIS_helper(X, i):
    if i == 0:
        return 1
    
    # recursively call only on previous characters that would from an IS
    smaller_precedessors = [j for j in range(i) if X[j] < X[i]]
    
    if len(smaller_precedessors) == 0:
        return 1
    else:
        Y = list(map(lambda j: recursive_LIS_helper(X,j), smaller_precedessors))
        return 1 + max(Y)

def recursive_LIS(X):
    # recursive implementation of longest increasing subsequence
    return max(list(map(lambda i: recursive_LIS_helper(X, i), range(len(X)))))

# the following instance doesn't compute in reasonable time
# s = "ABCDEFGHIJKLMNOPQRSTUVZ"
# recursive_LSS(s*3)

def dynamic_LIS(X):
    # dynamic programming implementation of LIS
    n = len(X)
    if n == 1:
        return 1
    V = [1]*n
    
    for i in range(1,n):       
        for j in range(i):
            if X[j] < X[i] and V[j]+1 > V[i]:
                V[i] = V[j] + 1
    
    return max(V)