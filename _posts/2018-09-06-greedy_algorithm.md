---
layout: post
title: "算法与数据结构之贪心算法"
tags: [算法与数据结构]
date: 2018-09-06
---

*Question 1*
+ *Introduction:* &nbsp;&nbsp; In this problem, you will design and implement an elementary greedy algorithm used by cashiers all over the world millions of times per day.
+ *Task:* &nbsp;&nbsp; The goal in this problem is to find the minimum number of coins needed to change the input value(an integer) into coins with denominations 1, 5, and 10.
+ *Input Format:* &nbsp;&nbsp; The input consists of a single integer 𝑚(1 ≤ 𝑚 ≤ 10<sup>3</sup>)
+ *Output Format:* &nbsp;&nbsp; Output the minimum number of coins with denominations 1, 5, 10 that changes 𝑚

```python
import sys
def get_change(m):
      #write your code here
      n10 = m//10
      n5 = (m%10)//5
      n1 = (m%10)%5
      return n1+n5+n10
### run
if __name__ == '__main__':
      m = int(sys.stdin.read())
      print(get_change(m))
```

*Question 2*
+ *Introduction:* &nbsp;&nbsp; A thief finds much more loot than his bag can fit. Help him to find the most valuable combination of items assuming that any fraction of a loot item can be put into his bag.
+ *Task:* &nbsp;&nbsp; The goal of this code problem is to implement an algorithm for the fractional knapsack problem.
+ *Input Format:* &nbsp;&nbsp; The first line of the input contains the number 𝑛 of items and the capacity 𝑊 of a knapsack. The next 𝑛 lines define the values and weights of the items. The 𝑖-th line contains integers 𝑣𝑖 and 𝑤𝑖—the value and the weight of 𝑖-th item, respectively(1 ≤ 𝑛 ≤ 10<sup>3</sup>, 0 ≤ 𝑊 ≤ 2 · 10<sup>6</sup>; 0 ≤ 𝑣𝑖 ≤ 2 · 10<sup>6</sup>, 0 < 𝑤𝑖 ≤ 2 · 10<sup>6</sup> for all 1 ≤ 𝑖 ≤ 𝑛. All the numbers are integers)
+ *Output Format:* &nbsp;&nbsp; Output the maximal value of fractions of items that fit into the knapsack.

```python
import sys
def get_optimal_value(capacity, weights, values):
    value = 0.
    # write your code here
    temp = [(w,v/w) for w,v in zip(weights, values)]
    temp = sorted(temp, key=lambda s:s[1], reverse=True)
    remain = capacity
    for i in range(len(temp)):
    	if remain==0: break
    	w_in = min(remain, temp[i][0])
    	remain -= w_in
    	value += w_in*temp[i][1]
    return value

if __name__ == "__main__":
    data = list(map(int, sys.stdin.read().split()))
    n, capacity = data[0:2]
    values = data[2:(2 * n + 2):2]
    weights = data[3:(2 * n + 2):2]
    opt_value = get_optimal_value(capacity, weights, values)
    print("{:.10f}".format(opt_value))
```

*Question 3*
+ *Introduction:* &nbsp;&nbsp; You have 𝑛 ads to place on a popular Internet page. For each ad, you know how much is the advertiser willing to pay for one click on this ad. You have set up 𝑛 slots on your page and estimated the expected number of clicks per day for each slot. Now, your goal is to distribute the ads among the slots to maximize the total revenue.
+ *Task:* &nbsp;&nbsp; Given two sequences 𝑎1, 𝑎2, . . . , 𝑎𝑛 (𝑎𝑖 is the profit per click of the 𝑖-th ad) and 𝑏1, 𝑏2, . . . , 𝑏𝑛 (𝑏𝑖 is the average number of clicks per day of the 𝑖-th slot), we need to partition them into 𝑛 pairs (𝑎𝑖, 𝑏𝑗 ) such that the sum of their products is maximized.
+ *Input Format:* &nbsp;&nbsp; The first line contains an integer 𝑛, the second one contains a sequence of integers 𝑎1, 𝑎2, . . . , 𝑎𝑛, the third one contains a sequence of integers 𝑏1, 𝑏2, . . . , 𝑏𝑛(1 ≤ 𝑛 ≤ 10<sup>3</sup>; −10<sup>5</sup> ≤ 𝑎𝑖, 𝑏𝑖 ≤ 10<sup>5</sup> for all 1 ≤ 𝑖 ≤ 𝑛.)
+ *Output Format:* &nbsp;&nbsp; Output the maximum value of $$\sum_1^n$$a<sub>i</sub>c<sub>i</sub>, where 𝑐1, 𝑐2, . . . , 𝑐𝑛 is a permutation of 𝑏1, 𝑏2, . . . , 𝑏𝑛.

```python
import sys
def max_dot_product(a, b):
    #write your code here
    return sum([x*y for x,y in zip(sorted(a),sorted(b))])

if __name__ == '__main__':
    input = sys.stdin.read()
    data = list(map(int, input.split()))
    n = data[0]
    a = data[1:(n + 1)]
    b = data[(n + 1):]
    print(max_dot_product(a, b))
```

*Question 4*
+ *Introduction:* &nbsp;&nbsp; You are responsible for collecting signatures from all tenants of a certain building. For each tenant, you know a period of time when he or she is at home. You would like to collect all signatures by visiting the building as few times as possible. The mathematical model for this problem is the following. You are given a set of segments on a line and your goal is to mark as few points on a line as possible so that each segment contains at least one marked point.
+ *Task:* &nbsp;&nbsp; Given a set of 𝑛 segments {[𝑎0, 𝑏0], [𝑎1, 𝑏1], . . . , [𝑎<sub>𝑛−1</sub>, 𝑏<sub>𝑛−1</sub>]} with integer coordinates on a line, find the minimum number 𝑚 of points such that each segment contains at least one point. That is, find a set of integers 𝑋 of the minimum size such that for any segment [𝑎𝑖, 𝑏𝑖] there is a point 𝑥 ∈ 𝑋 such that 𝑎𝑖 ≤ 𝑥 ≤ 𝑏𝑖.
+ *Input Format:* &nbsp;&nbsp; The first line of the input contains the number 𝑛 of segments. Each of the following 𝑛 lines contains two integers 𝑎𝑖 and 𝑏𝑖 (separated by a space) defining the coordinates of endpoints of the 𝑖-th segment(1 ≤ 𝑛 ≤ 100; 0 ≤ 𝑎𝑖 ≤ 𝑏𝑖 ≤ 10<sup>9</sup> for all 0 ≤ 𝑖 < 𝑛)
+ *Output Format:* &nbsp;&nbsp; Output the minimum number 𝑚 of points on the first line and the integer coordinates of 𝑚 points (separated by spaces) on the second line. You can output the points in any order. If there are many such sets of points, you can output any set. (It is not difficult to see that there always exist a set of points of the minimum size such that all the coordinates of the points are integers.)

```python
import sys
from collections import namedtuple
Segment = namedtuple('Segment', 'start end')
def optimal_points(segments):
    points = []
    #write your code here
    segments = sorted(segments, key=lambda s: s.end)
    points.append(segments[0].end)
    for s in segments:
        if s.start>points[-1]:
            points.append(s.end)
    return points

if __name__ == '__main__':
    input = sys.stdin.read()
    n, *data = map(int, input.split())
    segments = list(map(lambda x: Segment(x[0], x[1]), zip(data[::2], data[1::2])))
    points = optimal_points(segments)
    print(len(points))
    for p in points:
        print(p, end=' ')
```

*Question 5*
+ *Introduction:* &nbsp;&nbsp; You are organizing a funny competition for children. As a prize fund you have 𝑛 candies. You would like to use these candies for top 𝑘 places in a competition with a natural restriction that a higher place gets a larger number of candies. To make as many children happy as possible, you are going to find the largest value of 𝑘 for which it is possible.
+ *Task:* &nbsp;&nbsp; The goal of this problem is to represent a given positive integer 𝑛 as a sum of as many pairwise distinct positive integers as possible. That is, to find the maximum 𝑘 such that 𝑛 can be written as 𝑎1 + 𝑎2 + · · · + 𝑎𝑘 where 𝑎1, . . . , 𝑎𝑘 are positive integers and ai&ne;aj for all 1 ≤ 𝑖 < 𝑗 ≤ 𝑘.
+ *Input Format:* &nbsp;&nbsp; The input consists of a single integer 𝑛(1 ≤ 𝑛 ≤ 10<sup>9</sup>)
+ *Output Format:* &nbsp;&nbsp; In the first line, output the maximum number 𝑘 such that 𝑛 can be represented as a sum of 𝑘 pairwise distinct positive integers. In the second line, output 𝑘 pairwise distinct positive integers that sum up to 𝑛 (if there are many such representations, output any of them).

```python
import sys
def optimal_summands(n):
    summands = []
    #write your code here
    remain = n
    for i in range(1,n+1):
        last_c = i if remain-i>i else remain
        summands.append(last_c)
        remain -= last_c
        if remain==0: break
    return summands

if __name__ == '__main__':
    input = sys.stdin.read()
    n = int(input)
    summands = optimal_summands(n)
    print(len(summands))
    for x in summands:
        print(x, end=' ')
```

*Question 6*
+ *Introduction:* &nbsp;&nbsp; As the last question of a successful interview, your boss gives you a few pieces of paper with numbers on it and asks you to compose a largest number from these numbers. The resulting number is going to be your salary, so you are very much interested in maximizing this number. How can you do this?
+ *Task:* &nbsp;&nbsp; Compose the largest number out of a set of integers.
+ *Input Format:* &nbsp;&nbsp; The first line of the input contains an integer 𝑛. The second line contains integers 𝑎1, 𝑎2, . . . , 𝑎𝑛(1 ≤ 𝑛 ≤ 100; 1 ≤ 𝑎𝑖 ≤ 10<sup>3</sup> for all 1 ≤ 𝑖 ≤ 𝑛)
+ *Output Format:* &nbsp;&nbsp; Output the largest number that can be composed out of 𝑎1, 𝑎2, . . . , 𝑎𝑛.

```python
import sys
from functools import cmp_to_key

def comp(a,b):
	if a[0]>b[0]:
		return 1
	elif a[0]<b[0]:
		return -1
	else:
		ab = a+b
		ba = b+a
		return 1 if ab>ba else -1 if ab<ba else 0

def largest_number(a):
    #write your code here
    a = sorted(a, key=cmp_to_key(comp), reverse=True)
    return ''.join(a)

if __name__ == '__main__':
    input = sys.stdin.read()
    data = input.split()
    a = data[1:]
    print(largest_number(a))
```
