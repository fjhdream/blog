---
title: leetcode1414
categories:
  - Leetcode
tags:
  - 算法
halo:
  site: http://205.234.201.223:8090
  name: 70a17b7d-0a34-410c-9718-a6467b29a4e1
  publish: true
cssclasses:
---
# Leetcode1414

使用贪心算法来进行题解, 证明较难
题解[贪心证明](https://leetcode-cn.com/problems/find-the-minimum-number-of-fibonacci-numbers-whose-sum-is-k/solution/he-wei-k-de-zui-shao-fei-bo-na-qi-shu-zi-shu-mu-by)

给你数字 k ，请你返回和为 k 的斐波那契数字的最少数目，其中，每个斐波那契数字都可以被使用多次。

斐波那契数字定义为：

F1 = 1
F2 = 1
Fn = Fn-1 + Fn-2 ， 其中 n > 2 。
数据保证对于给定的 k ，一定能找到可行解。

示例 1：
输入：k = 7
输出：2
解释：斐波那契数字为：1，1，2，3，5，8，13，……
对于 k = 7 ，我们可以得到 2 + 5 = 7 。

示例 2：
输入：k = 10
输出：2
解释：对于 k = 10 ，我们可以得到 2 + 8 = 10 。

示例 3：
输入：k = 19
输出：3
解释：对于 k = 19 ，我们可以得到 1 + 5 + 13 = 19 。

提示：
1 <= k <= 10^9

``` rust
pub fn find_min_fibonacci_numbers(mut k: i32) -> i32 {
    let mut fibs = vec![1,1];
    let (mut a, mut b) = (1,1);
    while a + b <= k {
        let c= a + b;
        fibs.push(a + b);
        a = b;
        b = c;
    }
 
    let mut ans = 0;
    let mut i = fibs.len() - 1;
    while k > 0 {
        let num = fibs[i];
        if k >= num {
            k -= num;
            ans += 1;
        }
        i -= 1; 
    }
    ans
}
```
