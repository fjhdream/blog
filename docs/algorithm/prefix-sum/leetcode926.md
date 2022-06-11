# leetcode926
[Leecode926](https://leetcode.cn/problems/flip-string-to-monotone-increasing/) 将字符串翻转到单调递增

如果一个二进制字符串，是以一些 0（可能没有 0）后面跟着一些 1（也可能没有 1）的形式组成的，那么该字符串是 单调递增 的。
给你一个二进制字符串 s，你可以将任何 0 翻转为 1 或者将 1 翻转为 0 。
返回使 s 单调递增的最小翻转次数。

## 题解思路
1. 情况一:  001111 类似于这种情况(前面为0后面都为1)的字符串一定是单调递增的
   1. 我们就去枚举字符串的每一位去构造这种场景
   2. 基于这种情况, 我们就需要知道某一索引i, \[0..i-1\]区间上1的数量(即需要翻转的次数), \[i+1..n\]区间上0的数量. n为字符串s的长度
   3. 所以可以想到使用前缀和的方式来方便快速查询, 正向前缀和来存储\[0..i-1\]区间上1的数量, 逆向前缀和存储\[i+1..n\]区间上0的数量.
2. 情况二: 00000 全部为0的字符串也符合单调递增
3. 情况三: 11111 全部为1的字符串也符合单调递增
4. 以上三种情况中最小的翻转数即为答案

## 题解答案

``` rust
    impl Solution {
    pub fn min_flips_mono_incr(s: String) -> i32 {
        let s_chars = s.chars().collect::<Vec<_>>();
        let n = s_chars.len();
        let mut left_ones = vec![0; n];
        let mut right_zeros = vec![0; n];
        for (i, &ch) in s_chars.iter().enumerate() {
            if i == 0 {
                left_ones[i] = if ch == '1' { 1 } else { 0 };
            } else {
                left_ones[i] = left_ones[i - 1] + if ch == '1' { 1 } else { 0 };
            }
        }

        for (i, &ch) in s_chars.iter().enumerate().rev() {
            if i == n - 1 {
                right_zeros[i] = if ch == '0' { 1 } else { 0 };
            } else {
                right_zeros[i] = right_zeros[i + 1] + if ch == '0' { 1 } else { 0 };
            }
        }

        let mut res = i32::MAX;
        for i in 1..n - 1 {
            res = res.min(left_ones[i - 1] + right_zeros[i + 1]);
        }
        res = res.min(left_ones[n - 1]).min(right_zeros[0]);
        res
    }
}
```