# leetcode926
[Leecode926](https://leetcode.cn/problems/flip-string-to-monotone-increasing/) 将字符串翻转到单调递增

如果一个二进制字符串，是以一些 0（可能没有 0）后面跟着一些 1（也可能没有 1）的形式组成的，那么该字符串是 单调递增 的。
给你一个二进制字符串 s，你可以将任何 0 翻转为 1 或者将 1 翻转为 0 。
返回使 s 单调递增的最小翻转次数。

## 题解思路
直接根据题意来模拟, 从左到右的dp模拟, 当遍历到索引i时, dp\[i\]为\[0..i\]区间字符串的最小翻转数.
字符0前面只能为0, 字符1前面可以为0也可以为1

dp_0\[i\] 定义为当前索引为i, 字符为0 且\[0..n\]区间使字符串单调递增的最小反转次数. 
dp_1\[i\] 定义为当前索引为i, 字符为1 且\[0..n\]区间使字符串单调递增的最小反转次数. 

dp_0\[i\] = dp_0\[i-1\] + 是否需要变化的次数(当s\[i\]为0时不需要变化, s\[i\]为1时需要变化)
dp_1\[i\] = min(dp_0\[i-1\], dp_1\[i-1\]) + 是否需要变化的次数(当s\[i\]为1时不需要变化, s\[i\]为2时需要变化)


## 题解答案

``` rust
impl Solution {
    pub fn min_flips_mono_incr(s: String) -> i32 {
        let s_chars = s.chars().collect::<Vec<_>>();
        let n = s_chars.len();
        let mut dp_0 = if s_chars[0] == '0' { 0 } else { 1 };
        let mut dp_1 = if s_chars[0] == '1' { 0 } else { 1 };
        for i in 1..n {
            let pre_dp_0 = dp_0;
            dp_0 = dp_0 + if s_chars[i] == '0' { 0 } else { 1 };
            dp_1 = pre_dp_0.min(dp_1) + if s_chars[i] == '1' { 0 } else { 1 };
        }
        dp_0.min(dp_1)
    }
} 
```