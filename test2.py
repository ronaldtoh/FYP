from typing import Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class LinkedList(object):
    def __init__(self, sequence):
        self.head = ListNode(sequence[0])
        current = self.head
        for item in sequence[1:]:
            current.next = ListNode(item)
            current = current.next


class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:

        slow, fast, prev = head, head, None

        while fast and fast.next:
            fast = fast.next.next
            temp = slow.next
            slow.next = prev
            prev = slow
            slow = temp
        print(slow, fast, prev)
        if fast:
            slow = slow.next
        while slow and prev:
            if slow.val != prev.val:
                return False
            slow = slow.next
            prev = prev.next
        return True

    def longestPalindrome(self, s: str) -> str:
        longest = ""
        for idx, val in enumerate(s):
            curr = val
            pal = True
            left = idx
            right = idx
            while pal:
                left = left - 1
                right = right + 1
                if left < 0 or right >= len(s):
                    break
                if s[left] == s[right]:
                    curr = s[left] + curr + s[right]
                else:
                    break

            if len(longest) < len(curr):
                longest = curr
        return longest


# lst = [1, 2, 3, 4]
# lst = [1]
# storage = []
# idx = 0

# li = LinkedList(lst)
# current = li.head
# while current is not None:
#     current = current.next

sol = Solution()
print(sol.longestPalindrome("cbbd"))
