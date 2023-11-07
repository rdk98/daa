def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    left = merge_sort(left)
    right = merge_sort(right)

    return merge(left, right)

def merge(left, right):
    result = []
    i = 0
    j = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result

if __name__ == "__main__":
    n = int(input("Enter the size of the list: "))
    arr = []

    for i in range(n):
        element = int(input(f"Enter element {i + 1}: "))
        arr.append(element)

    # Merge Sort
    sorted_merge = merge_sort(arr.copy())
    print("Original List:", arr)
    print("Sorted List (Merge Sort):", sorted_merge)
