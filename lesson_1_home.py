import numpy as np

# Завдання 1
# Створіть масив:
# 1 2 3 4
# 5 6 7 8
# 9 10 11 12
# 13 14 15 16
# Використовуючи індекси виведіть:
# ● число 14
# ● третій рядок
# ● перший стовпчик
# ● верхню половину
# ● замініть числа в рядках 2-3 на 100
# ● зробіть другий рядок таким як останній рядок

def start_task1():
    nums = np.arange(1, 17).reshape(4, 4)

    print(f"число 14: {nums[-1, 1]}")
    print(f"третій рядок: {nums[2]}")
    print(f"перший стовпчик: {nums[:, 0]}")
    print(f"верхню половину: {nums[:2]}")

    nums[1:3] = 100
    print(f"замініть числа в рядках 2-3 на 100: {nums}")

    nums[1] = nums[-1]
    print(f"зробіть другий рядок таким як останній рядок: {nums}")


# start_task1()


# Завдання 2
# У масиві з попереднього завдання створіть маску для
# парних чисел. З її допомогою
# ● виведіть самі числа
# ● замініть їх на 100


def start_task2():
    nums = np.arange(1, 17).reshape(4, 4)
    mask = nums % 2 == 0

    print(f"{nums[mask] = }")

    nums[mask] = 100

    print(f"{nums = }")


# start_task2()


# Завдання 3
# Створіть 2 масиви типу uint8:
# Масив 1: 128 200 10
# Масив 2: 250 10 34
# Об’єднайте їх у пропорції 20% першого масив + 80%
# другого масиву. В результаті має бути тип даних uint8 та
# числа в діапазоні 0-255


def start_task3():
    nums = np.array([128, 200, 10], dtype=np.uint8)
    nums = nums * 0.2
    print(f"{nums = }")

    nums1 = np.array([250, 10, 34], dtype=np.uint8)
    nums1 = nums1 * 0.8
    print(f"{nums1 = }")

    result = nums + nums1
    mask = result > 255
    result[mask] = 255
    result = result.astype(np.uint8)

    print(f"{result = }")

start_task3()

