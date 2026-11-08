import numpy as np

numbers = [10, 20, 30, 40, 50]

x = np.array(numbers)
print(x.size,x.dtype, x.mean())

apartments = [
    [30, 1],
    [40, 1],
    [50, 2],
    [60, 2]
]

arr = np.array(apartments)
arr_square = arr[0]
counnt_room = arr[2]
last_rooms = arr[2:]
arr_square1 = arr[:, 0][:2]
areas = arr[:, 0]
rooms = arr[:, 1]
print(arr.shape,arr.size,arr_square, counnt_room,last_rooms,arr_square1)
print(areas.mean(), areas.max(), rooms.shape)
