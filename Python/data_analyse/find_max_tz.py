def get_tz_count(sequence):
    ...:     count_dict = defaultdict(int)
    ...:     for element in sequence:
    ...:         count_dict[element] += 1
    ...:     return count_dict
    ...:
    ...:

In [35]: count_dict = get_tz_count(time_zone)

In [36]: count_dict


Counter类：为hashable对象计数，是字典的子类