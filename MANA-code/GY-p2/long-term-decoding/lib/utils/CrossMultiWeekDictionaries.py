
# cross-week setup for GY-p2 dataset
# generate a domain dictionary and a training start-end day dictionary

first_days_of_week = [ # 每周的第一天
    0, 1, 3, 7, 12, 13, 16, 19, 22, 23, 24, 28, 32, 33, 36, 39, 41, 44, 46, 47, 50, 54, 57, 61, 64, 66, 71, 73, 76, 78, 83, 87, 89, 93, 98, 101, 104, 107, 111, 115, 119, 10000
]


class DayOutOfRange(Exception):
    def __init__(self, msg):
        self.message = msg
    
    def __str__(self):
        return self.message


def find_week_number_for_day(day_number): # 查找某一天在第几周
    if day_number in range(1, 120):
        for i in range(len(first_days_of_week)):
            if first_days_of_week[i] > day_number:
                return i - 1
    else:
        raise DayOutOfRange("Invalid day number. Expecting day number within range [1, ..., 119].")
            

def generate_domain_dict_and_day_from_to_dict(cross_week_num): # 根据跨周数，生成域长度字典和起止字典
    domain_dict = dict()
    day_from_to_dict = dict()
    if cross_week_num == 0:
        # train first 4 weeks
        for day in range(13, 51):
            domain_dict[str(day)] = [2, 4, 5, 1]
            day_from_to_dict[str(day)] = [1, 12]
        for day in range(64, 120):
            domain_dict[str(day)] = [4, 3, 4, 3]
            day_from_to_dict[str(day)] = [50, 63]
    elif cross_week_num == 0.1:
        # cross multi days
        for day in range(13, 51):
            domain_dict[str(day)] = [
                int((day + 2) / 4),
                int((day + 1) / 4),
                int((day + 0) / 4),
                int((day - 1) / 4),
            ]
            day_from_to_dict[str(day)] = [1, day - 1]
        for day in range(64, 120):
            domain_dict[str(day)] = [
                int((day - 47) / 4),
                int((day - 48) / 4),
                int((day - 49) / 4),
                int((day - 50) / 4),
            ]
            day_from_to_dict[str(day)] = [50, day - 1]
    else:
        cross_week_num = int(cross_week_num)
        for day in valid_range_for_cross_week_num(cross_week_num):
            # cross multi weeks
            week_num = find_week_number_for_day(day)
            train_start_week_num = week_num - cross_week_num - 3
            domain_dict[str(day)] = [
                first_days_of_week[train_start_week_num + 1] - first_days_of_week[train_start_week_num],
                first_days_of_week[train_start_week_num + 2] - first_days_of_week[train_start_week_num + 1],
                first_days_of_week[train_start_week_num + 3] - first_days_of_week[train_start_week_num + 2],
                first_days_of_week[train_start_week_num + 4] - first_days_of_week[train_start_week_num + 3],
            ]
            day_from_to_dict[str(day)] = [
                first_days_of_week[train_start_week_num], 
                first_days_of_week[train_start_week_num + 4] - 1,
            ]
    return domain_dict, day_from_to_dict


class IncompatibleDayWeek(Exception):
    def __init__(self, msg):
        self.message = msg
    
    def __str__(self):
        return self.message


def verify_day_in_test_range(day, cross_week_num):
    valid_day_list = valid_range_for_cross_week_num(cross_week_num)
    if day not in valid_day_list:
        raise IncompatibleDayWeek(
            f"Test day {day} out of range, since testing cross {cross_week_num} week. Test day must be within range [{valid_day_list[0]}, {valid_day_list[-1]}]."
        )
    

def valid_range_for_cross_week_num(cross_week_num):
    if cross_week_num == 0:
        return list(range(13, 51)) + list(range(64, 120))
    elif cross_week_num == 0.1:
        return list(range(13, 51)) + list(range(64, 120))
    else:
        cross_week_num = int(cross_week_num)
        start_day = first_days_of_week[cross_week_num + 4]
        end_day = 119
        return list(range(start_day, end_day + 1))
    

        
week_all = {
    '1': 20221103, '2': 20221104, 
    '3': 20221108, '4': 20221109, '5': 20221110, '6': 20221111, 
    '7': 20221114, '8': 20221115, '9': 20221116, '10': 20221117, '11': 20221118, 
    '12': 20221125, 
    '13': 20221128, '14': 20221129, '15': 20221202, 
    '16': 20221205, '17': 20221208, '18': 20221209, 
    '19': 20221213, '20': 20221214, '21': 20221215, 
    '22': 20221219, 
    '23': 20221230, 
    '24': 20230103, '25': 20230104, '26': 20230105, '27': 20230106, 
    '28': 20230109, '29': 20230111, '30': 20230112, '31': 20230113, 
    '32': 20230116, 
    '33': 20230208, '34': 20230209, '35': 20230210, 
    '36': 20230213, '37': 20230215, '38': 20230217, 
    '39': 20230227, '40': 20230303, 
    '41': 20230306, '42': 20230308, '43': 20230310, 
    '44': 20230313, '45': 20230316, 
    '46': 20230320, 
    '47': 20230327, '48': 20230328, '49': 20230329, 
    '50': 20230402, '51': 20230404, '52': 20230406, '53': 20230407, 
    '54': 20230410, '55': 20230411, '56': 20230414, 
    '57': 20230418, '58': 20230419, '59': 20230420, '60': 20230421, 
    '61': 20230423, '62': 20230425, '63': 20230428, 
    '64': 20230504, '65': 20230506, 
    '66': 20230508, '67': 20230509, '68': 20230510, '69': 20230511, '70': 20230512, 
    '71': 20230516, '72': 20230517, 
    '73': 20230612, '74': 20230614, '75': 20230616, 
    '76': 20230619, '77': 20230621, 
    '78': 20230625, '79': 20230626, '80': 20230628, '81': 20230629, '82': 20230630, 
    '83': 20230703, '84': 20230705, '85': 20230706, '86': 20230707, 
    '87': 20230710, '88': 20230713, 
    '89': 20230717, '90': 20230719, '91': 20230720, '92': 20230721, 
    '93': 20230724, '94': 20230725, '95': 20230726, '96': 20230727, '97': 20230728, 
    '98': 20230731, '99': 20230801, '100': 20230803, 
    '101': 20230807, '102': 20230808, '103': 20230811, 
    '104': 20230814, '105': 20230817, '106': 20230818, 
    '107': 20230821, '108': 20230822, '109': 20230824, '110': 20230825, 
    '111': 20230828, '112': 20230829, '113': 20230831, '114': 20230901, 
    '115': 20230904, '116': 20230905, '117': 20230906, '118': 20230908, 
    '119': 20230911,
}

day = {
    '0': 0, 
    '1': 5, '2': 10, 
    '3': 13, '4': 18, '5': 24, '6': 30, 
    '7': 36, '8': 40, '9': 42, '10': 46, '11': 48, 
    '12': 53, 
    '13': 55, '14': 58, '15': 62, 
    '16': 63, '17': 66, '18': 68, 
    '19': 70, '20': 71, '21': 74, 
    '22': 81, 
    '23': 86, 
    '24': 88, '25': 90, '26': 92, '27': 94, 
    '28': 96, '29': 98, '30': 100, '31': 101, 
    '32': 103, 
    '33': 106, '34': 110, '35': 114, 
    '36': 118, '37': 120, '38': 122, 
    '39': 124, '40': 128, 
    '41': 132, '42': 134, '43': 136, 
    '44': 138, '45': 140, 
    '46': 142, 
    '47': 143, '48': 146, '49': 149, 
    '50': 152, '51': 154, '52': 158, '53': 161, 
    '54': 165, '55': 168, '56': 173, 
    '57': 175, '58': 177, '59': 179, '60': 182, 
    '61': 184, '62': 189, '63': 195, 
    '64': 197, '65': 202, 
    '66': 205, '67': 208, '68': 210, '69': 212, '70': 215, 
    '71': 217, '72': 221, 
    '73': 223, '74': 227, '75': 229, 
    '76': 232, '77': 238, 
    '78': 241, '79': 245, '80': 251, '81': 255, '82': 259, 
    '83': 262, '84': 267, '85': 268, '86': 273, 
    '87': 277, '88': 281, 
    '89': 283, '90': 285, '91': 288, '92': 292, 
    '93': 295, '94': 298, '95': 300, '96': 304, '97': 307, 
    '98': 310, '99': 312, '100': 314, 
    '101': 315, '102': 317, '103': 321, 
    '104': 323, '105': 327, '106': 329, 
    '107': 332, '108': 335, '109': 338, '110': 342, 
    '111': 345, '112': 349, '113': 353, '114': 357, 
    '115': 360, '116': 363, '117': 366, '118': 369, 
    '119': 372,
}


