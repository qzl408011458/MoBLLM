import os
import numpy as np


class Info_Station():
    def __init__(self):
        self.path_ = './HKstion-line/stNo_stName_line.csv'
        self.path_lineDIR = './HKstion-line/line-station/'
        self.__init_valuables()
        self.__init_matrix_adjacent()

    def __init_valuables(self):
        self.lone_st_list = ['Lok Ma Chau', 'Racecourse', 'LOHAS Park']
        self.lone_stNo_list = ['78', '70', '57']
        self.map_stNo_to_No = {}
        self.map_No_to_stNo = {}
        self.map_stNo_to_stName = {}
        self.map_stName_to_stNo = {}
        self.station_list = []
        self.sum_stations = 0
        fr = open(self.path_, 'r', encoding='utf-8')
        line = fr.readline().strip()
        no = 0
        while line:
            self.sum_stations += 1
            line = line.split(',')
            self.map_No_to_stNo[no] = line[0]
            self.map_stNo_to_No[line[0]] = no
            self.map_stNo_to_stName[line[0]] = line[1]
            self.map_stName_to_stNo[line[1]] = line[0]
            no += 1
            self.station_list.append(int(line[0]))
            line = fr.readline().strip()
        fr.close()
        self.dict_line_st = {}
        for filename in os.listdir(self.path_lineDIR):
            name, suf = filename.split('.')
            self.dict_line_st[name] = {}
            fr = open(self.path_lineDIR + filename, 'r', encoding='utf-8')
            line = fr.readline().strip()
            while line:
                line = line.split(',')
                self.dict_line_st[name][line[0]] = line[1]
                line = fr.readline().strip()
        #print(self.dict_line_st)

    def __init_matrix_adjacent(self):
        self.matrix_adjacent = np.zeros((self.sum_stations, self.sum_stations))
        # Init all the connected states between two neighbor station in each line
        for line in self.dict_line_st:
            list_no = []
            for st_no in self.dict_line_st[line]:
                if st_no in self.lone_stNo_list:
                    continue
                list_no.append(self.map_stNo_to_No[st_no])
            for i in range(len(list_no)):
                if i + 1 < len(list_no):
                    self.matrix_adjacent[list_no[i]][list_no[i + 1]] = 1
                    self.matrix_adjacent[list_no[i + 1]][list_no[i]] = 1
        # Correction
        # Lok Ma Chau to Sheung Shui
        loneSt_no = self.map_stNo_to_No[self.map_stName_to_stNo['Lok Ma Chau']]
        triSt_no = self.map_stNo_to_No[self.map_stName_to_stNo['Sheung Shui']]
        self.matrix_adjacent[loneSt_no][triSt_no] = 1
        self.matrix_adjacent[triSt_no][loneSt_no] = 1

        loneSt_no = self.map_stNo_to_No[self.map_stName_to_stNo['LOHAS Park']]
        triSt_no = self.map_stNo_to_No[self.map_stName_to_stNo['Tseung Kwan O']]
        self.matrix_adjacent[loneSt_no][triSt_no] = 1
        self.matrix_adjacent[triSt_no][loneSt_no] = 1

        loneSt_no = self.map_stNo_to_No[self.map_stName_to_stNo['Racecourse']]
        st1_no = self.map_stNo_to_No[self.map_stName_to_stNo['University']]
        st2_no = self.map_stNo_to_No[self.map_stName_to_stNo['Sha Tin']]
        self.matrix_adjacent[loneSt_no][st1_no] = 1
        self.matrix_adjacent[st1_no][loneSt_no] = 1
        self.matrix_adjacent[loneSt_no][st2_no] = 1
        self.matrix_adjacent[st2_no][loneSt_no] = 1

    def checkMatrixADJ(self, stName):
        no = self.map_stNo_to_No[self.map_stName_to_stNo[stName]]
        list_neighborSt = []
        for i in range(self.sum_stations):
            if self.matrix_adjacent[no][i] == 1:
                list_neighborSt.append(self.map_stNo_to_stName[self.map_No_to_stNo[i]])
        print(list_neighborSt)

if __name__ == '__main__':
    info_st = Info_Station()
    print(info_st.map_stNo_to_No)