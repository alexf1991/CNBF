import numpy as np
import os
import csv

def csv_to_dict(path, data_dict=None):

    if data_dict == None:

        data_dict = {}

    with open(path, mode='r') as infile:

        reader = csv.reader(infile)

        first_row = True

        for rows in reader:

            if first_row:

                col_names = rows

                first_row = False

            col_ct = 0

            for name in col_names:

                try:

                    tmp_float = float(rows[col_ct])

                    is_integer = tmp_float.is_integer()

                    if is_integer:

                        data_dict[name] = int(tmp_float)

                    else:

                        data_dict[name] = tmp_float

                except:

                    data_dict[name] = rows[col_ct]

                col_ct += 1

    return data_dict





def write_params_csv(path, params):

    with open(os.path.join(path, 'model_params.csv'), 'w') as csvfile:

        header = [str(key) for key in params.keys()]

        wr = csv.DictWriter(csvfile, fieldnames=header)

        wr.writeheader()

        wr.writerow(params)

    return
