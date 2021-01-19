import os
import sys
import re
import pandas as pd
from tabulate import tabulate

class Labels():
    def __init__(self):
        self.labels_path = "labels"


    # ============================================================
    def find_labels(self):
        if os.path.exists(self.labels_path + "/Insert your labels csv file in this dir.txt"):
            os.remove(self.labels_path + "/Insert your labels csv file in this dir.txt")

        files = os.listdir(self.labels_path)
        if len(files) == 1:
            file = files[0]

        elif len(files) > 1:
            print("=" * 70)
            for n, i in enumerate(files):
                print("[{}]: {}".format(n+1, i))
            print("=" * 70)
            inp = int(input("There are multiple files in this directory. "
                            "(Choose file with number between '1' and '{}'): ".format(len(files))))
            file = files[inp-1]

        else:
            print("=" * 70 + "\n[1]: Create label file yourself\n"
                           "[2]: Exit program\n" + "=" * 70)
            inp = input(("No labels file found in this directory. Type either '1' or '2' (default = '1'): "))
            if inp == "2":
                sys.exit(1)
            else:
                file, column_name = self.create_labels()
                return file, column_name

        if file == "labels_generated.csv":
            column_name = "label"
            return file, column_name

        df = pd.read_csv(self.labels_path + "/" + file)
        print("\n" + tabulate(df[:3], headers='keys', tablefmt='psql', showindex=False))
        inp = int(input(
            "Choose the column with the label names (Type number between '1' and '{}'): ".format(df.shape[1])))
        col = list(df.columns)
        column_name = col[inp-1]
        return file, column_name


    # ============================================================
    def create_labels(self):
        def sorted_nicely(l):
            """ Sort the given iterable in the way that humans expect."""
            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
            return sorted(l, key=alphanum_key)

        path_data = "training_data"
        data = os.listdir(path_data)
        data = sorted_nicely(data)
        labels = []
        print()
        for n, folder in enumerate(data):
            inp = input("[{}/{}]: Choose label for folder '{}' "
                                "(if left blank the label name will be the folder name): ".format(n+1, len(data), folder))

            if inp == "":
                labels.append(folder)
            else:
                labels.append(inp)

        df = {"label": labels, "folder_name": data}
        df = pd.DataFrame(df)
        print("\n" + tabulate(df, headers='keys', tablefmt='psql', showindex=False))
        print("\n" + "=" * 70 + "\n[1]: Labels are correct. Continue with training\n[2]: Exit program\n" + "=" * 70)
        inp = input("Choose option '1' or '2' (default: '1'): ")
        if inp == "2":
            sys.exit(1)

        file = "labels_generated.csv"
        column_name = "label"
        df.to_csv(self.labels_path + "/" + file)
        return file, column_name


    # ============================================================
    def initialize(self):
        file_name, column_name = self.find_labels()
        return file_name, column_name
