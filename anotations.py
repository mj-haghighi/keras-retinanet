"""
anotations utils
put it out side Dataset folder
"""
import glob
import os
import csv

def collect_all_anotations(
    base_dir,
    anots_regx,
    out_path,
    class_name,
    anot_ext = 'csv',
    image_ext ='jpg',
    split_path_char = '\\'
):

    
    anots_paths = sorted(glob.glob(os.path.join(base_dir, anots_regx)))
    
    with open(out_path, mode='w', newline='') as out_csv_file:
        writer = csv.writer(out_csv_file)
        for path in anots_paths:
            file_name = path.split(split_path_char)[-1]
            img_name = file_name.replace(anot_ext, image_ext)
            with open(path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    x, y, alpha = row
                    writer.writerow([img_name, x, y, alpha, class_name])


if __name__ == "__main__":
    collect_all_anotations(
        base_dir = 'Dateset\\Train',
        anots_regx='*.csv',
        out_path = 'Dateset\\anots.csv',
        class_name = 'saffron'
    )
