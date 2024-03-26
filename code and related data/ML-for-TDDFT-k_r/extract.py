import os
import csv

# 获取工作路径
work_dir = os.getcwd()

# 创建CSV文件并写入表头
csv_file = open('output.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['direc_name', 'DeR2', 'RMSE', 'MAE', 'R2', 'rs number', 'model type'])

# 遍历文件夹
for folder in os.listdir(work_dir):
    folder_path = os.path.join(work_dir, folder)
    
    # 判断是否是文件夹
    if os.path.isdir(folder_path):
        # 获取best_perf.txt的路径
        file_path = os.path.join(folder_path, 'best_perf.txt')
        
        # 判断文件是否存在
        if os.path.exists(file_path):
            # 读取best_perf.txt的内容
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # 获取DeR2, RMSE, MAE, R2, rs number, model type的值
            der2 = lines[0].strip()
            rmse = lines[1].strip()
            mae = lines[2].strip()
            r2 = lines[3].strip()
            rs_number = lines[4].strip()
            model_type = lines[5].strip()
            
            # 写入CSV文件
            csv_writer.writerow([folder, der2, rmse, mae, r2, rs_number, model_type])

# 关闭CSV文件
csv_file.close()

# 提示信息
# print('文件夹内的best_perf.txt信息已成功写入output.csv文件中。')
print("complte")
