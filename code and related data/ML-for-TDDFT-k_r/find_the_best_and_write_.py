import pandas as pd
def cal_perf():
    dataset = pd.read_csv(r"./_/model_metrics.csv")
    print(type(dataset))
    print(type(dataset.shape))
    row_index= int()
    deR2= []
    for i in range(dataset.shape[0]):
        if i%13 == 10:
            deR2.append(float(dataset.iloc[i,11]))
    max_value = max(deR2)
    max_index = deR2.index(max_value)
    file = open('./_/best_perf.txt', 'w')

    print("DeR2:",float(dataset.iloc[max_index*13+10,11]),"+-",float(dataset.iloc[max_index*13+11,11]), file=file)
    print("RMSE:",float(dataset.iloc[max_index*13+10,12]),"+-",float(dataset.iloc[max_index*13+11,12]), file=file)

    print("MAE:",float(dataset.iloc[max_index*13+10,13]),"+-",float(dataset.iloc[max_index*13+11,13]), file=file)

    print("R2:",float(dataset.iloc[max_index*13+10,14]),"+-",float(dataset.iloc[max_index*13+11,14]), file=file)
    # print(i)
    print("rs number:", dataset.iloc[max_index*13+10,18], file=file)
    print("model type:", dataset.iloc[max_index*13+10,16], file=file)
    print("\n")
    file.close()

    print("DeR2:",float(dataset.iloc[max_index*13+10,11]),"+-",float(dataset.iloc[max_index*13+11,11]))
    print("RMSE:{:.3f}+-{:.3f}".format(float(dataset.iloc[max_index*13+10,12]),float(dataset.iloc[max_index*13+11,12])))

    print("MAE:{:.3f}+-{:.3f}".format(float(dataset.iloc[max_index*13+10,13]),float(dataset.iloc[max_index*13+11,13])))

    print("R2:{:.3f}+-{:.3f}".format(float(dataset.iloc[max_index*13+10,14]),float(dataset.iloc[max_index*13+11,14])))
    # print(i)
    print("rs number:", dataset.iloc[max_index*13+10,18])
    print("model type:", dataset.iloc[max_index*13+10,16])