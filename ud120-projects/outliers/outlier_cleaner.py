#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here

    # temp = []
    # for prediction,net_worth in zip(predictions,net_worths):
    #     temp.append(abs(prediction - net_worth))
    # temp = sorted(temp)[-9]
    # # print temp
    #
    # for prediction ,age, net_worth in zip(predictions,ages,net_worths):
    #     error = abs(prediction - net_worth)
    #     if error < temp:
    #         cleaned_data.append((age,net_worth,error))
    # return cleaned_data

    for prediction ,age, net_worth in zip(predictions,ages,net_worths):
        cleaned_data.append((age,net_worth,abs(net_worth-prediction)))

    cleaned_data = sorted(cleaned_data,key=lambda data:data[2])
    return cleaned_data[:81]


