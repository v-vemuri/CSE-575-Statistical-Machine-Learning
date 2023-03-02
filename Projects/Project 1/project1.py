import numpy
import scipy.io
import math
import geneNewData

def main():
    myID='0885'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    print([len(train0),len(train1),len(test0),len(test1)])
    print('Your trainset and testset are generated successfully!')

    #task1
    #digit0_train
    #feature1
    avg_brightness_train_digit0=[]
    for i in range (len(train0)):
        avg_brightness_train_digit0.append(numpy.mean(train0[i]))

    #feature2
    std_dev_train_digit0=[] 
    for i in range (len(train0)):
        std_dev_train_digit0.append(numpy.std(train0[i]))
    
    #digit0_test
    #feature1
    avg_brightness_test_digit0=[]
    for i in range (len(test0)):
        avg_brightness_test_digit0.append(numpy.mean(test0[i]))

    #feature2
    std_dev_test_digit0=[]
    for i in range (len(test0)):
        std_dev_test_digit0.append(numpy.std(test0[i]))
    
    #digit1_train
    #feature1
    avg_brightness_train_digit1=[]
    for i in range (len(train1)):
        avg_brightness_train_digit1.append(numpy.mean(train1[i]))

    #feature2
    std_dev_train_digit1=[]
    for i in range (len(train1)):
        std_dev_train_digit1.append(numpy.std(train1[i]))  
    
    #digit1_test
    #feature1
    avg_brightness_test_digit1=[]
    for i in range (len(test1)):
        avg_brightness_test_digit1.append(numpy.mean(test1[i]))

    #feature2
    std_dev_test_digit1=[]
    for i in range (len(test1)):
        std_dev_test_digit1.append(numpy.std(test1[i])) 
    
    #task2
    mean_feature1_digit0=numpy.mean(avg_brightness_train_digit0)
    print('1: Mean_of_feature1_for_digit0:', mean_feature1_digit0)

    sum=0
    for i in range(len(avg_brightness_train_digit0)):
        sum=sum+pow((avg_brightness_train_digit0[i]-mean_feature1_digit0),2)
    variance_feature1_digit0=sum/len(avg_brightness_train_digit0)
    print('2: Variance_of_feature1_for_digit0:', variance_feature1_digit0)
 
    mean_feature2_digit0=numpy.mean(std_dev_train_digit0)
    print('3: Mean_of_feature2_for_digit0:', mean_feature2_digit0)

    sum=0
    for i in range(len(std_dev_train_digit0)):
        sum=sum+pow((std_dev_train_digit0[i]-mean_feature2_digit0),2)
    variance_feature2_digit0=sum/len(std_dev_train_digit0)
    print('4: Variance_of_feature2_for_digit0:', variance_feature2_digit0)
    
    mean_feature1_digit1=numpy.mean(avg_brightness_train_digit1)
    print('5: Mean_of_feature1_for_digit1:', mean_feature1_digit1)

    sum=0
    for i in range(len(avg_brightness_train_digit1)):
        sum=sum+pow((avg_brightness_train_digit1[i]-mean_feature1_digit1),2)
    variance_feature1_digit1=sum/len(avg_brightness_train_digit1)
    print('6: Variance_of_feature1_for_digit1:', variance_feature1_digit1)
    
    mean_feature2_digit1=numpy.mean(std_dev_train_digit1)
    print('7: Mean_of_feature2_for_digit1:', mean_feature2_digit1)

    sum=0
    for i in range(len(std_dev_train_digit1)):
        sum=sum+pow((std_dev_train_digit1[i]-mean_feature2_digit1),2)
    variance_feature2_digit1=sum/len(std_dev_train_digit1)
    print('8: Variance_of_feature2_for_digit1:', variance_feature2_digit1)   
    
    #task3 & task4 
    ## test 0 - 980 images
    # p(xi/y) = 1/sqrt(2*pie*var)*exp(-((xi testdata-u train))^2)/2var train)

    test0_predicted_0 = []
    test0_predicted_1 = []

    for i in range(len(test0)):
        #digit 0
        exp_val = math.exp(-(math.pow(avg_brightness_test_digit0[i] - mean_feature1_digit0, 2)/ (2 * variance_feature1_digit0)))
        sqrt_val = 1/(math.sqrt(2*math.pi*variance_feature1_digit0)) 
        prob_feature_1_if_digit_0 = sqrt_val * exp_val

        exp_val = math.exp(-(math.pow(std_dev_test_digit0[i] - mean_feature2_digit0, 2)/ (2 * variance_feature2_digit0)))
        sqrt_val = 1/(math.sqrt(2*math.pi*variance_feature2_digit0)) 
        prob_feature_2_if_digit_0 = sqrt_val * exp_val

        total_probability_if_digit_predicted_0 = prob_feature_1_if_digit_0 * prob_feature_2_if_digit_0

        #digit1               
        exp_val = math.exp(-(math.pow(avg_brightness_test_digit0[i] - mean_feature1_digit1, 2)/ (2 * variance_feature1_digit1)))
        sqrt_val = 1/(math.sqrt(2*math.pi*variance_feature1_digit1)) 
        prob_feature_1_if_digit_1 = sqrt_val * exp_val   

        exp_val = math.exp(-(math.pow(std_dev_test_digit0[i] - mean_feature2_digit1, 2)/ (2 * variance_feature2_digit1)))
        sqrt_val = 1/(math.sqrt(2*math.pi*variance_feature2_digit1)) 
        prob_feature_2_if_digit_1 = sqrt_val * exp_val

        total_probability_if_digit_predicted_1 = prob_feature_1_if_digit_1 * prob_feature_2_if_digit_1       

        if total_probability_if_digit_predicted_0 > total_probability_if_digit_predicted_1:
            test0_predicted_0.append(test0[i])
        else:
            test0_predicted_1.append(test0[i])

    print('Test0 predicted as 0',len(test0_predicted_0))    
    print('Test0 predicted as 1',len(test0_predicted_1))    

    accuracy_test0 = len(test0_predicted_0)/len(test0)

    print('Accuracy_for_digit0testset', accuracy_test0)  
    
    ## test 1 - 1135 images
    # p(xi/y) = 1/sqrt(2*pie*var)*exp(-((xi testdata-u train))^2)/2var train)

    test1_predicted_0 = []
    test1_predicted_1 = []

    for i in range(len(test1)):
        #digit 0
        exp_val = math.exp(-(math.pow(avg_brightness_test_digit1[i] - mean_feature1_digit0, 2)/ (2 * variance_feature1_digit0)))
        sqrt_val = 1/(math.sqrt(2*math.pi*variance_feature1_digit0)) 
        prob_feature_1_if_digit_1 = sqrt_val * exp_val

        exp_val = math.exp(-(math.pow(std_dev_test_digit1[i] - mean_feature2_digit0, 2)/ (2 * variance_feature2_digit0)))
        sqrt_val = 1/(math.sqrt(2*math.pi*variance_feature2_digit0)) 
        prob_feature_2_if_digit_1 = sqrt_val * exp_val

        total_probability_if_digit_predicted_0 = prob_feature_1_if_digit_1 * prob_feature_2_if_digit_1

        #digit1               
        exp_val = math.exp(-(math.pow(avg_brightness_test_digit1[i] - mean_feature1_digit1, 2)/ (2 * variance_feature1_digit1)))
        sqrt_val = 1/(math.sqrt(2*math.pi*variance_feature1_digit1)) 
        prob_feature_1_if_digit_1 = sqrt_val * exp_val   

        exp_val = math.exp(-(math.pow(std_dev_test_digit1[i] - mean_feature2_digit1, 2)/ (2 * variance_feature2_digit1)))
        sqrt_val = 1/(math.sqrt(2*math.pi*variance_feature2_digit1)) 
        prob_feature_2_if_digit_1 = sqrt_val * exp_val

        total_probability_if_digit_predicted_1 = prob_feature_1_if_digit_1 * prob_feature_2_if_digit_1       

        if(total_probability_if_digit_predicted_0 > total_probability_if_digit_predicted_1):
            test1_predicted_0.append(test1[i])
        else:
            test1_predicted_1.append(test1[i])

    print('Test1 predicted as 0',len(test1_predicted_0))    
    print('Test1 predicted as 1',len(test1_predicted_1))      

    accuracy_test1 = len(test1_predicted_1)/len(test1)

    print('Accuracy_for_digit1testset', accuracy_test1)  

    
if __name__ == '__main__':
    main()