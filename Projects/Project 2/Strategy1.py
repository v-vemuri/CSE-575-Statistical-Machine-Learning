from Precode import *
import numpy
import math
import matplotlib.pyplot as plt

def objective_function_loss(list_of_all_clusters, updated_mean):

    list_of_clusters = []
    for cluster in list_of_all_clusters:
        if len(cluster) > 0:
            list_of_clusters.append(cluster)

    cluster_wise_sum = []
    for index, cluster_list in enumerate(list_of_clusters):
        distances_sum = 0
        for j in range(len(cluster_list)):
            distance = math.pow((updated_mean[index][0] - cluster_list[j][0]),2)+math.pow((updated_mean[index][1]-cluster_list[j][1]),2)
            distances_sum = distances_sum + distance
        cluster_wise_sum.append(distances_sum)

    computed_loss = np.sum(cluster_wise_sum)
    print(computed_loss)
    return computed_loss

def k_means_clustering(no_of_clusters, i_point):

    i_point = [list(np_array) for np_array in i_point]

    updated_mean = [[] for i in range(len(i_point))]

    initial_mean = i_point

    iter_number = 0

    while(updated_mean != initial_mean):

        d = {}
        cluster1_list = []
        cluster2_list = []
        cluster3_list = []
        cluster4_list = []
        cluster5_list = []
        cluster6_list = []
        cluster7_list = []
        cluster8_list = []
        cluster9_list = []
        cluster10_list = []
        print('***************')

        initial_mean = i_point
        print('initial mean', initial_mean)
        for k in range (len(data)):
            for i in range (len(i_point)):
                distance=math.sqrt(math.pow((data[k][0]-i_point[i][0]),2)+math.pow((data[k][1]-i_point[i][1]),2))
                key = 'c_'+str(i+1)
                d[key] = distance
            cluster_assignment=min(d, key=d.get)

            list_of_clusters = []

            if '1' in cluster_assignment:
                cluster1_list.append(data[k])
            if '2' in cluster_assignment:
                cluster2_list.append(data[k])
            if '3' in cluster_assignment:
                cluster3_list.append(data[k])
            if '4' in cluster_assignment:
                cluster4_list.append(data[k])
            if '5' in cluster_assignment:
                cluster5_list.append(data[k])
            if '6' in cluster_assignment:
                cluster6_list.append(data[k])
            if '7' in cluster_assignment:
                cluster7_list.append(data[k])
            if '8' in cluster_assignment:
                cluster8_list.append(data[k])
            if '9' in cluster_assignment:
                cluster9_list.append(data[k])
            if '10' in cluster_assignment:
                cluster10_list.append(data[k])

        print('cluster1_list', len(cluster1_list))
        print('cluster2_list', len(cluster2_list))
        print('cluster3_list', len(cluster3_list))
        print('cluster4_list', len(cluster4_list))
        print('cluster5_list', len(cluster5_list))
        print('cluster6_list', len(cluster6_list))
        print('cluster7_list', len(cluster7_list))
        print('cluster8_list', len(cluster8_list))
        print('cluster9_list', len(cluster9_list))
        print('cluster10_list', len(cluster10_list))

        i_point=[]
        list_of_all_clusters=[cluster1_list,cluster2_list,cluster3_list,cluster4_list,cluster5_list,cluster6_list, cluster7_list, cluster8_list, cluster9_list, cluster10_list]

        for new_list in list_of_all_clusters:
            if len(new_list) > 0:
                list_of_clusters.append(new_list)

        for cluster_list in list_of_clusters:
            sum_x = 0
            sum_y = 0
            for i in range (len(cluster_list)):
                sum_x = sum_x+cluster_list[i][0]
                sum_y = sum_y+cluster_list[i][1]
            mean_x = sum_x/len(cluster_list)
            mean_y = sum_y/len(cluster_list)
            i_point.append([mean_x, mean_y])
        iter_number = iter_number + 1
        print(f'mean_after_{iter_number}_iteration is {i_point}')
        updated_mean = i_point
        print('updated_mean', updated_mean)

        computed_loss = objective_function_loss(list_of_all_clusters, updated_mean)

    
    return updated_mean, computed_loss, list_of_clusters

def plot_results(data, list_of_clusters):
    
    plt.plot(np.array(data)[:, 0], np.array(data)[:, 1], 'o', label = 'given_data')
    plt.legend()
    plt.show()
    for i in range(len(list_of_clusters)):
        plt.plot(np.array(list_of_clusters[i])[:,0], np.array(list_of_clusters[i])[:, 1], 'o', label = 'cluster'+ str(i+1))
    plt.legend()
    plt.show()
    return 

if __name__ == '__main__':
    data = np.load('AllSamples.npy')
    k1,i_point1,k2,i_point2 = initial_S1('0885')
    print(k1)
    print(i_point1)
    print(k2)
    print(i_point2)

#     uncomment this to test for k=2 to 10 and change the number of clusters according to the requirement 
#     i_point = []
#     no_of_clusters = 5
#     i_point = random.choices(data, k=no_of_clusters)

    no_of_clusters= k2
    i_point = i_point2
    updated_mean, computed_loss, list_of_clusters = k_means_clustering(no_of_clusters, np.array(i_point))
    plot_results(data, list_of_clusters)
print('updated_mean', updated_mean)
print('computed_loss', computed_loss)