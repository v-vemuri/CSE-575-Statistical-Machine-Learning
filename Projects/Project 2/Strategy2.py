from Precode2 import *
import numpy
import random 
import math
import matplotlib.pyplot as plt


def calculating_centroids(centroids_list,no_of_clusters,data):
    while len(centroids_list) != no_of_clusters:
        d = []
        for i in range (len(data)):
#             print('datapoint', i)
            data_point_distance_list = []
            for cent_ind, cetroid in enumerate(centroids_list):
#                 print('----centroid----', cent_ind)
                distance = math.sqrt(math.pow(cetroid[0]-data[i][0],2)+math.pow(cetroid[1]-data[i][1],2))
                data_point_distance_list.append(distance)
#             print('data_point_distance_list', data_point_distance_list)
            avg_distance = np.mean(data_point_distance_list)
#             print('avg_distance', avg_distance)
            d.append(avg_distance)  
        
        max_value_index = d.index(max(d))
        next_centroid = list(data[max_value_index])
        print('next_centroid', next_centroid)
        centroids_list.insert(0, next_centroid)
        data.pop(max_value_index)
        print('length******',len(data))
        print('centroids_list', centroids_list)
        
    return centroids_list

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

def k_means_clustering(no_of_clusters, i_point, data):

    i_point = [list(np_array) for np_array in i_point]
    print('length', len(i_point))

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
        list_of_all_clusters=[cluster1_list,cluster2_list,cluster3_list,cluster4_list,cluster5_list,cluster6_list,cluster7_list,cluster8_list,cluster9_list,cluster10_list]

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
    
    original_data = np.load('AllSamples.npy')
    data = original_data
    k1,i_point1,k2,i_point2 = initial_S2('0885')
    print(k1)
    print(i_point1)
    print(k2)
    print(i_point2)
    
#     uncomment this to test for k=2 to 10 and change the number of clusters according to the requirement 
#     random.seed(500)
#     random_centroid = random.choice(data)
#     no_of_clusters = 6
#     print('random_centroid', random_centroid)

    random_centroid = list(i_point2)
    no_of_clusters = k2
    data = [list(data_point) for data_point in data]
    centroids_list = [list(random_centroid)]
    print('first centroid', list(random_centroid))
    random_point_index = data.index(list(random_centroid))
    data.pop(random_point_index)
    print('newlen', len(data))

    centroids_list=calculating_centroids(centroids_list,no_of_clusters, data)
    print('centroids_list------', centroids_list)
    updated_mean, computed_loss, list_of_clusters = k_means_clustering(no_of_clusters, centroids_list, original_data)
    
    plot_results(data, list_of_clusters)

print('updated_mean', updated_mean)
print('computed_loss', computed_loss)