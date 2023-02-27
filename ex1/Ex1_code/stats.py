# Name: Gili Gutfeld, ID: 209284512

'''
This file should be runnable to print map_statistics using 
$ python stats.py
'''
import collections
from collections import namedtuple
from ways import load_map_from_csv


# return the number of junctions in the map
def number_of_junctions(roads):
    return len(roads.keys())


# return the link list and the max/min/avg number of links per junction
def links_data(roads):

    max_link_count = 0
    min_link_count = 400
    link_list = []

    # check each junction
    for junction in roads.junctions():
        neighbor_count = 0
        neighbor_list = junction[3]

        for link in neighbor_list:
            neighbor_count = neighbor_count + 1
            link_list.append(link)

        # update the min/max count
        if neighbor_count > max_link_count:
            max_link_count = neighbor_count
        if neighbor_count < min_link_count:
            min_link_count = neighbor_count

    avg_link_count = len(link_list) / number_of_junctions(roads)
    return link_list, max_link_count, min_link_count, avg_link_count


# return the max/min/avg distance in the links
def link_distances(links):

    min_distance = links[0][2]
    max_distance = min_distance
    sum_distances = 0

    # check each link
    for link in links:
        dist = link[2]
        sum_distances += dist

        # update the min/max distance
        if dist < min_distance:
            min_distance = dist
        if dist > max_distance:
            max_distance = dist

    avg_distance = sum_distances / len(links)
    return max_distance, min_distance, avg_distance


# return dictionary of the number of road types
def roads_type(links):
    road_type_list = []
    for link in links:
        road_type_list.append(link[3])

    # make a dictionary from the list
    return collections.Counter(road_type_list)


def map_statistics(roads):
    '''return a dictionary containing the desired information
    You can edit this function as you wish'''

    junction_count = number_of_junctions(roads)
    link_list, max_link_count, min_link_count, avg_link_count = links_data(roads)
    max_dist, min_dist, avg_dist = link_distances(link_list)
    link_type_histogram = roads_type(link_list)
    Stat = namedtuple('Stat', ['max', 'min', 'avg'])

    return {
        'Number of junctions' : junction_count,
        'Number of links' : len(link_list),
        'Outgoing branching factor' : Stat(max=max_link_count, min=min_link_count, avg=avg_link_count),
        'Link distance' : Stat(max=max_dist, min=min_dist, avg=avg_dist),
        # value should be a dictionary
        # mapping each road_info.TYPE to the no' of links of this type
        'Link type histogram' : link_type_histogram,  # tip: use collections.Counter
    }


def print_stats():
    for k, v in map_statistics(load_map_from_csv()).items():
        print('{}: {}'.format(k, v))

        
if __name__ == '__main__':
    from sys import argv
    assert len(argv) == 1
    print_stats()

