# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 11:34:01 2021

@author: Kasia Kedziora
"""

import importlib
import sys
import re

import pandas as pd
from pandas.core.base import NoNewAttributesMixin
import numpy as np
from skimage import measure

fov_f = importlib.import_module('ring_functions')

def update_dataFrame(channel_list,my_labels,df,current_frame,active_label,object_properties,flag_list):
    
    '''
    Function to use viewer data to modify data frame with all data (for a specific object in a specific frame)
    
    input:
        channel_list
        my_labels - sent as a layer from the viewer
        df
        current_frame
        active_label
    
    output:
       df 
    '''
    
    # create intensity image
    signal_image = create_intensityImage(channel_list,current_frame)

    # create mask with only a selected object
    single_label_im = create_singleLabel(my_labels,current_frame,active_label)
    
    # characterize new nucleus
    cellData = characterize_newNucleus(single_label_im,signal_image,object_properties)
    
    # create ring image
    x = int(cellData['centroid-0'])
    y = int(cellData['centroid-1'])
    single_label_ring = make_ringImage(single_label_im,x,y,imSize=200)
    

    # measure properties of the ring
    ringData = characterize_newRing(single_label_ring,signal_image)
    
    # put data frames together
    labels_set = np.unique(my_labels[current_frame,:,:])
    df = mod_dataFrame(df,cellData,ringData,current_frame,labels_set,flag_list)
    
    return df

def create_singleLabel(my_labels,current_frame,active_label):
    
    '''
    Function to create a label image containing only a single cell
    
    input:
        my_labels
        current_frame
        active_label
    
    output:
       single_label_im 
    '''
    
    # create mask with only a selected object
    single_label_im = my_labels[current_frame,:,:].copy()
    single_label_im[single_label_im != active_label]=0
    
    return single_label_im
    
def create_intensityImage(channel_list,current_frame):

    '''
    Function to create intensity image for calculation for a single object.
    This has original size
    
    input:
        channel_list
        current_frame
        active_label
    
    output:
       signal_image 
    '''
    
    im_size_x = channel_list[0]['image'].shape[1]
    im_size_y = channel_list[0]['image'].shape[2]
    
    signal_image = np.zeros([im_size_x,im_size_y,len(channel_list)]).astype('uint16')
 
    for ch in channel_list:
        
        signal_image[:,:,ch['channel_number']] = ch['image'][current_frame,:,:] 
    
    return signal_image
    
def characterize_newNucleus(single_label_im,signal_image,object_properties):
    
    '''
    Function to get properties of a single cell
    
    input:
        single_label_im
        signal_image
        properties
    
    output:
        cellData - data frame with regionprops of a single object    
    '''
    
    # find features of the new object
    
    cellData = measure.regionprops_table(single_label_im, properties=object_properties,intensity_image=signal_image)
    
    cellData = pd.DataFrame(cellData)
    
    return cellData

def make_ringImage(single_label_im,x,y,imSize=200):
    
    '''
    Function to get properties of a single cell
    
    input:
        single_label_im
    
    output:
        single_label_ring  
    '''
    
    myFrame = int(imSize/2)
    
    # cut small image
    small_im = single_label_im[x-myFrame:x+myFrame,y-myFrame:y+myFrame]
    
    # change small image into a ring
    rings = fov_f.make_rings(small_im,width=6,gap=1)
    
    # put small rings image back into the whole frame
    single_label_ring = single_label_im.copy()
    single_label_ring[x-myFrame:x+myFrame,y-myFrame:y+myFrame]=rings
    
    return single_label_ring
    
def characterize_newRing(single_label_ring,signal_image):
    
    '''
    Function to get properties of a single cell
    
    input:
        single_label_im
        signal_image
    
    output:
        cellData - data frame with regionprops of a single object    
    '''
    # define properties to calculate
    properties_ring = ['label','mean_intensity']
    
    # find features of the new object
    ringData = measure.regionprops_table(single_label_ring, properties=properties_ring,intensity_image=signal_image)
    ringData = pd.DataFrame(ringData)
    
    return ringData

def mod_dataFrame(df,cellData,ringData,current_frame,labels_set,flag_list):
    
    '''
    function to modify gneral data frame with updated modified single object data
    
    input:
        df - original general data frame
        cellData
        ringData
        current_frame
        labels_set - set of labels present in the current frame
        
    output:
        df - modified general data frame
    '''
    
    # check which cell it is
    active_label = list(cellData['label'])[0]

    # put nucleus and ring data together
    cellData = pd.merge(cellData,ringData,how='inner',on='label',suffixes=('_nuc', '_ring'))
    
    # add aditional info
    cellData['t'] = current_frame
    cellData['track_id'] = active_label

    # add necessary tags
    for flag in flag_list:
        col = flag['flag_column']
        cellData[col] = False
    
    # collect information about this label and this time point to calculate 
    info_track = df.loc[:,['track_id','parent','root','generation','accepted','promise','rejected']].drop_duplicates()
    
    
    # merge it to the data of this frame
    cellData = cellData.merge(info_track,on='track_id',how='left')   
    
    # take care of the totally new tracks
    if (cellData.loc[0,'parent'] == cellData.loc[0,'parent']):
        pass
    else:
        cellData.parent = cellData.track_id
        cellData.generation = 0
        cellData.root = cellData.track_id
    
       
    # swap in the general data frame
    curr_df = df.loc[df.t==current_frame,:]
    
    drop_modified = (curr_df.track_id==active_label)
    
    # close overlaping objects
    #drop_overlaping_neighbours = ((abs(df['centroid-0']-cellData['centroid-0'][0])<10) & (abs(df['centroid-1']-cellData['centroid-1'][0])<10))
    
    # objects that were removed
    drop_missing = [not(x in labels_set) for x in curr_df.track_id]
    
    what_to_drop = (drop_modified | drop_missing)
    
    curr_df.drop(curr_df[what_to_drop].index,axis=0,inplace=True)
    curr_df = curr_df.append(cellData,ignore_index=True)
    
    # drop current frame 
    df.drop(df[df.t==current_frame].index,axis=0,inplace=True)
    df = df.append(curr_df,ignore_index=True)
    
    return df

def mod_trackLayer(data,properties,df,current_frame,active_label):
    
    '''
    function to modify tracking layer for the viewer
    
    input:
        data
        properties
        df
        current_frame
        active_label
        
    output:
        data
        properties
    '''
    # choose the data for the specific object
    selData = df.loc[((df.t == current_frame) & (df.track_id == active_label)),:]
    
    # prepare in the right format
    frameData = np.array(selData.loc[:,['label','t','centroid-0','centroid-1']])
    
    # find position of this cell in the tracking data structure
    changeIndex = ((data[:,1]==current_frame) & (data[:,0]==active_label))
    
    # change data
    data = np.delete(data,changeIndex,axis=0)
    data = np.vstack([data, frameData])
    
    # modify properties of the track layer

    selData.loc[:,'state'] = 5
    
    for tProp in properties.keys():
    
        properties[tProp] = np.delete(properties[tProp],changeIndex)
        properties[tProp] = np.append(properties[tProp], selData[tProp])
    
    return data, properties

def newTrack_number(vector):
    
    '''
    Function to find the smallest unused number for a track that can be used
    
    input:
        
        vector - array like with numbers used for tracks
        
    output:
        
        newTrack - number to be used for a new track
    
    '''
    # find number of independent tracks
    tracksSetLength = len(set(vector))
    
    # find maximum track number
    trackMax = np.max(vector)
    
    # check if all are used
    if (trackMax >= (tracksSetLength+1)):
        
        unusedTracks = set(vector).symmetric_difference(np.arange(trackMax+1))
        unusedTracks = np.array(list(unusedTracks))
        unusedTracks = unusedTracks[unusedTracks>0][0]
        newTrack = np.nanmin(unusedTracks)
        
    else:
        newTrack = trackMax + 1 
    
    return newTrack

def trackData_from_df(df,col_list=['promise'],create_graph = True):
    
    '''
    Function to extract tracking data from a data frame
    
    input:
        df - sorted
        create_graph - toggle if graph is needed
    
    output:
        data
        properties

    '''

    #############################################
    # prepare data
    #############################################
    
    # avoid objects without tracking data
    exist_vector = (df['track_id']==df['track_id'])
    
    # select only objects that have specific labels
    sel_vector = False*len(df)
    
    for i in range(len(col_list)):
 
        sel_vector = sel_vector | df[col_list[i]].astype('bool')

    selVector = exist_vector & sel_vector
    
    #gather data in a form of numpy array
    data = np.array(df.loc[selVector,['track_id','t','centroid-0','centroid-1']])
    
    # change format of tracks id
    data[:,0]=data[:,0].astype(int)

    if len(data)>0:

        #############################################
        # prepare properties
        #############################################
        # specify columns to extract properties
        properties = {}
        prop_prop = ['t', 'generation', 'root', 'parent']
        
        for tProp in prop_prop:
        
            properties[tProp] = df.loc[selVector,tProp]
        
        properties['state'] = [5]*len(properties['t'])
        
        #############################################
        # prepare graph
        #############################################
        if create_graph:
            graph = df.loc[(~(df.track_id == df.parent) & selVector),['track_id','parent']].drop_duplicates().to_numpy()
            
            graph = graph.astype(int)
            graph = dict(graph)
        else:
            graph = {}
    else:
        # create a dummy in case no data for this layer
        data = np.array([[0,0,0,0],[0,1,0,0]])
        properties = {'t':[0,1], 'generation':[0,0], 'root':[0,0], 'parent':[0,0], 'state':[5,5]}
        graph = {}

    return data,properties,graph

def labels_from_df(cell_data_all):
    
    '''
    Function to create labels based on the df
    
    input:
        df
    
    output:
        labels

    '''
    
    max_frame = int(np.max(cell_data_all.t))
    row_total = int(cell_data_all.size_x[0])
    column_total = int(cell_data_all.size_y[0])
    
    labels = []
    
    for i in np.arange(max_frame + 1):
    
        # choose data from this frame
        sel_data = cell_data_all.loc[cell_data_all.t==i,:]
    
        # create an empty image
        label_image = np.zeros([row_total,column_total]).astype('uint16')
    
        # add objects
        for ind,my_cell in sel_data.iterrows():
            
            if (my_cell.label == my_cell.label): #if it's a real object
    
                min_row = int(my_cell['bbox-0'])
                max_row = int(my_cell['bbox-2'])
                min_col = int(my_cell['bbox-1'])
                max_col = int(my_cell['bbox-3'])
        
                label_image[min_row:max_row,min_col:max_col]=label_image[min_row:max_row,min_col:max_col]+(my_cell.image*my_cell.track_id)
                                   
        labels.append(label_image)
    
    labels = np.array(labels)

    
    return labels

def tags_from_df(df,tag_list):
    
    '''
    Function to extract data for tags from df
    input:
        df
        tag_list
    output:
        tag_list
    '''
    
    tag_data = []
    
    for tag_column in [x['tag_column'] for x in tag_list]: 
        
        # select points for a given tag
        sel_data = df.loc[df[tag_column] == True,:]
        
        # create tag data
        tag_points = np.array([sel_data['t'],sel_data['centroid-0'],sel_data['centroid-1']]).T
        
        tag_data.append(tag_points)
        
    return tag_data
    
def find_all_paths(graph, node, path=[]):
    
    '''
    Function to find all the paths coming through a node in a graph 
    
    input:
        graph
        node
    output:
        list of paths
    '''
    
    path = path + [node]
    paths = [path]
    
    offspring_list = []
    for key, value in graph.items():   # iter on both keys and values
            if (value == [node]):
                offspring_list.append(key)
    
    for node in offspring_list:
        newpaths = find_all_paths(graph, node, path)
        for newpath in newpaths:
            paths.append(newpath)
            
    return paths

def forward_labels(my_labels,df,current_frame,active_label,newTrack):
    
    '''
    Function to modify labels layer.
    input:
        my_labels
        df
        current_frame
        active_label
        newTrack
    output:
        my_labels
    '''

    for myInd in df.index[(df.track_id==active_label) & (df.t>=current_frame)]:
        
        row_start = df.loc[myInd,'bbox-0']
        row_stop = df.loc[myInd,'bbox-2']
        column_start = df.loc[myInd,'bbox-1']
        column_stop = df.loc[myInd,'bbox-3']
        
        if np.isnan(row_start and row_stop and column_start and column_stop):
            
            pass
        
        else:
            
            myFrame = int(df.loc[myInd,'t'])
    
            # cut and replace
            temp = my_labels[myFrame,int(row_start):int(row_stop),int(column_start):int(column_stop)]
            temp[temp == active_label] = int(newTrack)
            my_labels[myFrame,int(row_start):int(row_stop),int(column_start):int(column_stop)] = temp
    
    return my_labels

def forward_df(df,current_frame,active_label,newTrack,connectTo=0):
    
    '''
    Function to modify forward data frame structure after linking changes
    input:
        df
        current_frame
        active_label
        newTrack
        graph
    output:
        df
    '''
    
    # find info about the cut track
    active_label_generation = list(df.loc[df.track_id==active_label,'generation'].drop_duplicates())[0]
    
    # find info about the new label
    genList = list(df.loc[df.track_id==newTrack,'generation'].drop_duplicates())
    if len(genList)>0:
        new_generation = genList[0]
        new_root = list(df.loc[df.track_id==newTrack,'root'].drop_duplicates())[0]
        new_parent = list(df.loc[df.track_id==newTrack,'parent'].drop_duplicates())[0]
        
    else: # so this is a completely new number for a track
        
        if connectTo == 0: # and nothing to connect to
            
            new_generation = 0
            new_root = newTrack
            new_parent = newTrack
        
        else: # check data of a track we connect to
        
            new_generation = list(df.loc[df.track_id==connectTo,'generation'].drop_duplicates())[0] + 1
            new_root = list(df.loc[df.track_id==connectTo,'root'].drop_duplicates())[0]
            new_parent = connectTo
            
    
    # get a graph
    data,properties,graph = trackData_from_df(df,col_list=['t'])
    
    # find kids
    kids_list = []
    for key, value in graph.items():   # iter on both keys and values
            if (value == [active_label]):
                kids_list.append(key)
    
    # find all family members
    all_paths = find_all_paths(graph, active_label)
    family_members = [item for sublist in all_paths for item in sublist]
    
    for myDescendant in family_members:
        
        # find which rows need to be changed
        changeIndex = (df.t>=current_frame) & (df.track_id==myDescendant)
        
        df.loc[changeIndex,'root'] = new_root
        df.loc[changeIndex,'generation'] = df.loc[changeIndex,'generation'] - active_label_generation + new_generation
        
        if(myDescendant == active_label):
        
            df.loc[changeIndex,'track_id'] = newTrack
            df.loc[changeIndex,'parent'] = new_parent
              
        elif (myDescendant in kids_list): #2nd generation
            
            df.loc[changeIndex,'parent'] = newTrack
            
    return df
  
def extract_graph_data(graph_list,df_sel):
    
    '''
    Function to translate input file info to signals for plotting.
    '''
 
    results_list = []   
 
    key_words = ['nuc','ring']
    
    for graph in graph_list:
        
        function = graph['function']

        if function=='family':

            function_value = np.zeros([len(df_sel),1])

        else:
            request_list = []
            replacement_list = []
        
            for key_word in key_words:
        
                signal_list = [x.end() for x in re.finditer(f'{key_word}_',function)]
        
                for signal in signal_list:
                    
                    # for which channel it's requested
                    ch_number = eval(function[signal])  
                    
                    # get a column name
                    col = f'mean_intensity-{ch_number}_{key_word}'
        
                    # get data
                    request_list.append(f'{key_word}_{function[signal]}')
                    replacement_list.append(f"df_sel['{col}']")


                    
            # translate the function
            for request_signal,replacement_name in zip(request_list,replacement_list): 
        
                function = function.replace(request_signal,replacement_name)
                
            # evaluate the function
            function_value = eval(function)
  
 
        # collect results
        results_list.append(function_value)
 
    return results_list

def calculate_graph_offset(df,current_track):

    
    sel_t = df.loc[df.track_id == current_track,'t']
    
    graph_offset = np.min(sel_t)
    
    return graph_offset

def find_empty_frames(t):

    '''
    Function to find empty frames in a time series.
    '''
     
    t_min = np.min(t)
    t_max = np.max(t)
    
    empty_frames_list = list(set(np.arange(t_min,t_max+1)) - set(t))
    empty_frames_list.sort()

    return empty_frames_list