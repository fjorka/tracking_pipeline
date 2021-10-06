# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 13:27:49 2021

@author: Kasia Kedziora
"""

import napari
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix
import numpy as np

import general_functions as gen

def display_set(viewer,stack_labels,stack_im_list,channel_list,label_contour=0):
    
    '''
    Function to create or update a viewer
        
    input: 
        viewer_stack
        stack_labels
        stack_im_list
        channel_list
        label_contour=0
    output:
        -
    '''
    try:
        viewer.layers['Labels'].data = stack_labels  
    except KeyError:
        viewer.add_labels(stack_labels,name='Labels',opacity = 0.5)
        viewer.layers['Labels'].contour = label_contour

    ################################
    for ch,ch_stack in zip(channel_list,stack_im_list):
        
        ch_name = ch['channel_name']
        
        try:
            # if the layer exists, update the data
            viewer.layers[ch_name].data = ch_stack
        except KeyError:
            # otherwise add it to the viewer  
            viewer.add_image(ch_stack,colormap=ch['color'],name = ch_name,opacity=0.5,blending='additive')
    
    ###############################
    return viewer


def create_graph_widget(graph_list,df,current_track):
    
    # select appropriate data
    df_sel = df.loc[df.track_id == current_track,:]
    df_sel = df_sel.sort_values(by='t')
    results_list = gen.extract_graph_data(graph_list,df_sel)

    # create widget
    mpl_widget = FigureCanvas(Figure(tight_layout=True))

    ax_number = len(graph_list)
    static_ax = mpl_widget.figure.subplots(ax_number,1)

    print(type(static_ax))

    if type(static_ax) == np.ndarray:
        pass
    else:
        static_ax = [static_ax]

    # populate
    for i,graph in enumerate(graph_list):
        
        signal = results_list[i]
        
        # plot from list or a single series
        if type(signal) == list:
            
            for sub_signal in signal:
        
                static_ax[i].plot(df_sel.t,sub_signal,color=graph['color'])
        
        else:
                static_ax[i].plot(df_sel.t,signal,color=graph['color'])
            
        static_ax[i].set_title(graph['graph_name'],color='black')
        static_ax[i].tick_params(axis='x', colors='black')
        static_ax[i].tick_params(axis='y', colors='black')
        static_ax[i].grid(color='0.95')
        
    return mpl_widget

def cut_track(viewer,df,gen_track_columns):
    
    '''
    Function to cut a track at a given point.
    '''
    
    # get images of objects
    my_labels = viewer.layers['Labels'].data

    # get the position in time
    current_frame = viewer.dims.current_step[0]

    # get my label
    active_label = viewer.layers['Labels'].selected_label

    # find new track number
    newTrack = gen.newTrack_number(df.track_id)

    #####################################################################
    # change labels layer
    #####################################################################

    my_labels = gen.forward_labels(my_labels,df,current_frame,active_label,newTrack)    
    viewer.layers['Labels'].data = my_labels

    #####################################################################
    # modify data frame
    #####################################################################
    df = gen.forward_df(df,current_frame,active_label,newTrack)

    #####################################################################
    # remove tags from affected tracks
    #####################################################################
    viewer = remove_tags(viewer, df,[active_label,newTrack])

    #####################################################################
    # change tracking layer
    #####################################################################

    # modify the data for the layer
    data,properties,graph = gen.trackData_from_df(df,col_list=gen_track_columns)

    # change tracks layer
    viewer.layers['Tracking'].data = data
    viewer.layers['Tracking'].properties = properties
    viewer.layers['Tracking'].graph = graph

    #####################################################################
    # change viewer status
    #####################################################################
    viewer.status = f'Track {active_label} was cut at frame {current_frame}.' 
    
    return viewer,df

def merge_track(viewer,df,gen_track_columns):
    
    '''
    Function to merge a track with a chosen track or the closest track in the previous frame
    
    input:
        viewer
        df
    output:
        
    '''
    # get images of objects
    my_labels = viewer.layers['Labels'].data
    
    # get the position in time
    current_frame = viewer.dims.current_step[0]
    
    # get my label
    active_label = viewer.layers['Labels'].selected_label
    
    if current_frame>0:
        
        connTrack=0
        
        # check if there is a point to merge too
        merge_to = viewer.layers['Helper Points'].data
        
        if len(merge_to)==1:
            
            merge_to = merge_to[0]
            
            if merge_to[0] == (current_frame-1):
                
                connTrack = my_labels[tuple(merge_to.astype(int))]
                
                viewer.layers['Helper Points'].data = []
                
            else:
                viewer.status = 'Merging cell does not match'
        
        elif len(merge_to)==0:
    
            # find the closest object in the previous layer
            object_data = df.loc[((df.track_id == active_label) & (df.t == current_frame)),['centroid-0','centroid-1']].to_numpy()
    
            candidate_objects = df.loc[(df.t == (current_frame-1)),['track_id','centroid-0','centroid-1']]
            candidate_objects_array = candidate_objects.loc[:,['centroid-0','centroid-1']].to_numpy()
    
            dist_mat = distance_matrix(object_data,candidate_objects_array)
            iloc_min = np.nanargmin(dist_mat)
    
            connTrack = int(candidate_objects.iloc[iloc_min,:].track_id)
            
            
        else:
            viewer.status = 'Only one point is allowed for merging.'
            
        if connTrack > 0:
            
            # check if there is another branch that needs to be cleaned
            deadBranch = df.loc[((df.track_id==connTrack) & (df.t>=current_frame)),:]
            
            if len(deadBranch) > 0:
                
                # find new track number
                newTrack = gen.newTrack_number(df.track_id)
                
                # modify labels
                my_labels = gen.forward_labels(my_labels,df,current_frame,connTrack,newTrack)    
                
                # modify data frame
                df = gen.forward_df(df,current_frame,connTrack,newTrack)
                
    
            #####################################################################
            # change labels layer
            #####################################################################
    
            my_labels = gen.forward_labels(my_labels,df,current_frame,active_label,connTrack)    
            viewer.layers['Labels'].data = my_labels
    
            #####################################################################
            # modify data frame
            #####################################################################
            df = gen.forward_df(df,current_frame,active_label,connTrack)
            
            #####################################################################
            # remove tags from affected tracks
            #####################################################################
            if len(deadBranch) > 0:
                
                viewer = remove_tags(viewer, df,[active_label,connTrack,newTrack])
                
            else:
                
                viewer = remove_tags(viewer, df,[active_label,connTrack])
            
    
            viewer.status = f'Track {active_label} was merged with {connTrack}.'
            
            #####################################################################
            # change tracking layer
            #####################################################################
    
            # modify the data for the layer
            data,properties,graph = gen.trackData_from_df(df,col_list=gen_track_columns)
    
            # change tracks layer
            viewer.layers['Tracking'].data = data
            viewer.layers['Tracking'].properties = properties
            viewer.layers['Tracking'].graph = graph
            
    else:
        viewer.status = 'It is not possible to merge objects from the first frame.'
    
        
    return viewer,df

def connect_track(viewer,df,gen_track_columns):
    
    # developing connecting function
    
    # get images of objects
    my_labels = viewer.layers['Labels'].data
    
    # get the position in time
    current_frame = viewer.dims.current_step[0]
    
    # get my label
    active_label = viewer.layers['Labels'].selected_label
    
    if current_frame>0:
    
        connTrack=0
    
        # check if there is a point to merge too
        merge_to = viewer.layers['Helper Points'].data
    
        if len(merge_to)==1:
    
            merge_to = merge_to[0]
    
            if merge_to[0] == (current_frame-1):
    
                connTrack = my_labels[tuple(merge_to.astype(int))]
    
                viewer.layers['Helper Points'].data = []
    
            else:
                viewer.status = 'Connecting cell does not match'
    
        elif len(merge_to)==0:
    
            # find the closest object in the previous layer
            object_data = df.loc[((df.track_id == active_label) & (df.t == current_frame)),['centroid-0','centroid-1']].to_numpy()
    
            candidate_objects = df.loc[(df.t == (current_frame-1)),['track_id','centroid-0','centroid-1']]
            candidate_objects_array = candidate_objects.loc[:,['centroid-0','centroid-1']].to_numpy()
    
            dist_mat = distance_matrix(object_data,candidate_objects_array)
            iloc_min = np.nanargmin(dist_mat)
    
            connTrack = int(candidate_objects.iloc[iloc_min,:].track_id)
    
    
        else:
            viewer.status = 'Only one mother object is allowed to be connected.'
    
        if connTrack > 0:
    
            # check if there is another branch that needs to be cleaned
            sisterBranch = df.loc[((df.track_id==connTrack) & (df.t>=current_frame)),:]
    
            if len(sisterBranch) > 0:
    
                # find new track number
                newTrack_sister = gen.newTrack_number(df.track_id)
    
                # modify labels
                my_labels = gen.forward_labels(my_labels,df,current_frame,connTrack,newTrack_sister)    
    
                # modify data frame
                df = gen.forward_df(df,current_frame,connTrack,newTrack_sister,connectTo=connTrack)
    
    
            #####################################################################
            # change labels layer
            #####################################################################
            
            # find new track number
            newTrack = gen.newTrack_number(df.track_id)
    
            my_labels = gen.forward_labels(my_labels,df,current_frame,active_label,newTrack)    
            viewer.layers['Labels'].data = my_labels
    
            #####################################################################
            # modify data frame
            #####################################################################
            df = gen.forward_df(df,current_frame,active_label,newTrack,connectTo=connTrack)
    
            #####################################################################
            # change tracking layer
            #####################################################################
    
            # modify the data for the layer
            data,properties,graph = gen.trackData_from_df(df,col_list=gen_track_columns)
    
            # change tracks layer
            viewer.layers['Tracking'].data = data
            viewer.layers['Tracking'].properties = properties
            viewer.layers['Tracking'].graph = graph
    
    
            viewer.status = f'Track {active_label} was merged with {connTrack}.'
    
    else:
        viewer.status = 'It is not possible to connect objects from the first frame.'
        
    return viewer,df

def update_single_object(viewer,df,channel_list,object_properties,gen_track_columns):
    
    
    # get images of objects
    my_labels = viewer.layers['Labels'].data
    
    ########################################################
    # modify data frame
    ########################################################

    # get the position in time
    current_frame = viewer.dims.current_step[0]

    # get my label
    active_label = viewer.layers['Labels'].selected_label

    # calculate features of a new cell and store in the general data frame
    df = gen.update_dataFrame(channel_list,my_labels,df,current_frame,active_label,object_properties)

    ########################################################
    # modify tracking layer
    ########################################################
    # this acually could be done only per track if extraction of data takes a lot of time

    # modify the data for the layer
    data,properties,graph = gen.trackData_from_df(df,col_list=gen_track_columns)

    # change tracks layer
    viewer.layers['Tracking'].data = data
    viewer.layers['Tracking'].properties = properties
    viewer.layers['Tracking'].graph = graph
    
    ########################################################
    # modify labeling points
    ########################################################

    # collect the information
    sel_data = df.loc[df.accepted==True,:]
    accepted_points = np.array([sel_data['t'],sel_data['centroid-0'],sel_data['centroid-1']]).T 
    
    sel_data = df.loc[df.rejected==True,:]
    rejected_points = np.array([sel_data['t'],sel_data['centroid-0'],sel_data['centroid-1']]).T 
    
    sel_data = df.loc[df.promise==True,:]
    promise_points = np.array([sel_data['t'],sel_data['centroid-0'],sel_data['centroid-1']]).T 
    
    viewer.layers['Accepted Tracks'].data = accepted_points
    viewer.layers['Rejected Tracks'].data = rejected_points
    viewer.layers['Promising Tracks'].data = promise_points

    ########################################################
    # change viewer status
    ########################################################
    viewer.status = f'Frame {current_frame} was modified.' 
    
    return viewer,df

def remove_tags(viewer, df, list_of_tracks,list_of_tags = ['accepted','rejected'],list_of_layers = ['Accepted Tracks','Rejected Tracks']):

    '''
    Function to remove tags from specified tracks.
    
    input:
        viewer
        list_of_tracks
        list_of_tags
    output:
        viewer
    '''
    
    for my_tag,my_layer in zip(list_of_tags,list_of_layers):
        
        for my_track in list_of_tracks:
            
            # change status of this track
            df.loc[df.track_id == my_track,my_tag] = False
    
            # regenerate points
            selData=df.loc[df[my_tag] == True,:]
            selPoints = np.array([selData['t'],selData['centroid-0'],selData['centroid-1']]).T 
            
            # update viewer
            viewer.layers[my_layer].data = selPoints
    
    return viewer